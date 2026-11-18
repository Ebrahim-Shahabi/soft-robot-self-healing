import numpy as np
import copy
import time
from pathlib import Path
from functools import partial

import jax
import jax.numpy as jnp
import haiku as hk
import optax

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import DeLaN_model_static as delan
from utils import ReplayMemory
import utils_dir as du
import tools


# -----------------------------
# SVD subspace helpers
# -----------------------------
def get_hidden_weight_matrix(params, param_key: str, layer: int):
    if param_key == "potential":
        module_params = params["potential"]
        module_name = "potential_energy"

    elif param_key == "input_transform":
        module_params = params["input_transform"]
        module_name = "input_transform_matrix"

    else:
        raise ValueError(
            f"Unknown param_key={param_key}. "
            "Expected 'potential' or 'input_transform'."
        )

    layer_key = f"{module_name}/~/linear_{layer}"

    if layer_key not in module_params:
        available = list(module_params.keys())
        raise ValueError(
            f"Layer {layer_key} not found. Available layers: {available}"
        )

    return module_params[layer_key]["w"]


def compute_svd_subspace_angles(
    params_model_1,
    params_model_2,
    param_key: str,
    layer: int,
    k: int = 2,
    direction_type: str = "right",
    degrees: bool = True,
    init_params = None
):
    """
    Computes principal angles between the top-k SVD subspaces
    of the selected hidden-layer weight matrices.

    W = U S Vh

    direction_type:
        "right" compares span(Vh[:k].T)
        "left" compares span(U[:, :k])
    """

    W1 = get_hidden_weight_matrix(
        params=params_model_1,
        param_key=param_key,
        layer=layer,
    )

    W2 = get_hidden_weight_matrix(
        params=params_model_2,
        param_key=param_key,
        layer=layer,
    )
    if init_params is not None:
        W1_init = get_hidden_weight_matrix(
            params=init_params,
            param_key=param_key,
            layer=layer,
        )

        W2_init = get_hidden_weight_matrix(
            params=init_params,
            param_key=param_key,
            layer=layer,
        )

        W1 = W1 - W1_init
        W2 = W2 - W2_init

    U1, S1, Vh1 = jnp.linalg.svd(W1, full_matrices=False)
    U2, S2, Vh2 = jnp.linalg.svd(W2, full_matrices=False)

    max_rank = min(S1.shape[0], S2.shape[0])

    if k > max_rank:
        raise ValueError(
            f"Requested k={k}, but maximum available rank is {max_rank}."
        )

    if direction_type == "right":
        Q1 = Vh1[:k].T
        Q2 = Vh2[:k].T

    elif direction_type == "left":
        Q1 = U1[:, :k]
        Q2 = U2[:, :k]

    else:
        raise ValueError("direction_type must be 'right' or 'left'.")

    # Singular values of Q1.T @ Q2 are cosines of principal angles
    cosines = jnp.linalg.svd(Q1.T @ Q2, compute_uv=False)
    cosines = jnp.clip(cosines, 0.0, 1.0)

    angles = jnp.arccos(cosines)

    if degrees:
        angles = angles * 180.0 / jnp.pi

    return angles


# -----------------------------
# Validation helpers
# -----------------------------
def compute_validation_loss(model):
    loss, logs = model["loss_fn"](
        model["params"],
        jnp.asarray(model["test_states"], dtype=jnp.float32),
        jnp.asarray(model["test_targets"], dtype=jnp.float32),
    )

    return float(logs["forward_mean"])


def find_t_valid(val_losses, patience=50):
    best = np.inf
    best_idx = 0

    for i, loss in enumerate(val_losses):
        if loss < best:
            best = loss
            best_idx = i

        if i - best_idx > patience:
            return best_idx

    return best_idx


def make_results_dir(script_dir, hyper, experiment_name):
    dirname = (
        f"svd_results_"
        f"{hyper['svd_param_key']}_"
        f"layer{hyper['svd_layer']}_"
        f"k{hyper['svd_num_directions']}_"
        f"{hyper['svd_direction_type']}_"
        f"{experiment_name}"
    )

    results_dir = script_dir / "results" / dirname
    results_dir.mkdir(parents=True, exist_ok=True)

    return results_dir




# -----------------------------
# Load data
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

TRACKER_ROOT = (
    SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"
)

ROBOTS = {
    "robot1_damage1": TRACKER_ROOT / "robot1_damage1_healed",
    "robot2_damage0": TRACKER_ROOT / "robot2",
}

robot1_dfs_0deg = {}
robot2_dfs_0deg = {}

runs_0deg = du.get_video_runs(
    video_dir=ROBOTS["robot1_damage1"] / "video",
    angle=0,
    run_name_filter=None,
)

for motion, video_path in runs_0deg.items():
    robot1_dfs_0deg[motion] = du.load_df(
        label=motion,
        video_path=video_path,
        online_dir=ROBOTS["robot1_damage1"] / "online",
    )

runs_0deg = du.get_video_runs(
    video_dir=ROBOTS["robot2_damage0"] / "video",
    angle=0,
    run_name_filter=None,
)

for motion, video_path in runs_0deg.items():
    robot2_dfs_0deg[motion] = du.load_df(
        label=motion,
        video_path=video_path,
        online_dir=ROBOTS["robot2_damage0"] / "online",
    )


def shuffle_df(df, seed: int = 42):
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


df_r1d1 = shuffle_df(robot1_dfs_0deg["rampUpwards"])
df_r2d0 = shuffle_df(robot2_dfs_0deg["rampUpwards"])

r1d1_x, r1d1_y = du.extract_states_and_target(df_r1d1)
r2d0_x, r2d0_y = du.extract_states_and_target(df_r2d0)

mask_r1d1 = r1d1_x[:, 0] != 0
mask_r2d0 = r2d0_x[:, 0] != 0

r1d1_x = r1d1_x[mask_r1d1]
r1d1_y = r1d1_y[mask_r1d1]

r2d0_x = r2d0_x[mask_r2d0]
r2d0_y = r2d0_y[mask_r2d0]


# -----------------------------
# Train / test split
# -----------------------------
r1d1_n_states = r1d1_x.shape[1]
r1d1_n_data = r1d1_x.shape[0]

print(f"For R1D1 there are {r1d1_n_states} states in this attempt.")
print("Set 80% data as the training set and 20% as the test set for R1D1")

div = r1d1_n_data * 8 // 10

r1d1_train_states = r1d1_x[:div, :].astype(np.float32)
r1d1_test_states = r1d1_x[div:, :].astype(np.float32)
r1d1_train_targets = r1d1_y[:div].reshape(-1, 1).astype(np.float32)
r1d1_test_targets = r1d1_y[div:].reshape(-1, 1).astype(np.float32)

r2d0_n_states = r2d0_x.shape[1]
r2d0_n_data = r2d0_x.shape[0]

print(f"For R2D0 there are {r2d0_n_states} states in this attempt.")
print("Set 80% data as the training set and 20% as the test set for R2D0")

div = r2d0_n_data * 8 // 10

r2d0_train_states = r2d0_x[:div, :].astype(np.float32)
r2d0_test_states = r2d0_x[div:, :].astype(np.float32)
r2d0_train_targets = r2d0_y[:div].reshape(-1, 1).astype(np.float32)
r2d0_test_targets = r2d0_y[div:].reshape(-1, 1).astype(np.float32)

actuator_dof = r2d0_train_targets.shape[1]
config_dof = r2d0_train_states.shape[1]

print(f"Config DOF = {config_dof}, Actuator DOF = {actuator_dof}")


# -----------------------------
# Hyperparameters
# -----------------------------
EXPERIMENT_NAME = "R1D1_R2D0"
hyper = {
    "n_dof": config_dof,
    "actuator_dof": actuator_dof,
    "n_width": 30,
    "n_depth": 3,
    "n_minibatch": 200,
    "activation1": "softplus",
    "activation2": "tanh",
    "learning_rate": 1e-4,
    "weight_decay": 3.2e-5,
    "max_epoch": int(6.0 * 1e3),

    # SVD subspace-angle hyperparameters
    "svd_param_key": "input_transform",        # "potential" or "input_transform"
    "svd_layer": 2,                      # hidden layer index
    "svd_num_directions": 2,             # compare span of top-k directions
    "svd_direction_type": "left",       # "right" = V directions, "left" = U directions
}


# -----------------------------
# Setup models
# -----------------------------
r1d1_model = tools.setup_static_delan(
    r1d1_train_states,
    r1d1_train_targets,
    r1d1_test_states,
    r1d1_test_targets,
    hyper,
    seed=42,
    model_name="R1D1",
)

r2d0_model = tools.setup_static_delan(
    r2d0_train_states,
    r2d0_train_targets,
    r2d0_test_states,
    r2d0_test_targets,
    hyper,
    seed=42,
    model_name="R2D0",
)


# -----------------------------
# Force same initialization
# -----------------------------
r2d0_model["params"] = copy.deepcopy(r1d1_model["params"])

optimizer = optax.adamw(
    learning_rate=hyper["learning_rate"],
    weight_decay=hyper["weight_decay"],
)

r1d1_model["opt1"] = optimizer.init(r1d1_model["params"]["potential"])
r1d1_model["opt2"] = optimizer.init(r1d1_model["params"]["input_transform"])

r2d0_model["opt1"] = optimizer.init(r2d0_model["params"]["potential"])
r2d0_model["opt2"] = optimizer.init(r2d0_model["params"]["input_transform"])


# -----------------------------
# Train and track SVD subspace angles
# -----------------------------
train_losses = {
    "r1d1": [],
    "r2d0": [],
}

val_losses = {
    "r1d1": [],
    "r2d0": [],
}

svd_angle_history = []

t0_start = time.perf_counter()
step = 50
epoch_i = 0
init_params = copy.deepcopy(r1d1_model["params"])
while epoch_i < hyper["max_epoch"]:

    r1d1_logs_epoch = jax.tree.map(lambda x: x * 0.0, r1d1_model["logs"])
    r2d0_logs_epoch = jax.tree.map(lambda x: x * 0.0, r2d0_model["logs"])

    n_batches = 0

    for (q_r1d1, tau_r1d1), (q_r2d0, tau_r2d0) in zip(
        r1d1_model["mem"],
        r2d0_model["mem"],
    ):
        q_r1d1 = jnp.asarray(q_r1d1, dtype=jnp.float32)
        tau_r1d1 = jnp.asarray(tau_r1d1, dtype=jnp.float32)

        q_r2d0 = jnp.asarray(q_r2d0, dtype=jnp.float32)
        tau_r2d0 = jnp.asarray(tau_r2d0, dtype=jnp.float32)

        r1d1_model["params"], r1d1_model["opt1"], r1d1_model["opt2"], r1d1_logs = (
            r1d1_model["update_fn"](
                r1d1_model["params"],
                r1d1_model["opt1"],
                r1d1_model["opt2"],
                q_r1d1,
                tau_r1d1,
            )
        )

        r2d0_model["params"], r2d0_model["opt1"], r2d0_model["opt2"], r2d0_logs = (
            r2d0_model["update_fn"](
                r2d0_model["params"],
                r2d0_model["opt1"],
                r2d0_model["opt2"],
                q_r2d0,
                tau_r2d0,
            )
        )

        angles = compute_svd_subspace_angles(
            init_params=init_params,
            params_model_1=r1d1_model["params"],
            params_model_2=r2d0_model["params"],
            param_key=hyper["svd_param_key"],
            layer=hyper["svd_layer"],
            k=hyper["svd_num_directions"],
            direction_type=hyper["svd_direction_type"],
            degrees=True,
        )

        svd_angle_history.append(np.asarray(angles))

        r1d1_logs_epoch = jax.tree.map(
            lambda x, y: x + y,
            r1d1_logs_epoch,
            r1d1_logs,
        )

        r2d0_logs_epoch = jax.tree.map(
            lambda x, y: x + y,
            r2d0_logs_epoch,
            r2d0_logs,
        )

        n_batches += 1

    epoch_i += 1

    r1d1_logs_epoch = jax.tree.map(
        lambda x: x / n_batches,
        r1d1_logs_epoch,
    )

    r2d0_logs_epoch = jax.tree.map(
        lambda x: x / n_batches,
        r2d0_logs_epoch,
    )

    train_losses["r1d1"].append(float(r1d1_logs_epoch["forward_mean"]))
    train_losses["r2d0"].append(float(r2d0_logs_epoch["forward_mean"]))

    val_losses["r1d1"].append(compute_validation_loss(r1d1_model))
    val_losses["r2d0"].append(compute_validation_loss(r2d0_model))

    if epoch_i == 1 or np.mod(epoch_i, step) == 0:
        t_elapsed = time.perf_counter() - t0_start
        latest_angles = svd_angle_history[-1]

        print(
            f"Epoch {epoch_i:05d} | "
            f"t={t_elapsed:05.1f}s | "
            f"R1D1 loss={r1d1_logs_epoch['forward_mean']:.3e} | "
            f"R2D0 loss={r2d0_logs_epoch['forward_mean']:.3e} | "
            f"SVD angles={latest_angles}"
        )


# -----------------------------
# Early-stopping point
# -----------------------------
svd_angle_history = np.stack(svd_angle_history, axis=0)

t_valid_r1d1 = find_t_valid(val_losses["r1d1"])
t_valid_r2d0 = find_t_valid(val_losses["r2d0"])

t_valid_epoch = min(t_valid_r1d1, t_valid_r2d0)

batches_per_epoch = len(svd_angle_history) // hyper["max_epoch"]
t_valid_batch = t_valid_epoch * batches_per_epoch

print("t_valid R1D1 epoch:", t_valid_r1d1)
print("t_valid R2D0 epoch:", t_valid_r2d0)
print("Using t_valid epoch:", t_valid_epoch)
print("Approximate t_valid batch:", t_valid_batch)
# -----------------------------
# Save directory
# -----------------------------
results_dir = make_results_dir(
    SCRIPT_DIR,
    hyper,
    EXPERIMENT_NAME,
)
print(f"Saving results to: {results_dir}")

svd_similarity_history = np.cos(np.deg2rad(svd_angle_history))

# -----------------------------
# Save text summary
# -----------------------------
summary_path = results_dir / "results_summary.txt"

with open(summary_path, "w") as f:
    f.write("SVD Subspace Angle Tracking Results\n")
    f.write("=" * 50 + "\n\n")

    f.write("Configuration\n")
    f.write("-" * 50 + "\n")
    f.write(f"svd_param_key       : {hyper['svd_param_key']}\n")
    f.write(f"svd_layer           : {hyper['svd_layer']}\n")
    f.write(f"svd_num_directions  : {hyper['svd_num_directions']}\n")
    f.write(f"svd_direction_type  : {hyper['svd_direction_type']}\n")
    f.write(f"activation          : {hyper['activation1']}\n")
    f.write(f"learning_rate       : {hyper['learning_rate']}\n")
    f.write(f"weight_decay        : {hyper['weight_decay']}\n")
    f.write(f"max_epoch           : {hyper['max_epoch']}\n")
    f.write(f"n_minibatch         : {hyper['n_minibatch']}\n\n")

    f.write("Early Stopping\n")
    f.write("-" * 50 + "\n")
    f.write(f"t_valid_r1d1_epoch  : {t_valid_r1d1}\n")
    f.write(f"t_valid_r2d0_epoch  : {t_valid_r2d0}\n")
    f.write(f"t_valid_epoch_used  : {t_valid_epoch}\n")
    f.write(f"t_valid_batch       : {t_valid_batch}\n\n")

    f.write("Losses at t_valid\n")
    f.write("-" * 50 + "\n")
    f.write(f"R1D1 train loss     : {train_losses['r1d1'][t_valid_epoch]:.8e}\n")
    f.write(f"R2D0 train loss     : {train_losses['r2d0'][t_valid_epoch]:.8e}\n")
    f.write(f"R1D1 val loss       : {val_losses['r1d1'][t_valid_epoch]:.8e}\n")
    f.write(f"R2D0 val loss       : {val_losses['r2d0'][t_valid_epoch]:.8e}\n\n")

    f.write("SVD Angles and Similarities at t_valid\n")
    f.write("-" * 50 + "\n")

    for i in range(hyper["svd_num_directions"]):
        angle_i = svd_angle_history[t_valid_batch, i]
        sim_i = svd_similarity_history[t_valid_batch, i]

        f.write(f"principal angle {i + 1} degrees : {angle_i:.8f}\n")
        f.write(f"similarity {i + 1}              : {sim_i:.8f}\n")

    f.write("\nFinal SVD Angles and Similarities\n")
    f.write("-" * 50 + "\n")

    for i in range(hyper["svd_num_directions"]):
        angle_i = svd_angle_history[-1, i]
        sim_i = svd_similarity_history[-1, i]

        f.write(f"final principal angle {i + 1} degrees : {angle_i:.8f}\n")
        f.write(f"final similarity {i + 1}              : {sim_i:.8f}\n")

    f.write("\nSaved Files\n")
    f.write("-" * 50 + "\n")
    f.write("svd_angles.png\n")
    f.write("svd_similarity.png\n")
    f.write("losses.png\n")
    f.write("results_summary.txt\n")
        
    # -----------------------------
    # Plot SVD subspace principal angles
    # -----------------------------
    plt.figure(figsize=(10, 4))

    for i in range(hyper["svd_num_directions"]):
        plt.plot(
            svd_angle_history[:, i],
            linewidth=1.0,
            label=f"principal angle {i + 1}",
        )

    plt.axvline(
        t_valid_batch,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="early stopping point",
    )

    plt.xlabel("batch")
    plt.ylabel("principal angle (degrees)")
    plt.title(
        f"SVD Subspace Angles — "
        f"{hyper['svd_param_key']}, "
        f"layer {hyper['svd_layer']}, "
        f"k={hyper['svd_num_directions']}, "
        f"{hyper['svd_direction_type']} directions"
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "svd_angles.png", dpi=300, bbox_inches="tight")

    # -----------------------------
    # Plot similarity values instead of angles
    # -----------------------------
    # svd_similarity_history = np.cos(np.deg2rad(svd_angle_history))

    plt.figure(figsize=(10, 4))

    for i in range(hyper["svd_num_directions"]):
        plt.plot(
            svd_similarity_history[:, i],
            linewidth=1.0,
            label=f"subspace similarity {i + 1}",
        )

    plt.axvline(
        t_valid_batch,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="early stopping point",
    )

    plt.xlabel("batch")
    plt.ylabel("cosine similarity")
    plt.ylim(0, 1.05)
    plt.title(
        f"SVD Subspace Similarity — "
        f"{hyper['svd_param_key']}, "
        f"layer {hyper['svd_layer']}, "
        f"k={hyper['svd_num_directions']}"
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "svd_similarity.png", dpi=300, bbox_inches="tight")

    # -----------------------------
    # Optional loss plot
    # -----------------------------
    plt.figure(figsize=(10, 4))

    plt.plot(train_losses["r1d1"], label="R1D1 train loss")
    plt.plot(train_losses["r2d0"], label="R2D0 train loss")
    plt.plot(val_losses["r1d1"], label="R1D1 validation loss")
    plt.plot(val_losses["r2d0"], label="R2D0 validation loss")

    plt.axvline(
        t_valid_epoch,
        color="black",
        linestyle="--",
        linewidth=1.0,
        label="early stopping point",
    )

    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training and validation losses")
    plt.legend()
    plt.tight_layout()
    plt.savefig(results_dir / "losses.png", dpi=300, bbox_inches="tight")
    plt.show()