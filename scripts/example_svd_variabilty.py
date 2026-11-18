# implementation based on:
# Improving Meta-Learning Generalization with Activation-Based Early-Stopping
# https://arxiv.org/abs/2208.02377
#
# Modified to track angles between SVD primary directions of hidden-layer
# weight matrices across two DeLaN models.

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


# -----------------------------
# SVD direction helpers
# -----------------------------
def get_hidden_weight_matrix(params, param_key: str, layer: int):
    """
    Gets the hidden-layer weight matrix for a selected DeLaN subnetwork.

    param_key:
        "potential" or "input_transform"

    layer:
        hidden layer index
    """
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


def get_primary_direction_from_weights(W, r: int = 0, direction_type: str = "right"):
    """
    Computes the r-th SVD direction of a weight matrix.

    W = U S Vh

    direction_type:
        "right" -> use rows of Vh
        "left"  -> use columns of U

    r:
        primary direction index.
        r=0 is the first singular direction.
        r=1 is the second singular direction.
    """
    U, S, Vh = jnp.linalg.svd(W, full_matrices=False)

    max_rank = S.shape[0]

    if r >= max_rank:
        raise ValueError(
            f"Requested r={r}, but weight matrix only has rank size {max_rank}."
        )

    if direction_type == "right":
        return Vh[r]

    if direction_type == "left":
        return U[:, r]

    raise ValueError("direction_type must be 'right' or 'left'.")


def compute_direction_angle(v1, v2, degrees=True):
    """
    Computes the angle between two directions.

    Uses abs(cos(theta)) because SVD directions are sign-ambiguous:
    v and -v represent the same singular direction.
    """
    v1 = jnp.asarray(v1)
    v2 = jnp.asarray(v2)

    denom = jnp.linalg.norm(v1) * jnp.linalg.norm(v2)

    cos_theta = jnp.dot(v1, v2) / denom
    cos_theta = jnp.clip(jnp.abs(cos_theta), 0.0, 1.0)

    angle = jnp.arccos(cos_theta)

    if degrees:
        angle = angle * 180.0 / jnp.pi

    return angle


def compute_svd_direction_angle(
    params_model_1,
    params_model_2,
    param_key: str,
    layer: int,
    r: int,
    direction_type: str = "right",
    degrees: bool = True,
):
    """
    Computes the angle between the r-th SVD directions of one hidden layer
    from two models.
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

    v1 = get_primary_direction_from_weights(
        W=W1,
        r=r,
        direction_type=direction_type,
    )

    v2 = get_primary_direction_from_weights(
        W=W2,
        r=r,
        direction_type=direction_type,
    )

    return compute_direction_angle(v1, v2, degrees=degrees)


# -----------------------------
# Validation helper
# -----------------------------
def compute_validation_loss(model):
    loss, logs = model["loss_fn"](
        model["params"],
        jnp.asarray(model["test_states"], dtype=jnp.float32),
        jnp.asarray(model["test_targets"], dtype=jnp.float32),
    )

    return float(logs["forward_mean"])


def find_t_valid(val_losses, patience=50):
    """
    Finds early stopping point based on validation loss.
    """
    best = np.inf
    best_idx = 0

    for i, loss in enumerate(val_losses):
        if loss < best:
            best = loss
            best_idx = i

        if i - best_idx > patience:
            return best_idx

    return best_idx


# -----------------------------
# Model setup
# -----------------------------
def setup_static_delan(
    train_states,
    train_targets,
    test_states,
    test_targets,
    hyper,
    seed=42,
    model_name="model",
):
    activations = {
        "tanh": jnp.tanh,
        "softplus": jax.nn.softplus,
        "sigmoid": jax.nn.sigmoid,
    }

    config_dof = train_states.shape[1]
    actuator_dof = train_targets.shape[1]

    potential_fn = hk.transform(
        partial(
            delan.potential_energy,
            shape=(hyper["n_width"],) * hyper["n_depth"],
            activation=activations[hyper["activation1"]],
        )
    )

    input_fn = hk.transform(
        partial(
            delan.input_transform_matrix,
            shape=(hyper["n_width"] // 2,) * (hyper["n_depth"] - 1),
            n_dof=config_dof,
            actuator_dof=actuator_dof,
            activation=activations[hyper["activation1"]],
        )
    )

    mem_dim = ((config_dof,), (actuator_dof,))
    mem = ReplayMemory(train_states.shape[0], hyper["n_minibatch"], mem_dim)
    mem.add_samples([train_states, train_targets])

    sample_q = jnp.zeros((config_dof,), dtype=jnp.float32)

    rng = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(rng)

    i_params = input_fn.init(key1, sample_q)
    g_params = potential_fn.init(key2, sample_q)

    params = {
        "potential": g_params,
        "input_transform": i_params,
    }

    pot_mat = potential_fn.apply
    input_mat = input_fn.apply

    loss_fn = partial(
        delan.loss_fn,
        pot_mat=pot_mat,
        input_mat=input_mat,
        lambda_data=1,
    )

    optimizer1 = optax.adamw(
        learning_rate=hyper["learning_rate"],
        weight_decay=hyper["weight_decay"],
    )

    optimizer2 = optax.adamw(
        learning_rate=hyper["learning_rate"],
        weight_decay=hyper["weight_decay"],
    )

    opt1 = optimizer1.init(params["potential"])
    opt2 = optimizer2.init(params["input_transform"])

    def update_fn(params, opt1, opt2, q, tau):
        (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(
            params, q, tau
        )

        updates1, opt1 = optimizer1.update(
            grads["potential"],
            opt1,
            params["potential"],
        )

        updates2, opt2 = optimizer2.update(
            grads["input_transform"],
            opt2,
            params["input_transform"],
        )

        params = {
            "potential": optax.apply_updates(
                params["potential"],
                updates1,
            ),
            "input_transform": optax.apply_updates(
                params["input_transform"],
                updates2,
            ),
        }

        return params, opt1, opt2, logs

    update_fn = jax.jit(update_fn)

    _, _, _, logs = update_fn(
        params,
        opt1,
        opt2,
        jnp.asarray(train_states[:2], dtype=jnp.float32),
        jnp.asarray(train_targets[:2], dtype=jnp.float32),
    )

    print(f"{model_name} setup complete")
    print(f"Config DOF = {config_dof}, Actuator DOF = {actuator_dof}")

    return {
        "name": model_name,
        "params": params,
        "opt1": opt1,
        "opt2": opt2,
        "mem": mem,
        "loss_fn": loss_fn,
        "update_fn": update_fn,
        "logs": logs,
        "train_states": train_states,
        "train_targets": train_targets,
        "test_states": test_states,
        "test_targets": test_targets,
        "pot_mat": pot_mat,
        "input_mat": input_mat,
    }


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

r1d1_x, r1d1_y = du.extract_states_targets(df_r1d1)
r2d0_x, r2d0_y = du.extract_states_targets(df_r2d0)

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

    # SVD angle-tracking hyperparameters
    "svd_param_key": "potential",       # "potential" or "input_transform"
    "svd_layer": 1,                     # selected hidden layer
    "svd_primary_direction_r": 2,       # r=0 first direction, r=1 second direction
    "svd_direction_type": "left",      # "right" for Vh[r], "left" for U[:, r]
}


# -----------------------------
# Setup models
# -----------------------------
r1d1_model = setup_static_delan(
    r1d1_train_states,
    r1d1_train_targets,
    r1d1_test_states,
    r1d1_test_targets,
    hyper,
    seed=42,
    model_name="R1D1",
)

r2d0_model = setup_static_delan(
    r2d0_train_states,
    r2d0_train_targets,
    r2d0_test_states,
    r2d0_test_targets,
    hyper,
    seed=42,
    model_name="R2D0",
)

# print(r2d0_model['params']['potential'].keys())

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
# Train and track SVD angles
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

        # Track SVD primary-direction angle after this batch update
        angle = compute_svd_direction_angle(
            params_model_1=r1d1_model["params"],
            params_model_2=r2d0_model["params"],
            param_key=hyper["svd_param_key"],
            layer=hyper["svd_layer"],
            r=hyper["svd_primary_direction_r"],
            direction_type=hyper["svd_direction_type"],
            degrees=True,
        )

        svd_angle_history.append(float(angle))

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

        print(
            f"Epoch {epoch_i:05d} | "
            f"t={t_elapsed:05.1f}s | "
            f"R1D1 loss={r1d1_logs_epoch['forward_mean']:.3e} | "
            f"R2D0 loss={r2d0_logs_epoch['forward_mean']:.3e} | "
            f"SVD angle={svd_angle_history[-1]:.3f} deg"
        )


# -----------------------------
# Early-stopping point
# -----------------------------
t_valid_r1d1 = find_t_valid(val_losses["r1d1"])
t_valid_r2d0 = find_t_valid(val_losses["r2d0"])

t_valid_epoch = min(t_valid_r1d1, t_valid_r2d0)

# Convert epoch index to approximate batch index
batches_per_epoch = len(svd_angle_history) // hyper["max_epoch"]
t_valid_batch = t_valid_epoch * batches_per_epoch

print("t_valid R1D1 epoch:", t_valid_r1d1)
print("t_valid R2D0 epoch:", t_valid_r2d0)
print("Using t_valid epoch:", t_valid_epoch)
print("Approximate t_valid batch:", t_valid_batch)


# -----------------------------
# Plot SVD angle history
# -----------------------------
plt.figure(figsize=(10, 4))

plt.plot(
    svd_angle_history,
    linewidth=1.0,
    label="SVD direction angle",
)

plt.axvline(
    t_valid_batch,
    color="black",
    linestyle="--",
    linewidth=1.0,
    label="early stopping point",
)

plt.xlabel("batch")
plt.ylabel("angle between SVD directions (degrees)")

plt.title(
    f"SVD Direction Angle — "
    f"{hyper['svd_param_key']}, "
    f"layer {hyper['svd_layer']}, "
    f"r={hyper['svd_primary_direction_r']}, "
    f"{hyper['svd_direction_type']} direction"
)

plt.legend()
plt.tight_layout()
plt.show()


# -----------------------------
# Optional: plot train and validation losses
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
plt.show()