# implementation of the algorithm found in:
# Improving Meta-Learning Generalization with Activation-Based Early-Stopping
# https://arxiv.org/abs/2208.02377

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
import dir_util as du


# -----------------------------
# Activation / moment helpers
# -----------------------------
def apply_activation(z, activation_name: str):
    '''
    Applies an activation function by name.

    @param[in] z               : input array.
    @param[in] activation_name : activation function name.

    @return z_act : activated array.
    '''
    if activation_name == "softplus":
        return jax.nn.softplus(z)

    if activation_name == "tanh":
        return jnp.tanh(z)

    if activation_name == "sigmoid":
        return jax.nn.sigmoid(z)

    raise ValueError(f"Unknown activation: {activation_name}")


def get_mlp_layer_activations(
    module_params,
    module_name: str,
    q_batch,
    layer: int,
    activation_name: str = "softplus",
):
    '''
    Computes hidden activations of a Haiku MLP directly from parameters.

    This function manually applies the linear layers in the Haiku parameter
    dictionary and returns the activation output of the selected hidden layer.
    It does not use Haiku interceptors.

    @param[in] module_params   : parameter dictionary for one network.
    @param[in] module_name     : Haiku module name, e.g. "potential_energy".
    @param[in] q_batch         : input batch with shape (N, n_states).
    @param[in] layer           : hidden layer index to extract.
    @param[in] activation_name : activation function used by the MLP.

    @return z : hidden activation array with shape (N, n_neurons).
    '''
    z = jnp.asarray(q_batch)

    layer_key = f"{module_name}/~/linear_{layer}"

    if layer_key not in module_params:
        available = list(module_params.keys())
        raise ValueError(
            f"Layer {layer_key} not found. Available layers: {available}"
        )

    for k in range(layer + 1):
        key = f"{module_name}/~/linear_{k}"

        W = module_params[key]["w"]
        b = module_params[key]["b"]

        z = z @ W + b
        z = apply_activation(z, activation_name)

    return z


def get_hidden_activations(
    params,
    q_batch,
    param_key: str,
    layer: int,
    activation_name: str = "softplus",
):
    '''
    Gets hidden activations for the potential or input-transform network.

    @param[in] params          : full parameter dictionary.
    @param[in] q_batch         : input batch.
    @param[in] param_key       : "potential" or "input_transform".
    @param[in] layer           : hidden layer index to extract.
    @param[in] activation_name : activation function used by the MLP.

    @return activations : hidden activations for the selected network.
    '''
    if param_key == "potential":
        return get_mlp_layer_activations(
            module_params=params["potential"],
            module_name="potential_energy",
            q_batch=q_batch,
            layer=layer,
            activation_name=activation_name,
        )

    if param_key == "input_transform":
        return get_mlp_layer_activations(
            module_params=params["input_transform"],
            module_name="input_transform_matrix",
            q_batch=q_batch,
            layer=layer,
            activation_name=activation_name,
        )

    raise ValueError(
        f"Unknown param_key={param_key}. "
        "Expected 'potential' or 'input_transform'."
    )


def aggregate_moments(z):
    '''
    Computes activation moments used for activation-based early stopping.

    @param[in] z : activation matrix with shape (batch_size, feature_dim).

    @return moments : array containing [m1_hat, m2_hat, m3_hat, m4_hat].
    '''
    z = jnp.asarray(z)

    mean = jnp.mean(z, axis=0)
    second = (z.T @ z) / z.shape[0]

    diag_second = jnp.diag(second)
    offdiag_second = second - jnp.diag(diag_second)

    m1_hat = jnp.sum(mean)
    m2_hat = jnp.sum(mean ** 2)
    m3_hat = jnp.sum(diag_second)
    m4_hat = jnp.sum(offdiag_second)

    return jnp.array([m1_hat, m2_hat, m3_hat, m4_hat])


def compute_model_moments(
    params,
    q_batch,
    layer: int,
    activation_name: str = "softplus",
):
    '''
    Computes activation moments for both model subnetworks.

    @param[in] params          : full parameter dictionary.
    @param[in] q_batch         : input batch.
    @param[in] layer           : hidden layer index.
    @param[in] activation_name : activation function used by the MLP.

    @return moments : dictionary with potential and input_transform moments.
    '''
    return {
        "potential": aggregate_moments(
            get_hidden_activations(
                params=params,
                q_batch=q_batch,
                param_key="potential",
                layer=layer,
                activation_name=activation_name,
            )
        ),
        "input_transform": aggregate_moments(
            get_hidden_activations(
                params=params,
                q_batch=q_batch,
                param_key="input_transform",
                layer=layer,
                activation_name=activation_name,
            )
        ),
    }


@partial(jax.jit, static_argnames=("layer", "activation_name"))
def compute_all_moments(
    params_r1d1,
    params_r2d0,
    q_r1d1,
    q_r2d0,
    layer: int,
    activation_name: str,
):
    '''
    Computes activation moments for both models.

    @param[in] params_r1d1     : parameters for robot1_damage1 model.
    @param[in] params_r2d0     : parameters for robot2_damage0 model.
    @param[in] q_r1d1          : input batch for robot1_damage1.
    @param[in] q_r2d0          : input batch for robot2_damage0.
    @param[in] layer           : hidden layer index.
    @param[in] activation_name : activation function used by the MLP.

    @return moments : nested dictionary of activation moments.
    '''
    return {
        "r1d1": compute_model_moments(
            params=params_r1d1,
            q_batch=q_r1d1,
            layer=layer,
            activation_name=activation_name,
        ),
        "r2d0": compute_model_moments(
            params=params_r2d0,
            q_batch=q_r2d0,
            layer=layer,
            activation_name=activation_name,
        ),
    }


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
    '''
    Sets up a static DeLaN model with potential and input-transform networks.

    @param[in] train_states  : training input states.
    @param[in] train_targets : training targets.
    @param[in] test_states   : testing input states.
    @param[in] test_targets  : testing targets.
    @param[in] hyper         : dictionary of hyperparameters.
    @param[in] seed          : random seed.
    @param[in] model_name    : display name for the model.

    @return model : dictionary containing parameters, optimizers, data, and update fn.
    '''
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
    "robot2_damage0": TRACKER_ROOT / "robot2_damage0",
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
    '''
    Shuffles rows of a dataframe.

    @param[in] df   : input dataframe.
    @param[in] seed : random seed for reproducibility.

    @return df_shuffled : shuffled dataframe with reset index.
    '''
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)
df_r1d1 = shuffle_df(robot1_dfs_0deg["rampUpwards"])
df_r2d0 = shuffle_df(robot2_dfs_0deg["rampUpwards"])


r1d1_x, r1d1_y = du.extract_states_and_target(
    df_r1d1
)


r2d0_x, r2d0_y = du.extract_states_and_target(
    df_r2d0
)

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
# Train and compute moments
# -----------------------------
HIDDEN_LAYER = 2

MOMENT_LABELS = [
    "m1 (Σ E[z])",
    "m2 (Σ E[z]²)",
    "m3 (Σ E[z²])",
    "m4 (Σ off-diag E[zᵢzⱼ])",
]

moments_history = {
    "r1d1": {
        "potential": [],
        "input_transform": [],
    },
    "r2d0": {
        "potential": [],
        "input_transform": [],
    },
}

train_losses = {
    "r1d1": [],
    "r2d0": [],
}

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

        all_moments = compute_all_moments(
            params_r1d1=r1d1_model["params"],
            params_r2d0=r2d0_model["params"],
            q_r1d1=q_r1d1,
            q_r2d0=q_r2d0,
            layer=HIDDEN_LAYER,
            activation_name=hyper["activation1"],
        )

        for model_name in ["r1d1", "r2d0"]:
            for param_key in ["potential", "input_transform"]:
                moments_history[model_name][param_key].append(
                    np.asarray(all_moments[model_name][param_key])
                )

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

    if epoch_i == 1 or np.mod(epoch_i, step) == 0:
        t_elapsed = time.perf_counter() - t0_start

        print(
            f"Epoch {epoch_i:05d} | "
            f"t={t_elapsed:05.1f}s | "
            f"R1D1 loss={r1d1_logs_epoch['forward_mean']:.3e} | "
            f"R2D0 loss={r2d0_logs_epoch['forward_mean']:.3e}"
        )


# -----------------------------
# Plot moments
# -----------------------------
def to_array(history):
    '''
    Converts list of moment arrays into a stacked numpy array.

    @param[in] history : list of moment arrays.

    @return arr : array with shape (n_steps, 4).
    '''
    return np.stack(history, axis=0)


fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True)
fig.suptitle(f"Activation Moments — Hidden Layer {HIDDEN_LAYER}", fontsize=14)

colors = {
    "r1d1": "#e05c5c",
    "r2d0": "#5c8de0",
}

for row, param_key in enumerate(["potential", "input_transform"]):
    for col, moment_label in enumerate(MOMENT_LABELS):
        ax = axes[row, col]

        for model_name in ["r1d1", "r2d0"]:
            arr = to_array(moments_history[model_name][param_key])
            ax.plot(
                arr[:, col],
                color=colors[model_name],
                label=model_name,
                linewidth=0.8,
                alpha=0.85,
            )

        ax.set_title(moment_label, fontsize=9)

        if col == 0:
            ax.set_ylabel(param_key, fontsize=9)

        if row == 1:
            ax.set_xlabel("batch", fontsize=8)

        if row == 0 and col == 0:
            ax.legend(fontsize=8)

plt.tight_layout()
plt.show()