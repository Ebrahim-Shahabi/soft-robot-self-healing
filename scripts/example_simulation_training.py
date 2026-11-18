import numpy as np
import pandas as pd
import jax.numpy as jnp
import random
import jax
from utils import ReplayMemory
import time 
import haiku as hk
from functools import partial
import DeLaN_model_v4 as delan
import pickle 
from pathlib import Path
import matplotlib
# matplotlib.use("Agg")  # use this if you only want to save images
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import optax

columns = [
    "time_s", "u_MPa",
    "x0_deg", "x1_deg", "x2_deg", "x3_deg",
    "dx0_deg_s", "dx1_deg_s", "dx2_deg_s", "dx3_deg_s",
    "s0x_m", "s1x_m", "s2x_m", "s3x_m", "sex_m",
    "ds0x_m_s", "ds1x_m_s", "ds2x_m_s", "ds3x_m_s", "dsex_m_s",
    "s0y_m", "s1y_m", "s2y_m", "s3y_m", "sey_m",
    "ds0y_m_s", "ds1y_m_s", "ds2y_m_s", "ds3y_m_s", "dsey_m_s",
    "s0z_m", "s1z_m", "s2z_m", "s3z_m", "sez_m",
    "ds0z_m_s", "ds1z_m_s", "ds2z_m_s", "ds3z_m_s", "dsez_m_s",
]

files = {
    "first": "first_robot_prbs_input.txt",
    "second": "second_robot_prbs_input.txt",
    "third": "third_robot_prbs_input.txt",
    "fourth": "fourth_robot_prbs_input.txt",
    "fifth": "fifth_robot_prbs_input.txt",
}



raw_data = {}
for robot_name, file_path in files.items():
    df_raw = pd.read_csv(file_path, sep=r"\s+", comment="#", names=columns)
    raw_data[robot_name] = {
        "time": df_raw["time_s"].to_numpy(),
        "pressure": df_raw["u_MPa"].to_numpy() * 10000,
        "angle_deg": df_raw["x3_deg"].to_numpy(),
    }

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Left: pressure signal sent to the 'first' robot
ax = axes[0]
ax.plot(raw_data["first"]["time"], raw_data["first"]["pressure"], color="#2563eb", linewidth=1)
ax.set_xlabel("time [s]")
ax.set_ylabel("pressure")
ax.set_title("Pressure signal — 'first' robot")
ax.grid(True, alpha=0.3)

# Right: bending angle vs pressure, all robots overlaid
ax = axes[1]
palette = plt.get_cmap("tab10")
for i, (robot_name, d) in enumerate(raw_data.items()):
    ax.scatter(d["pressure"], d["angle_deg"], s=4, alpha=0.4, color=palette(i), label=robot_name)
ax.set_xlabel("pressure")
ax.set_ylabel("bending angle [deg]")
ax.set_title("Bending angle vs pressure — all robots")
ax.legend(markerscale=3)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parents[0] / "models" / "models" / "simulation_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

'''
2. Set the model parameters/hyperparameters -- built once, shared by every robot
'''

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

config_dof = 4      # [ex, ey, ez, angle]
actuator_dof = 1     # [pressure]
print(f"Config DOF = {config_dof}, Actuator DOF = {actuator_dof}")

save_model = True
hyper = {
    'n_dof': config_dof,
    'actuator_dof': actuator_dof,
    'n_width': 64,
    'n_depth': 3,
    'n_minibatch': 400,
    'diagonal_epsilon': 0.01,
    'diagonal_shift': 0.005,
    'activation1': 'softplus',
    'activation2': 'tanh',
    'activation3': 'sigmoid',
    'learning_rate': 2.7e-04,
    'weight_decay': 3.2e-05,
    'max_epoch': int(6.0 * 1e3),
}

# 1. Construct DeLaN (architecture is the same for every robot):

lagrangian_fn = hk.transform(partial(
    delan.structured_lagrangian_fn,
    n_dof=hyper['n_dof'],
    shape=(hyper['n_width'],) * hyper['n_depth'],
    activation=activations[hyper['activation1']],
    epsilon=hyper['diagonal_epsilon'],
    shift=hyper['diagonal_shift'],
))

dissipative_fn = hk.transform(partial(
    delan.dissipative_matrix,
    n_dof=hyper['n_dof'],
    shape=(5,) * 3,
    activation=activations[hyper['activation2']]
))

input_mat_fn = hk.transform(partial(
    delan.input_transform_matrix,
    n_dof=hyper['n_dof'],
    actuator_dof=hyper['actuator_dof'],
    shape=(hyper['n_width']//2,) * (hyper['n_depth']-1),
    activation=activations[hyper['activation1']]
))

lagrangian = lagrangian_fn.apply
dissipative_mat = dissipative_fn.apply
input_mat = input_mat_fn.apply

loss_fn = partial(
    delan.loss_fn_experiment,
    lagrangian=lagrangian,
    dissipative_mat=dissipative_mat,
    input_mat=input_mat,
    n_dof=config_dof)

optimizer1 = optax.adamw(learning_rate=hyper['learning_rate'], weight_decay=hyper['weight_decay'])
optimizer2 = optax.adamw(learning_rate=hyper['learning_rate'], weight_decay=hyper['weight_decay'])
optimizer3 = optax.adamw(learning_rate=hyper['learning_rate'], weight_decay=hyper['weight_decay'])

def update_fn(params, opt1, opt2, opt3, q, dq, tau, q_next, dq_next, time_step):
    (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(params, q, dq, tau, q_next, dq_next, time_step=time_step)
    updates1, opt1 = optimizer1.update(grads["lagrangian"], opt1, params["lagrangian"])
    params["lagrangian"] = optax.apply_updates(params["lagrangian"], updates1)
    updates2, opt2 = optimizer2.update(grads["dissipative"], opt2, params["dissipative"])
    params["dissipative"] = optax.apply_updates(params["dissipative"], updates2)
    updates3, opt3 = optimizer3.update(grads["input_transform"], opt3, params["input_transform"])
    params["input_transform"] = optax.apply_updates(params["input_transform"], updates3)
    return params, opt1, opt2, opt3, logs

update_fn = jax.jit(update_fn)

rng = jax.random.PRNGKey(42)

W0 = None
all_params = {}
all_train_losses = {}
all_test_losses = {}

for robot_name, file_path in files.items():

    if robot_name == "fifth":
        print(f"\n========== Robot: {robot_name} ==========")
        print("Skipping 'fifth' here -- held out for the subspace adaptation script.")
        continue

    print(f"\n========== Robot: {robot_name} ==========")

    df = pd.read_csv(file_path, sep=r"\s+", comment="#", names=columns)

    time_ = df["time_s"].to_numpy()
    tau = df["u_MPa"].to_numpy().reshape(-1, 1) * 10000

    ex = df["sex_m"].to_numpy() / 100
    ey = df["sey_m"].to_numpy() / 100
    ez = df["sez_m"].to_numpy() / 100

    dex = df["dsex_m_s"].to_numpy() / 100
    dey = df["dsey_m_s"].to_numpy() / 100
    dez = df["dsez_m_s"].to_numpy() / 100

    angle = np.deg2rad(df["x3_deg"].to_numpy())
    dangle = np.deg2rad(df["dx3_deg_s"].to_numpy())

    states = np.column_stack((ex, ey, ez, angle, dex, dey, dez, dangle))
    n_states = states.shape[1]
    time_steps = time_[1:] - time_[:-1]

    data = states[:-1, :].copy()
    target = states[1:, :].copy()
    data = np.column_stack((time_steps, data, tau[:-1], target))
    random.shuffle(data)

    states = np.array(data[:, 1:n_states + 1])
    targets = np.array(data[:, -n_states:])
    inputs = np.array(data[:, n_states + 1]).reshape(-1, 1)
    time_steps = np.array(data[:, 0]).reshape(-1, 1)

    print(f"There are {states.shape[0]} data points for {robot_name}.")
    div = states.shape[0] * 8 // 10

    train_states, test_states = states[:div, :], states[div:, :]
    train_targets, test_targets = targets[:div, :], targets[div:, :]
    train_inputs, test_inputs = inputs[:div], inputs[div:]
    train_time_steps, test_time_steps = time_steps[:div], time_steps[div:]

    train_q, train_dq = jnp.split(train_states, 2, axis=1)
    train_q_next, train_dq_next = jnp.split(train_targets, 2, axis=1)
    test_q, test_dq = jnp.split(test_states, 2, axis=1)
    test_q_next, test_dq_next = jnp.split(test_targets, 2, axis=1)

    rng, init_key = jax.random.split(rng)

    mem_dim = ((config_dof,), (config_dof,), (actuator_dof,), (config_dof,), (config_dof,), (train_time_steps.shape[1],))
    mem = ReplayMemory(train_q.shape[0], hyper["n_minibatch"], mem_dim)
    mem.add_samples([train_q, train_dq, train_inputs, train_q_next, train_dq_next, train_time_steps])

    q, dq, tau_b, q_next, dq_next, _ = [jnp.array(x) for x in next(iter(mem))]

    if robot_name == "first":
        # Train from scratch -- this becomes W0
        l_params = lagrangian_fn.init(init_key, q[0], dq[0])
        d_params = dissipative_fn.init(init_key, dq[0])
        i_params = input_mat_fn.init(init_key, q[0])
        params = {"lagrangian": l_params, "dissipative": d_params, 'input_transform': i_params}
    else:
        # Initialize from a copy of W0, then fine-tune on this robot's data
        params = jax.tree_util.tree_map(lambda x: x, W0)

    forward_model = delan.forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat,
                                        input_mat=input_mat, n_dof=config_dof)
    state0 = jnp.concatenate([q[0], dq[0]])
    _ = forward_model(state0, tau_b[0])

    opt1 = optimizer1.init(params["lagrangian"])
    opt2 = optimizer2.init(params["dissipative"])
    opt3 = optimizer3.init(params["input_transform"])

    _, _, _, _, logs = update_fn(params, opt1, opt2, opt3, train_q[:2], train_dq[:2], train_inputs[:2], train_q_next[:2], train_dq_next[:2], time_step=train_time_steps[:2])

    t0_start = time.perf_counter()
    train_losses = {"forward_loss": [], "forward_var": []}
    test_losses = {"forward_loss": [], "forward_var": []}

    epoch_i = 0
    step = 50
    while epoch_i < hyper['max_epoch']:
        n_batches = 0
        logs = jax.tree.map(lambda x: x * 0.0, logs)

        for data_batch in mem:
            q, dq, tau_b, q_next, dq_next, time_steps_b = [jnp.array(x) for x in data_batch]
            params, opt1, opt2, opt3, batch_logs = update_fn(params, opt1, opt2, opt3, q, dq, tau_b, q_next, dq_next, time_step=time_steps_b)
            n_batches += 1
            logs = jax.tree.map(lambda x, y: x + y, logs, batch_logs)

        epoch_i += 1
        logs = jax.tree.map(lambda x: x / n_batches, logs)

        if epoch_i == 1 or np.mod(epoch_i, step) == 0:
            print(f"[{robot_name}] Epoch {epoch_i:05d}: ", end=" ")
            print(f"Time = {time.perf_counter() - t0_start:05.1f}s", end=", ")
            print(f"train For = {logs['forward_mean']:.3e} \u00B1 {1.96 * np.sqrt(logs['forward_var']):.2e}", end=";     ")
            train_losses["forward_loss"].append(logs['forward_mean'])
            train_losses["forward_var"].append(logs['forward_var'])

            test_loss, test_logs = loss_fn(params=params, q=test_q, qd=test_dq, tau=test_inputs, q_next=test_q_next, qd_next=test_dq_next, time_step=test_time_steps)
            print(f"test For = {test_logs['forward_mean']:.3e} \u00B1 {1.96 * np.sqrt(test_logs['forward_var']):.2e}")
            test_losses["forward_loss"].append(test_logs['forward_mean'])
            test_losses['forward_var'].append(test_logs['forward_var'])

    if robot_name == "first":
        W0 = params  # store for the other robots to initialize from

    all_params[robot_name] = params
    all_train_losses[robot_name] = train_losses
    all_test_losses[robot_name] = test_losses

    plt.style.use("seaborn-v0_8-whitegrid")
    palette = plt.get_cmap('Set1')
    iterations = list(range(0, hyper['max_epoch'] + step, step))

    f, ax = plt.subplots(1, 1)
    ax.plot(iterations, train_losses["forward_loss"], color=palette(1), label='train_forward_loss')
    r1 = list(map(lambda x: x[0] - x[1], zip(train_losses['forward_loss'], train_losses['forward_var'])))
    r2 = list(map(lambda x: x[0] + x[1], zip(train_losses['forward_loss'], train_losses['forward_var'])))
    ax.fill_between(iterations, r1, r2, color=palette(1), alpha=0.2)

    ax.plot(iterations, test_losses["forward_loss"], color=palette(2), label='test_forward_loss')
    t1 = list(map(lambda x: x[0] - x[1], zip(test_losses['forward_loss'], test_losses['forward_var'])))
    t2 = list(map(lambda x: x[0] + x[1], zip(test_losses['forward_loss'], test_losses['forward_var'])))
    ax.fill_between(iterations, t1, t2, color=palette(2), alpha=0.2)

    ax.legend(loc='upper right')
    ax.set_xlim(0, hyper['max_epoch'] + 1)
    ax.set_xlabel('epoch', fontsize=12)
    ax.set_ylabel('loss', fontsize=12)
    ax.set_title(robot_name)
    plt.show()

    if save_model:
        with open(MODEL_DIR / f"{robot_name}_delan.jax", "wb") as file:
            pickle.dump({"epoch": epoch_i, "hyper": hyper, "params": params}, file)
        print(f"Saved {robot_name} model to {MODEL_DIR / f'{robot_name}_delan.jax'}")

print("\nDone. Trained robots:", list(all_params.keys()))