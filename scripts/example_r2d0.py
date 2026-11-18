from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import utils_dir
from data_handle import get_states_from_data
import jax.numpy as jnp
from utils import ReplayMemory
import pickle
import jax
import time
import haiku as hk
from functools import partial
import DeLaN_model_static as delan
import optax


# -----------------------------
# Config
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

TRACKER_ROOT = (
    SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"
)

ROBOT_DIR = TRACKER_ROOT / "robot2_damage0"
VIDEO_DIR = ROBOT_DIR / "video"
ONLINE_DIR = ROBOT_DIR / "online"

SAVE_FIGURES = False
OUTPUT_DIR = SCRIPT_DIR / "robot2_damage0_all_inputs"

MOTIONS_TO_LOAD = [
        "rampUpwards",
        "rampDownwards",
        "Trapezoid",
]

def load_robot2_all_motions():
    dfs = {}
    
    for motion in MOTIONS_TO_LOAD:
        video_path = utils_dir.find_latest_video_for_motion(VIDEO_DIR, motion)

        if video_path is None:
            continue

        # This uses timestamp from the video to find the correct CSV
        df = utils_dir.load_df(
            label=motion,
            video_path=video_path,
            online_dir=ONLINE_DIR,
        )

        dfs[motion] = df

    if not dfs:
        raise RuntimeError("No motion data loaded for robot2_damage0.")

    return dfs


# -----------------------------
# Load data
# -----------------------------
dfs = load_robot2_all_motions()
print(dfs["rampUpwards"].keys())

# ignore data where angle is zero
dfs = {motion: df[df["angle[rad]"] != 0].copy()
    for motion, df in dfs.items()}
print("\nLoaded motions")

states = {motion: get_states_from_data(dfs[motion], ["angle", "endpoint", "pressure"], previous_states=0) for motion in dfs}

# slight preprocessing
for motion in MOTIONS_TO_LOAD:
    states[motion]["state"]["endpoint"]/=100
    states[motion]["state"]["pressure"]/=100
    # states[motion]["previous_state1"]["endpoint"]/=100
    # states[motion]["previous_state1"]["pressure"]/=100

def state_dict_to_matrix(state_dict):
    feature_order = ["angle", "endpoint", "pressure"]

    state_order = ["state"] + sorted(
        [key for key in state_dict if key.startswith("previous_state")],
        key=lambda x: int(x.replace("previous_state", ""))
    )

    parts = []
    columns = []

    for state_key in state_order:
        for name in feature_order:
            if name not in state_dict[state_key]:
                continue

            values = state_dict[state_key][name]
            parts.append(values)

            if name == "endpoint":
                columns.extend([
                    f"{state_key}_endpoint_dx",
                    f"{state_key}_endpoint_dy"
                ])
            else:
                columns.append(f"{state_key}_{name}")

    data = np.column_stack(parts)

    return data, columns


# randomize each motion, then unify all motions
all_data = []
columns = None

for motion in MOTIONS_TO_LOAD:
    data, columns = state_dict_to_matrix(states[motion])

    np.random.shuffle(data)

    all_data.append(data)

all_data = np.vstack(all_data)

pressure_idx = columns.index("state_pressure")
endpoint_dx_idx = columns.index("state_endpoint_dx")
endpoint_dy_idx = columns.index("state_endpoint_dy")
angle_idx = columns.index("state_angle")

# previous_pressure_idx = columns.index("previous_state1_pressure")
# previous_endpoint_dx_idx = columns.index("previous_state1_endpoint_dx")
# previous_endpoint_dy_idx = columns.index("previous_state1_endpoint_dy")
# previous_angle_idx = columns.index("previous_state1_angle")

targets = all_data[:, pressure_idx].reshape(-1,1)
states = all_data[:, [angle_idx, endpoint_dx_idx, endpoint_dy_idx]]


# train
print(f"There are {states.shape[0]} data in this attempt.")
print("Set 80%  data as the training set and 20% as the test set")
div = states.shape[0] * 8 // 10

train_states, test_states = states[:div, :], states[div:, :]
train_targets, test_targets = targets[:div, :], targets[div:, :] # input is the target.
train_states = train_states.astype(np.float32)
test_states = test_states.astype(np.float32)
train_targets = train_targets.astype(np.float32)
test_targets = test_targets.astype(np.float32)


'''
2. Set the model parameters/hyperparameters
'''
activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

actuator_dof = train_targets.shape[1]
config_dof = train_states.shape[1]
print(f"Config DOF = {config_dof}, Actuator DOF = {actuator_dof}")

hyper = {
    'n_dof': config_dof,
    'actuator_dof': actuator_dof,
    'n_width': 60,
    'n_depth': 3,
    'n_minibatch': 200, # 800.
    'activation1': 'softplus',
    'activation2': 'tanh',
    'learning_rate': 1e-04,
    'weight_decay': 3.2e-05,
    'max_epoch': int(6.0 * 1e3),
}

'''
3. Construct DeLaN static.
'''
t0 = time.perf_counter()

potential_fn = hk.transform(partial(delan.potential_energy, shape = (hyper["n_width"],)*hyper["n_depth"], \
                activation = activations[hyper['activation1']]))

input_fn = hk.transform(partial(delan.input_transform_matrix, shape = (hyper['n_width']//2,) * (hyper['n_depth']-1),\
                n_dof = config_dof, actuator_dof = actuator_dof, activation=activations[hyper['activation1']]))


mem_dim = ((config_dof,), (actuator_dof,))
mem = ReplayMemory(train_states.shape[0], hyper["n_minibatch"], mem_dim)
mem.add_samples([train_states, train_targets])


# Initialize parameters
sample_q = jnp.asarray(train_states[0], dtype=jnp.float32)
sample_tau = jnp.asarray(train_targets[0], dtype=jnp.float32)
rng = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(rng)

i_params = input_fn.init(key1, sample_q)
g_params = potential_fn.init(key2, sample_q)

params = {'potential': g_params, 'input_transform': i_params}

input_mat = input_fn.apply
pot_mat = potential_fn.apply

forward_model = delan.forward_model(params=params, key=None, pot_mat=pot_mat, input_mat=input_mat)

_ = forward_model(sample_q)

t_build = time.perf_counter() - t0
print(f"DeLaN Build Time    = {t_build:.2f}s")

'''
4. Generate and initialize optimizer
'''
t0 = time.perf_counter()
optimizer1 = optax.adamw(
    learning_rate=hyper['learning_rate'],
    weight_decay=hyper['weight_decay']
)

optimizer2 = optax.adamw(
    learning_rate=hyper['learning_rate'],
    weight_decay=hyper['weight_decay']
)

opt1 = optimizer1.init(params["potential"])
opt2 = optimizer2.init(params["input_transform"])

# loss_fn = partial(delan.loss_fn, pot_mat = pot_mat, input_mat=input_mat)
loss_fn = partial(
    delan.loss_fn,
    pot_mat=pot_mat,
    input_mat=input_mat,
    lambda_data=1,
)

def update_fn(params, opt1, opt2, q, tau):
    (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(params, q, tau)
    # print(grads)
    updates1, opt1 = optimizer1.update(grads["potential"], opt1, params["potential"])
    params["potential"] = optax.apply_updates(params["potential"], updates1)
    updates2, opt2 = optimizer2.update(grads["input_transform"], opt2, params["input_transform"])
    params["input_transform"] = optax.apply_updates(params["input_transform"], updates2)
    return params, opt1, opt2, logs



update_fn = jax.jit(update_fn)
_, _, _, logs = update_fn(params, opt1, opt2, train_states[:2], train_targets[:2])

t_build = time.perf_counter() - t0
print(f"Optimizer Build Time = {t_build:.2f}s")

'''
5. Start training loop
'''
t0_start = time.perf_counter()

train_losses = {"forward_loss": [],
                "forward_var": []}

test_losses = {"forward_loss": [],
               "forward_var": []}

print("")
epoch_i = 0
step = 50
while epoch_i < hyper['max_epoch']:
    n_batches = 0
    logs = jax.tree.map(lambda x: x * 0.0, logs)

    for data_batch in mem:
        t0_batch = time.perf_counter()

        q, tau = [jnp.array(x) for x in data_batch]        
        params, opt1, opt2, batch_logs = update_fn(params, opt1, opt2, q, tau)

        # Update logs:
        n_batches += 1
        print(batch_logs)
        logs = jax.tree.map(lambda x, y: x + y, logs, batch_logs)
        t_batch = time.perf_counter() - t0_batch


    # Update Epoch Loss & Computation Time:
    epoch_i += 1
    logs = jax.tree.map(lambda x: x / n_batches, logs)

    if epoch_i == 1 or np.mod(epoch_i, step) == 0:
        print("Epoch {0:05d}: ".format(epoch_i), end=" ")
        print("train: ", end=" ")
        print(f"Time = {time.perf_counter() - t0_start:05.1f}s", end=", ")
        print(f"For = {logs['forward_mean']:.3e} \u00B1 {1.96 * np.sqrt(logs['forward_var']):.2e}", end=";     ")
        train_losses["forward_loss"].append(logs['forward_mean'])
        train_losses["forward_var"].append(logs['forward_var'])

        # test loss computation:
        # (params, q, p, tau, q_next, p_next)
        test_loss, test_logs = loss_fn(params=params, q=test_states,  tau=test_targets)
        print("test: ", end=" ")
        print(f"For = {test_logs['forward_mean']:.3e} \u00B1 {1.96 * np.sqrt(test_logs['forward_var']):.2e}")
        test_losses["forward_loss"].append(test_logs['forward_mean'])
        test_losses['forward_var'].append(test_logs['forward_var'])

print(train_losses)
print(test_losses)

# Plot the loss picture
plt.style.use("seaborn-v0_8-whitegrid")
palette = plt.get_cmap('Set1')

iterations = list(range(0, hyper['max_epoch']+step, step))

f, ax = plt.subplots(1, 1)

# train forward loss
ax.plot(iterations, train_losses["forward_loss"], color=palette(1), label='train_forward_loss')
r1_forward = list(map(lambda x: x[0] - x[1], zip(train_losses['forward_loss'], train_losses['forward_var'])))
r2_forward = list(map(lambda x: x[0] + x[1], zip(train_losses['forward_loss'], train_losses['forward_var'])))
ax.fill_between(iterations, r1_forward, r2_forward, color=palette(1), alpha=0.2)


# test forward loss
ax.plot(iterations, test_losses["forward_loss"], color=palette(2), label='test_forward_loss')
t1_forward = list(map(lambda x: x[0] - x[1], zip(test_losses['forward_loss'], test_losses['forward_var'])))
t2_forward = list(map(lambda x: x[0] + x[1], zip(test_losses['forward_loss'], test_losses['forward_var'])))
ax.fill_between(iterations, t1_forward, t2_forward, color=palette(2), alpha=0.2)

ax.legend(loc='upper right')
ax.set_xlim(0, hyper['max_epoch']+1)
# ax.set_ylim(0, max(train_losses["forward_loss"][10:]))
ax.set_xlabel('epoch', fontsize=12)
ax.set_ylabel('loss', fontsize=12)
plt.show()

from os import path

basepath = path.dirname(__file__)
path = path.join(
    basepath,
    "..",
    "models",
    "static_models",
    "r2d0.jax"
)

with open(path, "wb") as file:
    pickle.dump(
        {"epoch": epoch_i,
            "hyper": hyper,
            "params": params},
        file)
