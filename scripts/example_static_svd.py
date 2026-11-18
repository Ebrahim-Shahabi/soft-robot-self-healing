import pandas as pd
import matplotlib
# matplotlib.use("Agg")  # use this if you only want to save images
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from os import path
import random
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
import time
import optax
from utils import ReplayMemory
import pickle
import os
import DeLaN_model_static_svd as delan
from haiku import data_structures as ds

RANK = None      # None = full rank; or 32/64 to compress
TRAIN_BIAS = True
'''
1. Prepare the data
'''
basepath = path.dirname(__file__)
csv_path = path.join(
    basepath,
    "../",
    "data",
    "experiments",
    "online_output20260311-125118.csv"
)

# Read the CSV file
df = pd.read_csv(csv_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()
# Clean whitespace inside FOUND columns too
df["FOUND_INITIAL"] = df["FOUND_INITIAL"].astype(str).str.strip()
df["FOUND_ENDPOINT"] = df["FOUND_ENDPOINT"].astype(str).str.strip()


# shuffle the data 
input = df["pressure[mBar]"]/100 # better scaling.

# take only x
pe = (df["xe[m]"] - df["x0[m]"])/100

q = np.array(df["angle[rad]"]).reshape(-1,1)
q = np.column_stack((pe, q))

nq_ = q.shape[1]

data = q[:-1,:].copy()
data = np.column_stack((data, input[:-1]))
random.shuffle(data)

states = data[:,:nq_]
targets = data[:,nq_:] # target is the input at this case.

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
1. Load model. This is a transfer learning method. 
The model is pretrained on a different dataset.
'''
t0 = time.perf_counter()
# Load model
model_path = os.path.join(os.getcwd(), "models/static_models", "r2d0.jax")

with open(model_path, "rb") as f:
    data = pickle.load(f)

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

hyper = data["hyper"]
params = data["params"]
print("hyperparameters:", hyper)
actuator_dof = train_targets.shape[1]
config_dof = train_states.shape[1]

# transform the functions to haiku modules.
potential_fn = hk.transform(partial(delan.potential_energy, shape = (hyper["n_width"],)*hyper["n_depth"], \
                activation = activations[hyper['activation1']]))

input_fn = hk.transform(partial(delan.input_transform_matrix, shape = (hyper['n_width']//2,) * (hyper['n_depth']-1),\
                n_dof = config_dof, actuator_dof = actuator_dof, activation=activations[hyper['activation1']]))



# store the data in easier format.
mem_dim = ((config_dof,), (actuator_dof,))
mem = ReplayMemory(train_states.shape[0], hyper["n_minibatch"], mem_dim)
mem.add_samples([train_states, train_targets])

rng = jax.random.PRNGKey(42)
rng_key, init_key = jax.random.split(rng)

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
_ = forward_model(q[0])

    

t_build = time.perf_counter() - t0
print(f"DeLaN Build Time    = {t_build:.2f}s")


# # 2. Generate and initialize the optimizer
t0 = time.perf_counter()


pot_dict = {'potential_energy/linear_0': True, 'potential_energy/linear_1': True, 'potential_energy/linear_2': True, 'potential_energy/linear_3': True}
input_dict = {'input_transform_matrix/linear_0': True, 'input_transform_matrix/linear_1': True, 'input_transform_matrix/linear_2': True, 'input_transform_matrix/linear_3': True}   


mask_l = delan.make_svd_mask(params["potential"],dix=pot_dict, train_bias=TRAIN_BIAS)
mask_i = delan.make_svd_mask(params["input_transform"], dix=input_dict, train_bias=TRAIN_BIAS)

optimizer1 = optax.masked(optax.adamw(hyper['learning_rate'], hyper['weight_decay']), mask_l)
optimizer2 = optax.masked(optax.adamw(hyper['learning_rate'], hyper['weight_decay']), mask_i)



opt1 = optimizer1.init(params["potential"])
opt2 = optimizer2.init(params["input_transform"])

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
3. Start Training Loop:
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

