import pandas as pd
import matplotlib
# matplotlib.use("Agg")  # use this if you only want to save images
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from os import path
import random
import MLP_model as network
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
from utils import ReplayMemory
import time
import optax
import pickle

save_model = True
'''
1. Prepare the data
'''
basepath = path.dirname(__file__)
csv_path = path.join(
    basepath,
    "../",
    "data",
    "experiments",
    "online_output20260428-132925.csv"
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
ye = (df["ye[m]"] - df["y0[m]"])/100

q = np.array(df["angle[rad]"]).reshape(-1,1)
q = np.column_stack((pe, ye, q))

nq_ = q.shape[1]

data = q.copy()
data = np.column_stack((data, input))
random.shuffle(data)

states = data[:,:nq_]
input = data[:,nq_:]


# train
print(f"There are {states.shape[0]} data in this attempt.")
print("Set 80%  data as the training set and 20% as the test set")
div = states.shape[0] * 8 // 10

train_x, test_x = states[:div, :], states[div:, :]
train_y, test_y = input[:div, :], input[div:, :]

train_y = train_y.astype(np.float32)
test_y = test_y.astype(np.float32)
train_x = train_x.astype(np.float32)
test_x = test_x.astype(np.float32)

'''
2. Set the model parameters/hyperparameters
'''
activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
}

actuator_dof = train_x.shape[1]
config_dof = train_y.shape[1]
print(f"Config DOF = {config_dof}, Actuator DOF = {actuator_dof}")

hyper = {
    'n_dof': config_dof,
    'actuator_dof': actuator_dof,
    'n_width': 64,
    'n_depth': 2,
    'n_minibatch': 200, # 800.
    'activation1': 'softplus', 
    'activation2': 'tanh',
    'learning_rate': 1e-04,
    'weight_decay': 3.2e-05,
    'max_epoch': int(6.0 * 1e3),
}

'''
3. Construct mlp model 
'''
t0 = time.perf_counter()
net_fn = hk.transform(partial(network.network, state_dof = config_dof, shape = (hyper["n_width"],)*hyper["n_depth"], \
                activation = activations[hyper['activation2']])) # tanh

mem_dim = ( (actuator_dof,), (config_dof,))
mem = ReplayMemory(train_x.shape[0], hyper["n_minibatch"], mem_dim)
mem.add_samples([train_x, train_y])

# Initialize parameters
sample_q = jnp.asarray(train_y[0], dtype=jnp.float32)
sample_tau = jnp.asarray(train_x[0], dtype=jnp.float32)
rng = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(rng)

net_params = net_fn.init(key1, sample_tau)

params = {'network': net_params}

net_ = net_fn.apply

forward_model = network.forward_model(params=params, key=None, net=net_)

_ = forward_model(sample_tau)

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

opt1 = optimizer1.init(params["network"])

# loss_fn = partial(delan.loss_fn, pot_mat = pot_mat, input_mat=input_mat)
loss_fn = partial(
    network.loss_fn,
    net = net_,
    lambda_data=1,
)

def update_fn(params, opt1, tau, q):
    (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(params, q=q, tau=tau)
    # print(grads)
    updates1, opt1 = optimizer1.update(grads["network"], opt1, params["network"])
    params["network"] = optax.apply_updates(params["network"], updates1)
    return params, opt1, logs



update_fn = jax.jit(update_fn)
_, _, logs = update_fn(params, opt1, tau = train_x[:2], q = train_y[:2])

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

        x, y = [jnp.array(x) for x in data_batch]        
        params, opt1, batch_logs = update_fn(params, opt1, x, y)

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
        test_loss, test_logs = loss_fn(params=params, q=test_y,  tau=test_x)
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

basepath = path.dirname(__file__)
path = path.join(
    basepath,
    "..",
    "models",
    "mlp_models",
    "20260311-125118.jax"
)

if save_model:
    with open(path, "wb") as file:
        pickle.dump(
            {"epoch": epoch_i,
             "hyper": hyper,
             "params": params},
            file)
