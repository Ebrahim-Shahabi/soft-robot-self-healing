import random
from random import shuffle
import DeLaN_model_v4 as delan
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
import time
import optax
import numpy as np
from utils import ReplayMemory
from utils import track_params, update_tracked_params
import pickle
import os

# package for plot
import matplotlib
# matplotlib.use("Agg")  # use this if you only want to save images
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

print("Loading Data:")
'''
1. Data Loading and data preprocess
'''
data = np.loadtxt("./data/data_rnd.txt", comments="#")
data2 = np.loadtxt("./data/data_sin.txt", comments="#")
data3 = np.loadtxt("./data/data_sudden.txt", comments="#")


data = np.vstack((data, data2, data3))
# plt.figure()
# plt.plot(data[:,10]*1000, data[:,4], marker='o', linestyle='None')
# plt.xlabel("Input [kPa]")
# plt.ylabel("Tip angle [deg]")
# plt.show()
print(f"Total data samples after merging: {data.shape[0]}")
random.shuffle(data) # shuffle along the first axis



time_step = 0.01

config_data = data[:, :5]
dconfig_data = data[:, 5:10]
# only get last element
config_data = config_data[:,-1]
dconfig_data = dconfig_data[:,-1]

# convert degree to radian
deg2rad = np.pi / 180.0
scale_input = 10000 # input is in MPa, scale to [0, 5]
config_data  = config_data     * deg2rad
dconfig_data = dconfig_data   * deg2rad
targets      = data[:, 11:]    * deg2rad
inputs       = data[:, 10]  * scale_input


targets = np.column_stack([targets[:,4], targets[:, -1]])
# normalize 
# format the data the way the code accepts them.
states = np.column_stack((config_data, dconfig_data))

# state_mean = states.mean(axis=0, keepdims=True)
# state_std  = states.std(axis=0, keepdims=True) + 1e-8
# states = (states - state_mean) / state_std

# tgt_mean = targets.mean(axis=0, keepdims=True)
# tgt_std  = targets.std(axis=0, keepdims=True) + 1e-8
# targets = (targets - tgt_mean) / tgt_std

# normalize inputs
# u_mean = inputs.mean(axis=0, keepdims=True)
# u_std  = inputs.std(axis=0, keepdims=True) + 1e-8
# inputs = (inputs - u_mean) / u_std


random.seed(42)
# shuffle
shuffle_data = True
if shuffle_data:
    c = list(zip(states, inputs, targets))
    shuffle(c)
    states, inputs, targets = zip(*c)

states = np.array(states)
inputs = np.array(inputs).reshape(-1, 1)
targets = np.array(targets)




print(f"There are {states.shape[0]} data in this attempt.")
print("Set 80%  data as the training set and 20% as the test set")
div = states.shape[0] * 8 // 10
train_states, test_states = states[:div, :], states[div:, :]
train_targets, test_targets = targets[:div, :], targets[div:, :]
train_inputs, test_inputs = inputs[:div], inputs[div:]


train_q, train_dq = jnp.split(train_states, 2, axis=1)
train_q_next, train_dq_next = jnp.split(train_targets, 2, axis=1)

test_q, test_dq = jnp.split(test_states, 2, axis=1)
test_q_next, test_dq_next = jnp.split(test_targets, 2, axis=1)

'''
2. Set the model parameters/hyperparameters
'''

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

actuator_dof = train_inputs.shape[1]
config_dof = train_q.shape[1]
print(f"Config DOF = {config_dof}, Actuator DOF = {actuator_dof}")

save_model = True
hyper = {
    'n_dof': config_dof,
    'actuator_dof': actuator_dof,
    'n_width': 30,
    'n_depth': 3,
    'n_minibatch': 400, # 800.
    'diagonal_epsilon': 0.01,
    'diagonal_shift': 0.005,
    'activation1': 'softplus',
    'activation2': 'tanh',
    'activation3': 'sigmoid',
    'learning_rate': 2.7e-04,
    'weight_decay': 3.2e-05,
    'max_epoch': int(6.0 * 1e3),
}

# 1. Construct DeLaN:
t0 = time.perf_counter()


#def structured_lagrangian_fn(q, qd, n_dof, shape, activation, epsilon, shift)
lagrangian_fn = hk.transform(partial(
    delan.structured_lagrangian_fn,
    n_dof=hyper['n_dof'],
    shape=(hyper['n_width'],) * hyper['n_depth'],
    activation=activations[hyper['activation1']],
    epsilon=hyper['diagonal_epsilon'],
    shift=hyper['diagonal_shift'],
))

#dissipative_matrix(qd, n_dof, shape, activation)
dissipative_fn = hk.transform(partial(
    delan.dissipative_matrix,
    n_dof=hyper['n_dof'],
    shape=(5,) * 3,
    activation=activations[hyper['activation2']]
))

#input_transform_matrix(q, n_dof, actuator_dof, shape, activation)
input_mat_fn = hk.transform(partial(
    delan.input_transform_matrix,
    n_dof=hyper['n_dof'],
    actuator_dof=hyper['actuator_dof'],
    shape=(hyper['n_width']//2,) * (hyper['n_depth']-1),
    activation=activations[hyper['activation1']]
))
rng = jax.random.PRNGKey(42)

# q, dq, tau, q_next, dq_next
mem_dim = ((config_dof,), (config_dof,), (actuator_dof,), (config_dof,), (config_dof,))
mem = ReplayMemory(train_q.shape[0], hyper["n_minibatch"], mem_dim)
mem.add_samples([train_q, train_dq, train_inputs, train_q_next, train_dq_next])

rng_key, init_key = jax.random.split(rng)
# Initialize Parameters:
q, dq, tau, q_next, dq_next = [jnp.array(x) for x in next(iter(mem))]
l_params = lagrangian_fn.init(init_key, q[0], dq[0])
d_params = dissipative_fn.init(init_key, dq[0])
i_params = input_mat_fn.init(init_key, q[0])

params = {"lagrangian": l_params, "dissipative": d_params, 'input_transform': i_params}

lagrangian = lagrangian_fn.apply
dissipative_mat = dissipative_fn.apply
input_mat = input_mat_fn.apply

# def forward_model(params, key, lagrangian, dissipative_mat, input_mat, n_dof):
#     def equation_of_motion(state, tau, t=None):
forward_model = delan.forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat,
                                    input_mat=input_mat, n_dof=config_dof)

state0 = jnp.concatenate([q[0], dq[0]])
_ = forward_model(state0, tau[0])

t_build = time.perf_counter() - t0
print(f"DeLaN Build Time    = {t_build:.2f}s")


# # 2. Generate and initialize the optimizer
t0 = time.perf_counter()

optimizer1 = optax.adamw(
    learning_rate=hyper['learning_rate'],
    weight_decay=hyper['weight_decay']
)

optimizer2 = optax.adamw(
    learning_rate=hyper['learning_rate'],
    weight_decay=hyper['weight_decay']
)

optimizer3 = optax.adamw(
    learning_rate=hyper['learning_rate'],
    weight_decay=hyper['weight_decay']
)

opt1 = optimizer1.init(params["lagrangian"])
opt2 = optimizer2.init(params["dissipative"])
opt3 = optimizer3.init(params["input_transform"])


# def loss_fn(params, q, qd, tau, q_next, qd_next, lagrangian,
#             dissipative_mat, input_mat, n_dof, time_step=None):
loss_fn = partial(
    delan.loss_fn,
    lagrangian=lagrangian,
    dissipative_mat=dissipative_mat,
    input_mat=input_mat,
    n_dof=config_dof,
    time_step=time_step)

def update_fn(params, opt1, opt2, opt3, q, dq, tau, q_next, dq_next):
    (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(params, q, dq, tau, q_next, dq_next)
    # print(grads)
    updates1, opt1 = optimizer1.update(grads["lagrangian"], opt1, params["lagrangian"])
    params["lagrangian"] = optax.apply_updates(params["lagrangian"], updates1)
    updates2, opt2 = optimizer2.update(grads["dissipative"], opt2, params["dissipative"])
    params["dissipative"] = optax.apply_updates(params["dissipative"], updates2)
    updates3, opt3 = optimizer3.update(grads["input_transform"], opt3, params["input_transform"])
    params["input_transform"] = optax.apply_updates(params["input_transform"], updates3)
    return params, opt1, opt2, opt3, logs

update_fn = jax.jit(update_fn)
_, _, _, _, logs = update_fn(params, opt1, opt2, opt3, train_q[:2], train_dq[:2], train_inputs[:2], train_q_next[:2], train_dq_next[:2])


t_build = time.perf_counter() - t0
print(f"Optimizer Build Time = {t_build:.2f}s")


############################
"""
Store some random parameters and check how they evolve during
training. Later this will be expanded to include to also find weights of networks
that are simillar. 
"""
tracked_params = track_params(params, 4)
# print(tracked_params)
############################

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
# print(tracked_params)
# print("\n")
                
while epoch_i < hyper['max_epoch']:
    n_batches = 0
    logs = jax.tree.map(lambda x: x * 0.0, logs)

    for data_batch in mem:
        t0_batch = time.perf_counter()

        q, dq, tau, q_next, dq_next= [jnp.array(x) for x in data_batch]        
        params, opt1, opt2, opt3, batch_logs = update_fn(params, opt1, opt2, opt3, q, dq, tau, q_next, dq_next)

        # Update logs:
        n_batches += 1
        print(batch_logs)
        logs = jax.tree.map(lambda x, y: x + y, logs, batch_logs)
        t_batch = time.perf_counter() - t0_batch

        # track weight updates of the tracked parameters
        tracked_params = update_tracked_params(params, tracked_params)

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
        test_loss, test_logs = loss_fn(params=params, q=test_q, qd=test_dq, tau=test_inputs, q_next=test_q_next, qd_next=test_dq_next)
        print("test: ", end=" ")
        print(f"For = {test_logs['forward_mean']:.3e} \u00B1 {1.96 * np.sqrt(test_logs['forward_var']):.2e}")
        test_losses["forward_loss"].append(test_logs['forward_mean'])
        test_losses['forward_var'].append(test_logs['forward_var'])
    
import json

def to_serializable(obj):
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)

with open("tracked_params.json", "w") as f:
    json.dump(to_serializable(tracked_params), f, indent=2)


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


if save_model:
    with open(f"./models/one_segment_spatial_soft_robot_delan.jax", "wb") as file:
        pickle.dump(
            {"epoch": epoch_i,
             "hyper": hyper,
             "params": params},
            file)

