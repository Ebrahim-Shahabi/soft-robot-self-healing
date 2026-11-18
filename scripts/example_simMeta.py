import random
from random import shuffle
import DeLaN_model_meta as delan
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial
import time
import optax
import numpy as np
from utils import ReplayMemory
import pickle
import os
from jax.flatten_util import ravel_pytree
# package for plot
import matplotlib
# matplotlib.use("Agg")  # use this if you only want to save images
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


class MultiSubspaceAdapter:
    """
    Adapt:
      - lagrangian/mass
      - lagrangian/potential
      - dissipative
      - input_transform

    with separate subspace coefficients.
    """
    def __init__(self, full_base_params, U_mass, U_pot, U_diss, U_inp):
        self.full_base_params = full_base_params

        # ----- split lagrangian into mass / potential -----
        lagr = full_base_params["lagrangian"]

        self.mass_keys = [k for k in lagr.keys() if "mass" in k]
        self.pot_keys  = [k for k in lagr.keys() if "potential" in k]

        self.base_mass = {k: lagr[k] for k in self.mass_keys}
        self.base_pot  = {k: lagr[k] for k in self.pot_keys}
        self.base_diss = full_base_params["dissipative"]
        self.base_inp  = full_base_params["input_transform"]

        self.base_mass_flat, self.unravel_mass = ravel_pytree(self.base_mass)
        self.base_pot_flat,  self.unravel_pot  = ravel_pytree(self.base_pot)
        self.base_diss_flat, self.unravel_diss = ravel_pytree(self.base_diss)
        self.base_inp_flat,  self.unravel_inp  = ravel_pytree(self.base_inp)

        self.U_mass = jnp.asarray(U_mass)
        self.U_pot  = jnp.asarray(U_pot)
        self.U_diss = jnp.asarray(U_diss)
        self.U_inp  = jnp.asarray(U_inp)

        assert self.U_mass.shape[0] == self.base_mass_flat.shape[0], \
            f"U_mass rows {self.U_mass.shape[0]} != mass params {self.base_mass_flat.shape[0]}"
        assert self.U_pot.shape[0] == self.base_pot_flat.shape[0], \
            f"U_pot rows {self.U_pot.shape[0]} != potential params {self.base_pot_flat.shape[0]}"
        assert self.U_diss.shape[0] == self.base_diss_flat.shape[0], \
            f"U_diss rows {self.U_diss.shape[0]} != dissipative params {self.base_diss_flat.shape[0]}"
        assert self.U_inp.shape[0] == self.base_inp_flat.shape[0], \
            f"U_inp rows {self.U_inp.shape[0]} != input params {self.base_inp_flat.shape[0]}"

        self.rank_mass = self.U_mass.shape[1]
        self.rank_pot  = self.U_pot.shape[1]
        self.rank_diss = self.U_diss.shape[1]
        self.rank_inp  = self.U_inp.shape[1]

    def init_meta_params(self):
        return {
            "a_mass": jnp.zeros((self.rank_mass,)),
            "a_pot":  jnp.zeros((self.rank_pot,)),
            "a_diss": jnp.zeros((self.rank_diss,)),
            "a_inp":  jnp.zeros((self.rank_inp,)),

            "r_mass": jnp.zeros_like(self.base_mass_flat),
            "r_pot":  jnp.zeros_like(self.base_pot_flat),
            "r_diss": jnp.zeros_like(self.base_diss_flat),
            "r_inp":  jnp.zeros_like(self.base_inp_flat),
        }

    def adapted_params(self, meta_params, lambda_residual=1.0):
        delta_mass = self.U_mass @ meta_params["a_mass"] + lambda_residual * meta_params["r_mass"]
        delta_pot  = self.U_pot  @ meta_params["a_pot"]  + lambda_residual * meta_params["r_pot"]
        delta_diss = self.U_diss @ meta_params["a_diss"] + lambda_residual * meta_params["r_diss"]
        delta_inp  = self.U_inp  @ meta_params["a_inp"]  + lambda_residual * meta_params["r_inp"]

        adapted_mass = self.unravel_mass(self.base_mass_flat + delta_mass)
        adapted_pot  = self.unravel_pot(self.base_pot_flat + delta_pot)
        adapted_diss = self.unravel_diss(self.base_diss_flat + delta_diss)
        adapted_inp  = self.unravel_inp(self.base_inp_flat + delta_inp)

        new_lagr = dict(self.full_base_params["lagrangian"])
        new_lagr.update(adapted_mass)
        new_lagr.update(adapted_pot)

        new_params = dict(self.full_base_params)
        new_params["lagrangian"] = new_lagr
        new_params["dissipative"] = adapted_diss
        new_params["input_transform"] = adapted_inp

        return new_params


print("Loading Data:")
'''
1. Data Loading and data preprocess
'''
data = np.loadtxt("./../data/new_new_data_rnd.txt", comments="#")
data2 = np.loadtxt("./../data/new_new_data_sin.txt", comments="#")
data3 = np.loadtxt("./../data/new_new_data_const.txt", comments="#")
data = np.vstack((data, data2, data3))


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
states = np.column_stack((config_data, dconfig_data))

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

'Load Model'
with open(f"./../models/one_segment_spatial_soft_robot_delan.jax", 'rb') as f:
    data = pickle.load(f)

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

hyper = data["hyper"]
params = data["params"]
# hyper['learning_rate'] = 2.7e-03
print("hyperparameters:", hyper)

actuator_dof = train_inputs.shape[1]
config_dof = train_q.shape[1]
print(f"Config DOF = {config_dof}, Actuator DOF = {actuator_dof}")

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


forward_model = delan.forward_model(params=params, key=None, lagrangian=lagrangian, dissipative_mat=dissipative_mat,
                                    input_mat=input_mat, n_dof=config_dof)

state0 = jnp.concatenate([q[0], dq[0]])
_ = forward_model(state0, tau[0])

t_build = time.perf_counter() - t0
print(f"DeLaN Build Time    = {t_build:.2f}s")


# # 2. Generate and initialize the optimizer
t0 = time.perf_counter()
U_mass = jnp.load("../models/U_mass.npy")
U_pot  = jnp.load("../models/U_potential.npy")
U_diss = jnp.load("../models/U_dissipative.npy")
U_inp  = jnp.load("../models/U_input.npy")

adapter = MultiSubspaceAdapter(
    full_base_params=params,
    U_mass=U_mass,
    U_pot=U_pot,
    U_diss=U_diss,
    U_inp=U_inp,
)

meta_params = adapter.init_meta_params()

meta_optimizer = optax.adamw(hyper['learning_rate'], hyper['weight_decay'])
meta_opt_state = meta_optimizer.init(meta_params)


loss_fn = partial(
    delan.loss_fn,
    lagrangian=lagrangian,
    dissipative_mat=dissipative_mat,
    input_mat=input_mat,
    n_dof=config_dof,
    time_step=time_step)

def residual_penalty(meta_params):
    return (
        jnp.sum(meta_params["r_mass"] ** 2) +
        jnp.sum(meta_params["r_pot"] ** 2) +
        jnp.sum(meta_params["r_diss"] ** 2) +
        jnp.sum(meta_params["r_inp"] ** 2)
    )

def meta_loss_fn(meta_params, q, dq, tau, q_next, dq_next):
    adapted_params = adapter.adapted_params(meta_params, lambda_residual=1e-2)
    loss, logs = loss_fn(
        params=adapted_params,
        q=q,
        qd=dq,
        tau=tau,
        q_next=q_next,
        qd_next=dq_next
    )

    reg = 1e-6 * residual_penalty(meta_params)
    total_loss = loss + reg

    logs = dict(logs)
    print(logs)
    logs["residual_reg"] = reg
    logs["total_loss"] = total_loss
    return total_loss, logs

def update_fn(meta_params, meta_opt_state, q, dq, tau, q_next, dq_next):
    (_, logs), grads = jax.value_and_grad(meta_loss_fn, argnums=0, has_aux=True)(
        meta_params, q, dq, tau, q_next, dq_next
    )

    updates, meta_opt_state = meta_optimizer.update(grads, meta_opt_state, meta_params)
    meta_params = optax.apply_updates(meta_params, updates)

    return meta_params, meta_opt_state, logs

update_fn = jax.jit(update_fn)
meta_params, meta_opt_state, logs = update_fn(
    meta_params, meta_opt_state,
    train_q[:2], train_dq[:2], train_inputs[:2], train_q_next[:2], train_dq_next[:2]
)

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

        q, dq, tau, q_next, dq_next= [jnp.array(x) for x in data_batch]        
        meta_params, meta_opt_state, batch_logs = update_fn(
            meta_params, meta_opt_state,
            q, dq, tau, q_next, dq_next
        )
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
        # test loss computation using the adapted parameters
        test_loss, test_logs = meta_loss_fn(
            meta_params=meta_params,
            q=test_q,
            dq=test_dq,
            tau=test_inputs,
            q_next=test_q_next,
            dq_next=test_dq_next
        )        
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


if save_model:
    with open("./models/meta.jax", "wb") as file:
        pickle.dump(
            {
                "epoch": epoch_i,
                "hyper": hyper,
                "base_params": params,
                "meta_params": meta_params,
                "subtree_key": "lagrangian",
            },
            file
        )

