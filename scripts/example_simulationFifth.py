import pickle
import random
from pathlib import Path
from functools import partial

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import DeLaN_model_v4 as delan
from utils import ReplayMemory
from plots import plot_subspace_analysis




SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_DIR = SCRIPT_DIR.parents[0] / "models" / "models" / "simulation_models"

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

'''
1. Load W0 and the task models trained earlier
'''

with open(MODEL_DIR / "first_delan.jax", "rb") as f:
    base_data = pickle.load(f)

hyper = base_data["hyper"]
W0 = base_data["params"]

TASK_MODEL_FILES = {
    "second": MODEL_DIR / "second_delan.jax",
    "third": MODEL_DIR / "third_delan.jax",
    "fourth": MODEL_DIR / "fourth_delan.jax",
}

task_weights = {}
for name, fpath in TASK_MODEL_FILES.items():
    with open(fpath, "rb") as f:
        task_weights[name] = pickle.load(f)["params"]

print("Loaded W0 and task models:", list(task_weights.keys()))

'''
2. Subspace utilities -- full SVD (no truncation) so every rank is available
'''

def flatten_params(params: dict):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = [leaf.shape for leaf in leaves]
    flat = np.concatenate([np.array(leaf).ravel() for leaf in leaves])
    return flat, (treedef, shapes)


def unflatten_params(flat: jnp.ndarray, structure):
    treedef, shapes = structure
    sizes = [int(np.prod(s)) for s in shapes]
    splits = jnp.split(flat, np.cumsum(sizes[:-1]))
    leaves = [s.reshape(sh) for s, sh in zip(splits, shapes)]
    return treedef.unflatten(leaves)


def build_free_mask_from_structure(params: dict, structure, free_layer_names=None) -> np.ndarray:
    free_layer_names = free_layer_names or []
    treedef, shapes = structure

    paths = []

    def collect_path(path, leaf):
        paths.append("/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path))
        return leaf

    jax.tree_util.tree_map_with_path(collect_path, params)

    sizes = [int(np.prod(s)) for s in shapes]
    free_mask = np.zeros(int(np.sum(sizes)), dtype=bool)

    offset = 0
    for path_str, size in zip(paths, sizes):
        is_free = any(name in path_str for name in free_layer_names)
        if not is_free and path_str.endswith("b"):
            is_free = True
        free_mask[offset: offset + size] = is_free
        offset += size

    return free_mask


def compute_task_subspace_full(w0_star, task_weights, free_layer_names=None):
    """SVD of weight deltas without truncating by a variance threshold --
    returns every singular vector/value so any rank can be tried later."""
    w0_flat, structure = flatten_params(w0_star)
    d = len(w0_flat)
    K = len(task_weights)

    free_mask = build_free_mask_from_structure(w0_star, structure, free_layer_names)
    constrained_mask = ~free_mask

    print(f"\n[SVD] Total parameter dimensionality      d = {d:,}")
    print(f"[SVD] Free parameters excluded              = {int(free_mask.sum()):,}")
    print(f"[SVD] Constrained parameters used for SVD   = {int(constrained_mask.sum()):,}")
    print(f"[SVD] Number of tasks                       K = {K}")

    Delta = np.zeros((d, K), dtype=np.float64)
    for k, (name, wk) in enumerate(task_weights.items()):
        wk_flat, _ = flatten_params(wk)
        Delta[:, k] = wk_flat - w0_flat
        print(f"  Task '{name}': ||\u0394W_full|| = {np.linalg.norm(Delta[:, k]):.4f}")

    Delta_constrained = Delta[constrained_mask, :]
    U, S, _ = np.linalg.svd(Delta_constrained, full_matrices=False)  # U: (n_constrained, K)

    denom = np.sum(S ** 2)
    var_explained = np.cumsum(S ** 2) / denom

    print(f"[SVD] Singular values: {np.round(S, 4)}")
    print(f"[SVD] Variance explained per rank: {np.round(var_explained, 4)}")

    return {
        "U_full": U, "S_full": S, "variance_explained": var_explained,
        "max_rank": K,
        "w0_flat": w0_flat, "structure": structure,
        "free_mask": free_mask, "constrained_mask": constrained_mask,
    }

# def plot_subspace_analysis(subspace: dict, save_path=None):
#     S = subspace["S"]          # already truncated to rank r
#     var = subspace["variance_explained"]   # full K components
#     r = subspace["rank"]

#     K = len(var)               # total number of tasks
#     ks_full = np.arange(1, K + 1)   # for variance plot (all K)
#     ks_trunc = np.arange(1, r + 1)  # for singular values (only top r)

#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
#     fig.suptitle("Task Subspace SVD Analysis", fontsize=14, fontweight="bold")

#     ax1.bar(ks_trunc, S, color="#2563eb", alpha=0.8)
#     ax1.set_xlabel("Component")
#     ax1.set_ylabel("Singular value")
#     ax1.set_title("Singular Value Spectrum (kept components)")
#     ax1.legend()
#     ax1.grid(True, alpha=0.3)

#     ax2.plot(ks_full, var * 100, "o-", color="#2563eb", linewidth=2)
#     ax2.axhline(90, color="orange", linestyle="--", label="90%")
#     ax2.axhline(95, color="red", linestyle="--", label="95%")
#     ax2.set_xlabel("Number of components")
#     ax2.set_ylabel("Cumulative variance explained (%)")
#     ax2.set_title("Variance Explained")
#     ax2.legend()
#     ax2.grid(True, alpha=0.3)
#     ax2.set_ylim(0, 105)

#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, dpi=150, bbox_inches="tight")
#     plt.show()

def subspace_for_rank(full_svd, rank):
    """Slice the full SVD down to a chosen rank."""
    return {
        "U": full_svd["U_full"][:, :rank],
        "S": full_svd["S_full"][:rank],
        "variance_explained": full_svd["variance_explained"],
        "rank": rank,
        "w0_flat": full_svd["w0_flat"],
        "structure": full_svd["structure"],
        "free_mask": full_svd["free_mask"],
        "constrained_mask": full_svd["constrained_mask"],
    }


class SubspaceParams:
    def __init__(self, subspace, w0_star, free_layer_names=None):
        self.subspace = subspace
        self.w0_flat = jnp.array(subspace["w0_flat"])
        self.r = subspace["rank"]
        self.structure = subspace["structure"]
        self.free_mask = subspace["free_mask"]
        self.constrained_mask = subspace["constrained_mask"]
        self.free_layer_names = free_layer_names or []

        self.w0_free = self.w0_flat[self.free_mask]
        self.w0_constrained = self.w0_flat[self.constrained_mask]
        self.U_constrained = jnp.array(subspace["U"])

        print(f"\n[SubspaceParams] Free parameters        : {int(self.free_mask.sum()):,}")
        print(f"[SubspaceParams] Constrained parameters : {int(self.constrained_mask.sum()):,}  (rank-{self.r} subspace)")

    def init_trainable(self):
        return {"alpha": jnp.zeros(self.r), "free": jnp.array(self.w0_free)}

    def reconstruct(self, trainable):
        alpha = trainable["alpha"]
        free = trainable["free"]
        w_constrained = self.w0_constrained + self.U_constrained @ alpha
        d = len(self.w0_flat)
        w_full = jnp.zeros(d)
        w_full = w_full.at[self.constrained_mask].set(w_constrained)
        w_full = w_full.at[self.free_mask].set(free)
        return unflatten_params(w_full, self.structure)


full_svd = compute_task_subspace_full(w0_star=W0, task_weights=task_weights, free_layer_names=None)
max_rank = full_svd["max_rank"]

plot_subspace_analysis(
    subspace_for_rank(full_svd, max_rank),
    save_path=MODEL_DIR / "svd_analysis.png",
)


print(f"\nWill sweep ranks 1..{max_rank}")

'''
3. Load the held-out 'fifth' robot data
'''

df = pd.read_csv("fifth_robot_prbs_input.txt", sep=r"\s+", comment="#", names=columns)

time_ = df["time_s"].to_numpy()
tau = df["u_MPa"].to_numpy().reshape(-1, 1) * 10000


plt.figure()
plt.plot(time_, tau)
plt.xlabel('Time [s]')
plt.ylabel('Pressure [kPa]')
plt.grid()
plt.show()

ex = df["sex_m"].to_numpy() / 100
ey = df["sey_m"].to_numpy() / 100
ez = df["sez_m"].to_numpy() / 100

dex = df["dsex_m_s"].to_numpy() / 100
dey = df["dsey_m_s"].to_numpy() / 100
dez = df["dsez_m_s"].to_numpy() / 100

angle = np.deg2rad(df["x3_deg"].to_numpy())
dangle = np.deg2rad(df["dx3_deg_s"].to_numpy())

states = np.column_stack((ex, ey, ez, angle, dex, dey, dez, dangle))


target_angle_deg = 35
idx = np.argmin(np.abs(df["x3_deg"].to_numpy() - target_angle_deg))

x, y, z, ang = states[idx, 0], states[idx, 1], states[idx, 2], np.rad2deg(states[idx, 3])
print(f"angle={ang:.2f} deg -> x={x:.5f}, y={y:.5f}, z={z:.5f}")



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

print(f"\nThere are {states.shape[0]} data points for 'fifth'.")
div = states.shape[0] * 8 // 10

train_states, test_states = states[:div, :], states[div:, :]
train_targets, test_targets = targets[:div, :], targets[div:, :]
train_inputs, test_inputs = inputs[:div], inputs[div:]
train_time_steps, test_time_steps = time_steps[:div], time_steps[div:]

train_q, train_dq = jnp.split(train_states, 2, axis=1)
train_q_next, train_dq_next = jnp.split(train_targets, 2, axis=1)
test_q, test_dq = jnp.split(test_states, 2, axis=1)
test_q_next, test_dq_next = jnp.split(test_targets, 2, axis=1)

'''
4. Build apply functions matching W0's hyperparameters exactly
'''

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

config_dof = hyper['n_dof']
actuator_dof = hyper['actuator_dof']

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

'''
5a. Baseline A: full fine-tuning from W0
'''

optimizer1 = optax.adamw(learning_rate=hyper['learning_rate'], weight_decay=hyper['weight_decay'])
optimizer2 = optax.adamw(learning_rate=hyper['learning_rate'], weight_decay=hyper['weight_decay'])
optimizer3 = optax.adamw(learning_rate=hyper['learning_rate'], weight_decay=hyper['weight_decay'])

def update_fn(params, opt1, opt2, opt3, q, dq, tau_b, q_next, dq_next, time_step):
    (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(params, q, dq, tau_b, q_next, dq_next, time_step=time_step)
    updates1, opt1 = optimizer1.update(grads["lagrangian"], opt1, params["lagrangian"])
    params["lagrangian"] = optax.apply_updates(params["lagrangian"], updates1)
    updates2, opt2 = optimizer2.update(grads["dissipative"], opt2, params["dissipative"])
    params["dissipative"] = optax.apply_updates(params["dissipative"], updates2)
    updates3, opt3 = optimizer3.update(grads["input_transform"], opt3, params["input_transform"])
    params["input_transform"] = optax.apply_updates(params["input_transform"], updates3)
    return params, opt1, opt2, opt3, logs

update_fn = jax.jit(update_fn)

# ft_params = jax.tree_util.tree_map(lambda x: x, W0)

# scratch
rng = jax.random.PRNGKey(42)
ft_params = jax.tree_util.tree_map(
    lambda x: jax.random.normal(rng, x.shape) * 0.01, W0
)

mem_dim = ((config_dof,), (config_dof,), (actuator_dof,), (config_dof,), (config_dof,), (train_time_steps.shape[1],))
mem = ReplayMemory(train_q.shape[0], 128, mem_dim)
mem.add_samples([train_q, train_dq, train_inputs, train_q_next, train_dq_next, train_time_steps])

opt1 = optimizer1.init(ft_params["lagrangian"])
opt2 = optimizer2.init(ft_params["dissipative"])
opt3 = optimizer3.init(ft_params["input_transform"])

_, _, _, _, logs = update_fn(ft_params, opt1, opt2, opt3, train_q[:2], train_dq[:2], train_inputs[:2], train_q_next[:2], train_dq_next[:2], time_step=train_time_steps[:2])

print("\n=== Full fine-tuning from W0 on 'fifth' ===")
ft_iterations = []
ft_train_losses = {"forward_loss": [], "forward_var": []}
ft_test_losses = {"forward_loss": [], "forward_var": []}

epoch_i = 0
step = 50
while epoch_i < hyper['max_epoch']:
    n_batches = 0
    logs = jax.tree.map(lambda x: x * 0.0, logs)

    for data_batch in mem:
        q, dq, tau_b, q_next, dq_next, time_steps_b = [jnp.array(x) for x in data_batch]
        ft_params, opt1, opt2, opt3, batch_logs = update_fn(ft_params, opt1, opt2, opt3, q, dq, tau_b, q_next, dq_next, time_step=time_steps_b)
        n_batches += 1
        logs = jax.tree.map(lambda x, y: x + y, logs, batch_logs)

    epoch_i += 1
    logs = jax.tree.map(lambda x: x / n_batches, logs)

    if epoch_i == 1 or np.mod(epoch_i, step) == 0:
        test_loss, test_logs = loss_fn(params=ft_params, q=test_q, qd=test_dq, tau=test_inputs, q_next=test_q_next, qd_next=test_dq_next, time_step=test_time_steps)
        print(f"[fine-tune] epoch {epoch_i:05d}: train For = {logs['forward_mean']:.3e} | test For = {test_logs['forward_mean']:.3e}")
        ft_iterations.append(epoch_i)
        ft_train_losses["forward_loss"].append(float(logs['forward_mean']))
        ft_train_losses["forward_var"].append(float(logs['forward_var']))
        ft_test_losses["forward_loss"].append(float(test_logs['forward_mean']))
        ft_test_losses["forward_var"].append(float(test_logs['forward_var']))

'''
5b. Baseline B: subspace adaptation -- swept over EVERY rank, recorded every 50 epochs
'''

N_EPOCHS_SUBSPACE = 6000
BATCH_SIZE_SUBSPACE = 128
RECORD_EVERY = 50

# LR_SUBSPACE = optax.warmup_cosine_decay_schedule(
#     init_value=1e-1,
#     peak_value=5e-1,
#     warmup_steps=200,
#     decay_steps=N_EPOCHS_SUBSPACE,
#     end_value=1e-3,
# )

LR_SUBSPACE = hyper['learning_rate']


rank_results = {}

for rank in range(1, max_rank + 1):
    print(f"\n=== Subspace adaptation on 'fifth' -- rank {rank} ===")

    subspace_r = subspace_for_rank(full_svd, rank)
    sp_r = SubspaceParams(subspace=subspace_r, w0_star=W0, free_layer_names=None)

    @jax.jit
    def subspace_loss(trainable, q_b, qd_b, tau_b, q_next_b, qd_next_b, dt_b, sp_r=sp_r):
        full_params = sp_r.reconstruct(trainable)
        loss, logs = delan.loss_fn_experiment(
            full_params, q_b, qd_b, tau_b, q_next_b, qd_next_b,
            lagrangian, dissipative_mat, input_mat, config_dof, dt_b,
        )
        alpha_reg = 1e-1 * jnp.sum(trainable["alpha"] ** 2)
        return loss + alpha_reg, logs

    trainable = sp_r.init_trainable()
    opt_sub = optax.adam(LR_SUBSPACE)
    state_sub = opt_sub.init(trainable)

    @jax.jit
    def step_sub(tr, st, q_b, qd_b, tau_b, q_next_b, qd_next_b, dt_b):
        (l, logs), g = jax.value_and_grad(subspace_loss, has_aux=True)(tr, q_b, qd_b, tau_b, q_next_b, qd_next_b, dt_b)
        upd, st = opt_sub.update(g, st)
        return optax.apply_updates(tr, upd), st, l, logs

    N_fifth_train = train_q.shape[0]
    iterations_r = []
    train_losses_r = {"forward_loss": [], "forward_var": []}
    test_losses_r = {"forward_loss": [], "forward_var": []}

    for epoch in range(N_EPOCHS_SUBSPACE):
        bi = np.random.permutation(N_fifth_train)[:BATCH_SIZE_SUBSPACE]
        trainable, state_sub, l, logs = step_sub(
            trainable, state_sub, train_q[bi], train_dq[bi], train_inputs[bi],
            train_q_next[bi], train_dq_next[bi], train_time_steps[bi],
        )

        if epoch == 0 or epoch % RECORD_EVERY == 0 or epoch == N_EPOCHS_SUBSPACE - 1:
            sub_params_now = sp_r.reconstruct(trainable)
            test_loss, test_logs = loss_fn(params=sub_params_now, q=test_q, qd=test_dq, tau=test_inputs, q_next=test_q_next, qd_next=test_dq_next, time_step=test_time_steps)
            if epoch % 1000 == 0 or epoch == N_EPOCHS_SUBSPACE - 1:
                print(f"[rank {rank}] epoch {epoch:05d}: train For = {logs['forward_mean']:.3e} | test For = {test_logs['forward_mean']:.3e}")
            iterations_r.append(epoch)
            train_losses_r["forward_loss"].append(float(logs['forward_mean']))
            train_losses_r["forward_var"].append(float(logs['forward_var']))
            test_losses_r["forward_loss"].append(float(test_logs['forward_mean']))
            test_losses_r["forward_var"].append(float(test_logs['forward_var']))

    rank_results[rank] = {
        "iterations": iterations_r,
        "train_losses": train_losses_r,
        "test_losses": test_losses_r,
        "params": sp_r.reconstruct(trainable),
        "alpha": np.array(trainable["alpha"]),
        "n_trainable": rank + int(sp_r.free_mask.sum()),
    }

'''
6. Compare
'''

w0_test_loss, _ = loss_fn(params=W0, q=test_q, qd=test_dq, tau=test_inputs, q_next=test_q_next, qd_next=test_dq_next, time_step=test_time_steps)
ft_test_loss, _ = loss_fn(params=ft_params, q=test_q, qd=test_dq, tau=test_inputs, q_next=test_q_next, qd_next=test_dq_next, time_step=test_time_steps)

print("\n[Result] Final test loss on 'fifth':")
print(f"  W0 (no adaptation) : {float(w0_test_loss):.8f}")
print(f"  Full fine-tune      : {float(ft_test_loss):.8f}")
for rank, res in rank_results.items():
    final_loss = res["test_losses"]["forward_loss"][-1]
    print(f"  Subspace rank {rank}      : {final_loss:.8f}  ({res['n_trainable']} trainable params)")

total_params = sum(int(np.prod(s)) for s in full_svd["structure"][1])
print("\n[Parameter efficiency]")
print(f"  Total parameters in network : {total_params:,}")
print(f"  Full fine-tune trains        : {total_params:,}  (100%)")
for rank, res in rank_results.items():
    print(f"  Subspace rank {rank} trains    : {res['n_trainable']:,}  ({100*res['n_trainable']/total_params:.2f}%)")

plt.figure(figsize=(11, 5))
plt.semilogy(ft_iterations, ft_test_losses["forward_loss"], label="fine-tune (test)", linewidth=2, color="black")
cmap = plt.get_cmap("viridis")
for i, (rank, res) in enumerate(rank_results.items()):
    plt.semilogy(res["iterations"], res["test_losses"]["forward_loss"],
                 label=f"subspace rank={rank} (test)", color=cmap(i / max(1, max_rank - 1)))
plt.axhline(float(w0_test_loss), color="grey", linestyle="--", label="W0 (no adaptation)")
plt.xlabel("epoch")
plt.ylabel("test loss")
plt.title("Subspace (all ranks) vs. full fine-tuning — 'fifth' robot")
plt.legend(fontsize=8)
plt.grid(True, alpha=0.3)
plt.show()

plt.figure(figsize=(7, 5))
ranks_list = list(rank_results.keys())
final_losses = [rank_results[r]["test_losses"]["forward_loss"][-1] for r in ranks_list]
n_trainable_list = [rank_results[r]["n_trainable"] for r in ranks_list]
plt.scatter(n_trainable_list, final_losses, color="#CE0FFE", s=100, zorder=5, label="subspace")
for r, p, l in zip(ranks_list, n_trainable_list, final_losses):
    plt.annotate(f"rank={r}", (p, l), textcoords="offset points", xytext=(6, 4), fontsize=8)
plt.scatter([total_params], [float(ft_test_loss)], color="#16a34a", s=100, zorder=5, label="full fine-tune")
plt.axhline(float(w0_test_loss), color="grey", linestyle="--", label="W0 (no adaptation)")
plt.xlabel("trainable parameters")
plt.ylabel("final test loss")
plt.xscale("log")
plt.title("Parameter efficiency: subspace rank vs. fine-tuning")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


'''
6b. Train vs. Eval (test) loss plot — full fine-tune and subspace rank=2
'''

plt.figure(figsize=(10, 6))

# Full fine-tune
plt.semilogy(ft_iterations, ft_train_losses["forward_loss"],
             label="fine-tune (train)", color="black", linestyle="-")
plt.semilogy(ft_iterations, ft_test_losses["forward_loss"],
             label="fine-tune (eval)", color="black", linestyle="--")

# Subspace rank 2 (or loop over all ranks if you want every one)
rank_to_plot = 2
res = rank_results[rank_to_plot]
plt.semilogy(res["iterations"], res["train_losses"]["forward_loss"],
             label=f"subspace rank={rank_to_plot} (train)", color="#CE0FFE", linestyle="-")
plt.semilogy(res["iterations"], res["test_losses"]["forward_loss"],
             label=f"subspace rank={rank_to_plot} (eval)", color="#CE0FFE", linestyle="--")

plt.xlabel("Epoch")
plt.ylabel("Forward loss")
plt.title("Train vs. Eval Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

'''
7. Save final models
'''

SAVE_DIR = MODEL_DIR / "fifth_robot_adaptation_results"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ---- Save full fine-tuned model ----
with open(SAVE_DIR / "fine_tune.jax", "wb") as f:
    pickle.dump(
        {
            "model_name": "fine_tune",
            "hyper": hyper,
            "params": ft_params,
            "iterations": ft_iterations,
            "train_losses": ft_train_losses,
            "test_losses": ft_test_losses,
            "final_test_loss": float(ft_test_loss),
        },
        f,
    )
print(f"Saved fine-tune model to: {SAVE_DIR / 'fine_tune.jax'}")

# ---- Save W0 (no-adaptation baseline) for reference ----
with open(SAVE_DIR / "w0_baseline.jax", "wb") as f:
    pickle.dump(
        {
            "model_name": "w0_baseline",
            "hyper": hyper,
            "params": W0,
            "final_test_loss": float(w0_test_loss),
        },
        f,
    )
print(f"Saved W0 baseline to: {SAVE_DIR / 'w0_baseline.jax'}")

# ---- Save subspace-adapted model for every rank ----
for rank, res in rank_results.items():
    fname = SAVE_DIR / f"subspace_rank{rank}.jax"
    with open(fname, "wb") as f:
        pickle.dump(
            {
                "model_name": f"subspace_rank{rank}",
                "hyper": hyper,
                "rank": rank,
                "params": res["params"],
                "alpha": res["alpha"],
                "n_trainable": res["n_trainable"],
                "iterations": res["iterations"],
                "train_losses": res["train_losses"],
                "test_losses": res["test_losses"],
                "final_test_loss": res["test_losses"]["forward_loss"][-1],
                # Subspace basis needed to reconstruct params from alpha later
                "U": np.array(subspace_for_rank(full_svd, rank)["U"]),
                "w0_flat": np.array(full_svd["w0_flat"]),
                "free_mask": np.array(full_svd["free_mask"]),
                "constrained_mask": np.array(full_svd["constrained_mask"]),
            },
            f,
        )
    print(f"Saved subspace rank-{rank} model to: {fname}")

# ---- Save the full SVD (so any rank can be re-sliced later without recomputation) ----
with open(SAVE_DIR / "full_svd.jax", "wb") as f:
    pickle.dump(full_svd, f)
print(f"Saved full SVD decomposition to: {SAVE_DIR / 'full_svd.jax'}")

print(f"\nAll models saved under: {SAVE_DIR}")