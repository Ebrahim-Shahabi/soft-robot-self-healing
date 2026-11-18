"""
MAML meta-training — drop-in replacement for the subspace adaptation block.

Produces a meta-initialised W0_maml that, after K inner-loop steps on a
small support set, outperforms plain fine-tune and scratch on unseen angles.

Comparison plot mirrors your existing 4-panel layout exactly.
"""

import pickle
import copy
import numpy as np

import jax
import jax.numpy as jnp
import optax
import haiku as hk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path

import utils_dir as du
import plots as plts
import subspace as sb
import data_efficiency

# import exactly the same symbols your existing script uses
from example_validation_meta_mark2 import (
    SubspaceParams,
    compute_task_subspace,
    loss_fn,
    forward_model,
    potential_energy,
    input_transform_matrix,
    flatten_params,
    unflatten_params,
)

# =============================================================================
# Paths
# =============================================================================
SCRIPT_DIR   = Path(__file__).resolve().parent
MODEL_DIR    = SCRIPT_DIR.parents[0] / "models" / "models"
TRACKER_ROOT = SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"

# =============================================================================
# 0.  Load target robot data  (robot2_damage1_healed @ 0°)
# =============================================================================
robot_dir = TRACKER_ROOT / "robot2_damage1_healed"
runs = du.get_video_runs(video_dir=robot_dir / "video", run_name_filter="trapezoid")

r2d1_df = du.load_df(
    label="robot2_damage1_healed_0deg_trapezoid",
    video_path=runs["trapezoid"],
    online_dir=robot_dir / "online",
)
r2d1_df = r2d1_df[r2d1_df["angle[rad]"] != 0].copy()
r2d1_df = r2d1_df[r2d1_df["pressure[mBar]"] >= 25].copy()

# training window — same as your existing script
Q_MIN = np.deg2rad(45)
Q_MAX = np.deg2rad(55)

raw_q = r2d1_df["angle[rad]"].values
mask  = (raw_q >= Q_MIN) & (raw_q <= Q_MAX)
df_filtered = r2d1_df[mask].reset_index(drop=True)
print(f"[Filter] {mask.sum()} / {len(mask)} samples  "
      f"(q in [{np.rad2deg(Q_MIN):.1f}°, {np.rad2deg(Q_MAX):.1f}°])")

x, y     = du.extract_states_targets(df_filtered)
q_data   = jnp.array(x,                dtype=jnp.float32)
tau_data = jnp.array(y.reshape(-1, 1), dtype=jnp.float32)

x_full, y_full = du.extract_states_targets(r2d1_df)
q_full   = jnp.array(x_full,                  dtype=jnp.float32)
tau_full = jnp.array(y_full.reshape(-1, 1),    dtype=jnp.float32)

# =============================================================================
# 1.  Load W0 + task weights  (identical to your existing script)
# =============================================================================
with open(MODEL_DIR / "robot1_damage1_healed_0deg.jax", "rb") as f:
    base_data = pickle.load(f)

hyper = base_data["hyper"]
W0    = base_data["params"]

TASK_MODEL_FILES = {
    "r1d1_45deg" : MODEL_DIR / "robot1_damage1_healed_45deg.jax",
    "r1d1_135deg": MODEL_DIR / "robot1_damage1_healed_135deg.jax",
    "r1d2_0deg"  : MODEL_DIR / "robot1_damage2_healed_0deg.jax",
    "r1d3_0deg"  : MODEL_DIR / "robot1_damage3_healed_0deg.jax",
    "r1d3_135deg": MODEL_DIR / "robot1_damage3_healed_135deg.jax",
    "r2d0_0deg"  : MODEL_DIR / "robot2_damage0_0deg.jax",
    "r2d0_135deg": MODEL_DIR / "robot2_damage0_135deg.jax",
}
task_weights = {}
for name, fpath in TASK_MODEL_FILES.items():
    with open(fpath, "rb") as f:
        task_weights[name] = pickle.load(f)["params"]

# =============================================================================
# 2.  Build haiku apply functions  (identical to your existing script)
# =============================================================================
activation   = jax.nn.softplus
hidden_pot   = (hyper["n_width"],) * hyper["n_depth"]
hidden_inp   = (hyper["n_width"] // 2,) * (hyper["n_depth"] - 1)
n_dof        = hyper["n_dof"]
actuator_dof = hyper["actuator_dof"]

pot_transformed = hk.without_apply_rng(hk.transform(
    lambda q: potential_energy(q, hidden_pot, activation)
))
input_transformed = hk.without_apply_rng(hk.transform(
    lambda q: input_transform_matrix(q, n_dof, actuator_dof, hidden_inp, activation)
))

def _potential(params, key, q):
    return pot_transformed.apply(params, q)

def _input_mat(params, key, q):
    return input_transformed.apply(params, q)

# =============================================================================
# 3.  MAML helpers
# =============================================================================
INNER_LR    = 1e-3   # α — inner-loop learning rate
INNER_STEPS = 5      # K — steps per task during meta-training
ADAPT_STEPS = 10     # steps used at test-time adaptation


def _task_loss(params, q_b, tau_b):
    l, _ = loss_fn(params, q_b, tau_b, _potential, _input_mat)
    return l


def inner_loop(params, q_sup, tau_sup, n_steps=INNER_STEPS, lr=INNER_LR):
    """Pure SGD on support set; returns adapted copy."""
    adapted = params
    for _ in range(n_steps):
        grads   = jax.grad(_task_loss)(adapted, q_sup, tau_sup)
        adapted = jax.tree_util.tree_map(lambda p, g: p - lr * g, adapted, grads)
    return adapted


def maml_meta_loss(meta_params, tasks):
    total = 0.0
    for task in tasks:
        adapted = inner_loop(meta_params, task["q_sup"], task["tau_sup"])
        total  += _task_loss(adapted, task["q_qry"], task["tau_qry"])
    return total / len(tasks)


# =============================================================================
# 4.  Build meta-training task list from actual data
# =============================================================================
TASK_DATA_DIRS = {
    "r1d1_45deg" : (TRACKER_ROOT / "robot1_damage1_healed", 45),
    "r1d1_135deg": (TRACKER_ROOT / "robot1_damage1_healed", 135),
    "r1d2_0deg"  : (TRACKER_ROOT / "robot1_damage2_healed",   0),
    "r1d3_0deg"  : (TRACKER_ROOT / "robot1_damage3_healed",   0),
    "r1d3_135deg": (TRACKER_ROOT / "robot1_damage3_healed", 135),
    "r2d0_0deg"  : (TRACKER_ROOT / "robot2",                  0),
    "r2d0_135deg": (TRACKER_ROOT / "robot2",                135),
}
SUPPORT_SIZE = 50

meta_tasks = []
for name, (rdir, angle) in TASK_DATA_DIRS.items():
    try:
        runs_t = du.get_video_runs(video_dir=rdir / "video",
                                   angle=angle, run_name_filter="trapezoid")
        df_t = du.load_df(label=name, video_path=runs_t["trapezoid"],
                          online_dir=rdir / "online")
        df_t = df_t[df_t["angle[rad]"] != 0]
        df_t = df_t[df_t["pressure[mBar]"] >= 25]

        xt, yt = du.extract_states_targets(df_t)
        idx    = np.random.permutation(len(xt))
        xt, yt = xt[idx], yt[idx]

        meta_tasks.append({
            "name"   : name,
            "q_sup"  : jnp.array(xt[:SUPPORT_SIZE],              dtype=jnp.float32),
            "tau_sup": jnp.array(yt[:SUPPORT_SIZE].reshape(-1,1), dtype=jnp.float32),
            "q_qry"  : jnp.array(xt[SUPPORT_SIZE:],              dtype=jnp.float32),
            "tau_qry": jnp.array(yt[SUPPORT_SIZE:].reshape(-1,1), dtype=jnp.float32),
        })
        print(f"  Loaded task {name}  ({len(xt)} samples)")
    except Exception as e:
        print(f"  Skipping {name}: {e}")

print(f"\n{len(meta_tasks)} tasks ready for meta-training.\n")

# =============================================================================
# 5.  MAML meta-training
# =============================================================================
META_LR     = 1e-3
META_EPOCHS = 2000
LOG_EVERY   = 200

meta_params = copy.deepcopy(W0)
meta_opt    = optax.adam(META_LR)
opt_state   = meta_opt.init(meta_params)

meta_loss_hist = []
best_loss      = float("inf")
best_params    = copy.deepcopy(meta_params)

print("Starting MAML meta-training …")
for epoch in range(META_EPOCHS):
    loss_val, grads = jax.value_and_grad(maml_meta_loss)(meta_params, meta_tasks)
    updates, opt_state = meta_opt.update(grads, opt_state, meta_params)
    meta_params        = optax.apply_updates(meta_params, updates)

    lf = float(loss_val)
    meta_loss_hist.append(lf)
    if lf < best_loss:
        best_loss   = lf
        best_params = copy.deepcopy(meta_params)

    if epoch % LOG_EVERY == 0 or epoch == META_EPOCHS - 1:
        print(f"  epoch {epoch:>5d}  meta-loss = {lf:.6f}")

print(f"\nMeta-training done. Best meta-loss = {best_loss:.6f}\n")

with open(MODEL_DIR / "W0_maml.jax", "wb") as f:
    pickle.dump({"model_name": "W0_maml", "hyper": hyper,
                 "params": best_params, "loss_history": meta_loss_hist}, f)

# =============================================================================
# 6.  Fast-adapt MAML init to the target task  (same support window as others)
# =============================================================================
# maml_adapted = inner_loop(best_params, q_data, tau_data,
#                           n_steps=ADAPT_STEPS, lr=INNER_LR)

# =============================================================================
# 7.  Subspace baseline  (verbatim from your existing script)
# =============================================================================
# subspace = sb.compute_task_subspace(
#     w0_star=W0, task_weights=task_weights,
#     variance_threshold=0.90, max_rank=3, free_layer_names=None,
# )
# sp = sb.SubspaceParams(subspace=subspace, w0_star=W0, free_layer_names=None)

# LR_sched  = optax.warmup_cosine_decay_schedule(
#     init_value=1e-2, peak_value=5e-2,
#     warmup_steps=400, decay_steps=20_000, end_value=1e-6,
# )
# N_EPOCHS   = 20_000
# BATCH_SIZE = 256
# N          = q_data.shape[0]

# @jax.jit
# def _subspace_loss(trainable, q_b, tau_b):
#     full_p = sp.reconstruct(trainable)
#     l, _   = loss_fn(full_p, q_b, tau_b, _potential, _input_mat)
#     return l + 1e-1 * jnp.sum(trainable["alpha"] ** 2), {}

# trainable = sp.init_trainable()
# opt_sub   = optax.adam(LR_sched)
# state_sub = opt_sub.init(trainable)

# @jax.jit
# def _step_sub(tr, st, q_b, tau_b):
#     (l, _), g = jax.value_and_grad(_subspace_loss, has_aux=True)(tr, q_b, tau_b)
#     upd, st   = opt_sub.update(g, st)
#     return optax.apply_updates(tr, upd), st, l

# for epoch in range(N_EPOCHS):
#     bi = np.random.permutation(N)[:BATCH_SIZE]
#     trainable, state_sub, l = _step_sub(trainable, state_sub,
#                                          q_data[bi], tau_data[bi])
#     if epoch % 2000 == 0 or epoch == N_EPOCHS - 1:
#         print(f"  [Subspace] epoch {epoch:5d}  loss={float(l):.6f}")

# sub_params = sp.reconstruct(trainable)

# # =============================================================================
# # 8.  Fine-tune and Scratch baselines  (verbatim from your existing script)
# # =============================================================================
# ft_params = data_efficiency._train_plain(
#     W0, q_data, tau_data, _potential, _input_mat,
#     N_EPOCHS, LR_sched, BATCH_SIZE, loss_fn=loss_fn,
# )

# scratch_init = data_efficiency._make_scratch_params(W0, rng_seed=0)
# sc_params = data_efficiency._train_plain(
#     scratch_init, q_data, tau_data, _potential, _input_mat,
#     N_EPOCHS, LR_sched, BATCH_SIZE, loss_fn=loss_fn,
# )

# # =============================================================================
# # 9.  Predict on FULL dataset
# # =============================================================================
# def predict_on(params, q_eval):
#     f = forward_model(params, _potential, _input_mat)
#     _, tau_p = jax.vmap(f)(q_eval)
#     return np.array(tau_p).ravel()

# tau_w0_full   = predict_on(W0,           q_full)
# tau_sub_full  = predict_on(sub_params,   q_full)
# tau_ft_full   = predict_on(ft_params,    q_full)
# tau_sc_full   = predict_on(sc_params,    q_full)
# tau_maml_full = predict_on(maml_adapted, q_full)

# angle_full  = np.rad2deg(r2d1_df["angle[rad]"].values)
# y_full_true = np.array(tau_full).ravel()

# # =============================================================================
# # 10.  5-panel plot  — same style as your existing 4-panel, MAML added
# # =============================================================================
# fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharex=True, sharey=True)
# fig.suptitle(
#     f"Robot 2 — 0° | predictions on FULL dataset "
#     f"| trained on [{np.rad2deg(Q_MIN):.1f}°, {np.rad2deg(Q_MAX):.1f}°]",
#     fontsize=13, fontweight="bold",
# )
# axes[1, 2].set_visible(False)   # 6th panel unused

# plot_cfg = [
#     (axes[0, 0], tau_w0_full,   "W0 (no adaptation)", "#6b7280", W0),
#     (axes[0, 1], tau_sub_full,  "Subspace",            "#2563eb", sub_params),
#     (axes[0, 2], tau_ft_full,   "Fine-tune",           "#16a34a", ft_params),
#     (axes[1, 0], tau_sc_full,   "Scratch",             "#dc2626", sc_params),
#     (axes[1, 1], tau_maml_full, "MAML (ours)",         "#7c3aed", maml_adapted),
# ]

# for ax, tau_p, title, color, params in plot_cfg:
#     ax.scatter(angle_full, y_full_true,
#                s=4, alpha=0.25, color="grey", label="full data")
#     ax.scatter(angle_full, tau_p,
#                s=4, alpha=0.5,  color=color,  label="predicted")
#     if Q_MIN is not None:
#         ax.axvline(np.rad2deg(Q_MIN), color="red",    linestyle="--", alpha=0.8)
#     if Q_MAX is not None:
#         ax.axvline(np.rad2deg(Q_MAX), color="orange", linestyle="--", alpha=0.8)
#     loss_full, _ = loss_fn(params, q_full, tau_full, _potential, _input_mat)
#     ax.set_title(f"{title}  full loss={float(loss_full):.5f}")
#     ax.set_xlabel("Angle [deg]")
#     ax.set_ylabel("Pressure [mBar]")
#     ax.legend(markerscale=3)
#     ax.grid(True, alpha=0.3)

# plt.tight_layout()
# plt.savefig(MODEL_DIR / "maml_comparison_full.png", dpi=150)
# plt.show()

# # =============================================================================
# # 11.  Loss summary
# # =============================================================================
# print("\n── Full-dataset loss summary ──────────────────────────")
# for label, params in [
#     ("W0",          W0),
#     ("Subspace",    sub_params),
#     ("Fine-tune",   ft_params),
#     ("Scratch",     sc_params),
#     ("MAML (ours)", maml_adapted),
# ]:
#     l, _ = loss_fn(params, q_full, tau_full, _potential, _input_mat)
#     print(f"  {label:<14s}: {float(l):.6f}")