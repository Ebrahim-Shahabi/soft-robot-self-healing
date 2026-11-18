"""
Boundary-filtered training experiment
======================================
1. Plot raw robot2 data so you can inspect it and pick q_min / q_max.
2. Filter to only the samples inside [q_min, q_max].
3. Train scratch / finetune / subspace on the filtered data.
4. Evaluate on the filtered data (and optionally the full set).
"""

import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── import your existing modules ──────────────────────────────────────────────
import utils_dir as du
import pandas as pd

# ── copy in the pieces you need from your main script ────────────────────────
# import example_validation_meta_mark2 as subspace_training
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
import data_efficiency

# =============================================================================
# CONFIG — edit these
# =============================================================================
MODEL_DIR    = Path("./../models/static_models")
TRACKER_ROOT = Path(__file__).resolve().parents[2] / "exp_setup_ws" / "data" / "tracker_online"

# ── boundaries: set these after looking at the first plot ────────────────────
Q_MIN = None    # e.g. -0.8   (radians)   set to None to use all data
Q_MAX = None    # e.g.  0.8

N_EPOCHS   = 20_000
N_REPEATS  = 3
BATCH_SIZE = 256

LR = optax.warmup_cosine_decay_schedule(
    init_value=1e-2,
    peak_value=5e-2,
    warmup_steps=400,
    decay_steps=N_EPOCHS,
    end_value=1e-6,
)

FREE_LAYERS = None

# =============================================================================
# STEP 1 — load robot2 data
# =============================================================================
robot_dir = TRACKER_ROOT / "robot2"
ANGLES     = [0, 45, 135]

dfs_by_angle = {}
for angle in ANGLES:
    runs = du.get_video_runs(video_dir=robot_dir / "video", angle=angle, run_name_filter=None)
    if not runs:
        continue
    frames = []
    for motion, video_path in runs.items():
        df = du.load_df(
            label=f"robot2_{angle}deg_{motion}",
            video_path=video_path,
            online_dir=robot_dir / "online",
        )
        mask = (df["pressure[mBar]"] >= 60) & (df["angle[rad]"] != 0)
        frames.append(df[mask])
    if frames:
        dfs_by_angle[angle] = pd.concat(frames, ignore_index=True)

# =============================================================================
# STEP 2 — plot raw data so you can pick q_min / q_max
# =============================================================================
fig, axes = plt.subplots(1, len(dfs_by_angle), figsize=(5 * len(dfs_by_angle), 4), sharey=False)
if len(dfs_by_angle) == 1:
    axes = [axes]

for ax, (angle, df) in zip(axes, sorted(dfs_by_angle.items())):
    angles_deg = np.rad2deg(df["angle[rad]"].values)
    pressure   = df["pressure[mBar]"].values
    ax.scatter(angles_deg, pressure, s=4, alpha=0.4, color="#2563eb")
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Pressure [mBar]")
    ax.set_title(f"Robot 2 — {angle}°")
    ax.grid(True, alpha=0.3)

fig.suptitle("Robot 2 raw data — pick your Q_MIN / Q_MAX boundaries", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.show()

# =============================================================================
# STEP 3 — define boundaries and filter (edit Q_MIN / Q_MAX at top then rerun)
# =============================================================================
# We work with 0deg for the experiment — change as needed
TARGET_ANGLE = 0
df_target = dfs_by_angle[TARGET_ANGLE].copy()

raw_q   = df_target["angle[rad]"].values
raw_tau = df_target["pressure[mBar]"].values
# FULL DATASET (for final evaluation)
x_full, y_full = du.extract_states_targets(df_target)

q_full   = jnp.array(x_full, dtype=jnp.float32)
tau_full = jnp.array(y_full.reshape(-1, 1), dtype=jnp.float32)

# 45 - 55 worked
Q_MIN = np.deg2rad(45)
Q_MAX = np.deg2rad(55)
if Q_MIN is not None or Q_MAX is not None:
    q_lo = Q_MIN if Q_MIN is not None else -np.inf
    q_hi = Q_MAX if Q_MAX is not None else  np.inf
    keep = (raw_q >= q_lo) & (raw_q <= q_hi) # | (raw_q >= np.deg2rad(50)) & (raw_q <= np.deg2rad(55))
    df_filtered = df_target[keep].reset_index(drop=True)
    print(f"\n[Filter] Kept {keep.sum()} / {len(keep)} samples  "
          f"(q in [{np.rad2deg(q_lo):.1f}°, {np.rad2deg(q_hi):.1f}°])")
else:
    df_filtered = df_target
    print(f"\n[Filter] No boundary set — using all {len(df_filtered)} samples")

x, y = du.extract_states_targets(df_filtered)
q_data   = jnp.array(x,                dtype=jnp.float32)
tau_data = jnp.array(y.reshape(-1, 1), dtype=jnp.float32)

# plot filtered vs full to confirm
fig, ax = plt.subplots(figsize=(8, 4))
ax.scatter(np.rad2deg(raw_q), raw_tau,
           s=4, alpha=0.2, color="grey",  label="excluded")
ax.scatter(np.rad2deg(df_filtered["angle[rad]"].values),
           df_filtered["pressure[mBar]"].values,
           s=4, alpha=0.5, color="#2563eb", label="training region")

if Q_MIN is not None:
    ax.axvline(np.rad2deg(Q_MIN), color="red",    linestyle="--", label="Q_MIN")
if Q_MAX is not None:
    ax.axvline(np.rad2deg(Q_MAX), color="orange",  linestyle="--", label="Q_MAX")

ax.set_xlabel("Angle [deg]")
ax.set_ylabel("Pressure [mBar]")
ax.set_title(f"Robot 2 — {TARGET_ANGLE}° with boundaries")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# =============================================================================
# STEP 4 — load base model + task weights, build subspace
# =============================================================================
with open(MODEL_DIR / "r1d1.jax", "rb") as f:
    base_data = pickle.load(f)

hyper   = base_data["hyper"]
W0_star = base_data["params"]

TASK_MODEL_FILES = {
    "r1d1_45deg" : MODEL_DIR / "r1d1_45deg.jax",
    "r1d1_135deg": MODEL_DIR / "r1d1_135deg.jax",
    "r1d2_0deg"  : MODEL_DIR / "r1d2_0deg.jax",
    "r2d0_135deg": MODEL_DIR / "r2d0_135deg.jax",
}

task_weights = {}
for name, fpath in TASK_MODEL_FILES.items():
    with open(fpath, "rb") as f:
        task_weights[name] = pickle.load(f)["params"]

subspace = compute_task_subspace(
    w0_star=W0_star,
    task_weights=task_weights,
    variance_threshold=0.90,
    max_rank = 1,
    free_layer_names=FREE_LAYERS,
)

sp = SubspaceParams(subspace=subspace, w0_star=W0_star, free_layer_names=FREE_LAYERS)

# =============================================================================
# STEP 5 — build haiku apply functions
# =============================================================================
activation  = jax.nn.softplus
hidden_pot  = (hyper["n_width"],) * hyper["n_depth"]
hidden_inp  = (hyper["n_width"] // 2,) * (hyper["n_depth"] - 1)
n_dof       = hyper["n_dof"]
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
# STEP 6 — train all three methods on the filtered data
# =============================================================================
N = q_data.shape[0]
print(f"\nTraining on {N} filtered samples\n")

# ---- subspace ---------------------------------------------------------------
@jax.jit
def _subspace_loss(trainable, q_b, tau_b):
    full_p = sp.reconstruct(trainable)
    loss, _ = loss_fn(full_p, q_b, tau_b, _potential, _input_mat)
    return loss + 1e-1 * jnp.sum(trainable["alpha"] ** 2), {}

trainable = sp.init_trainable()
opt       = optax.adam(LR)
state     = opt.init(trainable)

@jax.jit
def _step_sub(tr, st, q_b, tau_b):
    (l, _), g = jax.value_and_grad(_subspace_loss, has_aux=True)(tr, q_b, tau_b)
    upd, st   = opt.update(g, st)
    return optax.apply_updates(tr, upd), st, l

sub_losses = []
for epoch in range(N_EPOCHS):
    bi = np.random.permutation(N)[:BATCH_SIZE]
    trainable, state, l = _step_sub(trainable, state, q_data[bi], tau_data[bi])
    sub_losses.append(float(l))
    if epoch % 2000 == 0 or epoch == N_EPOCHS - 1:
        print(f"  [Subspace] epoch {epoch:5d}  loss={float(l):.6f}")

sub_params = sp.reconstruct(trainable)

# ---- fine-tune from W0_star -------------------------------------------------
ft_params = data_efficiency._train_plain(
    W0_star, q_data, tau_data, _potential, _input_mat,
    N_EPOCHS, LR, BATCH_SIZE, loss_fn=loss_fn,
)

# ---- scratch ----------------------------------------------------------------
scratch_init = data_efficiency._make_scratch_params(W0_star, rng_seed=0)
sc_params = data_efficiency._train_plain(
    scratch_init, q_data, tau_data, _potential, _input_mat,
    N_EPOCHS, LR, BATCH_SIZE, loss_fn=loss_fn,
)

# =============================================================================
# STEP 7 — predict and plot
# =============================================================================
angle_plot = np.rad2deg(df_filtered["angle[rad]"].values)

def predict(params):
    f = forward_model(params, _potential, _input_mat)
    _, tau_p = jax.vmap(f)(q_data)
    return np.array(tau_p).ravel()

tau_sub = predict(sub_params)
tau_ft  = predict(ft_params)
tau_sc  = predict(sc_params)
tau_w0  = predict(W0_star)
y_true  = np.array(tau_data).ravel()

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
fig.suptitle(
    f"Robot 2 — {TARGET_ANGLE}° | boundary [{np.rad2deg(Q_MIN) if Q_MIN else '–'}°, "
    f"{np.rad2deg(Q_MAX) if Q_MAX else '–'}°]",
    fontsize=13, fontweight="bold",
)

plot_cfg = [
    (axes[0, 0], tau_w0,  "W0_star (no adaptation)", "#6b7280"),
    (axes[0, 1], tau_sub, "Subspace",                 "#2563eb"),
    (axes[1, 0], tau_ft,  "Fine-tune",                "#16a34a"),
    (axes[1, 1], tau_sc,  "Scratch",                  "#dc2626"),
]

for ax, tau_p, title, color in plot_cfg:
    ax.scatter(angle_plot, y_true,  s=4, alpha=0.3, color="grey",  label="data")
    ax.scatter(angle_plot, tau_p,   s=4, alpha=0.5, color=color,   label="predicted")
    loss_val, _ = loss_fn(
        sub_params if "Subspace" in title else
        ft_params  if "Fine"     in title else
        sc_params  if "Scratch"  in title else W0_star,
        q_data, tau_data, _potential, _input_mat,
    )
    ax.set_title(f"{title}  (loss={float(loss_val):.5f})")
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Pressure [mBar]")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# final loss summary
for name, params in [("W0_star", W0_star), ("Subspace", sub_params),
                     ("Fine-tune", ft_params), ("Scratch", sc_params)]:
    l, _ = loss_fn(params, q_data, tau_data, _potential, _input_mat)
    print(f"  {name:12s}: {float(l):.6f}")

# =============================================================================
# STEP 8 — predict and plot on FULL DATASET
# =============================================================================

angle_full = np.rad2deg(df_target["angle[rad]"].values)
y_full_true = np.array(tau_full).ravel()

def predict_on(params, q_eval):
    f = forward_model(params, _potential, _input_mat)
    _, tau_p = jax.vmap(f)(q_eval)
    return np.array(tau_p).ravel()

tau_sub_full = predict_on(sub_params, q_full)
tau_ft_full  = predict_on(ft_params, q_full)
tau_sc_full  = predict_on(sc_params, q_full)
tau_w0_full  = predict_on(W0_star, q_full)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
fig.suptitle(
    f"Robot 2 — {TARGET_ANGLE}° | predictions on FULL dataset "
    f"| trained on [{np.rad2deg(Q_MIN):.1f}°, {np.rad2deg(Q_MAX):.1f}°]",
    fontsize=13,
    fontweight="bold",
)

plot_cfg_full = [
    (axes[0, 0], tau_w0_full,  "W0_star (no adaptation)", "#6b7280", W0_star),
    (axes[0, 1], tau_sub_full, "Subspace",                 "#2563eb", sub_params),
    (axes[1, 0], tau_ft_full,  "Fine-tune",                "#16a34a", ft_params),
    (axes[1, 1], tau_sc_full,  "Scratch",                  "#dc2626", sc_params),
]

for ax, tau_p, title, color, params in plot_cfg_full:
    ax.scatter(angle_full, y_full_true,
               s=4, alpha=0.25, color="grey", label="full data")

    ax.scatter(angle_full, tau_p,
               s=4, alpha=0.5, color=color, label="predicted")

    # show training boundaries
    if Q_MIN is not None:
        ax.axvline(np.rad2deg(Q_MIN), color="red", linestyle="--", alpha=0.8)
    if Q_MAX is not None:
        ax.axvline(np.rad2deg(Q_MAX), color="orange", linestyle="--", alpha=0.8)

    loss_full, _ = loss_fn(params, q_full, tau_full,
                           _potential, _input_mat)

    ax.set_title(f"{title}  full loss={float(loss_full):.5f}")
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Pressure [mBar]")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()