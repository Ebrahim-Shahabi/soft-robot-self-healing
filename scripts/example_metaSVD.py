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
import utils_dir as du
import pandas as pd
import plots as plts
import subspace as sb
import DeLaN_model_static as DeLan
import tools
import LoRA as lora

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_DIR = SCRIPT_DIR.parents[0] / "models" / "models"
TRACKER_ROOT = (
    SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"
)

Q_MIN = None    
Q_MAX = None    


robot_dir = TRACKER_ROOT / "robot2_damage1_healed"
deg_key = "0deg"
runs = du.get_video_runs(
    video_dir= robot_dir / "video",
    run_name_filter="trapezoid",
)
print(runs.keys())
print(f"robot2_damage1_healed | {deg_key}: found {len(runs)} runs in directory : \n")
print(runs["trapezoid"])

r2d1_path = du.find_csv_for_video(video_path= runs["trapezoid"], online_dir=robot_dir / "online")

r2d1_df = du.load_df(
    label=f"robot2_damage1_healed_{deg_key}_trapezoid",
    video_path=runs["trapezoid"],
    online_dir=robot_dir / "online",
)

r2d1_df = r2d1_df[r2d1_df["angle[rad]"] != 0].copy() # remove zero angle rows because they are not real data points but just the initial state.
r2d1_df = r2d1_df[r2d1_df["pressure[mBar]"] >= 25].copy() # remove low pressure rows because they are not real data points but just the initial noise.

# plts.plot_one_motion(
#     robot_name="robot2_damage1_healed",
#     motion="0deg_trapezoid",
#     dfs={"trapezoid": r2d1_df},
#     save_flag=False,
# )

'''
1. Define boundaries for training.
'''
raw_q   = r2d1_df["angle[rad]"].values
raw_tau = r2d1_df["pressure[mBar]"].values
Q_MIN = np.deg2rad(35)
Q_MAX =  np.deg2rad(45)

if Q_MIN is not None or Q_MAX is not None:
    q_lo = Q_MIN if Q_MIN is not None else -np.inf
    q_hi = Q_MAX if Q_MAX is not None else  np.inf
    
    mask = (raw_q >= q_lo) & (raw_q <= q_hi) # | (raw_q >= np.deg2rad(50)) & (raw_q <= np.deg2rad(55))
    df_filtered = r2d1_df[mask].reset_index(drop=True)
    print(f"\n[Filter] Kept {mask.sum()} / {len(mask)} samples  "
          f"(q in [{np.rad2deg(q_lo):.1f}°, {np.rad2deg(q_hi):.1f}°])")
else:
    df_filtered = r2d1_df
    print(f"\n[Filter] No boundary set — using all {len(df_filtered)} samples")

x, y = du.extract_states_targets(df_filtered)

q_data   = jnp.array(x,                dtype=jnp.float32)
tau_data = jnp.array(y.reshape(-1, 1), dtype=jnp.float32)
N = q_data.shape[0]
print(f"\nTraining on {N} filtered samples\n")
plts.plot_boundaries(r2d1_df, df_filtered, Q_MIN=Q_MIN, Q_MAX=Q_MAX)

'''
2. Load base model.
'''
with open(MODEL_DIR / "robot1_damage1_healed_0deg.jax", "rb") as f:
    base_data = pickle.load(f)
# with open(MODEL_DIR / "W0_maml.jax", "rb") as f:
#     base_data = pickle.load(f)

hyper   = base_data["hyper"]
W0 = base_data["params"]

TASK_MODEL_FILES = {
    "r1d1_45deg" : MODEL_DIR / "robot1_damage1_healed_45deg.jax",
    "r1d1_135deg": MODEL_DIR / "robot1_damage1_healed_135deg.jax",
    "r1d2_0deg"  : MODEL_DIR / "robot1_damage2_healed_0deg.jax",
    "r1d3_0deg": MODEL_DIR / "robot1_damage3_healed_0deg.jax",
    "r1d3_135deg": MODEL_DIR / "robot1_damage3_healed_135deg.jax",
    "r2d0_0deg"  : MODEL_DIR / "robot2_damage0_0deg.jax",
    "r2d0_135deg": MODEL_DIR / "robot2_damage0_135deg.jax"
}

# TASK_MODEL_FILES = {
#     "r1d1_45deg" : MODEL_DIR / "robot1_damage1_healed_45deg_maml_onlyangle.jax",
#     "r1d1_135deg": MODEL_DIR / "robot1_damage1_healed_135deg_maml_onlyangle.jax",
#     "r1d2_0deg"  : MODEL_DIR / "robot1_damage2_healed_0deg_maml_onlyangle.jax",
#     "r1d3_0deg": MODEL_DIR / "robot1_damage3_healed_0deg_maml_onlyangle.jax",
#     "r1d3_135deg": MODEL_DIR / "robot1_damage3_healed_135deg_maml_onlyangle.jax",
#     "r2d0_0deg"  : MODEL_DIR / "robot2_damage0_0deg_maml_onlyangle.jax",
#     "r2d0_135deg": MODEL_DIR / "robot2_damage0_135deg_maml_onlyangle.jax"
# }

task_weights = {}
for name, fpath in TASK_MODEL_FILES.items():
    with open(fpath, "rb") as f:
        task_weights[name] = pickle.load(f)["params"]

# print(len(task_weights))

subspace = sb.compute_task_subspace(
    w0_star=W0,
    task_weights=task_weights,
    variance_threshold=0.95,
    max_rank = None,
    free_layer_names=None,
)

# plts.plot_subspace_analysis(subspace)

sp = sb.SubspaceParams(subspace=subspace, w0_star=W0, free_layer_names=None)

'''
STEP 5 — build haiku apply functions
'''

N_EPOCHS   = 20000
N_REPEATS  = 3
BATCH_SIZE = 256
TARGET_ANGLE = 0

LR = optax.warmup_cosine_decay_schedule(
    init_value=1e-2,
    peak_value=5e-2,
    warmup_steps=100,
    decay_steps=N_EPOCHS,
    end_value=1e-3,
)
activations = {
        "tanh": jnp.tanh,
        "softplus": jax.nn.softplus,
        "sigmoid": jax.nn.sigmoid,
}
activation  = activations[hyper["activation1"]]
hidden_pot  = (hyper["n_width"],) * hyper["n_depth"]
hidden_inp  = (hyper["n_width"] // 2,) * (hyper["n_depth"] - 1)
n_dof       = hyper["n_dof"]
actuator_dof = hyper["actuator_dof"]

pot_transformed = hk.without_apply_rng(hk.transform(
    lambda q: DeLan.potential_energy(q, hidden_pot, activation)
))
input_transformed = hk.without_apply_rng(hk.transform(
    lambda q: DeLan.input_transform_matrix(q, n_dof, actuator_dof, hidden_inp, activation)
))

def _potential(params, key, q):
    return pot_transformed.apply(params, q)

def _input_mat(params, key, q):
    return input_transformed.apply(params, q)

'''
3. Train models.
'''
# ---- subspace ---------------------------------------------------------------
@jax.jit
def _subspace_loss(trainable, q_b, tau_b, _potential=_potential, _input_mat=_input_mat):
    full_p = sp.reconstruct(trainable)
    loss, _ = DeLan.loss_fn(full_p, q_b, tau_b, _potential, _input_mat)
    return loss + 1e-3 * jnp.sum(trainable["alpha"] ** 2), {}

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

# ---- fine-tune from W0 -------------------------------------------------
model_name = "fine_tune"
model = tools.setup_static_delan(q_data, tau_data, hyper, seed=42, model_name=model_name, initial_params=W0)
results = tools.train_static_model(model, hyper, make_plot=False)
ft_params = results["params"]


# ---- scratch ----------------------------------------------------------------
# model_name = "scratch"
# model = tools.setup_static_delan(q_data, tau_data, hyper, seed=42, model_name=model_name, initial_params=None)
# results = tools.train_static_model(model, hyper)
# sc_params = results["params"]

# ---- one shot svd ----------------------------------------------------------------

W2 = task_weights["r2d0_0deg"]
model_name = "svd"
model = tools.setup_oneshot_delan_svd(q_data, tau_data, hyper, seed=42, model_name=model_name, initial_params=W2)
results = tools.train_static_model(model, hyper)
sc_params = results["params"]

def svd_params_to_dense(svd_params):
    dense = {}
    for mod_name, leaf in svd_params.items():
        new_leaf = {}

        if "U" in leaf and "log_s" in leaf and "Vt" in leaf:
            # hidden layers: reconstruct W and transpose
            U     = leaf["U"]
            log_s = leaf["log_s"]
            Vt    = leaf["Vt"]
            W     = (U * jnp.exp(log_s)[None, :]) @ Vt  # (out, in)
            new_leaf["w"] = W.T                           # (in, out) for hk.Linear

        elif "w" in leaf:
            w = leaf["w"]
            if w.ndim == 1:
                # first layer: (out,) -> (1, out)  or last layer: (in,) -> (in, 1)
                # disambiguate by module name
                if mod_name.endswith("linear_0"):
                    new_leaf["w"] = w.reshape(1, -1)   # (1, out)
                else:
                    new_leaf["w"] = w.reshape(-1, 1)   # (in, 1)
            else:
                new_leaf["w"] = w

        if "b" in leaf:
            new_leaf["b"] = leaf["b"]

        dense_name = mod_name.replace("/linear_", "/~/linear_")
        dense[dense_name] = new_leaf

    return dense

sc_params_dense = {
    "potential":       svd_params_to_dense(sc_params["potential"]),
    "input_transform": svd_params_to_dense(sc_params["input_transform"]),
}
sc_params = sc_params_dense
# ---- LoRA ----------------------------------------------------------------
model_name = "LoRA"
model = lora.LoRAParams(w0_star=W0, rank=6, rng_seed=42)
results = tools.train_lora_delan(model, q_data, tau_data, _potential, _input_mat, n_epochs=N_EPOCHS)
lora_params = results["params"]
print(lora_params.keys())


'''
4. Make plots.
'''
angle_plot = np.rad2deg(df_filtered["angle[rad]"].values)

def predict(params):
    f = DeLan.forward_model(params, None, _potential, _input_mat)
    _, tau_p = jax.vmap(f)(q_data)
    return np.array(tau_p).ravel()*100 # convert to mBar for plotting

tau_sub = predict(sub_params)
tau_ft  = predict(ft_params)
tau_sc  = predict(sc_params)

tau_lora = predict(lora_params)
y_true  = np.array(tau_data).ravel()*100 # convert to mBar for plotting

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
fig.suptitle(
    f"Robot 2 — {TARGET_ANGLE}° | boundary [{np.rad2deg(Q_MIN) if Q_MIN else '–'}°, "
    f"{np.rad2deg(Q_MAX) if Q_MAX else '–'}°]",
    fontsize=13, fontweight="bold",
)

plot_cfg = [
    
    (axes[0, 0], tau_sc, "Scratch",                 "#eb2525"),
    (axes[0, 1], tau_ft,  "Fine-tune",                "#16a34a"),
    (axes[1, 0], tau_lora, f"LoRA rank={model.rank}",                   "#0c2dea"),
    (axes[1, 1], tau_sub,  "Subspace",     "#CE0FFE"),

]

for ax, tau_p, title, color in plot_cfg:
    ax.scatter(angle_plot, y_true,  s=4, alpha=0.3, color="grey",  label="data")
    ax.scatter(angle_plot, tau_p,   s=4, alpha=0.5, color=color,   label="predicted")
    loss_val, _ = DeLan.loss_fn(
        sub_params if "Subspace" in title else
        ft_params  if "Fine"     in title else
        sc_params  if "Scratch"  in title else 
        lora_params if "LoRA"     in title else None,
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
for name, params in [("Subspace", sub_params),("Fine-tune", ft_params), 
                     ("Scratch", sc_params), ("LoRA", lora_params)]:
    l, _ = DeLan.loss_fn(params, q_data, tau_data, _potential, _input_mat)
    print(f"  {name:12s}: {float(l):.6f}")

# predict and plot on FULL DATASET
raw_q   = r2d1_df["angle[rad]"].values
raw_tau = r2d1_df["pressure[mBar]"].values
# FULL DATASET (for final evaluation)
x_full, y_full = du.extract_states_targets(r2d1_df)

q_full   = jnp.array(x_full, dtype=jnp.float32)
tau_full = jnp.array(y_full.reshape(-1, 1), dtype=jnp.float32)

angle_full = np.rad2deg(r2d1_df["angle[rad]"].values)
y_full_true = np.array(tau_full).ravel()*100 # convert to mBar for plotting

def predict_on(params, q_eval):
    f = DeLan.forward_model(params, None, _potential, _input_mat)
    _, tau_p = jax.vmap(f)(q_eval)
    return np.array(tau_p).ravel()*100

tau_sub_full = predict_on(sub_params, q_full)
tau_ft_full  = predict_on(ft_params, q_full)
tau_sc_full  = predict_on(sc_params, q_full)
tau_lora_full = predict_on(lora_params, q_full)
tau_w0_full = predict_on(W0, q_full)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
fig.suptitle(
    f"Robot 2 — {TARGET_ANGLE}° | predictions on FULL dataset "
    f"| trained on [{np.rad2deg(Q_MIN):.1f}°, {np.rad2deg(Q_MAX):.1f}°]",
    fontsize=13,
    fontweight="bold",
)

plot_cfg_full = [
    (axes[0,0], tau_sc_full,  "Scratch",                 "#eb2525", sc_params),
    (axes[0,1], tau_ft_full,  "Fine-tune",                "#16a34a", ft_params),
    (axes[1,0], tau_lora_full, f"LoRA rank={model.rank}",                   "#0c2dea", lora_params),
    (axes[1,1], tau_sub_full,  "Subspace",     "#CE0FFE", sub_params),
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

    loss_full, _ = DeLan.loss_fn(params, q_full, tau_full,
                           _potential, _input_mat)

    ax.set_title(f"{title}  full loss={float(loss_full):.5f}")
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Pressure [mBar]")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

plt.figure()
plt.title("W0 loss on FULL dataset")
plt.scatter(angle_full, tau_w0_full, s=4, alpha=0.5, color="blue", label="W0 predicted")
plt.xlabel("Angle [deg]")
plt.ylabel("Pressure [mBar]")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""
eval_comparison.py
==================
Drop-in extension of the original training script.
Paste this block AFTER all models have been trained and
`tau_*_full`, `q_full`, `tau_full`, `angle_full`, `y_full_true` are defined.

Produces three new figures:
  Fig A — per-region bar chart  (interpolation / extrap-left / extrap-right)
  Fig B — signed-error (bias) curves per method
  Fig C — parameter-count vs full-loss scatter  (efficiency frontier)

And prints a LaTeX-ready metrics table.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import jax.numpy as jnp

# ── helpers ──────────────────────────────────────────────────────────────────

def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))

def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))

def max_ae(pred, true):
    return float(np.max(np.abs(pred - true)))

def signed_error(pred, true):
    return pred - true   # positive = overestimate

# ── region masks  (angles in degrees) ─────────────────────────────────────────
train_lo = np.rad2deg(Q_MIN)   # 35°
train_hi = np.rad2deg(Q_MAX)   # 45°

angles = angle_full            # 1-D array, degrees

mask_interp = (angles >= train_lo) & (angles <= train_hi)
mask_ext_lo = angles < train_lo
mask_ext_hi = angles > train_hi
mask_extrap = mask_ext_lo | mask_ext_hi

# ── collect predictions & param counts ────────────────────────────────────────
methods = {
    "Scratch":   (tau_sc_full,   sc_params),
    "Fine-tune": (tau_ft_full,   ft_params),
    "LoRA r=6":  (tau_lora_full, lora_params),
    "Subspace":  (tau_sub_full,  sub_params),
}

colors = {
    "Scratch":   "#eb2525",
    "Fine-tune": "#16a34a",
    "LoRA r=6":  "#0c2dea",
    "Subspace":  "#CE0FFE",
}

def count_params(params):
    """Recursively count all float32 scalars in a nested dict/array tree."""
    total = 0
    if isinstance(params, dict):
        for v in params.values():
            total += count_params(v)
    elif hasattr(params, "size"):
        total += int(np.asarray(params).size)
    return total

def count_trainable_subspace(sp_obj, trainable_dict):
    """For subspace: count only the alpha coefficients that are actually optimised."""
    return int(np.asarray(trainable_dict["alpha"]).size)

def count_lora_trainable(lora_model):
    return lora_model.n_trainable

# param counts (trainable only)
n_params = {
    "Scratch":   count_params(sc_params),
    "Fine-tune": count_params(ft_params),
    "LoRA r=6":  count_lora_trainable(model),   # `model` is LoRAParams object
    "Subspace":  count_trainable_subspace(sp, trainable),
}

# ── per-region metrics table ───────────────────────────────────────────────────
regions = {
    "Interp\n[35°–45°]":    mask_interp,
    "Extrap-lo\n[<35°]":    mask_ext_lo,
    "Extrap-hi\n[>45°]":    mask_ext_hi,
    "Full":                 np.ones(len(angles), dtype=bool),
}

print("\n" + "="*80)
print(f"{'Method':<14}", end="")
for rname in regions:
    clean = rname.replace("\n", " ")
    print(f"  {clean:>18} RMSE", end="")
print()
print("-"*80)

metric_table = {}   # method -> region -> rmse
for mname, (tau_pred, _) in methods.items():
    metric_table[mname] = {}
    print(f"{mname:<14}", end="")
    for rname, mask in regions.items():
        r = rmse(tau_pred[mask], y_full_true[mask])
        metric_table[mname][rname] = r
        clean = rname.replace("\n", " ")
        print(f"  {r:>22.2f}", end="")
    print()
print("="*80)

# LaTeX table
print("\n--- LaTeX table ---")
print(r"\begin{tabular}{lrrrr}")
print(r"\toprule")
header_cols = [r.replace('\n', ' ') for r in regions]
print("Method & " + " & ".join(header_cols) + r" \\")
print(r"\midrule")
for mname in methods:
    row = mname
    for rname in regions:
        row += f" & {metric_table[mname][rname]:.2f}"
    row += r" \\"
    print(row)
print(r"\bottomrule")
print(r"\end{tabular}")

# ── FIG A — per-region bar chart ──────────────────────────────────────────────
region_labels = ["Interp\n[35°–45°]", "Extrap-lo\n[<35°]", "Extrap-hi\n[>45°]", "Full"]
n_methods = len(methods)
n_regions = len(region_labels)
x = np.arange(n_regions)
width = 0.18

fig_a, ax_a = plt.subplots(figsize=(11, 5))
for i, (mname, _) in enumerate(methods.items()):
    vals = [metric_table[mname][r] for r in region_labels]
    bars = ax_a.bar(x + i * width - (n_methods - 1) * width / 2,
                    vals, width, label=mname, color=colors[mname], alpha=0.85,
                    edgecolor="white", linewidth=0.5)
    for bar, v in zip(bars, vals):
        ax_a.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                  f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)

ax_a.set_xticks(x)
ax_a.set_xticklabels(region_labels, fontsize=10)
ax_a.set_ylabel("RMSE [mBar]", fontsize=11)
ax_a.set_title("Per-region RMSE — lower is better\n"
               "(vertical dashed lines separate interpolation from extrapolation)",
               fontsize=11)
ax_a.legend(fontsize=9)
ax_a.grid(axis="y", alpha=0.3)

# shade extrapolation bars with a light background hint
for xi in [1, 2]:
    ax_a.axvspan(xi - 0.45, xi + 0.45, color="lightyellow", zorder=0, alpha=0.6)
ax_a.axvspan(3 - 0.45, 3 + 0.45, color="whitesmoke", zorder=0, alpha=0.6)

plt.tight_layout()
plt.savefig("fig_A_per_region_rmse.png", dpi=150)
plt.show()

# ── FIG B — signed-error bias curves ─────────────────────────────────────────
fig_b, axes_b = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
fig_b.suptitle("Signed prediction error  (pred − true)  by angle\n"
               "Zero = perfect; positive = overestimate", fontsize=11)

for ax, region_mask, region_title in zip(
        axes_b,
        [mask_interp, mask_extrap],
        ["Interpolation region [35°–45°]", "Extrapolation region (outside training)"]):

    ang_r = angles[region_mask]
    sort_idx = np.argsort(ang_r)
    ang_sorted = ang_r[sort_idx]

    for mname, (tau_pred, _) in methods.items():
        err = signed_error(tau_pred, y_full_true)[region_mask][sort_idx]
        # running mean for clarity
        window = max(1, len(err) // 40)
        kernel = np.ones(window) / window
        err_smooth = np.convolve(err, kernel, mode="same")
        ax.plot(ang_sorted, err_smooth, color=colors[mname],
                label=mname, linewidth=1.8, alpha=0.9)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    if Q_MIN is not None:
        ax.axvline(train_lo, color="red",    linestyle="--", alpha=0.5)
    if Q_MAX is not None:
        ax.axvline(train_hi, color="orange", linestyle="--", alpha=0.5)
    ax.set_xlabel("Angle [deg]", fontsize=10)
    ax.set_ylabel("Error [mBar]", fontsize=10)
    ax.set_title(region_title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("fig_B_signed_error.png", dpi=150)
plt.show()

# ── FIG C — efficiency frontier (params vs full RMSE) ─────────────────────────
fig_c, ax_c = plt.subplots(figsize=(7, 5))
ax_c.set_title("Efficiency frontier: trainable parameters vs full-dataset RMSE\n"
               "Bottom-left = better", fontsize=11)

for mname, (tau_pred, _) in methods.items():
    r = rmse(tau_pred, y_full_true)
    p = n_params[mname]
    ax_c.scatter(p, r, color=colors[mname], s=140, zorder=5,
                 edgecolors="white", linewidths=1.2)
    ax_c.annotate(mname, (p, r),
                  textcoords="offset points", xytext=(8, 4),
                  fontsize=9, color=colors[mname])

ax_c.set_xlabel("Trainable parameters", fontsize=10)
ax_c.set_ylabel("Full-dataset RMSE [mBar]", fontsize=10)
ax_c.grid(alpha=0.3)
ax_c.set_xscale("log")

plt.tight_layout()
plt.savefig("fig_C_efficiency_frontier.png", dpi=150)
plt.show()

print("\nSaved: fig_A_per_region_rmse.png | fig_B_signed_error.png | fig_C_efficiency_frontier.png")


"""
eval_data_efficiency.py
=======================
Evaluates adaptation methods under LIMITED DATA conditions.

Key idea: how quickly does each method converge to a good solution
as we increase the number of training samples?

This is the strongest argument for subspace/structured methods:
they need LESS data to generalize because the subspace already
encodes prior task knowledge from MAML.

Paste this after all models and imports are defined in your main script,
but BEFORE the final plotting block.

Requires in scope:
    W0, hyper, sp, subspace, trainable (subspace init),
    model (LoRAParams), q_data, tau_data, q_full, tau_full,
    _potential, _input_mat, DeLan, tools, lora (module),
    Q_MIN, Q_MAX, angle_full, y_full_true
"""

import numpy as np
import jax
import jax.numpy as jnp
import optax
import haiku as hk
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import subspace as sb
import LoRA as lora_module

# ── config ────────────────────────────────────────────────────────────────────

N_EPOCHS_EVAL  = 5000          # fewer epochs for speed across many runs
BATCH_SIZE     = 64
N_REPEATS      = 3             # repeat each experiment to reduce noise
LR_EVAL        = 1e-3

# Fraction of training data to try
DATA_FRACTIONS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]

COLORS = {
    "Fine-tune": "#16a34a",
    "LoRA r=6":  "#0c2dea",
    "Subspace":  "#CE0FFE",
}

# ── LR schedule (fixed LR for fair comparison) ────────────────────────────────

def make_lr(lr=LR_EVAL):
    return optax.adam(lr)

# ── training functions ────────────────────────────────────────────────────────

def train_finetune(q_tr, tau_tr, n_epochs):
    """Full fine-tune from W0."""
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), W0)
    opt    = make_lr()
    state  = opt.init(params)

    @jax.jit
    def loss_fn(p, qb, tb):
        l, _ = DeLan.loss_fn(p, qb, tb, _potential, _input_mat)
        return l

    @jax.jit
    def step(p, s, qb, tb):
        l, g = jax.value_and_grad(loss_fn)(p, qb, tb)
        upd, s = opt.update(g, s)
        return optax.apply_updates(p, upd), s, l

    N = q_tr.shape[0]
    for _ in range(n_epochs):
        bi = np.random.permutation(N)[:BATCH_SIZE]
        params, state, _ = step(params, state, q_tr[bi], tau_tr[bi])
    return params


def train_lora(q_tr, tau_tr, n_epochs):
    """LoRA adaptation from W0."""
    lora_model = lora_module.LoRAParams(w0_star=W0, rank=6, rng_seed=42)
    trainable  = lora_model.init_trainable()
    opt        = make_lr()
    state      = opt.init(trainable)

    @jax.jit
    def loss_fn(tr, qb, tb):
        full_p = lora_model.reconstruct(tr)
        l, _ = DeLan.loss_fn(full_p, qb, tb, _potential, _input_mat)
        return l

    @jax.jit
    def step(tr, s, qb, tb):
        l, g = jax.value_and_grad(loss_fn)(tr, qb, tb)
        upd, s = opt.update(g, s)
        return optax.apply_updates(tr, upd), s, l

    N = q_tr.shape[0]
    for _ in range(n_epochs):
        bi = np.random.permutation(N)[:BATCH_SIZE]
        trainable, state, _ = step(trainable, state, q_tr[bi], tau_tr[bi])
    return lora_model.reconstruct(trainable)


def train_subspace_local(q_tr, tau_tr, n_epochs):
    """Subspace adaptation — reinitialised fresh each call."""
    sp_local   = sb.SubspaceParams(subspace=subspace, w0_star=W0, free_layer_names=None)
    tr_local   = sp_local.init_trainable()
    opt        = make_lr()
    state      = opt.init(tr_local)

    @jax.jit
    def loss_fn(tr, qb, tb):
        full_p = sp_local.reconstruct(tr)
        l, _ = DeLan.loss_fn(full_p, qb, tb, _potential, _input_mat)
        return l + 1e-3 * jnp.sum(tr["alpha"] ** 2)

    @jax.jit
    def step(tr, s, qb, tb):
        l, g = jax.value_and_grad(loss_fn)(tr, qb, tb)
        upd, s = opt.update(g, s)
        return optax.apply_updates(tr, upd), s, l

    N = q_tr.shape[0]
    for _ in range(n_epochs):
        bi = np.random.permutation(N)[:min(BATCH_SIZE, N)]
        tr_local, state, _ = step(tr_local, state, q_tr[bi], tau_tr[bi])
    return sp_local.reconstruct(tr_local)


def eval_rmse_extrap(params, q_eval, tau_eval_true, angle_eval, train_lo, train_hi):
    """RMSE on extrapolation region only."""
    f   = DeLan.forward_model(params, None, _potential, _input_mat)
    _, tau_p = jax.vmap(f)(q_eval)
    tau_p_mbar = np.array(tau_p).ravel() * 100
    tau_true_mbar = np.array(tau_eval_true).ravel() * 100

    mask_extrap = (angle_eval < train_lo) | (angle_eval > train_hi)
    if mask_extrap.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((tau_p_mbar[mask_extrap] - tau_true_mbar[mask_extrap]) ** 2)))


def eval_rmse_full(params, q_eval, tau_eval_true):
    """RMSE on full dataset."""
    f   = DeLan.forward_model(params, None, _potential, _input_mat)
    _, tau_p = jax.vmap(f)(q_eval)
    tau_p_mbar = np.array(tau_p).ravel() * 100
    tau_true_mbar = np.array(tau_eval_true).ravel() * 100
    return float(np.sqrt(np.mean((tau_p_mbar - tau_true_mbar) ** 2)))


# ── main experiment loop ──────────────────────────────────────────────────────

train_lo = np.rad2deg(Q_MIN)
train_hi = np.rad2deg(Q_MAX)

N_total = q_data.shape[0]
print(f"\n[Data efficiency eval] Total training samples: {N_total}")
print(f"Fractions: {DATA_FRACTIONS}")
print(f"Epochs per run: {N_EPOCHS_EVAL}  |  Repeats: {N_REPEATS}\n")

results = {
    "Fine-tune": {"full": [], "extrap": []},
    "LoRA r=6":  {"full": [], "extrap": []},
    "Subspace":  {"full": [], "extrap": []},
}
sample_counts = []

for frac in DATA_FRACTIONS:
    n_samples = max(BATCH_SIZE, int(frac * N_total))
    sample_counts.append(n_samples)
    print(f"  -- fraction={frac:.0%}  n={n_samples} --")

    rep_full   = {"Fine-tune": [], "LoRA r=6": [], "Subspace": []}
    rep_extrap = {"Fine-tune": [], "LoRA r=6": [], "Subspace": []}

    for rep in range(N_REPEATS):
        idx = np.random.permutation(N_total)[:n_samples]
        q_tr   = q_data[idx]
        tau_tr = tau_data[idx]

        # Fine-tune
        p = train_finetune(q_tr, tau_tr, N_EPOCHS_EVAL)
        rep_full["Fine-tune"].append(eval_rmse_full(p, q_full, tau_full))
        rep_extrap["Fine-tune"].append(eval_rmse_extrap(p, q_full, tau_full, angle_full, train_lo, train_hi))

        # LoRA
        p = train_lora(q_tr, tau_tr, N_EPOCHS_EVAL)
        rep_full["LoRA r=6"].append(eval_rmse_full(p, q_full, tau_full))
        rep_extrap["LoRA r=6"].append(eval_rmse_extrap(p, q_full, tau_full, angle_full, train_lo, train_hi))

        # Subspace
        p = train_subspace_local(q_tr, tau_tr, N_EPOCHS_EVAL)
        rep_full["Subspace"].append(eval_rmse_full(p, q_full, tau_full))
        rep_extrap["Subspace"].append(eval_rmse_extrap(p, q_full, tau_full, angle_full, train_lo, train_hi))

        print(f"    rep {rep+1}/{N_REPEATS} done")

    for mname in results:
        results[mname]["full"].append((
            np.mean(rep_full[mname]),
            np.std(rep_full[mname]),
        ))
        results[mname]["extrap"].append((
            np.mean(rep_extrap[mname]),
            np.std(rep_extrap[mname]),
        ))

# ── plotting ──────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(
    "Data efficiency: RMSE vs number of training samples\n"
    f"(trained on [{train_lo:.0f}°–{train_hi:.0f}°], evaluated on full range, "
    f"{N_EPOCHS_EVAL} epochs, {N_REPEATS} repeats)",
    fontsize=11,
)

for ax, metric_key, title in zip(
    axes,
    ["full", "extrap"],
    ["Full dataset RMSE", "Extrapolation-only RMSE\n(outside training range)"],
):
    for mname, color in COLORS.items():
        means = [v[0] for v in results[mname][metric_key]]
        stds  = [v[1] for v in results[mname][metric_key]]
        means = np.array(means)
        stds  = np.array(stds)

        ax.plot(sample_counts, means, color=color, label=mname,
                linewidth=2, marker="o", markersize=6)
        ax.fill_between(sample_counts,
                        means - stds, means + stds,
                        color=color, alpha=0.15)

    ax.set_xlabel("Training samples", fontsize=10)
    ax.set_ylabel("RMSE [mBar]", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xscale("log")

plt.tight_layout()
plt.savefig("fig_data_efficiency.png", dpi=150)
plt.show()

# ── print summary table ───────────────────────────────────────────────────────
print("\n--- Data Efficiency Summary (Extrapolation RMSE) ---")
header = f"{'Samples':>10}" + "".join(f"  {m:>14}" for m in results)
print(header)
print("-" * len(header))
for i, n in enumerate(sample_counts):
    row = f"{n:>10}"
    for mname in results:
        mu, sd = results[mname]["extrap"][i]
        row += f"  {mu:>8.2f}±{sd:>4.2f}"
    print(row)

print("\nSaved: fig_data_efficiency.png")