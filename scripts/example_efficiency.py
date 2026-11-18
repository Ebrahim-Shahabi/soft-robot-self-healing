"""
eval_cross_robot_LOO_subspace.py
================================
Cross-robot / leave-out evaluation.

For each target (robot, damage, angle):
  1. Build the subspace WITHOUT the target task/robot.
  2. Adapt W0, Fine-tune, LoRA, and Subspace using only a small target window.
  3. Evaluate on the full target angle range.
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

import utils_dir as du
import subspace as sb
import LoRA as lora_module
import DeLaN_model_static as DeLan


# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR   = Path(__file__).resolve().parent
MODEL_DIR    = SCRIPT_DIR.parents[0] / "models" / "models"
TRACKER_ROOT = SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"


# ── config ────────────────────────────────────────────────────────────────────
N_EPOCHS   = 6000
BATCH_SIZE = 128
LR         = 1e-3
LORA_RANK  = 1

# Choose exclusion mode:
#   "task"  = exclude only exact target task
#   "robot" = exclude all tasks from target robot
#   "damage" = exclude same robot/damage condition
EXCLUDE_MODE = "task"

COLORS = {
    "Scratch":   "#e85d04",
    "Fine-tune": "#16a34a",
    "LoRA r=1":  "#0c2dea",
    "Subspace":  "#CE0FFE",
}

ROBOTS = {
    "robot1_damage1_healed": TRACKER_ROOT / "robot1_damage1_healed",
    "robot1_damage2_healed": TRACKER_ROOT / "robot1_damage2_healed",
    "robot1_damage3_healed": TRACKER_ROOT / "robot1_damage3_healed",
    "robot2_damage0":        TRACKER_ROOT / "robot2",
}

ANGLES = [0, 45, 135]


# ── load W0 ───────────────────────────────────────────────────────────────────
print("Loading W0 ...")
with open(MODEL_DIR / "W0_maml.jax", "rb") as f:
    base_data = pickle.load(f)

hyper = base_data["hyper"]
W0    = base_data["params"]


# ── task-weight files for subspace construction ───────────────────────────────
TASK_MODEL_FILES = {
    "robot1_damage1_healed_45":  MODEL_DIR / "robot1_damage1_healed_45deg_maml_onlyangle.jax",
    "robot1_damage1_healed_135": MODEL_DIR / "robot1_damage1_healed_135deg_maml_onlyangle.jax",
    "robot1_damage2_healed_0":   MODEL_DIR / "robot1_damage2_healed_0deg_maml_onlyangle.jax",
    "robot1_damage3_healed_0":   MODEL_DIR / "robot1_damage3_healed_0deg_maml_onlyangle.jax",
    "robot1_damage3_healed_135": MODEL_DIR / "robot1_damage3_healed_135deg_maml_onlyangle.jax",
    "robot2_damage0_0":          MODEL_DIR / "robot2_damage0_0deg_maml_onlyangle.jax",
    "robot2_damage0_135":        MODEL_DIR / "robot2_damage0_135deg_maml_onlyangle.jax",
}

print("Loading task weights ...")
task_weights_all = {}
for name, fpath in TASK_MODEL_FILES.items():
    with open(fpath, "rb") as f:
        task_weights_all[name] = pickle.load(f)["params"]


# ── Haiku apply functions ─────────────────────────────────────────────────────
activations = {
    "tanh":      jnp.tanh,
    "softplus":  jax.nn.softplus,
    "sigmoid":   jax.nn.sigmoid,
}

activation   = activations[hyper["activation1"]]
hidden_pot   = (hyper["n_width"],) * hyper["n_depth"]
hidden_inp   = (hyper["n_width"] // 2,) * (hyper["n_depth"] - 1)
n_dof        = hyper["n_dof"]
actuator_dof = hyper["actuator_dof"]

pot_transformed = hk.without_apply_rng(hk.transform(
    lambda q: DeLan.potential_energy(q, hidden_pot, activation)
))

input_transformed = hk.without_apply_rng(hk.transform(
    lambda q: DeLan.input_transform_matrix(
        q, n_dof, actuator_dof, hidden_inp, activation
    )
))

def _potential(params, key, q):
    return pot_transformed.apply(params, q)

def _input_mat(params, key, q):
    return input_transformed.apply(params, q)


# ── helpers ───────────────────────────────────────────────────────────────────
def rmse(pred, true):
    return float(np.sqrt(np.mean((pred - true) ** 2)))

def predict_mbar(params, q_eval):
    f = DeLan.forward_model(params, None, _potential, _input_mat)
    _, tau_p = jax.vmap(f)(q_eval)
    return np.array(tau_p).ravel() *100

def make_opt():
    return optax.adam(LR)

def target_key(robot, angle):
    return f"{robot}_{angle}"

def should_exclude_subspace_task(task_name, robot, angle, mode):
    """
    task_name examples:
      robot1_damage1_healed_45
      robot2_damage0_135

    robot examples:
      robot1_damage1_healed
      robot2_damage0
    """
    exact_target = f"{robot}_{angle}"

    if mode == "task":
        return task_name == exact_target

    if mode == "robot":
        if robot.startswith("robot1"):
            return task_name.startswith("robot1")
        if robot.startswith("robot2"):
            return task_name.startswith("robot2")
        return task_name.startswith(robot)

    if mode == "damage":
        return task_name.startswith(robot)

    raise ValueError(f"Unknown EXCLUDE_MODE: {mode}")

def build_subspace_for_target(robot, angle):
    task_weights_train = {
        name: params
        for name, params in task_weights_all.items()
        if not should_exclude_subspace_task(name, robot, angle, EXCLUDE_MODE)
    }

    if len(task_weights_train) == 0:
        raise RuntimeError(
            f"No task weights left to build subspace for target {robot}, {angle}."
        )

    print(f"  Building subspace using {len(task_weights_train)} tasks:")
    for name in task_weights_train:
        print(f"    - {name}")

    return sb.compute_task_subspace(
        w0_star=W0,
        task_weights=task_weights_train,
        variance_threshold=0.95,
        max_rank=None,
        free_layer_names=None,
    )


# ── adaptation methods ────────────────────────────────────────────────────────
def adapt_scratch(q_tr, tau_tr):
    rng = jax.random.PRNGKey(34)

    # initialize each sub-network properly using hk.init
    dummy_q = q_tr[:1]  # single sample for shape inference

    pot_params = pot_transformed.init(rng, dummy_q[0])
    inp_params = input_transformed.init(rng, dummy_q[0])

    params = {
        "potential":       pot_params,
        "input_transform": inp_params,
    }

    opt   = make_opt()
    state = opt.init(params)

    @jax.jit
    def step(p, s, qb, tb):
        loss, grads = jax.value_and_grad(
            lambda p: DeLan.loss_fn(p, qb, tb, _potential, _input_mat)[0]
        )(p)
        updates, s = opt.update(grads, s)
        p = optax.apply_updates(p, updates)
        return p, s, loss

    N = q_tr.shape[0]
    for _ in range(N_EPOCHS):
        bi = np.random.permutation(N)[:min(BATCH_SIZE, N)]
        params, state, _ = step(params, state, q_tr[bi], tau_tr[bi])

    return params

def adapt_finetune(q_tr, tau_tr):
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), W0)
    opt    = make_opt()
    state  = opt.init(params)

    @jax.jit
    def step(p, s, qb, tb):
        loss, grads = jax.value_and_grad(
            lambda p: DeLan.loss_fn(p, qb, tb, _potential, _input_mat)[0]
        )(p)
        updates, s = opt.update(grads, s)
        p = optax.apply_updates(p, updates)
        return p, s, loss

    N = q_tr.shape[0]

    for _ in range(N_EPOCHS):
        bi = np.random.permutation(N)[:min(BATCH_SIZE, N)]
        params, state, _ = step(params, state, q_tr[bi], tau_tr[bi])

    return params


def adapt_lora(q_tr, tau_tr):
    lm        = lora_module.LoRAParams(w0_star=W0, rank=LORA_RANK, rng_seed=42)
    trainable = lm.init_trainable()
    opt       = make_opt()
    state     = opt.init(trainable)

    @jax.jit
    def step(tr, s, qb, tb):
        loss, grads = jax.value_and_grad(
            lambda tr: DeLan.loss_fn(
                lm.reconstruct(tr), qb, tb, _potential, _input_mat
            )[0]
        )(tr)
        updates, s = opt.update(grads, s)
        tr = optax.apply_updates(tr, updates)
        return tr, s, loss

    N = q_tr.shape[0]

    for _ in range(N_EPOCHS):
        bi = np.random.permutation(N)[:min(BATCH_SIZE, N)]
        trainable, state, _ = step(trainable, state, q_tr[bi], tau_tr[bi])

    return lm.reconstruct(trainable)


def adapt_subspace(q_tr, tau_tr, target_subspace):
    sp    = sb.SubspaceParams(
        subspace=target_subspace,
        w0_star=W0,
        free_layer_names=None,
    )
    tr    = sp.init_trainable()
    opt   = make_opt()
    state = opt.init(tr)

    @jax.jit
    def step(tr, s, qb, tb):
        loss, grads = jax.value_and_grad(
            lambda tr: (
                DeLan.loss_fn(
                    sp.reconstruct(tr), qb, tb, _potential, _input_mat
                )[0]
                + 1e-3 * jnp.sum(tr["alpha"] ** 2)
            )
        )(tr)
        updates, s = opt.update(grads, s)
        tr = optax.apply_updates(tr, updates)
        return tr, s, loss

    N = q_tr.shape[0]

    for _ in range(N_EPOCHS):
        bi = np.random.permutation(N)[:min(BATCH_SIZE, N)]
        tr, state, _ = step(tr, state, q_tr[bi], tau_tr[bi])

    return sp.reconstruct(tr)


# ── load robot data ───────────────────────────────────────────────────────────
print("Loading all robot data ...")
all_robot_dfs = du.get_all_robot_dfs(ROBOTS, ANGLES)


# ── main evaluation ───────────────────────────────────────────────────────────
method_names = ["Scratch", "Fine-tune", "LoRA r=1", "Subspace"]

results = {m: {} for m in method_names}
task_labels = []

for robot in all_robot_dfs:
    for angle in all_robot_dfs[robot]:
        df = all_robot_dfs[robot][angle]

        if df is None or len(df) == 0:
            continue

        df = df[df["angle[rad]"] != 0].copy()
        df = df[df["pressure[mBar]"] >= 25].copy()

        if len(df) < BATCH_SIZE:
            print(f"Skipping {robot} {angle}° — too few samples ({len(df)})")
            continue

        label = f"{robot.replace('robot','R').replace('_damage','D').replace('_healed','')}_{angle}°"
        task_labels.append(label)

        print("\n" + "=" * 70)
        print(f"Target task: {label}")
        print(f"Raw samples: {len(df)}")
        print(f"Exclusion mode: {EXCLUDE_MODE}")

        target_subspace = build_subspace_for_target(robot, angle)

        # full target data
        x_full, y_full = du.extract_states_targets(df)
        q_full = jnp.array(x_full, dtype=jnp.float32)
        tau_full = jnp.array(y_full.reshape(-1, 1), dtype=jnp.float32)
        tau_true_mbar = np.array(tau_full).ravel() * 100

        # target adaptation window: middle 30%
        q_vals = df["angle[rad]"].values
        q_lo_task = np.percentile(q_vals, 35)
        q_hi_task = np.percentile(q_vals, 65)

        mask_tr = (q_vals >= q_lo_task) & (q_vals <= q_hi_task)
        df_tr = df[mask_tr].reset_index(drop=True)

        if len(df_tr) < 10:
            print("Skipping — fewer than 10 samples in training window")
            task_labels.pop()
            continue

        x_tr, y_tr = du.extract_states_targets(df_tr)
        q_tr = jnp.array(x_tr, dtype=jnp.float32)
        tau_tr = jnp.array(y_tr.reshape(-1, 1), dtype=jnp.float32)

        print(
            f"Target adaptation window: "
            f"[{np.rad2deg(q_lo_task):.1f}°–{np.rad2deg(q_hi_task):.1f}°], "
            f"{len(df_tr)} samples"
        )

        for mname in method_names:
            print(f"  Adapting {mname} ...", end=" ", flush=True)

            if mname == "Scratch":
                params = adapt_scratch(q_tr, tau_tr)
            elif mname == "Fine-tune":
                params = adapt_finetune(q_tr, tau_tr)
            elif mname == "LoRA r=1":
                params = adapt_lora(q_tr, tau_tr)
            elif mname == "Subspace":
                params = adapt_subspace(q_tr, tau_tr, target_subspace)
            else:
                raise ValueError(mname)

            tau_pred = predict_mbar(params, q_full) / 10
            r_full = rmse(tau_pred, tau_true_mbar / 10)
            results[mname][label] = r_full

            print(f"full RMSE = {r_full:.1f} mBar")


# ── summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("LEAVE-OUT CROSS-ROBOT EVALUATION SUMMARY")
print("=" * 80)
print(f"Subspace exclusion mode: {EXCLUDE_MODE}")

header = f"{'Task':<35}" + "".join(f"  {m:>14}" for m in method_names)
print(header)
print("-" * len(header))

for label in task_labels:
    row = f"{label:<35}"
    for mname in method_names:
        v = results[mname].get(label, float("nan"))
        row += f"  {v:>14.1f}"
    print(row)

print("-" * len(header))
row = f"{'MEAN':<35}"
for mname in method_names:
    vals = [results[mname][l] for l in task_labels if l in results[mname]]
    row += f"  {np.nanmean(vals):>14.1f}"
print(row)


# ── Figure A: heatmap ─────────────────────────────────────────────────────────
n_tasks = len(task_labels)
n_methods = len(method_names)

data_matrix = np.array([
    [results[m].get(l, np.nan) for l in task_labels]
    for m in method_names
])

fig, ax = plt.subplots(figsize=(max(10, n_tasks * 0.9), 4))
im = ax.imshow(data_matrix, aspect="auto", cmap="RdYlGn_r")

ax.set_xticks(range(n_tasks))
ax.set_xticklabels(task_labels, rotation=40, ha="right", fontsize=8)
ax.set_yticks(range(n_methods))
ax.set_yticklabels(method_names, fontsize=9)

plt.colorbar(im, ax=ax, label="RMSE [kPa]")

ax.set_title(
    f"Leave-out cross-robot RMSE",
    fontsize=11,
)

for i in range(n_methods):
    for j in range(n_tasks):
        v = data_matrix[i, j]
        if not np.isnan(v):
            ax.text(
                j, i, f"{v:.0f}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="black",
            )

plt.tight_layout()
plt.savefig(f"fig_cross_robot_heatmap_exclude_{EXCLUDE_MODE}.png", dpi=150)
print(f"\nSaved: fig_cross_robot_heatmap_exclude_{EXCLUDE_MODE}.png")


# ── Figure B: summary plots ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle(
    "Leave-out cross-robot evaluation",
    fontsize=12,
)
means = [
    np.nanmean([results[m][l] for l in task_labels if l in results[m]])
    for m in method_names
]

stds = [
    np.nanstd([results[m][l] for l in task_labels if l in results[m]])
    for m in method_names
]

# mean bar chart
ax = axes[0]
bars = ax.bar(
    method_names,
    means,
    yerr=stds,
    color=[COLORS[m] for m in method_names],
    capsize=5,
    alpha=0.85,
    edgecolor="white",
)

# remove the text labels on bars
ax.set_title("Mean RMSE across target tasks", fontsize=10)
ax.set_ylabel("RMSE [kPa]", fontsize=10)
ax.set_xticks(range(len(method_names)))
ax.set_xticklabels(method_names, rotation=15, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.3)

for bar, v in zip(bars, means):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 1,
        f"{v:.1f}",
        ha="center",
        va="bottom",
        fontsize=9,
    )

ax.set_title("Mean RMSE across target tasks", fontsize=10)
ax.set_ylabel("RMSE [kPa]", fontsize=10)
ax.set_xticks(range(len(method_names)))
ax.set_xticklabels(method_names, rotation=15, ha="right", fontsize=9)
ax.grid(axis="y", alpha=0.3)


# per-task grouped bar chart
ax2 = axes[1]
x = np.arange(len(task_labels))
w = 0.18

for i, mname in enumerate(method_names):
    vals = [results[mname].get(l, np.nan) for l in task_labels]
    ax2.bar(
        x + i * w - (len(method_names) - 1) * w / 2,
        vals,
        w,
        label=mname,
        color=COLORS[mname],
        alpha=0.85,
        edgecolor="white",
    )

ax2.set_xticks(x)
ax2.set_xticklabels(task_labels, rotation=40, ha="right", fontsize=7.5)
ax2.set_ylabel("RMSE [kPa]", fontsize=10)
ax2.set_title("Per-task RMSE", fontsize=10)
ax2.legend(fontsize=8)
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(f"fig_cross_robot_summary_exclude_{EXCLUDE_MODE}.png", dpi=150)
print(f"Saved: fig_cross_robot_summary_exclude_{EXCLUDE_MODE}.png")

plt.show()