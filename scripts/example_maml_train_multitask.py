"""
Multi-task W0_star Training
============================
Trains a single shared model on all available tasks simultaneously,
using an equal-weighted sum of per-task losses.

The result is a better W0_star initialisation for subspace adaptation:
instead of starting from a model trained on one robot, you start from
a model that already sits in a low-loss region for *all* tasks.

Workflow:
  1. Load all task datasets.
  2. Train one shared model with loss = mean(loss_task_1, ..., loss_task_K).
  3. Save the result as the new W0_star.
  4. Use it as drop-in replacement in your subspace script.
"""

from __future__ import annotations
from pathlib import Path
from functools import partial
import pickle
import time

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Import your model definitions — adjust path if needed
# ---------------------------------------------------------------------------
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

import utils_dir as du


# ---------------------------------------------------------------------------
# Model definitions (copied from your main script)
# ---------------------------------------------------------------------------

def input_transform_matrix(q, n_dof, actuator_dof, shape, activation):
    net = hk.nets.MLP(
        output_sizes=shape + (n_dof * actuator_dof,),
        activation=activation,
        name="input_transform_matrix",
    )
    raw_A = net(q)
    A = raw_A.reshape(n_dof, actuator_dof)
    A = A + 1e-4
    return A


def potential_energy(q, shape, activation):
    net = hk.nets.MLP(
        output_sizes=shape + (1,),
        activation=activation,
        name="potential_energy",
    )
    return net(q).squeeze(-1)


def forward_model(params, pot_mat, input_mat):
    def model(q):
        def V_fn(q_):
            return pot_mat(params["potential"], None, q_)
        G = jax.grad(V_fn)(q)
        A = input_mat(params["input_transform"], None, q)
        tau_hat = jnp.linalg.pinv(A) @ G
        return A, tau_hat
    return model


def loss_fn(params, q, tau, pot_mat, input_mat):
    f = forward_model(params=params, pot_mat=pot_mat, input_mat=input_mat)
    A_pred, tau_pred = jax.vmap(f)(q)
    if tau_pred.ndim == 1:
        tau_pred = tau_pred[:, None]
    residual = tau_pred - tau
    per_sample = jnp.sum(residual ** 2, axis=-1)
    return jnp.mean(per_sample), {"loss": jnp.mean(per_sample), "forward_var": jnp.var(per_sample)}


# ---------------------------------------------------------------------------
# Multi-task loss: equal average over all tasks
# ---------------------------------------------------------------------------

def multitask_loss(params, task_batches, pot_mat, input_mat):
    """
    task_batches: list of (q_batch, tau_batch) tuples, one per task.
    Returns mean loss across tasks + per-task losses for logging.
    """
    task_losses = []
    for q_b, tau_b in task_batches:
        l, _ = loss_fn(params, q_b, tau_b, pot_mat, input_mat)
        task_losses.append(l)

    task_losses = jnp.stack(task_losses)
    return jnp.mean(task_losses), task_losses


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_robot_data(robot_dir: Path, angles: list[int], du) -> dict:
    """Returns {angle_deg: combined_df} for all available angles."""
    data = {}
    for angle in angles:
        runs = du.get_video_runs(
            video_dir=robot_dir / "video",
            angle=angle,
            run_name_filter=None,
        )
        if not runs:
            continue
        dfs = []
        for motion, video_path in runs.items():
            df = du.load_df(
                label=f"{angle}deg_{motion}",
                video_path=video_path,
                online_dir=robot_dir / "online",
            )
            mask = (df["pressure[mBar]"] >= 15) & (df != 0).any(axis=1)
            dfs.append(df[mask])
        if dfs:
            combined = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
            data[f"{angle}deg"] = combined
    return data


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    SCRIPT_DIR = Path(__file__).resolve().parent
    TRACKER_ROOT = SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"
    MODEL_DIR = SCRIPT_DIR.parent / "models" / "static_models"

    # ------------------------------------------------------------------ #
    #  STEP A: Load hyperparameters from existing W0_star                  #
    #  (architecture must match so the result is a drop-in replacement)   #
    # ------------------------------------------------------------------ #
    with open(MODEL_DIR / "r1d1.jax", "rb") as f:
        base_data = pickle.load(f)

    hyper = base_data["hyper"]
    print("Loaded hyper:", hyper)

    activation = jax.nn.softplus
    hidden_pot = (hyper["n_width"],) * hyper["n_depth"]              # (64, 64, 64)
    hidden_inp = (hyper["n_width"] // 2,) * (hyper["n_depth"] - 1)  # (32, 32)
    n_dof        = hyper["n_dof"]
    actuator_dof = hyper["actuator_dof"]

    # ------------------------------------------------------------------ #
    #  STEP B: Load all task datasets                                      #
    # ------------------------------------------------------------------ #
    ROBOTS = {
        "robot1_damage1_healed": TRACKER_ROOT / "robot1_damage1_healed",
        "robot1_damage2_healed": TRACKER_ROOT / "robot1_damage2_healed",
        # "robot2_damage0":        TRACKER_ROOT / "robot2",
    }
    ANGLES = [0, 45, 135]

    # Collect (q, tau) arrays per task — each robot x angle combination is one task
    task_datasets = {}   # {task_name: (q_array, tau_array)}

    for robot_name, robot_dir in ROBOTS.items():
        robot_data = load_robot_data(robot_dir, ANGLES, du)
        for deg_key, df in robot_data.items():
            x, y = du.extract_states_targets(df)
            mask = x[:, 0] != 0
            x, y = x[mask], y[mask]
            task_name = f"{robot_name}_{deg_key}"
            task_datasets[task_name] = (
                jnp.array(x, dtype=jnp.float32),
                jnp.array(y.reshape(-1, 1), dtype=jnp.float32),
            )
            print(f"  Task '{task_name}': N={x.shape[0]}")

    K = len(task_datasets)
    task_names = list(task_datasets.keys())
    print(f"\nTotal tasks K={K}: {task_names}")

    # ------------------------------------------------------------------ #
    #  STEP C: Build model apply functions                                 #
    # ------------------------------------------------------------------ #
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

    # ------------------------------------------------------------------ #
    #  STEP D: Initialise shared params from existing W0_star              #
    #  (warm start — already reasonable, multi-task loss refines it)       #
    # ------------------------------------------------------------------ #
    params = jax.tree_util.tree_map(jnp.array, base_data["params"])

    # ------------------------------------------------------------------ #
    #  STEP E: Multi-task training                                         #
    # ------------------------------------------------------------------ #
    N_EPOCHS   = 20_000
    BATCH_SIZE = 512
    lr_schedule = optax.cosine_decay_schedule(
        init_value=1e-3,
        decay_steps=N_EPOCHS,
        alpha=1e-5,
    )
    optimizer  = optax.adam(lr_schedule)
    opt_state  = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, task_batches):
        def _loss(p):
            total, per_task = multitask_loss(p, task_batches, _potential, _input_mat)
            return total, per_task

        (total_loss, per_task_losses), grads = jax.value_and_grad(_loss, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, total_loss, per_task_losses

    print(f"\n[MultiTask] Training W0_star on {K} tasks for {N_EPOCHS} epochs")
    print(f"[MultiTask] batch_size={BATCH_SIZE}  lr_schedule=cosine 1e-3→1e-5\n")

    loss_history = []
    per_task_history = {name: [] for name in task_names}

    t0 = time.perf_counter()

    for epoch in range(N_EPOCHS):
        # Sample a mini-batch from each task
        task_batches = []
        for name in task_names:
            q_all, tau_all = task_datasets[name]
            N = q_all.shape[0]
            idx = np.random.permutation(N)[:BATCH_SIZE]
            task_batches.append((q_all[idx], tau_all[idx]))

        params, opt_state, total_loss, per_task_losses = step(
            params, opt_state, task_batches
        )

        loss_history.append(float(total_loss))
        for i, name in enumerate(task_names):
            per_task_history[name].append(float(per_task_losses[i]))

        if epoch % 1000 == 0 or epoch == N_EPOCHS - 1:
            elapsed = time.perf_counter() - t0
            per_task_str = "  ".join(
                f"{name.split('_')[-2]}_{name.split('_')[-1]}={float(per_task_losses[i]):.4f}"
                for i, name in enumerate(task_names)
            )
            print(f"  Epoch {epoch:6d} | total={float(total_loss):.5f} | {per_task_str} | t={elapsed:.1f}s")

    print(f"\n[MultiTask] Done in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------ #
    #  STEP F: Evaluate vs original W0_star on each task                   #
    # ------------------------------------------------------------------ #
    print("\n[Eval] Per-task loss comparison:")
    print(f"  {'Task':<35} {'Original W0':>12}  {'MultiTask W0':>12}  {'Delta':>10}")
    print(f"  {'-'*35} {'-'*12}  {'-'*12}  {'-'*10}")

    orig_params = jax.tree_util.tree_map(jnp.array, base_data["params"])
    for name in task_names:
        q_all, tau_all = task_datasets[name]
        l_orig, _ = loss_fn(orig_params, q_all, tau_all, _potential, _input_mat)
        l_new,  _ = loss_fn(params,      q_all, tau_all, _potential, _input_mat)
        delta = float(l_new) - float(l_orig)
        sign  = "▲" if delta > 0 else "▼"
        print(f"  {name:<35} {float(l_orig):>12.5f}  {float(l_new):>12.5f}  {sign}{abs(delta):>9.5f}")

    # ------------------------------------------------------------------ #
    #  STEP G: Save new W0_star                                            #
    # ------------------------------------------------------------------ #
    save_path = MODEL_DIR / "w0_star_multitask.jax"
    with open(save_path, "wb") as f:
        pickle.dump({"hyper": hyper, "params": params}, f)
    print(f"\n[Save] Saved multitask W0_star to {save_path}")
    print(f"       Drop-in replacement: load this file instead of r1d1.jax in your subspace script.")

    # ------------------------------------------------------------------ #
    #  STEP H: Plot training curves                                        #
    # ------------------------------------------------------------------ #
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Multi-task W0★ Training", fontsize=13, fontweight="bold")

    ax1.semilogy(loss_history, color="#38bdf8", lw=1.5)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean loss across tasks (log)")
    ax1.set_title("Total Multi-task Loss")
    ax1.grid(True, alpha=0.3)

    colors = plt.cm.tab10(np.linspace(0, 1, K))
    for i, name in enumerate(task_names):
        label = name.replace("robot", "r").replace("_damage", "d").replace("_healed", "")
        ax2.semilogy(per_task_history[name], color=colors[i], lw=1, label=label, alpha=0.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Per-task loss (log)")
    ax2.set_title("Per-task Loss Curves")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(MODEL_DIR / "multitask_w0_training.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\n[Done] To use the new W0_star in your subspace script, change:")
    print('  with open(MODEL_DIR / "r1d1.jax", "rb") as f:')
    print("to:")
    print('  with open(MODEL_DIR / "w0_star_multitask.jax", "rb") as f:')