"""
Data Efficiency Experiment
==========================
Compares three training regimes on increasing fractions of new-task data:
  1. Subspace DeLaN  — alpha + free params, U fixed from SVD
  2. Fine-tune        — start from W0_star, all weights free
  3. Scratch          — random init, all weights free
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_scratch_params(w0_star: dict, rng_seed: int = 0) -> dict:
    """Random reinit with Kaiming scaling — same tree structure as w0_star."""
    rng = jax.random.PRNGKey(rng_seed)
    leaves, treedef = jax.tree_util.tree_flatten(w0_star)
    new_leaves = []
    for leaf in leaves:
        rng, subkey = jax.random.split(rng)
        fan_in = leaf.shape[0] if leaf.ndim >= 2 else leaf.shape[0]
        scale  = 1.0 / np.sqrt(max(fan_in, 1))
        new_leaves.append(jax.random.normal(subkey, leaf.shape) * scale)
    return treedef.unflatten(new_leaves)


def _make_subspace_step(sp, loss_fn, pot_mat, input_mat, optimizer):
    """
    Factory that closes over a *specific* sp instance and returns a jit-compiled
    step function. Called once per (frac, seed) so each iteration gets its own
    compiled function with the correct sp — avoids the stale-closure bug that
    occurs when @jax.jit decorators are used inside a loop.
    """
    def _subspace_loss(trainable, q_b, tau_b):
        full_p = sp.reconstruct(trainable)
        loss, _ = loss_fn(full_p, q_b, tau_b, pot_mat, input_mat)
        alpha_reg = 1e-3 * jnp.sum(trainable["alpha"] ** 2)
        return loss + alpha_reg, {}

    @jax.jit
    def step(tr, st, q_b, tau_b):
        (l, _), g = jax.value_and_grad(_subspace_loss, has_aux=True)(tr, q_b, tau_b)
        upd, st   = optimizer.update(g, st)
        return optax.apply_updates(tr, upd), st, l   # JAX scalar

    return step


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_data_efficiency_experiment(
    q_all:        jnp.ndarray,
    tau_all:      jnp.ndarray,
    w0_star:      dict,
    subspace:     dict,
    pot_mat,
    input_mat,
    SubspaceParams,
    loss_fn,
    data_fractions: list[float] | None = None,
    n_epochs:       int   = 5_000,
    lr             = None,
    batch_size:     int   = 256,
    n_repeats:      int   = 3,
    free_layer_names: list[str] | None = None,
    save_fig_path:  Path | str | None = None,
) -> dict:

    if data_fractions is None:
        data_fractions = [0.01, 0.05, 0.10, 0.25, 0.50, 1.00]

    if lr is None:
        lr = optax.cosine_decay_schedule(init_value=5e-3, decay_steps=n_epochs, alpha=1e-5)

    N_total = q_all.shape[0]
    print(f"\n{'='*60}")
    print(f"  Data Efficiency Experiment")
    print(f"  fractions : {data_fractions}")
    print(f"  N_total   : {N_total}")
    print(f"  epochs    : {n_epochs}  |  repeats: {n_repeats}")
    print(f"{'='*60}")

    results = {
        "fractions":      data_fractions,
        "subspace_train": [],
        "subspace_val":   [],
        "finetune_train": [],
        "finetune_val":   [],
        "scratch_train":  [],
        "scratch_val":    [],
    }

    for frac in data_fractions:
        N_frac = max(1, int(N_total * frac))
        print(f"\n--- fraction={frac:.0%}  N_train={N_frac}  N_val={N_total - N_frac} ---")

        sub_train_losses, sub_val_losses = [], []
        ft_train_losses,  ft_val_losses  = [], []
        sc_train_losses,  sc_val_losses  = [], []

        for seed in range(n_repeats):
            rng = np.random.default_rng(seed)
            idx = rng.choice(N_total, size=N_frac, replace=False)

            # complement indices for validation
            val_mask = np.ones(N_total, dtype=bool)
            val_mask[idx] = False
            val_idx = np.where(val_mask)[0]

            q_sub   = q_all[idx]
            tau_sub = tau_all[idx]
            # frac=1.0 has no complement — fall back to full set
            q_val   = q_all[val_idx]   if len(val_idx) > 0 else q_all
            tau_val = tau_all[val_idx] if len(val_idx) > 0 else tau_all

            # ----------------------------------------------------------------
            # 1. Subspace model
            # ----------------------------------------------------------------
            sp = SubspaceParams(
                subspace=subspace,
                w0_star=w0_star,
                free_layer_names=free_layer_names,
            )

            @jax.jit
            def _subspace_loss_fn(trainable, q_b, tau_b):
                full_p = sp.reconstruct(trainable)
                loss, _ = loss_fn(full_p, q_b, tau_b, pot_mat, input_mat)
                alpha_reg = 1e-3 * jnp.sum(trainable["alpha"] ** 2)
                return loss + alpha_reg, {}

            trainable = sp.init_trainable()
            opt_sub   = optax.adam(lr)
            state_sub = opt_sub.init(trainable)

            @jax.jit
            def _step_sub(tr, st, q_b, tau_b):
                (l, _), g = jax.value_and_grad(_subspace_loss_fn, has_aux=True)(tr, q_b, tau_b)
                upd, st   = opt_sub.update(g, st)
                return optax.apply_updates(tr, upd), st

            for _ in range(n_epochs):
                bi = np.random.permutation(N_frac)[:batch_size]
                trainable, state_sub = _step_sub(trainable, state_sub, q_sub[bi], tau_sub[bi])

            full_p = sp.reconstruct(trainable)
            l_sub_tr, _ = loss_fn(full_p, q_sub,  tau_sub,  pot_mat, input_mat)
            l_sub_va, _ = loss_fn(full_p, q_val,  tau_val,  pot_mat, input_mat)
            sub_train_losses.append(float(l_sub_tr))
            sub_val_losses.append(float(l_sub_va))

            # ----------------------------------------------------------------
            # 2. Fine-tune from W0_star
            # ----------------------------------------------------------------
            ft_params = _train_plain(
                w0_star, q_sub, tau_sub, pot_mat, input_mat,
                n_epochs, lr, batch_size, loss_fn=loss_fn,
            )
            l_ft_tr, _ = loss_fn(ft_params, q_sub,  tau_sub,  pot_mat, input_mat)
            l_ft_va, _ = loss_fn(ft_params, q_val,  tau_val,  pot_mat, input_mat)
            ft_train_losses.append(float(l_ft_tr))
            ft_val_losses.append(float(l_ft_va))

            # ----------------------------------------------------------------
            # 3. Train from scratch
            # ----------------------------------------------------------------
            scratch_init = _make_scratch_params(w0_star, rng_seed=seed * 100)
            sc_params = _train_plain(
                scratch_init, q_sub, tau_sub, pot_mat, input_mat,
                n_epochs, lr, batch_size, loss_fn=loss_fn,
            )
            l_sc_tr, _ = loss_fn(sc_params, q_sub,  tau_sub,  pot_mat, input_mat)
            l_sc_va, _ = loss_fn(sc_params, q_val,  tau_val,  pot_mat, input_mat)
            sc_train_losses.append(float(l_sc_tr))
            sc_val_losses.append(float(l_sc_va))

            print(
                f"  seed={seed} | "
                f"sub  tr={l_sub_tr:.5f} va={l_sub_va:.5f} | "
                f"ft   tr={l_ft_tr:.5f} va={l_ft_va:.5f} | "
                f"sc   tr={l_sc_tr:.5f} va={l_sc_va:.5f}"
            )

        results["subspace_train"].append(np.mean(sub_train_losses))
        results["subspace_val"].append(np.mean(sub_val_losses))
        results["finetune_train"].append(np.mean(ft_train_losses))
        results["finetune_val"].append(np.mean(ft_val_losses))
        results["scratch_train"].append(np.mean(sc_train_losses))
        results["scratch_val"].append(np.mean(sc_val_losses))

    for k in results:
        if k != "fractions":
            results[k] = np.array(results[k])

    _plot_data_efficiency(results, save_fig_path)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_data_efficiency(results, save_fig_path=None):
    fracs = [f * 100 for f in results["fractions"]]

    colors = {
        "subspace": "#2563eb",
        "finetune": "#16a34a",
        "scratch":  "#dc2626",
    }
    labels = {
        "subspace": "Subspace",
        "finetune": "Fine-tune",
        "scratch":  "Scratch",
    }

    fig, (ax_tr, ax_va) = plt.subplots(1, 2, figsize=(14, 5), sharey=False)
    fig.suptitle("Data Efficiency: Train vs Validation Loss", fontsize=13, fontweight="bold")

    for method in ("subspace", "finetune", "scratch"):
        c   = colors[method]
        lbl = labels[method]
        ax_tr.plot(fracs, results[f"{method}_train"], "o-", color=c, linewidth=2, label=lbl)
        ax_va.plot(fracs, results[f"{method}_val"],   "o-", color=c, linewidth=2, label=lbl)

    for ax, title in [
        (ax_tr, "Train loss  (eval on training subset)"),
        (ax_va, "Val loss  (eval on complement)"),
    ]:
        ax.set_xlabel("Training data (%)")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.tight_layout()
    if save_fig_path:
        plt.savefig(save_fig_path, dpi=150, bbox_inches="tight")
    plt.show()

def _train_plain(
    init_params: dict,
    q_train: jnp.ndarray,
    tau_train: jnp.ndarray,
    pot_mat,
    input_mat,
    n_epochs: int,
    lr,
    batch_size: int,
    loss_fn=None,
) -> dict:
    """Train all weights freely (fine-tune or scratch baseline)."""
    if loss_fn is None:
        raise ValueError("Pass loss_fn=loss_fn from your main script.")

    # Wrap loss_fn in a plain function (not jit) so the factory below can jit it
    def _loss(params, q_b, tau_b):
        return loss_fn(params, q_b, tau_b, pot_mat, input_mat)

    optimizer = optax.adam(lr)
    params    = jax.tree_util.tree_map(jnp.array, init_params)
    opt_state = optimizer.init(params)

    @jax.jit
    def step(params, opt_state, q_b, tau_b):
        (loss, _), grads = jax.value_and_grad(_loss, has_aux=True)(params, q_b, tau_b)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss   # JAX scalar — no float() inside jit

    N = q_train.shape[0]
    for _ in range(n_epochs):
        idx = np.random.permutation(N)[:batch_size]
        params, opt_state, _ = step(params, opt_state, q_train[idx], tau_train[idx])

    return params
