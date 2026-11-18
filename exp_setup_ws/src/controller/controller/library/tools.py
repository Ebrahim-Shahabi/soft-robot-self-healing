import pandas
import jax 
import jax.numpy as jnp
from .. import DeLaN_model_static as delan
from functools import partial
import haiku as hk
import optax
import numpy as np
import copy
import time
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from . import LoRA as lora
from . import plots as plts
from pathlib import Path

def shuffle_df(df, seed: int = 42):
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)

# -----------------------------
# Model setup
# -----------------------------
def setup_static_delan(
    train_states,
    train_targets,
    hyper,
    seed=42,
    model_name="model",
    initial_params=None,
):
    activations = {
        "tanh": jnp.tanh,
        "softplus": jax.nn.softplus,
        "sigmoid": jax.nn.sigmoid,
    }

    config_dof = train_states.shape[1]
    actuator_dof = train_targets.shape[1]

    potential_fn = hk.transform(
        partial(
            delan.potential_energy,
            shape=(hyper["n_width"],) * hyper["n_depth"],
            activation=activations[hyper["activation1"]],
        )
    )

    input_fn = hk.transform(
        partial(
            delan.input_transform_matrix,
            shape=(hyper["n_width"] // 2,) * (hyper["n_depth"] - 1),
            n_dof=config_dof,
            actuator_dof=actuator_dof,
            activation=activations[hyper["activation1"]],
        )
    )

    mem_dim = ((config_dof,), (actuator_dof,))
    mem = ReplayMemory(train_states.shape[0], hyper["n_minibatch"], mem_dim)
    mem.add_samples([train_states, train_targets])

    sample_q = jnp.zeros((config_dof,), dtype=jnp.float32)

    rng = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(rng)

    i_params = input_fn.init(key1, sample_q)
    g_params = potential_fn.init(key2, sample_q)

    params = {
        "potential": g_params,
        "input_transform": i_params,
    }
    if initial_params is not None:
        params = copy.deepcopy(initial_params)

    pot_mat = potential_fn.apply
    input_mat = input_fn.apply

    loss_fn = partial(
        delan.loss_fn,
        pot_mat=pot_mat,
        input_mat=input_mat,
        lambda_data=1,
    )
    loss_fn = jax.jit(loss_fn)

    optimizer1 = optax.adamw(
        learning_rate=hyper["learning_rate"],
        weight_decay=hyper["weight_decay"],
    )

    optimizer2 = optax.adamw(
        learning_rate=hyper["learning_rate"],
        weight_decay=hyper["weight_decay"],
    )

    opt1 = optimizer1.init(params["potential"])
    opt2 = optimizer2.init(params["input_transform"])

    def update_fn(params, opt1, opt2, q, tau):
        (_, logs), grads = jax.value_and_grad(loss_fn, 0, has_aux=True)(
            params, q, tau
        )

        updates1, opt1 = optimizer1.update(
            grads["potential"],
            opt1,
            params["potential"],
        )

        updates2, opt2 = optimizer2.update(
            grads["input_transform"],
            opt2,
            params["input_transform"],
        )

        params = {
            "potential": optax.apply_updates(
                params["potential"],
                updates1,
            ),
            "input_transform": optax.apply_updates(
                params["input_transform"],
                updates2,
            ),
        }

        return params, opt1, opt2, logs

    update_fn = jax.jit(update_fn)

    warmup_n = min(hyper["n_minibatch"], train_states.shape[0])

    _, _, _, logs = update_fn(
        params,
        opt1,
        opt2,
        jnp.asarray(train_states[:warmup_n], dtype=jnp.float32),
        jnp.asarray(train_targets[:warmup_n], dtype=jnp.float32),
    )

    print(f"{model_name} setup complete")
    print(f"Config DOF = {config_dof}, Actuator DOF = {actuator_dof}")

    return {
        "name": model_name,
        "params": params,
        "opt1": opt1,
        "opt2": opt2,
        "mem": mem,
        "loss_fn": loss_fn,
        "update_fn": update_fn,
        "logs": logs,
        "train_states": train_states,
        "train_targets": train_targets,
        "pot_mat": pot_mat,
        "input_mat": input_mat,
    }


def split_states_targets(x, y, train_ratio=0.8):
    n_data = x.shape[0]
    div = n_data * int(train_ratio * 10) // 10

    train_states = x[:div, :].astype(np.float32)
    test_states = x[div:, :].astype(np.float32)

    train_targets = y[:div].reshape(-1, 1).astype(np.float32)
    test_targets = y[div:].reshape(-1, 1).astype(np.float32)

    return train_states, train_targets, test_states, test_targets


def setup_model_from_df(df, hyper, model_name, seed=42, initial_params=None):
    x, y = extract_states_targets(df)

    train_states, train_targets, test_states, test_targets = split_states_targets(
        x,
        y,
        train_ratio=0.8,
    )

    model = setup_static_delan(
        train_states,
        train_targets,
        test_states,
        test_targets,
        hyper,
        seed=seed,
        model_name=model_name,
    )

    # Start from another robot's weights
    if initial_params is not None:
        model["params"] = copy.deepcopy(initial_params)

    return model

def plot_training_losses(iterations, train_losses, test_losses=None, title=None):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    train_mean = np.array(train_losses["forward_loss"])
    train_var = np.array(train_losses["forward_var"])

    if test_losses is not None:
        test_mean = np.array(test_losses["forward_loss"])
        test_var = np.array(test_losses["forward_var"])
    else:
        test_mean = None
        test_var = None

    ax.plot(iterations, train_mean, label="train_forward_loss")
    ax.fill_between(
        iterations,
        train_mean - train_var,
        train_mean + train_var,
        alpha=0.2,
    )

    if test_mean is not None and test_var is not None:
        ax.plot(iterations, test_mean, label="test_forward_loss")
        ax.fill_between(
            iterations,
            test_mean - test_var,
            test_mean + test_var,
            alpha=0.2,
    )

    ax.legend(loc="upper right")
    ax.set_xlim(0, max(iterations) + 1)
    ax.set_xlabel("epoch", fontsize=12)
    ax.set_ylabel("loss", fontsize=12)

    if title is not None:
        ax.set_title(title)

    plt.tight_layout()

    return fig, ax

def train_static_model(model, hyper, test_states=None, test_targets=None, step=50, make_plot=False):
    t0_start = time.perf_counter()

    params = model["params"]
    opt1 = model["opt1"]
    opt2 = model["opt2"]
    logs = model["logs"]

    loss_fn = model["loss_fn"]
    update_fn = model["update_fn"]
    mem = model["mem"]

    if test_states is not None and test_targets is not None:
        test_states = jnp.asarray(test_states, dtype=jnp.float32)
        test_targets = jnp.asarray(test_targets, dtype=jnp.float32)

    train_losses = {"forward_loss": [], "forward_var": []}
    test_losses = {"forward_loss": [], "forward_var": []}
    iterations = []

    epoch_i = 0
    # TODO: implement early stopping.
    while epoch_i < hyper["max_epoch"]:

        n_batches = 0
        logs = jax.tree.map(lambda x: x * 0.0, logs)

        for data_batch in mem:
            q = jnp.asarray(data_batch[0], dtype=jnp.float32)
            tau = jnp.asarray(data_batch[1], dtype=jnp.float32)

            params, opt1, opt2, batch_logs = update_fn(
                params, opt1, opt2, q, tau
            )

            n_batches += 1
            logs = jax.tree.map(lambda x, y: x + y, logs, batch_logs)

        epoch_i += 1
        logs = jax.tree.map(lambda x: x / n_batches, logs)

        if epoch_i == 1 or epoch_i % step == 0:

            iterations.append(epoch_i)

            print(f"{model['name']} | Epoch {epoch_i:05d}: ", end="")
            print(f"Time = {time.perf_counter() - t0_start:05.1f}s, ", end="")
            print(
                f"train For = {logs['forward_mean']:.3e} ± "
                f"{1.96 * np.sqrt(logs['forward_var']):.2e}",
                end="; "
            )

            train_losses["forward_loss"].append(float(logs["forward_mean"]))
            train_losses["forward_var"].append(float(logs["forward_var"]))
            if test_states is not None and test_targets is not None:
                _, test_logs = loss_fn(
                    params=params,
                    q=test_states,
                    tau=test_targets,
                )

                print(
                    f"test For = {test_logs['forward_mean']:.3e} ± "
                    f"{1.96 * np.sqrt(test_logs['forward_var']):.2e}"
                )

                test_losses["forward_loss"].append(float(test_logs["forward_mean"]))
                test_losses["forward_var"].append(float(test_logs["forward_var"]))

    model["params"] = params
    model["opt1"] = opt1
    model["opt2"] = opt2
    model["logs"] = logs
    model["train_losses"] = train_losses
    if test_states is not None and test_targets is not None:
        model["test_losses"] = test_losses
    else:
        model["test_losses"] = None
    model["iterations"] = iterations

    fig = None
    ax = None

    if make_plot:
        if test_states is None or test_targets is None:
            print(f"{model['name']} | No test set provided, skipping test loss plot.")
            fig, ax = plot_training_losses(
                iterations,
                train_losses,
                title=model["name"],
            )
        else:   
            fig, ax = plot_training_losses(
                iterations,
                train_losses,
                test_losses,
                title=model["name"],
            )
        plt.show()

    return {
        "model": model,
        "params": params,
        "train_losses": train_losses,
        "test_losses": test_losses,
        "iterations": iterations,
        "fig": fig,
        "ax": ax,
    }

'''
Extracts state variables and target from a dataframe.

The function builds a feature matrix consisting of endpoint position and
bending angle, and a target vector consisting of pressure. It optionally
filters invalid angle measurements and converts angles to degrees.

States:
    - xe[m]
    - ye[m]
    - angle[rad] (or degrees)

Target:
    - pressure[mBar]

@param[in] df         : input dataframe containing robot data.
    should have the entry dx_calc and dy_calc which is obtained from load_df.
@param[in] use_degrees: if True, converts angle from radians to degrees.
@param[in] only_valid : if True, only keeps rows where ANGLE_VALID is True.

@return states : array of shape (N, 3) -> [xe, ye, angle]
@return target : array of shape (N,)   -> pressure
'''
def extract_states_targets(df, only_valid=False, use_degrees=False):

    # Optionally filter valid rows
    if only_valid:
        df = df[df["ANGLE_VALID"]].copy()
    else:
        df = df.copy()

    # Check if dx_calc / dy_calc exist
    if "dx_calc" not in df.columns:
        print("Warning: 'dx_calc' not found. Calculating dx_calc = xe[m] - x0[m]")
        df["dx_calc"] = df["xe[m]"] - df["x0[m]"]

    if "dy_calc" not in df.columns:
        print("Warning: 'dy_calc' not found. Calculating dy_calc = ye[m] - y0[m]")
        df["dy_calc"] = df["ye[m]"] - df["y0[m]"]

    # Extract states
    dx = df["dx_calc"].to_numpy()
    dy = df["dy_calc"].to_numpy()
    angle = df["angle[rad]"].to_numpy()

    if use_degrees:
        angle = np.rad2deg(angle)

    states = np.column_stack((dx, dy, angle))

    # Extract target
    target = df["pressure[mBar]"].to_numpy() / 100

    return states, target


def flatten_params(params: dict) -> tuple[np.ndarray, list]:
    """
    Flatten a (possibly nested) haiku param dict into a single 1-D numpy array.
    Returns (flat_vector, structure_info) where structure_info lets you
    reconstruct the dict.
    """
    leaves, treedef = jax.tree_util.tree_flatten(params)
    shapes = [l.shape for l in leaves]
    flat = np.concatenate([np.array(l).ravel() for l in leaves])
    return flat, (treedef, shapes)

def unflatten_params(flat: jnp.ndarray, structure) -> dict:
    """Inverse of flatten_params — JAX-safe (no np.array conversion)."""
    treedef, shapes = structure
    sizes = [int(np.prod(s)) for s in shapes]
    # Use jnp.split instead of np.split so the traced array stays in JAX
    splits = jnp.split(flat, np.cumsum(sizes[:-1]))
    leaves = [s.reshape(sh) for s, sh in zip(splits, shapes)]
    return treedef.unflatten(leaves)


def build_free_mask_from_structure(
    params: dict,
    structure,
    free_layer_names: list[str] | None = None,
) -> np.ndarray:
    free_layer_names = free_layer_names or []
    treedef, shapes = structure

    paths = []

    def collect_path(path, leaf):
        paths.append(
            "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
        )
        return leaf

    jax.tree_util.tree_map_with_path(collect_path, params)

    sizes = [int(np.prod(s)) for s in shapes]
    free_mask = np.zeros(int(np.sum(sizes)), dtype=bool)

    offset = 0
    for path_str, size in zip(paths, sizes):
        is_free = False

        for free_name in free_layer_names:
            if free_name in path_str:
                is_free = True
                break

        if not is_free and path_str.endswith("b"):
            is_free = True

        free_mask[offset : offset + size] = is_free
        offset += size

    return free_mask

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

def train_lora_delan(
    lora:        lora.LoRAParams,
    q_train:     jnp.ndarray,
    tau_train:   jnp.ndarray,
    pot_mat,
    input_mat,
    n_epochs:    int   = 20_000,
    lr                 = None,
    batch_size:  int   = 600,
    print_every: int   = 1_000,
) -> dict:
    """Train LoRA params (B, A matrices + biases) while W0 is frozen."""

    if lr is None:
        lr = optax.cosine_decay_schedule(init_value=5e-3, decay_steps=n_epochs, alpha=1e-5)

    loss_fn = partial(
        delan.loss_fn,
        pot_mat=pot_mat,
        input_mat=input_mat,
        lambda_data=1,
    )
    loss_fn = jax.jit(loss_fn)

    def _loss(trainable, q_b, tau_b):
        full_params = lora.reconstruct(trainable)
        l, _ = loss_fn(full_params, q_b, tau_b)
        return l

    trainable = lora.init_trainable()
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(trainable)

    @jax.jit
    def step(trainable, opt_state, q_b, tau_b):
        loss, grads = jax.value_and_grad(_loss)(trainable, q_b, tau_b)
        updates, opt_state = optimizer.update(grads, opt_state)
        return optax.apply_updates(trainable, updates), opt_state, loss

    N = q_train.shape[0]
    losses = []

    print(f"\n[LoRA train] rank={lora.rank}  trainable={lora.n_trainable:,}  "
          f"epochs={n_epochs}  N={N}")

    for epoch in range(n_epochs):
        idx = np.random.permutation(N)[:batch_size]
        trainable, opt_state, loss = step(trainable, opt_state, q_train[idx], tau_train[idx])
        losses.append(float(loss))

        if epoch % print_every == 0 or epoch == n_epochs - 1:
            print(f"  Epoch {epoch:6d} | loss={float(loss):.6f}")

    return {
        "trainable":   trainable,
        "params": lora.reconstruct(trainable),
        "losses":      losses,
    }



def run_lora_vs_subspace(
    q_all:          jnp.ndarray,
    tau_all:        jnp.ndarray,
    w0_star:        dict,
    subspace:       dict,           # output of compute_task_subspace()
    SubspaceParams,                 # class from your main script
    pot_mat,
    input_mat,
    loss_fn,
    lora_ranks:     list[int] = [1, 2, 4, 8],
    data_fractions: list[float] | None = None,
    n_epochs:       int   = 20_000,
    lr                    = None,
    batch_size:     int   = 512,
    n_repeats:      int   = 3,
    free_layer_names: list[str] | None = None,
    save_fig_path:  Path | str | None = None,
) -> dict:
    """
    Compare LoRA at multiple ranks vs SVD subspace across data fractions.

    Returns
    -------
    dict with keys:
      "fractions"         : list of fractions
      "subspace"          : array (n_fractions,)
      "lora_r{rank}"      : array (n_fractions,)  for each rank
    """
    if data_fractions is None:
        data_fractions = [0.02, 0.05, 0.10, 0.25, 0.50, 1.00]

    if lr is None:
        lr = optax.cosine_decay_schedule(init_value=5e-3, decay_steps=n_epochs, alpha=1e-5)

    N_total = q_all.shape[0]
    results = {"fractions": data_fractions, "subspace": []}
    for r in lora_ranks:
        results[f"lora_r{r}"] = []

    print(f"\n{'='*65}")
    print(f"  LoRA vs Subspace Experiment")
    print(f"  LoRA ranks    : {lora_ranks}")
    print(f"  fractions     : {data_fractions}")
    print(f"  N_total       : {N_total}  |  epochs: {n_epochs}  |  repeats: {n_repeats}")
    print(f"{'='*65}")

    for frac in data_fractions:
        N_frac = max(1, int(N_total * frac))
        print(f"\n--- fraction={frac:.0%}  N={N_frac} ---")

        sub_losses = []
        lora_losses = {r: [] for r in lora_ranks}

        for seed in range(n_repeats):
            rng = np.random.default_rng(seed)
            idx   = rng.choice(N_total, size=N_frac, replace=False)
            q_sub   = q_all[idx]
            tau_sub = tau_all[idx]

            # ---- SVD Subspace ----
            sp = SubspaceParams(subspace=subspace, w0_star=w0_star,
                                free_layer_names=free_layer_names)
            trainable = sp.init_trainable()
            opt_sub   = optax.adam(lr)
            state_sub = opt_sub.init(trainable)

            def _make_sub_step(sp_, loss_fn_, pot_mat_, input_mat_, opt_):
                def _loss(tr, q_b, tau_b):
                    full_p = sp_.reconstruct(tr)
                    l, _ = loss_fn_(full_p, q_b, tau_b, pot_mat_, input_mat_)
                    return l + 1e-3 * jnp.sum(tr["alpha"] ** 2)
                @jax.jit
                def step(tr, st, q_b, tau_b):
                    loss, g = jax.value_and_grad(_loss)(tr, q_b, tau_b)
                    upd, st = opt_.update(g, st)
                    return optax.apply_updates(tr, upd), st, loss
                return step

            sub_step = _make_sub_step(sp, loss_fn, pot_mat, input_mat, opt_sub)

            for _ in range(n_epochs):
                bi = np.random.permutation(N_frac)[:batch_size]
                trainable, state_sub, _ = sub_step(trainable, state_sub, q_sub[bi], tau_sub[bi])

            full_p = sp.reconstruct(trainable)
            l_sub, _ = loss_fn(full_p, q_all, tau_all, pot_mat, input_mat)
            sub_losses.append(float(l_sub))

            # ---- LoRA at each rank ----
            for r in lora_ranks:
                lora = LoRAParams(w0_star, rank=r, rng_seed=seed)

                def _make_lora_step(lora_, loss_fn_, pot_mat_, input_mat_, opt_):
                    def _loss(tr, q_b, tau_b):
                        full_p = lora_.reconstruct(tr)
                        l, _ = loss_fn_(full_p, q_b, tau_b, pot_mat_, input_mat_)
                        return l
                    @jax.jit
                    def step(tr, st, q_b, tau_b):
                        loss, g = jax.value_and_grad(_loss)(tr, q_b, tau_b)
                        upd, st = opt_.update(g, st)
                        return optax.apply_updates(tr, upd), st, loss
                    return step

                tr_lora   = lora.init_trainable()
                opt_lora  = optax.adam(lr)
                st_lora   = opt_lora.init(tr_lora)
                lora_step = _make_lora_step(lora, loss_fn, pot_mat, input_mat, opt_lora)

                for _ in range(n_epochs):
                    bi = np.random.permutation(N_frac)[:batch_size]
                    tr_lora, st_lora, _ = lora_step(tr_lora, st_lora, q_sub[bi], tau_sub[bi])

                full_p_lora = lora.reconstruct(tr_lora)
                l_lora, _ = loss_fn(full_p_lora, q_all, tau_all, pot_mat, input_mat)
                lora_losses[r].append(float(l_lora))

            # Print seed summary
            lora_str = "  ".join(f"LoRA-r{r}={np.mean(lora_losses[r]):.5f}" for r in lora_ranks)
            print(f"  seed={seed}  sub={float(l_sub):.5f}  {lora_str}")

        results["subspace"].append(np.mean(sub_losses))
        for r in lora_ranks:
            results[f"lora_r{r}"].append(np.mean(lora_losses[r]))

    for k in results:
        if k != "fractions":
            results[k] = np.array(results[k])

    plts.plot_lora_vs_subspace(results, lora_ranks, save_fig_path)
    return results
def early_stopping_train_static_model(
    model,
    hyper,
    source_data,
    all_robot_dfs,
    patience=100,
    min_delta=1e-5,
    step=50,
    make_plot=True,
):
    """
    Train W0 with early stopping based on cosine similarity alignment of
    (Wi - W0) weight vectors across all robots.

    Each epoch:
      1. One gradient step on W0 (source_data).
      2. One gradient step on each Wi (robot-specific data), starting from W0.
      3. Compute delta_i = flatten(Wi) - flatten(W0) for every robot i.
      4. Compute mean pairwise cosine similarity across all delta_i vectors.
      5. Stop when similarity stops improving by min_delta for `patience` evals.

    Returns the W0 params at the stopping epoch.
    """

    # ------------------------------------------------------------------ #
    # 1.  Set up W0 state
    # ------------------------------------------------------------------ #
    params_w0 = model["params"]
    opt1_w0   = model["opt1"]
    opt2_w0   = model["opt2"]
    update_fn = model["update_fn"]
    mem_w0    = model["mem"]
    logs_w0   = model["logs"]

    train_states_w0, train_targets_w0 = source_data

    # ------------------------------------------------------------------ #
    # 2.  Set up one Wi model per (robot, angle) using W0 as init
    # ------------------------------------------------------------------ #
    wi_models = {}
    for robot, angle_dict in all_robot_dfs.items():
        for angle, df in angle_dict.items():
            if df is None or len(df) == 0:
                continue
            key = f"{robot}_{angle}"
            x, y = extract_states_targets(df)
            train_s, train_t, _, _ = split_states_targets(x, y)
            wi = setup_static_delan(
                train_s, train_t, hyper,
                seed=42,
                model_name=key,
                initial_params=params_w0,
            )
            wi_models[key] = wi

    if len(wi_models) == 0:
        raise ValueError("No valid robot dataframes found in all_robot_dfs.")

    print(f"Early stopping: tracking {len(wi_models)} Wi models alongside W0.")

    # ------------------------------------------------------------------ #
    # 3.  Helper: mean pairwise cosine similarity of a list of vectors
    # ------------------------------------------------------------------ #
    def mean_pairwise_cosine(vecs: list[np.ndarray]) -> float:
        n = len(vecs)
        if n < 2:
            return 1.0
        sims = []
        for i in range(n):
            for j in range(i + 1, n):
                a, b = vecs[i], vecs[j]
                norm_a = np.linalg.norm(a)
                norm_b = np.linalg.norm(b)
                if norm_a < 1e-12 or norm_b < 1e-12:
                    sims.append(0.0)
                else:
                    sims.append(float(np.dot(a, b) / (norm_a * norm_b)))
        return float(np.mean(sims))

    # ------------------------------------------------------------------ #
    # 4.  Main training loop
    # ------------------------------------------------------------------ #
    best_similarity    = -np.inf
    patience_counter   = 0
    similarity_history = []
    w0_loss_history    = []
    delta_norm_history = {key: [] for key in wi_models}  # per-robot delta norms
    iterations         = []

    epoch_i = 0
    while epoch_i < hyper["max_epoch"]:

        # --- 4a. One epoch of W0 ---
        logs_w0 = jax.tree.map(lambda x: x * 0.0, logs_w0)
        n_batches = 0
        for batch in mem_w0:
            q   = jnp.asarray(batch[0], dtype=jnp.float32)
            tau = jnp.asarray(batch[1], dtype=jnp.float32)
            params_w0, opt1_w0, opt2_w0, batch_log = update_fn(
                params_w0, opt1_w0, opt2_w0, q, tau
            )
            logs_w0 = jax.tree.map(lambda a, b: a + b, logs_w0, batch_log)
            n_batches += 1
        logs_w0 = jax.tree.map(lambda a: a / n_batches, logs_w0)

        # --- 4b. One epoch of each Wi ---
        for key, wi in wi_models.items():
            wi_logs = jax.tree.map(lambda x: x * 0.0, wi["logs"])
            nb = 0
            for batch in wi["mem"]:
                q   = jnp.asarray(batch[0], dtype=jnp.float32)
                tau = jnp.asarray(batch[1], dtype=jnp.float32)
                wi["params"], wi["opt1"], wi["opt2"], bl = wi["update_fn"](
                    wi["params"], wi["opt1"], wi["opt2"], q, tau
                )
                wi_logs = jax.tree.map(lambda a, b: a + b, wi_logs, bl)
                nb += 1
            wi["logs"] = jax.tree.map(lambda a: a / nb, wi_logs)

        epoch_i += 1

        # --- 4c. Evaluate every `step` epochs ---
        if epoch_i == 1 or epoch_i % step == 0:
            flat_w0, _ = flatten_params(params_w0)

            deltas = []
            for key, wi in wi_models.items():
                flat_wi, _ = flatten_params(wi["params"])
                delta = flat_wi - flat_w0
                deltas.append(delta)
                delta_norm_history[key].append(float(np.linalg.norm(delta)))

            similarity = mean_pairwise_cosine(deltas)
            similarity_history.append(similarity)
            w0_loss_history.append(float(logs_w0["forward_mean"]))
            iterations.append(epoch_i)

            print(
                f"Epoch {epoch_i:05d} | "
                f"W0 loss = {float(logs_w0['forward_mean']):.3e} | "
                f"cos-sim = {similarity:.4f} | "
                f"patience = {patience_counter}/{patience}"
            )

            # --- 4d. Early stopping check ---
            if similarity > best_similarity + min_delta:
                best_similarity  = similarity
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch_i} "
                      f"(best cos-sim = {best_similarity:.4f})")
                break

    # ------------------------------------------------------------------ #
    # 5.  Update model
    # ------------------------------------------------------------------ #
    model["params"] = params_w0
    model["opt1"]   = opt1_w0
    model["opt2"]   = opt2_w0
    model["logs"]   = logs_w0

    # ------------------------------------------------------------------ #
    # 6.  Summary plots
    # ------------------------------------------------------------------ #
    fig = None
    if make_plot:
        fig = _plot_early_stopping_summary(
            iterations=iterations,
            similarity_history=similarity_history,
            w0_loss_history=w0_loss_history,
            delta_norm_history=delta_norm_history,
            stopped_epoch=epoch_i,
            model_name=model["name"],
        )
        plt.show()

    return {
        "model":              model,
        "params":             params_w0,
        "wi_models":          wi_models,
        "similarity_history": similarity_history,
        "w0_loss_history":    w0_loss_history,
        "delta_norm_history": delta_norm_history,
        "iterations":         iterations,
        "best_similarity":    best_similarity,
        "stopped_epoch":      epoch_i,
        "fig":                fig,
    }


def _plot_early_stopping_summary(
    iterations,
    similarity_history,
    w0_loss_history,
    delta_norm_history,
    stopped_epoch,
    model_name="W0",
):
    """3-panel summary plot for early stopping diagnostics."""

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    fig.suptitle(f"Early Stopping Summary — {model_name}", fontsize=13, fontweight="bold")

    iters = np.array(iterations)

    # ── Panel 1: W0 training loss ──────────────────────────────────────
    ax = axes[0]
    ax.plot(iters, w0_loss_history, color="steelblue", linewidth=1.5, label="W0 train loss")
    ax.axvline(stopped_epoch, color="red", linestyle="--", linewidth=1.2, label=f"stop @ {stopped_epoch}")
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_title("W0 Training Loss", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 2: Mean pairwise cosine similarity ───────────────────────
    ax = axes[1]
    ax.plot(iters, similarity_history, color="darkorange", linewidth=1.5, label="mean pairwise cos-sim")
    ax.axvline(stopped_epoch, color="red", linestyle="--", linewidth=1.2)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_ylabel("Cosine Similarity", fontsize=11)
    ax.set_title("Alignment of (Wi − W0) Vectors", fontsize=11)
    ax.set_ylim(-1.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Panel 3: Per-robot delta norms ─────────────────────────────────
    ax = axes[2]
    cmap = plt.cm.get_cmap("tab10", len(delta_norm_history))
    for idx, (key, norms) in enumerate(delta_norm_history.items()):
        ax.plot(iters, norms, linewidth=1.2, label=key, color=cmap(idx))
    ax.axvline(stopped_epoch, color="red", linestyle="--", linewidth=1.2)
    ax.set_ylabel("||Wi − W0||", fontsize=11)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_title("Per-Robot Weight Delta Norm", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig