"""
DeLaN Subspace Training via SVD of Weight Deltas
=================================================
Workflow:
  1. Load W0_star and all task weights {Wk_star}.
  2. Compute deltas ΔWk = Wk_star - W0_star per layer/param.
  3. Stack deltas -> SVD -> extract top-r directions (task subspace).
  4. Parameterise a new network as:
       W_l = W0_star_l + U_l_r @ alpha_l
     where U_l_r is FIXED (from SVD) and alpha_l is trained.
  5. Some parameters are left fully free (biases, chosen layers).
  6. Train on a new task with the standard DeLaN loss.
"""

import pickle
import copy
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import haiku as hk
import optax
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# 1.  DeLaN model (unchanged from your code)
# ---------------------------------------------------------------------------

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
    loss = jnp.mean(per_sample)
    return loss, {"loss": loss, "forward_var": jnp.var(per_sample)}


# ---------------------------------------------------------------------------
# 2.  Utilities: flatten / unflatten nested param dicts
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 3.  SVD subspace analysis
# ---------------------------------------------------------------------------

def compute_task_subspace(
    w0_star: dict,
    task_weights: dict[str, dict],
    variance_threshold: float = 0.90,
    max_rank: int | None = None,
    free_layer_names: list[str] | None = None,
) -> dict:
    """
    Compute task-adaptation subspace via SVD of weight deltas,
    but only on constrained parameters.

    Free parameters, such as biases or explicitly selected layers, are excluded
    before SVD. Therefore the variance explained corresponds exactly to the
    subspace that will actually be trained through alpha.
    """

    w0_flat, structure = flatten_params(w0_star)
    d = len(w0_flat)
    K = len(task_weights)

    free_mask = build_free_mask_from_structure(
        params=w0_star,
        structure=structure,
        free_layer_names=free_layer_names,
    )
    constrained_mask = ~free_mask

    n_free = int(free_mask.sum())
    n_constrained = int(constrained_mask.sum())

    print(f"\n[SVD] Total parameter dimensionality       d = {d:,}")
    print(f"[SVD] Free parameters excluded              = {n_free:,}")
    print(f"[SVD] Constrained parameters used for SVD    = {n_constrained:,}")
    print(f"[SVD] Number of tasks                        K = {K}")

    Delta = np.zeros((d, K), dtype=np.float64)

    for k, (name, wk) in enumerate(task_weights.items()):
        wk_flat, _ = flatten_params(wk)
        Delta[:, k] = wk_flat - w0_flat
        print(f"  Task '{name}': ||ΔW_full|| = {np.linalg.norm(Delta[:, k]):.4f}")

    Delta_constrained = Delta[constrained_mask, :]

    print(
        f"[SVD] Delta_constrained shape = "
        f"{Delta_constrained.shape[0]:,} x {Delta_constrained.shape[1]}"
    )

    U, S, Vt = np.linalg.svd(Delta_constrained, full_matrices=False)

    denom = np.sum(S ** 2)
    if denom == 0:
        raise ValueError("All constrained task deltas are zero; cannot compute SVD subspace.")

    var_explained = np.cumsum(S ** 2) / denom

    r = int(np.searchsorted(var_explained, variance_threshold) + 1)
    r = min(r, K)

    if max_rank is not None:
        r = min(r, max_rank)

    print(f"\n[SVD] Singular values: {np.round(S, 4)}")
    print(f"[SVD] Variance explained: {np.round(var_explained, 4)}")
    print(
        f"[SVD] Chosen rank r = {r} "
        f"(explains {var_explained[r - 1] * 100:.1f}% of constrained variance)"
    )

    return {
        "U": U[:, :r],  # now shape: (n_constrained, r)
        "S": S[:r],
        "variance_explained": var_explained,
        "rank": r,
        "w0_flat": w0_flat,
        "structure": structure,
        "free_mask": free_mask,
        "constrained_mask": constrained_mask,
    }

def plot_subspace_analysis(subspace: dict, save_path=None):
    S = subspace["S"]          # already truncated to rank r
    var = subspace["variance_explained"]   # full K components
    r = subspace["rank"]

    K = len(var)               # total number of tasks
    ks_full = np.arange(1, K + 1)   # for variance plot (all K)
    ks_trunc = np.arange(1, r + 1)  # for singular values (only top r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Task Subspace SVD Analysis", fontsize=14, fontweight="bold")

    ax1.bar(ks_trunc, S, color="#2563eb", alpha=0.8)
    ax1.axvline(r + 0.5, color="red", linestyle="--", label=f"rank={r}")
    ax1.set_xlabel("Component")
    ax1.set_ylabel("Singular value")
    ax1.set_title("Singular Value Spectrum (kept components)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks_full, var * 100, "o-", color="#2563eb", linewidth=2)
    ax2.axhline(90, color="orange", linestyle="--", label="90%")
    ax2.axhline(95, color="red", linestyle="--", label="95%")
    ax2.axvline(r + 0.5, color="grey", linestyle="--", label=f"rank={r}")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative variance explained (%)")
    ax2.set_title("Variance Explained")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
# ---------------------------------------------------------------------------
# 4.  Constrained parameter class
# ---------------------------------------------------------------------------

class SubspaceParams:
    """
    Manages the mixed parameterisation:

        W_constrained = W0_star + U @ alpha       (subspace part)
        W_free        = freely trainable           (biases, chosen layers, etc.)

    The 'trainable' pytree exposed to the optimiser is:
        {"alpha": jnp.array(shape=(r,)),  "free": {...free params...}}

    Usage
    -----
    sp = SubspaceParams(subspace, w0_star, free_layer_names=["input_transform/~/linear_0"])
    trainable = sp.init_trainable()           # what you pass to the optimiser
    full_params = sp.reconstruct(trainable)   # what you pass to the DeLaN loss
    """

    def __init__(
        self,
        subspace: dict,
        w0_star: dict,
        free_layer_names: list[str] | None = None,
    ):
        """
        Parameters
        ----------
        subspace         : output of compute_task_subspace()
        w0_star          : base weights
        free_layer_names : list of haiku module path prefixes that should be
                           fully free (e.g. ["potential_energy/~/linear_2"])
                           If None, only biases are freed.
        """
        self.subspace = subspace
        self.w0_flat = jnp.array(subspace["w0_flat"])        # (d,)
        self.U = jnp.array(subspace["U"])                    # (d, r)
        self.r = subspace["rank"]
        self.structure = subspace["structure"]
        self.free_layer_names = free_layer_names or []

        # Build masks: which flat indices are "free" vs "constrained"
        self._build_masks(w0_star)

    # ------------------------------------------------------------------
    def _build_masks(self, w0_star: dict):
        self.free_mask = self.subspace["free_mask"]
        self.constrained_mask = self.subspace["constrained_mask"]

        self.w0_free = self.w0_flat[self.free_mask]
        self.w0_constrained = self.w0_flat[self.constrained_mask]

        # U is already restricted to constrained parameters
        self.U_constrained = jnp.array(self.subspace["U"])

        n_free = int(self.free_mask.sum())
        n_con = int(self.constrained_mask.sum())

        print(f"\n[SubspaceParams] Free parameters       : {n_free:,}")
        print(f"[SubspaceParams] Constrained parameters: {n_con:,}  (rank-{self.r} subspace)")
    # ------------------------------------------------------------------
    def init_trainable(self) -> dict:
        """
        Initialise the trainable pytree:
          alpha  : zero init (starts exactly at W0_star)
          free   : 1-D array of free param values (from W0_star)
        """
        return {
            "alpha": jnp.zeros(self.r),
            "free": jnp.array(self.w0_free),
        }

    # ------------------------------------------------------------------
    def reconstruct(self, trainable: dict) -> dict:
        alpha = trainable["alpha"]
        free  = trainable["free"]

        w_constrained = self.w0_constrained + self.U_constrained @ alpha

        d = len(self.w0_flat)
        w_full = jnp.zeros(d)
        w_full = w_full.at[self.constrained_mask].set(w_constrained)
        w_full = w_full.at[self.free_mask].set(free)

        # No np.array() here — keep w_full as a traced JAX array
        return unflatten_params(w_full, self.structure)


# ---------------------------------------------------------------------------
# 5.  Training loop for subspace-constrained DeLaN
# ---------------------------------------------------------------------------

def train_subspace_delan(
    sp: SubspaceParams,
    q_train: jnp.ndarray,
    tau_train: jnp.ndarray,
    pot_mat,
    input_mat,
    n_epochs: int = 2000,
    lr: float = 1e-3,
    batch_size: int = 256,
    print_every: int = 100,
) -> dict:
    """
    Train alpha (and free params) while U is fixed.

    Parameters
    ----------
    sp         : SubspaceParams instance
    q_train    : (N, n_dof) joint angles
    tau_train  : (N, actuator_dof) pressures / torques
    pot_mat    : partial-applied potential energy haiku function
    input_mat  : partial-applied input transform haiku function

    Returns
    -------
    dict with trained trainable params, full params, and loss history
    """

    # ---- loss over trainable ----
    @jax.jit
    def subspace_loss(trainable, q_batch, tau_batch):
        full_params = sp.reconstruct(trainable)
        loss, logs = loss_fn(full_params, q_batch, tau_batch, pot_mat, input_mat)
        
        # Penalize large alpha — keeps adaptation close to the subspace origin
        alpha_reg = 1e-3 * jnp.sum(trainable["alpha"] ** 2)
        total_loss = loss + alpha_reg
        
        logs["alpha_reg"] = alpha_reg
        return total_loss, logs

    # ---- optimiser ----
    optimizer = optax.adam(lr)
    trainable = sp.init_trainable()
    opt_state = optimizer.init(trainable)

    @jax.jit
    def step(trainable, opt_state, q_batch, tau_batch):
        (loss, logs), grads = jax.value_and_grad(subspace_loss, has_aux=True)(
            trainable, q_batch, tau_batch
        )
        updates, opt_state = optimizer.update(grads, opt_state)
        trainable = optax.apply_updates(trainable, updates)
        return trainable, opt_state, loss, logs

    # ---- train ----
    N = q_train.shape[0]
    losses = []
    alpha_norms = []

    print(f"\n[Train] Starting subspace-constrained DeLaN training")
    print(f"[Train] rank={sp.r}  |  N={N}  |  epochs={n_epochs}  |  lr={lr}")

    for epoch in range(n_epochs):
        # Mini-batch
        idx = np.random.permutation(N)[:batch_size]
        q_b   = q_train[idx]
        tau_b = tau_train[idx]

        trainable, opt_state, loss, logs = step(trainable, opt_state, q_b, tau_b)
        losses.append(float(loss))
        alpha_norms.append(float(jnp.linalg.norm(trainable["alpha"])))

        if epoch % print_every == 0 or epoch == n_epochs - 1:
            print(
                f"  Epoch {epoch:5d} | loss={loss:.6f} "
                f"| ||alpha||={alpha_norms[-1]:.4f}"
            )

    full_params = sp.reconstruct(trainable)

    return {
        "trainable": trainable,
        "full_params": full_params,
        "losses": losses,
        "alpha_norms": alpha_norms,
    }


def plot_training(result: dict, title: str = "Subspace DeLaN Training"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    ax1.semilogy(result["losses"])
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (log scale)")
    ax1.set_title("Training Loss")
    ax1.grid(True, alpha=0.3)

    ax2.plot(result["alpha_norms"])
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("||alpha||")
    ax2.set_title("Subspace Coefficient Norm")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 6.  Main – wire everything together
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # ------------------------------------------------------------------ #
    #  STEP A: Load W0_star (your base trained model)                      #
    # ------------------------------------------------------------------ #
    MODEL_DIR = Path("./../models/static_models")

    with open(MODEL_DIR / "r1d1.jax", "rb") as f:
        base_data = pickle.load(f)

    hyper    = base_data["hyper"]
    W0_star  = base_data["params"]          # <-- your W0_star

    # ------------------------------------------------------------------ #
    #  STEP B: Load all task weights {Wk_star}                             #
    #  These are the models you already trained starting from W0_star      #
    # ------------------------------------------------------------------ #
    TASK_MODEL_FILES = {
        "r1d1_45deg" : MODEL_DIR / "r1d1_45deg.jax",
        "r1d1_135deg": MODEL_DIR / "r1d1_135deg.jax",
        "r1d2_0deg"  : MODEL_DIR / "r1d2_0deg.jax",
        "r2d0_135deg": MODEL_DIR / "r2d0_135deg.jax",
    }

    task_weights = {}
    for name, fpath in TASK_MODEL_FILES.items():
        with open(fpath, "rb") as f:
            d = pickle.load(f)
        task_weights[name] = d["params"]
        print(f"Loaded task weights: {name}")

    # ------------------------------------------------------------------ #
    #  STEP C: Compute the task-adaptation subspace via SVD               #
    # ------------------------------------------------------------------ #
    free_layers = None

    subspace = compute_task_subspace(
        w0_star=W0_star,
        task_weights=task_weights,
        variance_threshold=0.90,
        max_rank=None,
        free_layer_names=free_layers,
    )

    plot_subspace_analysis(subspace, save_path=MODEL_DIR / "svd_analysis.png")

    # ------------------------------------------------------------------ #
    #  STEP D: Build SubspaceParams for a new task                         #
    # ------------------------------------------------------------------ #
    #  free_layer_names: haiku paths of layers you want fully free.
    #  You can inspect the keys with:
    #      print(jax.tree_util.tree_leaves(jax.tree_util.tree_map_with_path(lambda p,v: "/".join(str(k.key) for k in p), W0_star)))
    #  Biases are always freed automatically.
    #  Example: free the last layer of potential_energy entirely:
    

    sp = SubspaceParams(
        subspace=subspace,
        w0_star=W0_star,
        free_layer_names=free_layers,
    )

    # ------------------------------------------------------------------ #
    #  STEP E: Load new-task data (e.g. r2d0 at 0 deg)                    #
    #  Replace this block with your actual data loading                    #
    # ------------------------------------------------------------------ #
    # ------------------------------------------------------------------ #
    #  STEP E: Real data — r2d0 0deg                                      #
    # ------------------------------------------------------------------ #
    import utils_dir as du
    # ------------------------------------------------------------------ #
    #  STEP E-PREP: Load and combine robot data                           #
    # ------------------------------------------------------------------ #
    import pandas as pd
    import utils_dir as du

    SCRIPT_DIR = Path(__file__).resolve().parent
    TRACKER_ROOT = SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"

    ROBOTS = {
        "robot1_damage1_healed": TRACKER_ROOT / "robot1_damage1_healed",
        "robot1_damage2_healed": TRACKER_ROOT / "robot1_damage2_healed",
        "robot2_damage0":        TRACKER_ROOT / "robot2",
    }

    ANGLES = [0, 45, 135]

    all_robot_dfs = {}
    for robot_name, robot_dir in ROBOTS.items():
        all_robot_dfs[robot_name] = {}
        for angle in ANGLES:
            deg_key = f"{angle}deg"
            runs = du.get_video_runs(video_dir=robot_dir / "video", angle=angle, run_name_filter=None)
            if not runs:
                continue
            all_robot_dfs[robot_name][deg_key] = {}
            for motion, video_path in runs.items():
                df = du.load_df(
                    label=f"{robot_name}_{deg_key}_{motion}",
                    video_path=video_path,
                    online_dir=robot_dir / "online",
                )
                mask = (df["pressure[mBar]"] >= 15) & (df != 0).any(axis=1)
                filtered_df = df[mask]
                key = "trapezoid" if "trapezoid" in motion else motion
                all_robot_dfs[robot_name][deg_key][key] = filtered_df

    combined_robot_dfs = {}
    for robot_name in all_robot_dfs:
        combined_robot_dfs[robot_name] = {}
        for deg_key in all_robot_dfs[robot_name]:
            temp = []
            for r in ["rampUpwards", "rampDownwards", "trapezoid"]:
                if r in all_robot_dfs[robot_name][deg_key]:
                    temp.append(all_robot_dfs[robot_name][deg_key][r].copy())
            if not temp:
                continue
            combined_df = pd.concat(temp, ignore_index=True).sample(frac=1).reset_index(drop=True)
            combined_robot_dfs[robot_name][deg_key] = combined_df

    print("Available keys:", {r: list(d.keys()) for r, d in combined_robot_dfs.items()})

    # Do this once and reuse — don't reshuffle df_new after extracting x, y
    df_new = combined_robot_dfs["robot2_damage0"]["0deg"]
    x, y = du.extract_states_targets(df_new)
    raw_angles_deg = np.rad2deg(df_new["angle[rad]"].values)  # same row order as x, y

    q_new   = jnp.array(x, dtype=jnp.float32)
    tau_new = jnp.array(y.reshape(-1, 1), dtype=jnp.float32)

    print(f"q shape: {q_new.shape}, tau shape: {tau_new.shape}")

    # ------------------------------------------------------------------ #
    #  STEP F: Build apply functions matching tools.py exactly            #
    # ------------------------------------------------------------------ #
    activation = jax.nn.softplus  # activation1 from hyper

    # Mirrors setup_static_delan in tools.py exactly:
    # potential : shape=(n_width,) * n_depth       = (64, 64, 64)
    # input     : shape=(n_width//2,) * (n_depth-1) = (32, 32)
    hidden_pot = (hyper["n_width"],) * hyper["n_depth"]             # (64, 64, 64)
    hidden_inp = (hyper["n_width"] // 2,) * (hyper["n_depth"] - 1)  # (32, 32)

    n_dof        = hyper["n_dof"]        # 3
    actuator_dof = hyper["actuator_dof"] # 1

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

    # Sanity check — run before training to catch shape mismatches early
    test_q = jnp.ones((n_dof,))
    print("pot shape:", pot_transformed.apply(W0_star["potential"], test_q).shape)         # expect: ()
    print("inp shape:", input_transformed.apply(W0_star["input_transform"], test_q).shape)  # expect: (3, 1)

    # ------------------------------------------------------------------ #
    #  STEP G: Train!                                                      #
    # ------------------------------------------------------------------ #
    import time 
    t0 = time.perf_counter()
    print("hyper keys:", hyper)
    print("W0_star keys:", jax.tree_util.tree_map_with_path(lambda p, v: ("/".join(str(k.key) for k in p), v.shape), W0_star))
    lr_schedule = optax.warmup_cosine_decay_schedule(
    init_value=1e-3,
    peak_value=5e-2,        # aggressive peak
    warmup_steps=200,
    decay_steps=20000,
    end_value=1e-6,
    )

    result = train_subspace_delan(
        sp=sp,
        q_train=q_new,
        tau_train=tau_new,
        pot_mat=_potential,
        input_mat=_input_mat,
        n_epochs=20000,
        lr=lr_schedule,
        batch_size=600,       # larger batch = less noise
        print_every=1000,
    )
    print(time.perf_counter() - t0)
    # After training, evaluate on the full dataset
    full_params = result["full_params"]
    f = forward_model(full_params, _potential, _input_mat)
    _, tau_pred = jax.vmap(f)(q_new)

    # Use the raw angle from the dataframe, not the engineered feature x[:,0]
    raw_angles_deg = np.rad2deg(df_new["angle[rad]"].values)

    full_params = result["full_params"]
    f = forward_model(full_params, _potential, _input_mat)
    _, tau_pred = jax.vmap(f)(q_new)
    tau_pred = np.array(tau_pred).ravel()

    plt.figure(figsize=(10, 4))
    plt.scatter(raw_angles_deg, y, s=5, alpha=0.5, label="data")
    plt.scatter(raw_angles_deg, tau_pred, s=5, alpha=0.5, label="predicted")
    plt.xlabel("Angle [deg]")
    plt.ylabel("Pressure [mBar]")
    plt.legend()
    plt.title("r2d0 0deg — Subspace DeLaN prediction")
    plt.tight_layout()
    plt.show()

    plot_training(result, title="Subspace DeLaN – New Task")

    # ------------------------------------------------------------------ #
    #  STEP H: Inspect what the network learned                            #
    # ------------------------------------------------------------------ #
    alpha = result["trainable"]["alpha"]
    print("\nLearned alpha:")
    for i, (a, s) in enumerate(zip(alpha, subspace["S"])):
        print(f"  dir {i+1}: alpha={float(a):+.4f}  singular_value={s:.4f}  contribution={float(a)*s:.4f}")
    print("\n[Result] Full params reconstructed. Ready to use / save.")

    # ------------------------------------------------------------------ #
    #  STEP I: Save                                                        #
    # ------------------------------------------------------------------ #
    save_path = MODEL_DIR / "new_task_subspace.jax"
    with open(save_path, "wb") as f:
        pickle.dump(
            {
                "hyper": hyper,
                "params": result["full_params"],
                "alpha": np.array(alpha),
                "subspace_rank": subspace["rank"],
                "losses": result["losses"],
            },
            f,
        )
    print(f"\nSaved to {save_path}")

    # Unpack the (loss, logs) tuple that loss_fn returns
    w0_loss, _         = loss_fn(W0_star, q_new, tau_new, _potential, _input_mat)
    subspace_loss, _   = loss_fn(result["full_params"], q_new, tau_new, _potential, _input_mat)

    print(f"W0 loss:       {float(w0_loss):.6f}")
    print(f"Subspace loss: {float(subspace_loss):.6f}")

    # Check if biases alone are doing the work (alpha frozen at zero)
    free_only_params = sp.reconstruct({"alpha": jnp.zeros(sp.r), "free": result["trainable"]["free"]})
    free_only_loss, _ = loss_fn(free_only_params, q_new, tau_new, _potential, _input_mat)
    print(f"Free-only loss (alpha=0): {float(free_only_loss):.6f}")

    # Also print how much alpha actually moved
    print(f"||alpha||: {float(jnp.linalg.norm(result['trainable']['alpha'])):.6f}")

    import data_efficiency

    eff_results = data_efficiency.run_data_efficiency_experiment(
    q_all=q_new,
    tau_all=tau_new,
    w0_star=W0_star,
    subspace=subspace,
    pot_mat=_potential,
    input_mat=_input_mat,
    SubspaceParams=SubspaceParams,   # pass the class directly
    loss_fn=loss_fn,                 # pass the function directly
    data_fractions=[0.02, 0.05, 0.10, 0.25, 0.50, 1.0],
    n_epochs=20_000,
    n_repeats=3,
    free_layer_names=None,
    save_fig_path=MODEL_DIR / "data_efficiency.png",
    )
    # ------------------------------------------------------------------ #
    #  Parameter count breakdown                                           #
    # ------------------------------------------------------------------ #
    total_params = sum(int(np.prod(s)) for s in sp.structure[1])
    n_constrained = int(sp.constrained_mask.sum())
    n_free = int(sp.free_mask.sum())
    r = sp.r

    print("\n[Parameter Efficiency]")
    print(f"  Total parameters in network : {total_params:,}")
    print(f"  Subspace trainable (alpha)  : {r}  (the directions)")
    print(f"  Free trainable (biases etc) : {n_free:,}")
    print(f"  Total subspace adaptation   : {r + n_free:,}  ({100*(r+n_free)/total_params:.1f}% of network)")
    print(f"  Frozen (constrained, fixed) : {n_constrained - r:,}")

    # ------------------------------------------------------------------ #
    #  Loss comparison at full data                                        #
    # ------------------------------------------------------------------ #
    w0_loss, _       = loss_fn(W0_star,              q_new, tau_new, _potential, _input_mat)
    sub_loss, _      = loss_fn(result["full_params"], q_new, tau_new, _potential, _input_mat)
    free_only, _     = loss_fn(
        sp.reconstruct({"alpha": jnp.zeros(sp.r), "free": result["trainable"]["free"]}),
        q_new, tau_new, _potential, _input_mat
    )

    print("\n[Loss Comparison]")
    print(f"  W0_star (no adaptation)     : {float(w0_loss):.6f}")
    print(f"  Subspace (alpha + biases)   : {float(sub_loss):.6f}")
    print(f"  Biases only (alpha=0)       : {float(free_only):.6f}")
    print(f"  Improvement over W0_star    : {100*(float(w0_loss)-float(sub_loss))/float(w0_loss):.1f}%")

    import LoRA as lora_mod

    lora_results = lora_mod.run_lora_vs_subspace(
        q_all=q_new,
        tau_all=tau_new,
        w0_star=W0_star,
        subspace=subspace,
        SubspaceParams=SubspaceParams,
        pot_mat=_potential,
        input_mat=_input_mat,
        loss_fn=loss_fn,
        lora_ranks=[1, 2, 4, 8],
        data_fractions=[0.02, 0.05, 0.10, 0.25, 0.50, 1.0],
        n_epochs=20_000,
        n_repeats=3,
        save_fig_path=MODEL_DIR / "lora_vs_subspace.png",
    )

    # Parameter efficiency bar chart (uses 100% data results)
    lora_mod.plot_parameter_efficiency(
        w0_star=W0_star,
        subspace=subspace,
        SubspaceParams=SubspaceParams,
        lora_ranks=[1, 2, 4, 8],
        eval_losses={k: float(v[-1]) for k, v in lora_results.items() if k != "fractions"},
        save_path=MODEL_DIR / "parameter_efficiency.png",
    )
 
 