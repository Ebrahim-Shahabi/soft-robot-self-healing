import numpy as np
import jax
import jax.numpy as jnp
import tools 
import DeLaN_model_static as delan

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
        return tools.unflatten_params(w_full, self.structure)

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

    w0_flat, structure = tools.flatten_params(w0_star)
    d = len(w0_flat)
    K = len(task_weights)

    free_mask = tools.build_free_mask_from_structure(
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
        wk_flat, _ = tools.flatten_params(wk)
        Delta[:, k] = wk_flat - w0_flat
        print(f"  Task '{name}': ||ΔW_full|| = {np.linalg.norm(Delta[:, k]):.4f}")

    Delta_constrained = Delta[constrained_mask, :] # keep only non-free parameters for SVD

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

