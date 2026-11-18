from __future__ import annotations
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
import optax
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import plots as plts

# LoRA parameter manager
class LoRAParams:
    """
    Manages LoRA parameterisation for a haiku param dict.

    For every weight matrix W0 (any leaf whose path ends in 'w'):
        W = W0 + B @ A      B: (d, r),  A: (r, k)

    Biases (path ends in 'b') are always fully free.

    The trainable pytree exposed to the optimiser:
        {
          "lora": { layer_path: {"B": ..., "A": ...} },
          "biases": { layer_path: {"b": ...} }
        }
    """

    def __init__(self, w0_star: dict, rank: int, rng_seed: int = 0):
        self.rank = rank
        self.w0_star = w0_star

        # Flatten to get paths and shapes
        leaves_with_paths = []
        def collect(path, leaf):
            key = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
            leaves_with_paths.append((key, leaf))
            return leaf
        jax.tree_util.tree_map_with_path(collect, w0_star)

        self._weight_paths = {}   # path -> (d, k) shape
        self._bias_paths   = {}   # path -> shape

        for path_str, leaf in leaves_with_paths:
            if path_str.endswith("/w") and leaf.ndim == 2:
                self._weight_paths[path_str] = leaf.shape   # (d, k)
            elif path_str.endswith("/b"):
                self._bias_paths[path_str] = leaf.shape

        n_lora_params = sum(
            d * rank + rank * k
            for (d, k) in self._weight_paths.values()
        )
        n_bias_params = sum(int(np.prod(s)) for s in self._bias_paths.values())
        n_total = sum(int(np.prod(v.shape)) for _, v in leaves_with_paths)

        print(f"\n[LoRA rank={rank}]")
        print(f"  Weight matrices with LoRA : {len(self._weight_paths)}")
        print(f"  LoRA parameters (B+A)     : {n_lora_params:,}")
        print(f"  Free bias parameters      : {n_bias_params:,}")
        print(f"  Total trainable           : {n_lora_params + n_bias_params:,}  "
              f"({100*(n_lora_params+n_bias_params)/n_total:.1f}% of network)")
        print(f"  Frozen                    : {n_total - n_lora_params - n_bias_params:,}")

        self.n_trainable = n_lora_params + n_bias_params
        self.n_total     = n_total
        self._rng_seed   = rng_seed

    # ------------------------------------------------------------------
    def init_trainable(self) -> dict:
        """
        Initialise trainable params.
        B: zero init  (so W = W0 + 0 @ A = W0 at start)
        A: random normal scaled by 1/sqrt(rank)
        biases: from W0_star
        """
        rng = jax.random.PRNGKey(self._rng_seed)

        lora = {}
        for path_str, (d, k) in self._weight_paths.items():
            rng, k1, k2 = jax.random.split(rng, 3)
            lora[path_str] = {
                "B": jnp.zeros((d, self.rank)),                          # zero init
                "A": jax.random.normal(k2, (self.rank, k)) * (1.0 / self.rank),
            }

        # Extract biases from w0_star
        biases = {}
        def collect_bias(path, leaf):
            key = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
            if key in self._bias_paths:
                biases[key] = jnp.array(leaf)
            return leaf
        jax.tree_util.tree_map_with_path(collect_bias, self.w0_star)

        return {"lora": lora, "biases": biases}

    # ------------------------------------------------------------------
    def reconstruct(self, trainable: dict) -> dict:
        """
        Rebuild the full haiku param dict:
            W_layer = W0_layer + B @ A
            b_layer = trainable bias
        """
        lora    = trainable["lora"]
        biases  = trainable["biases"]

        def update_leaf(path, leaf):
            key = "/".join(str(k.key) if hasattr(k, "key") else str(k) for k in path)
            if key in self._weight_paths:
                B, A = lora[key]["B"], lora[key]["A"]
                return jnp.array(leaf) + B @ A
            elif key in self._bias_paths:
                return biases[key]
            return jnp.array(leaf)

        return jax.tree_util.tree_map_with_path(update_leaf, self.w0_star)


