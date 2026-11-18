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
from plots import plot_subspace_analysis

SCRIPT_DIR = Path(__file__).resolve().parent

MODEL_DIR = SCRIPT_DIR.parents[0] / "models" / "models"
TRACKER_ROOT = (
    SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"
)

robot_dir = TRACKER_ROOT / "robot2_damage1_healed"
deg_key = "0deg"
runs = du.get_video_runs(
    video_dir=robot_dir / "video",
    run_name_filter="trapezoid",
)
print(runs.keys())
print(f"robot2_damage1_healed | {deg_key}: found {len(runs)} runs in directory : \n")
print(runs["trapezoid"])

r2d1_path = du.find_csv_for_video(video_path=runs["trapezoid"], online_dir=robot_dir / "online")

r2d1_df = du.load_df(
    label=f"robot2_damage1_healed_{deg_key}_trapezoid",
    video_path=runs["trapezoid"],
    online_dir=robot_dir / "online",
)

r2d1_df = r2d1_df[r2d1_df["angle[rad]"] != 0].copy()        # drop initial-state zero-angle rows
r2d1_df = r2d1_df[r2d1_df["pressure[mBar]"] >= 25].copy()    # drop low-pressure noise rows

'''
1. Use the whole (cleaned) dataset for training -- no angle/pressure boundary filtering.
'''
x, y = du.extract_states_targets(r2d1_df)

q_data = jnp.array(x, dtype=jnp.float32)
tau_data = jnp.array(y.reshape(-1, 1), dtype=jnp.float32)
N = q_data.shape[0]
print(f"\nTraining on {N} samples (full dataset, no boundary filter)\n")

'''
2. Load base model and task models for the subspace.
'''
# with open(MODEL_DIR / "W0_maml.jax", "rb") as f:
#     base_data = pickle.load(f)

with open(MODEL_DIR / "robot1_damage1_healed_0deg.jax", "rb") as f:
    base_data = pickle.load(f)

hyper = base_data["hyper"]
W0 = base_data["params"]

TASK_MODEL_FILES = {
    "r1d1_45deg":  MODEL_DIR / "robot1_damage1_healed_45deg.jax",
    "r1d1_135deg": MODEL_DIR / "robot1_damage1_healed_135deg.jax",
    "r1d2_0deg":   MODEL_DIR / "robot1_damage2_healed_0deg.jax",
    "r1d3_0deg":   MODEL_DIR / "robot1_damage3_healed_0deg.jax",
    "r1d3_135deg": MODEL_DIR / "robot1_damage3_healed_135deg.jax",
    "r2d0_0deg":   MODEL_DIR / "robot2_damage0_0deg.jax",
    "r2d0_135deg": MODEL_DIR / "robot2_damage0_135deg.jax",
}

task_weights = {}
for name, fpath in TASK_MODEL_FILES.items():
    with open(fpath, "rb") as f:
        task_weights[name] = pickle.load(f)["params"]

subspace = sb.compute_task_subspace(
    w0_star=W0,
    task_weights=task_weights,
    variance_threshold=0.95,
    max_rank=None,
    free_layer_names=None,
)



sp = sb.SubspaceParams(subspace=subspace, w0_star=W0, free_layer_names=None)

plot_subspace_analysis(subspace)

'''
3. Build haiku apply functions.
'''
N_EPOCHS = 20000
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
activation = activations[hyper["activation1"]]
hidden_pot = (hyper["n_width"],) * hyper["n_depth"]
hidden_inp = (hyper["n_width"] // 2,) * (hyper["n_depth"] - 1)
n_dof = hyper["n_dof"]
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
4. Train models on the full dataset.
'''
# ---- subspace ----------------------------------------------------------------
@jax.jit
def _subspace_loss(trainable, q_b, tau_b, _potential=_potential, _input_mat=_input_mat):
    full_p = sp.reconstruct(trainable)
    loss, _ = DeLan.loss_fn(full_p, q_b, tau_b, _potential, _input_mat)
    return loss + 1e-3 * jnp.sum(trainable["alpha"] ** 2), {}

trainable = sp.init_trainable()
opt = optax.adam(LR)
state = opt.init(trainable)

@jax.jit
def _step_sub(tr, st, q_b, tau_b):
    (l, _), g = jax.value_and_grad(_subspace_loss, has_aux=True)(tr, q_b, tau_b)
    upd, st = opt.update(g, st)
    return optax.apply_updates(tr, upd), st, l

import time
t0 = time.time()
sub_losses = []
flag = True
for epoch in range(N_EPOCHS):
    bi = np.random.permutation(N)[:BATCH_SIZE]
    trainable, state, l = _step_sub(trainable, state, q_data[bi], tau_data[bi])
    sub_losses.append(float(l))
    if (l <= 0.02) and flag:
        flag = False
        print('Time for subspace method is:', time.time() - t0)
    if epoch % 2000 == 0 or epoch == N_EPOCHS - 1:
        print(f"  [Subspace] epoch {epoch:5d}  loss={float(l):.6f}")

sub_params = sp.reconstruct(trainable)


# ---- fine-tune from W0 ---------------------------------------------------
ft_model = tools.setup_static_delan(
    q_data, tau_data, hyper, seed=42, model_name="fine_tune", initial_params=W0
)
results = tools.train_static_model(ft_model, hyper, make_plot=False)
ft_params = results["params"]
print('Time for fine-tune method is:', time.time() - t0)

# ---- one-shot SVD (from a task model) ------------------------------------
# W2 = task_weights["r2d0_0deg"]
# svd_model = tools.setup_oneshot_delan_svd(
#     q_data, tau_data, hyper, seed=42, model_name="svd", initial_params=W2
# )
# results = tools.train_static_model(svd_model, hyper)
# sc_params = results["params"]

# def svd_params_to_dense(svd_params):
#     dense = {}
#     for mod_name, leaf in svd_params.items():
#         new_leaf = {}

#         if "U" in leaf and "log_s" in leaf and "Vt" in leaf:
#             U = leaf["U"]
#             log_s = leaf["log_s"]
#             Vt = leaf["Vt"]
#             W = (U * jnp.exp(log_s)[None, :]) @ Vt   # (out, in)
#             new_leaf["w"] = W.T                       # (in, out) for hk.Linear

#         elif "w" in leaf:
#             w = leaf["w"]
#             if w.ndim == 1:
#                 if mod_name.endswith("linear_0"):
#                     new_leaf["w"] = w.reshape(1, -1)   # (1, out)
#                 else:
#                     new_leaf["w"] = w.reshape(-1, 1)   # (in, 1)
#             else:
#                 new_leaf["w"] = w

#         if "b" in leaf:
#             new_leaf["b"] = leaf["b"]

#         dense_name = mod_name.replace("/linear_", "/~/linear_")
#         dense[dense_name] = new_leaf

#     return dense

# sc_params = {
#     "potential":       svd_params_to_dense(sc_params["potential"]),
#     "input_transform": svd_params_to_dense(sc_params["input_transform"]),
# }



model_name = "scratch"
model = tools.setup_static_delan(q_data, tau_data, hyper, seed=42, model_name=model_name, initial_params=None)
results = tools.train_static_model(model, hyper)
sc_params = results["params"]

# ---- LoRA -----------------------------------------------------------------
lora_model = lora.LoRAParams(w0_star=W0, rank=1, rng_seed=42)
results = tools.train_lora_delan(
    lora_model, q_data, tau_data, _potential, _input_mat, n_epochs=N_EPOCHS
)
lora_params = results["params"]
print(lora_params.keys())

'''
5. Plot predictions vs. ground truth on the (full) training data.
'''
angle_plot = np.rad2deg(r2d1_df["angle[rad]"].values)

def predict(params):
    f = DeLan.forward_model(params, None, _potential, _input_mat)
    _, tau_p = jax.vmap(f)(q_data)
    return np.array(tau_p).ravel() * 100  # convert to mBar for plotting

tau_sub = predict(sub_params)
tau_ft = predict(ft_params)
tau_sc = predict(sc_params)
tau_lora = predict(lora_params)
y_true = np.array(tau_data).ravel() * 100

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
fig.suptitle(
    f"Robot 2 — {TARGET_ANGLE}° | trained on full dataset",
    fontsize=13, fontweight="bold",
)

plot_cfg = [
    (axes[0, 0], tau_sc,   sc_params,   "Scratch",        "#eb2525"),
    (axes[0, 1], tau_ft,   ft_params,   "Fine-tune",                      "#16a34a"),
    (axes[1, 0], tau_lora, lora_params, f"LoRA rank={lora_model.rank}",   "#0c2dea"),
    (axes[1, 1], tau_sub,  sub_params,  "Subspace",                       "#CE0FFE"),
]

for ax, tau_p, params, title, color in plot_cfg:
    ax.scatter(angle_plot, y_true/10, s=4, alpha=0.3, color="grey", label="data")
    ax.scatter(angle_plot, tau_p/10, s=4, alpha=0.5, color=color, label="predicted")
    loss_val, _ = DeLan.loss_fn(params, q_data, tau_data, _potential, _input_mat)
    ax.set_title(f"{title}  (loss={float(loss_val):.5f})")
    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Pressure [kPa]")
    ax.legend(markerscale=3)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

'''
6. Final loss summary.
'''
print("\n[Final loss on full dataset]")
for name, params in [
    ("Subspace", sub_params),
    ("Fine-tune", ft_params),
    ("Scratch (SVD)", sc_params),
    ("LoRA", lora_params),
]:
    l, _ = DeLan.loss_fn(params, q_data, tau_data, _potential, _input_mat)
    print(f"  {name:15s}: {float(l):.6f}")