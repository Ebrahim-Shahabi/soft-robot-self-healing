from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import utils_dir as du
import tools
import pickle
import plots as plts
import DeLaN_model_static as DeLan
import jax
import numpy as np

'''
0. Load data.
'''
SCRIPT_DIR = Path(__file__).resolve().parent

TRACKER_ROOT = (
    SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online" / "robot1_damage1_healed"
)


deg_key = "0deg"
runs = du.get_video_runs(
    video_dir= TRACKER_ROOT / "video",
    run_name_filter="trapezoid",
)

print(f"robot1_damage1_healed | {deg_key}: found {len(runs)} runs in directory : \n")
print(runs["trapezoid"])

r1d1_path = du.find_csv_for_video(video_path= runs["trapezoid"], online_dir=TRACKER_ROOT / "online")

print(r1d1_path)
r1d1_df = du.load_df(
    label=f"robot1_damage1_healed_{deg_key}_trapezoid",
    video_path=runs["trapezoid"],
    online_dir=TRACKER_ROOT / "online",
)

r1d1_df = r1d1_df[r1d1_df["angle[rad]"] != 0].copy() # remove zero angle rows because they are not real data points but just the initial state.

# plts.plot_one_motion(
#     robot_name="robot1_damage1_healed",
#     motion="0deg_trapezoid",
#     dfs={"trapezoid": r1d1_df},
#     save_flag=False,
# )


'''
1. Train / test split.
'''
# train the first model to get W0.
r1d1_x, r1d1_y = tools.extract_states_targets(r1d1_df)
r1d1_x = r1d1_x[:,-1].reshape(-1, 1) # only use angle as state input for W0 model.
train_states, train_targets, test_states, test_targets = tools.split_states_targets(r1d1_x, r1d1_y)

actuator_dof = train_targets.shape[1]
config_dof = train_states.shape[1]

'''
2. Hyperparameters.
'''
hyper = {
    "n_dof": config_dof,
    "actuator_dof": actuator_dof,
    "n_width": 64,
    "n_depth": 3,
    "n_minibatch": 200,
    "activation1": "softplus",
    "activation2": "tanh",
    "learning_rate": 1e-4,
    "weight_decay": 3.2e-5,
    "max_epoch": int(6.0 * 1e3),
}


'''
3. Setting up  and training the model.
'''
model_name = "robot1_damage1_healed_0deg"   
model = tools.setup_static_delan(train_states, train_targets, hyper, seed=42, model_name=model_name, initial_params=None)
results = tools.train_static_model(model, hyper, test_states=test_states, test_targets=test_targets)

'''
5. Plot W0 predictions on full dataset.
'''
import jax.numpy as jnp
import numpy as np
import haiku as hk
import DeLaN_model_static as DeLan

activation_map = {
    "tanh": jax.numpy.tanh,
    "softplus": jax.nn.softplus,
    "sigmoid": jax.nn.sigmoid,
}
activation = activation_map[hyper["activation1"]]
hidden_pot = (hyper["n_width"],) * hyper["n_depth"]
hidden_inp = (hyper["n_width"] // 2,) * (hyper["n_depth"] - 1)

pot_transformed = hk.without_apply_rng(hk.transform(
    lambda q: DeLan.potential_energy(q, hidden_pot, activation)
))
input_transformed = hk.without_apply_rng(hk.transform(
    lambda q: DeLan.input_transform_matrix(q, hyper["n_dof"], hyper["actuator_dof"], hidden_inp, activation)
))

def _potential(params, key, q):
    return pot_transformed.apply(params, q)

def _input_mat(params, key, q):
    return input_transformed.apply(params, q)

# full dataset (no filtering)
x_full, y_full = tools.extract_states_targets(r1d1_df)
q_full   = jnp.array(x_full, dtype=jnp.float32)
q_full = q_full[:,-1].reshape(-1, 1) # only use angle as state input for W0 model.
tau_full = jnp.array(y_full.reshape(-1, 1), dtype=jnp.float32)
angle_full = np.rad2deg(r1d1_df["angle[rad]"].values)

W0_params = results["params"]
f = DeLan.forward_model(W0_params, None, _potential, _input_mat)
_, tau_w0_full = jax.vmap(f)(q_full)
tau_w0_full = np.array(tau_w0_full).ravel()

plt.figure()
plt.title("W0 predictions on FULL dataset")
plt.scatter(angle_full, np.array(tau_full).ravel(), s=4, alpha=0.25, color="grey", label="full data")
plt.scatter(angle_full, tau_w0_full, s=4, alpha=0.5, color="blue", label="W0 predicted")
plt.xlabel("Angle [deg]")
plt.ylabel("Pressure [mBar]")
plt.legend(markerscale=3)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


'''
4. Saving up the model
'''
model_dir = SCRIPT_DIR.parents[0] / "models" / "models"


with open(model_dir / f"{model_name}_onlyangle.jax", "wb") as file:
    pickle.dump(
        {
            "model_name": model_name,
            "hyper": hyper,
            "params":  results["params"],
            "train_losses": results["train_losses"],
            "test_losses": results["test_losses"],
            "iterations": results["iterations"],
        },
        file,
    )

print(f"Saved model to: {model_dir / f'{model_name}_onlyangle.jax'}")