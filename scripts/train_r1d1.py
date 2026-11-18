import numpy as np
import time
import pandas as pd
from pathlib import Path
import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import utils_dir as du
import tools
from os import path
import pickle
import copy

def add_state_difference_features(df):
    df = df.copy()

    # Current state features
    state_cols = ["xe[m]", "ye[m]", "angle[rad]"]

    # Difference to previous timestep
    for col in state_cols:
        df[f"d_{col}"] = df[col].diff()

    # First row has no previous state, so set difference to zero
    for col in state_cols:
        df[f"d_{col}"] = df[f"d_{col}"].fillna(0.0)

    return df

def show_plot(robots):
    plt.figure(figsize=(16, 10))

    # Generate MANY distinct colors
    num_lines = sum(
        len(motion_dict)
        for deg_dict in robots.values()
        for motion_dict in deg_dict.values()
    )

    colors = plt.cm.nipy_spectral(np.linspace(0, 1, num_lines))

    color_idx = 0

    for robot_name, deg_dict in robots.items():

        for deg, motion_dict in deg_dict.items():

            for motion_name, df in motion_dict.items():

                # Skip empty dfs
                if df is None or df.empty:
                    continue

                # Data
                angle_deg = np.rad2deg(df["angle[rad]"])
                pressure = df["pressure[mBar]"]

                # Unique label
                label = f"{robot_name} | {deg} | {motion_name}"

                # Plot
                plt.scatter(
                    angle_deg,
                    pressure,
                    color=colors[color_idx],
                    s=15,          # marker size
                    alpha=0.8,
                    label=label
                )

                color_idx += 1

    # Labels
    plt.xlabel("Angle [deg]", fontsize=14)
    plt.ylabel("Pressure [mBar]", fontsize=14)
    plt.title("Pressure vs Angle", fontsize=16)

    # Grid
    plt.grid(True, alpha=0.3)

    # LEGEND
    plt.legend(
        loc='center left',
        bbox_to_anchor=(1.02, 0.5),
        fontsize=9,
        frameon=True
    )

    plt.tight_layout()
    plt.show()

# first i have to load the data from all the robots. 
# all the robots are in dir : ~/Desktop/tu_delft/thesis/code/exp_setup_ws/data/tracker_online
# inside folders : robot1_damage0, robot1_damage1_healed, robot1_damage2_healed, robot2_damage0, robot2_damage1
# when loading the data i have to check in the video directory of every folder how many _xdeg.csv files are there. 
# These will count as different robots.
 
'''
0. Load data.
'''
SCRIPT_DIR = Path(__file__).resolve().parent

TRACKER_ROOT = (
    SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"
)

ROBOTS = {
    # "robot1_damage0": TRACKER_ROOT / "robot1_damage0",
    "robot1_damage1_healed": TRACKER_ROOT / "robot1_damage1_healed",
    "robot1_damage2_healed": TRACKER_ROOT / "robot1_damage2_healed",
    "robot2_damage0": TRACKER_ROOT / "robot2",
    # "robot2_damage1": TRACKER_ROOT / "robot2_damage1",
}

ANGLES = [0, 45, 135]

all_robot_dfs = {}

for robot_name, robot_dir in ROBOTS.items():

    all_robot_dfs[robot_name] = {}

    for angle in ANGLES:

        deg_key = f"{angle}deg"
        runs = du.get_video_runs(
            video_dir=robot_dir / "video",
            angle=angle,
            run_name_filter=None,
        )

        if not runs:
            continue

        all_robot_dfs[robot_name][deg_key] = {}
        # print(f"{robot_name} | {deg_key}: found {len(runs)} runs")
        for motion, video_path in runs.items():

            df = du.load_df(
                label=f"{robot_name}_{deg_key}_{motion}",
                video_path=video_path,
                online_dir=robot_dir / "online",
            )
            # remove rows with pressure below 15
            pressure_mask = df["pressure[mBar]"] >= 15

            # optional: also remove all-zero rows
            nonzero_mask = (df != 0).any(axis=1)

            # combine masks
            mask = pressure_mask & nonzero_mask

            # filtered_df = df[mask]
            filtered_df = df[mask].copy()

            # Add hysteresis / direction information before shuffling
            filtered_df = add_state_difference_features(filtered_df)

            if "trapezoid" in motion:
                all_robot_dfs[robot_name][deg_key]["trapezoid"] = filtered_df
            else:
                all_robot_dfs[robot_name][deg_key][motion] = filtered_df

            # TODO: filter out non clustered elements because some points end up with angles that do not match the pressure.


print(all_robot_dfs["robot1_damage1_healed"]["0deg"]["rampDownwards"].keys())

# show_plot(all_robot_dfs)

# filtering and randomizing the data.
# also storing the motions of one robot into one dataframe for every robot.
combined_robot_dfs = {}
for robot_name, _ in all_robot_dfs.items():

    combined_robot_dfs[robot_name] = {}


    for deg_key, _ in all_robot_dfs[robot_name].items():

        temp = []

        print(f"{robot_name} | {deg_key} | {all_robot_dfs[robot_name][deg_key].keys()}")

        for r in ["rampUpwards", "rampDownwards", "trapezoid"]:
            if r in all_robot_dfs[robot_name][deg_key]:
                temp.append(all_robot_dfs[robot_name][deg_key][r].copy())
        
        # combine all motions
        combined_df = pd.concat(temp, ignore_index=True)

        # shuffle final combined dataframe
        combined_df = combined_df.sample(frac=1).reset_index(drop=True)

        # store separately
        combined_robot_dfs[robot_name][deg_key] = combined_df
            
# df = all_robot_dfs["robot1_damage1_healed"]["45deg"]["trapezoid"].copy()

# df["xe_minus_x0[m]"] = df["xe[m]"] - df["x0[m]"]
# df["ye_minus_y0[m]"] = df["ye[m]"] - df["y0[m]"]


# plt.figure(figsize=(7, 5))
# plt.scatter(df["pressure[mBar]"], df["dx_calc"], alpha=0.6)
# plt.xlabel("Pressure [mBar]")
# plt.ylabel("xe - x0 [m]")
# plt.title("Pressure vs xe - x0")
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(7, 5))
# plt.scatter(df["pressure[mBar]"], df["dy_calc"], alpha=0.6)
# plt.xlabel("Pressure [mBar]")
# plt.ylabel("ye - y0 [m]")
# plt.title("Pressure vs ye - y0")
# plt.grid(True)
# plt.show()

'''
1. Train / test split.
'''
# train the first model to get W0.
r1d1_x, r1d1_y = du.extract_states_targets(combined_robot_dfs["robot1_damage1_healed"]["0deg"])

# train_df = combined_robot_dfs["robot1_damage1_healed"]["0deg"]

# state_cols = [
#     "xe[m]",
#     "ye[m]",
#     "angle[rad]",
#     "d_xe[m]",
#     "d_ye[m]",
#     "d_angle[rad]",
# ]

# target_cols = ["pressure[mBar]"]

# r1d1_x = train_df[state_cols].values
# r1d1_y = train_df[target_cols].values


robot1_train_states, robot1_train_targets, robot1_test_states, robot1_test_targets = tools.split_states_targets(r1d1_x, r1d1_y)

actuator_dof = robot1_train_targets.shape[1]
config_dof = robot1_train_states.shape[1]
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
3. Setting up the models.
'''
robot1_model = tools.setup_static_delan(
    robot1_train_states,
    robot1_train_targets,
    robot1_test_states,
    robot1_test_targets,
    hyper,
    seed=42,
    model_name="Robot1",
)

# '''
# 4. Training loop.
# '''
# t0 = time.perf_counter()
# robot1_result = tools.train_static_model(
#     robot1_model,
#     hyper=hyper,
#     step=50,
#     make_plot=True,
# )
# print(time.perf_counter() - t0)

# robot1_model = robot1_result["model"]

# plt.show()

# basepath = path.dirname(__file__)
# path = path.join(
#     basepath,
#     "..",
#     "models",
#     "static_models",
#     "r1d1.jax"
# )

# with open(path, "wb") as file:
#     pickle.dump(
#         {
#             "hyper": hyper,
#             "params": robot1_result["params"],
#             "train_losses": robot1_result["train_losses"],
#             "test_losses": robot1_result["test_losses"],
#             "iterations": robot1_result["iterations"],
#         },
#         file,
#     )


'''
3. Load pretrained model.
'''
basepath = path.dirname(__file__)
model_path = path.join(basepath, "..", "models", "static_models", "r1d1.jax")

with open(model_path, "rb") as f:
    saved = pickle.load(f)

hyper  = saved["hyper"]
params = saved["params"]
# robot1_result["params"] = saved["params"]
print("Loaded r1d1 model, hyper:", hyper)

'''
4. Forward prediction plot (no retraining).
'''
import DeLaN_model_static as delan
import haiku as hk
from functools import partial

activations = {
    "tanh": jnp.tanh,
    "softplus": jax.nn.softplus,
    "sigmoid": jax.nn.sigmoid,
}

potential_fn = hk.without_apply_rng(hk.transform(
    partial(
        delan.potential_energy,
        shape=(hyper["n_width"],) * hyper["n_depth"],
        activation=activations[hyper["activation1"]],
    )
))

input_fn = hk.without_apply_rng(hk.transform(
    partial(
        delan.input_transform_matrix,
        shape=(hyper["n_width"] // 2,) * (hyper["n_depth"] - 1),
        n_dof=hyper["n_dof"],
        actuator_dof=hyper["actuator_dof"],
        activation=activations[hyper["activation1"]],
    )
))

def pot_mat(p, key, q):
    return potential_fn.apply(p, q)

def input_mat(p, key, q):
    return input_fn.apply(p, q)

r1d1_full_q = jnp.array(r1d1_x, dtype=jnp.float32)
raw_angles_deg = np.rad2deg(
    combined_robot_dfs["robot1_damage1_healed"]["0deg"]["angle[rad]"].values
)

f = delan.forward_model(params, key=None, pot_mat=pot_mat, input_mat=input_mat)
_, tau_pred = jax.vmap(f)(r1d1_full_q)
tau_pred = np.array(tau_pred).ravel()

fig, ax = plt.subplots(figsize=(10, 4))
ax.scatter(raw_angles_deg, r1d1_y, s=5, alpha=0.5, label="data")
ax.scatter(raw_angles_deg, tau_pred, s=5, alpha=0.5, label="predicted")
ax.set_xlabel("Angle [deg]")
ax.set_ylabel("Pressure [mBar]")
ax.set_title("r1d1 0deg — DeLaN prediction (loaded)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()