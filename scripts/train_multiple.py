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

            filtered_df = df[mask]

            if "trapezoid" in motion:
                all_robot_dfs[robot_name][deg_key]["trapezoid"] = filtered_df
            else:
                all_robot_dfs[robot_name][deg_key][motion] = filtered_df

            # TODO: filter out non clustered elements because some points end up with angles that do not match the pressure.

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
            
print(all_robot_dfs["robot1_damage1_healed"]["0deg"]["rampDownwards"].keys())


# load the model obtained from r1d1 data.
'''
1. Load Model.
'''
# This will be W0.
with open(f"./../models/static_models/r1d1.jax", 'rb') as f:
    data = pickle.load(f)

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

hyper = data["hyper"]
params = data["params"]
# hyper['learning_rate'] = 2.7e-03
print("hyperparameters:", hyper)
W0 = copy.deepcopy(params)
# Target robot is : R2D0 0 deg.
'''
2. Setup other models.
'''
models = {}

datasets_to_train = {
    "r1d1_45deg": combined_robot_dfs["robot1_damage1_healed"]["45deg"],
    "r1d1_135deg": combined_robot_dfs["robot1_damage1_healed"]["135deg"],
    "r1d2_0deg": combined_robot_dfs["robot1_damage2_healed"]["0deg"],
    "r2d0_135deg": combined_robot_dfs["robot2_damage0"]["135deg"],
}

SAVE_DIR = Path("./../models/static_models")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

models = {}
results = {}

for model_name, df in datasets_to_train.items():

    print(f"\n==============================")
    print(f"Training {model_name}")
    print(f"==============================")

    model = tools.setup_model_from_df(
        df,
        hyper=hyper,
        model_name=model_name,
        seed=42,
        initial_params=W0,
    )

    result = tools.train_static_model(
        model,
        hyper=hyper,
        step=50,
        make_plot=True,
    )

    models[model_name] = result["model"]
    results[model_name] = result

    save_path = SAVE_DIR / f"{model_name}.jax"

    with open(save_path, "wb") as file:
        pickle.dump(
            {
                "model_name": model_name,
                "hyper": hyper,
                "params": result["params"],
                "train_losses": result["train_losses"],
                "test_losses": result["test_losses"],
                "iterations": result["iterations"],
            },
            file,
        )

    print(f"Saved model to: {save_path}")

plt.show()