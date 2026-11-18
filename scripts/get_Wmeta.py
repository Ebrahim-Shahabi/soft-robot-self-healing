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
import plots as plts
import pickle, copy
import utils_dir as du
 
'''
0. Load data.
'''
SCRIPT_DIR = Path(__file__).resolve().parent

TRACKER_ROOT = (
    SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"
)


MODEL_DIR = SCRIPT_DIR.parents[0] / "models" / "models"


ROBOTS = {
    "robot1_damage1_healed": TRACKER_ROOT / "robot1_damage1_healed",
    "robot1_damage2_healed": TRACKER_ROOT / "robot1_damage2_healed",
    "robot1_damage3_healed": TRACKER_ROOT / "robot1_damage3_healed",
    "robot2_damage0": TRACKER_ROOT / "robot2",
}

ANGLES = [0, 45, 135]


# robot1_damage1 has already been trained and we will use it as W0
all_robot_dfs = du.get_all_robot_dfs(ROBOTS, ANGLES)

'''
1. Train / test models.
'''
# load W0 and its hyperparameters
'''
1. Load Model.
'''
print("Loading W0 model\n")
# This will be W0.
with open(MODEL_DIR / "W0_maml.jax", 'rb') as f:
    w0 = pickle.load(f)

activations = {
    'tanh': jnp.tanh,
    'softplus': jax.nn.softplus,
    'sigmoid': jax.nn.sigmoid,
}

hyper = w0["hyper"]
params = w0["params"]
print("W0 hyperparameters:", hyper)
W0 = copy.deepcopy(params)


# train the models
for i, robot in enumerate(all_robot_dfs.keys()):
    for angle in all_robot_dfs[robot].keys():
        if len(all_robot_dfs[robot][angle]) == 0:
            continue
        print(f"Training model for {robot} at {angle}")
        df = all_robot_dfs[robot][angle]
        x, y = tools.extract_states_targets(df)
        # x = x[:,-1].reshape(-1, 1) # only use angle as state input for W0 model.
        train_states, train_targets, test_states, test_targets = tools.split_states_targets(x, y)
        model_name = f"{robot}_{angle}"
        model = tools.setup_static_delan(train_states, train_targets, hyper, seed=42, model_name=model_name, initial_params=params)
        results = tools.train_static_model(model, hyper, test_states=test_states, test_targets=test_targets)

        # saving models
        with open(MODEL_DIR / f"{model_name}_maml_onlyangle.jax", "wb") as file:
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
        print(f"Saved model to: {MODEL_DIR / f'{model_name}_maml_onlyangle.jax'}")

