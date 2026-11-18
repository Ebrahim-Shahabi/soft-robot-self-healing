from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import utils_dir as du
import tools
import pickle
import plots as plts

'''
0. Load data.
'''
SCRIPT_DIR = Path(__file__).resolve().parent

TRACKER_ROOT = (
    SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"
)


R1D1_ROOT = (
    SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online" / "robot1_damage1_healed"
)


deg_key = "0deg"
runs = du.get_video_runs(
    video_dir= R1D1_ROOT / "video",
    run_name_filter="trapezoid",
)

print(f"robot1_damage1_healed | {deg_key}: found {len(runs)} runs in directory : \n")
print(runs["trapezoid"])

r1d1_path = du.find_csv_for_video(video_path= runs["trapezoid"], online_dir=R1D1_ROOT / "online")

print(r1d1_path)
r1d1_df = du.load_df(
    label=f"robot1_damage1_healed_{deg_key}_trapezoid",
    video_path=runs["trapezoid"],
    online_dir=R1D1_ROOT / "online",
)

r1d1_df = r1d1_df[r1d1_df["angle[rad]"] != 0].copy() # remove zero angle rows because they are not real data points but just the initial state.

'''
1. Load other robots' data for meta-learning.
'''
ROBOTS = {
    "robot1_damage1_healed": TRACKER_ROOT / "robot1_damage1_healed",
    "robot1_damage2_healed": TRACKER_ROOT / "robot1_damage2_healed",
    "robot1_damage3_healed": TRACKER_ROOT / "robot1_damage3_healed",
    "robot2_damage0": TRACKER_ROOT / "robot2",
}

ANGLES = [0, 45, 135]


# robot1_damage1 has already been trained and we will use it as W0

all_robot_dfs = du.get_all_robot_dfs(ROBOTS, ANGLES, name_filter="trapezoid")

'''
1. Train / test split.
'''
# train the first model to get W0.
r1d1_x, r1d1_y = tools.extract_states_targets(r1d1_df)
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
model_name = "early_stopping"   
model = tools.setup_static_delan(train_states, train_targets, hyper, seed=42, model_name=model_name, initial_params=None)
results = tools.early_stopping_train_static_model(
    model=model,          # your W0 model dict from setup_static_delan
    hyper=hyper,
    source_data=(train_states, train_targets),
    all_robot_dfs=all_robot_dfs,
    patience=20,          # in units of `step` evaluations
    min_delta=1e-4,
    step=50,
)
#  return {
#         "model":              model,
#         "params":             params_w0,
#         "wi_models":          wi_models,
#         "similarity_history": similarity_history,
#         "w0_loss_history":    w0_loss_history,
#         "delta_norm_history": delta_norm_history,
#         "iterations":         iterations,
#         "best_similarity":    best_similarity,
#         "stopped_epoch":      epoch_i,
#         "fig":                fig,
#     }

'''
4. Saving up the model
'''
model_dir = SCRIPT_DIR.parents[0] / "models" / "models"


with open(model_dir / f"{model_name}.jax", "wb") as file:
    pickle.dump(
        {
            "model_name": model_name,
            "hyper": hyper,
            "params":  results["params"],
            "train_losses": results["w0_loss_history"],
            "iterations": results["iterations"],
        },
        file,
    )

print(f"Saved model to: {model_dir / f'{model_name}.jax'}")