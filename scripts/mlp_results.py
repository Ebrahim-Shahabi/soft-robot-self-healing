import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from os import path
from functools import partial
import pickle

import haiku as hk
import jax
import jax.numpy as jnp
import MLP_model as network


basepath = path.dirname(__file__)

csv_path = path.join(
    basepath, "../data/experiments/online_output20260428-132925.csv"
)

model_path = path.join(
    basepath, "../models/mlp_models/20260311-125118.jax"
)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

pressure_true = df["pressure[mBar]"].to_numpy()

x = (df["xe[m]"] - df["x0[m]"]).to_numpy() / 100
y = (df["ye[m]"] - df["y0[m]"]).to_numpy() / 100
angle = df["angle[rad]"].to_numpy()

states = np.column_stack((x, y, angle)).astype(np.float32)

# Try to find a time column automatically
time_col = None
for col in df.columns:
    if "time" in col.lower():
        time_col = col
        break

if time_col is not None:
    t = df[time_col].to_numpy()
else:
    t = np.arange(len(df))


# -------------------------
# Load trained model
# -------------------------
with open(model_path, "rb") as file:
    saved = pickle.load(file)

params = saved["params"]
hyper = saved["hyper"]

activations = {
    "tanh": jnp.tanh,
    "softplus": jax.nn.softplus,
}

config_dof = hyper["n_dof"]
actuator_dof = hyper["actuator_dof"]

net_fn = hk.transform(
    partial(
        network.network,
        state_dof=config_dof,
        shape=(hyper["n_width"],) * hyper["n_depth"],
        activation=activations[hyper["activation2"]],
    )
)

net_ = net_fn.apply

# Predict pressure
states_jax = jnp.asarray(states, dtype=jnp.float32)

pressure_pred_scaled = net_(params["network"], None, states_jax)
pressure_pred_scaled = np.asarray(pressure_pred_scaled).reshape(-1)

# Your training scaled pressure as pressure[mBar] / 100
pressure_pred = pressure_pred_scaled * 100

# -------------------------
# Plotting
# -------------------------

# Use actual time column if it exists
time_axis = df["t_system[s]"].to_numpy()

# Convert scaled pressure -> kPa
# pressure[mBar]/100 * 10 = kPa
true_kpa = pressure_true * 0.1
pred_kpa = pressure_pred * 0.1

x_rel = x
y_rel = y
angle_deg = np.rad2deg(angle)

plt.style.use("seaborn-v0_8-whitegrid")

# --- Time series ---
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_title("Pressure: Measured vs MLP")
ax.plot(time_axis, true_kpa, label="Measured")
ax.plot(time_axis, pred_kpa, "--", label="Predicted")
ax.set_xlabel("time [s]")
ax.set_ylabel("pressure [kPa]")
ax.legend()

# --- vs Angle ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_title("Pressure vs Angle")
ax.scatter(angle_deg, true_kpa, s=12, alpha=0.6, label="Measured")
ax.scatter(angle_deg, pred_kpa, s=12, alpha=0.6, label="Predicted")
ax.set_xlabel("angle [deg]")
ax.set_ylabel("pressure [kPa]")
ax.legend()

# --- vs X ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_title("Pressure vs Relative X Position")
ax.scatter(x_rel, true_kpa, s=12, alpha=0.6, label="Measured")
ax.scatter(x_rel, pred_kpa, s=12, alpha=0.6, label="Predicted")
ax.set_xlabel("relative x [m/100]")
ax.set_ylabel("pressure [kPa]")
ax.legend()

# --- vs Y ---
fig, ax = plt.subplots(figsize=(7, 5))
ax.set_title("Pressure vs Relative Y Position")
ax.scatter(y_rel, true_kpa, s=12, alpha=0.6, label="Measured")
ax.scatter(y_rel, pred_kpa, s=12, alpha=0.6, label="Predicted")
ax.set_xlabel("relative y [m/100]")
ax.set_ylabel("pressure [kPa]")
ax.legend()

plt.show()