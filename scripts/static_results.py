import os
import pickle
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import haiku as hk
from functools import partial

import DeLaN_model_static as delan


def load_data(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    # Exactly mirrors training code preprocessing
    pressure = df["pressure[mBar]"] / 100  # mBar -> scaled (same as training 'input')

    pe = (df["xe[m]"] - df["x0[m]"]) / 100
    ye = (df["ye[m]"] - df["y0[m]"]) / 100
    angle = np.array(df["angle[rad]"]).reshape(-1, 1)

    q = np.column_stack((pe, ye, angle)).astype(np.float32)
    pressure = np.array(pressure).reshape(-1, 1).astype(np.float32)

    return df, q, pressure


def load_static_delan(model_path):
    with open(model_path, "rb") as file:
        saved = pickle.load(file)

    hyper = saved["hyper"]
    params = saved["params"]

    print("Loaded model hyperparameters:", hyper)

    activations = {
        "tanh": jnp.tanh,
        "softplus": jax.nn.softplus,
        "sigmoid": jax.nn.sigmoid,
    }

    # Mirror exactly how training code builds these
    potential_fn = hk.transform(
        partial(
            delan.potential_energy,
            shape=(hyper["n_width"],) * hyper["n_depth"],
            activation=activations[hyper["activation1"]],
        )
    )

    input_fn = hk.transform(
        partial(
            delan.input_transform_matrix,
            shape=(hyper["n_width"] // 2,) * (hyper["n_depth"] - 1),
            n_dof=hyper["n_dof"],
            actuator_dof=hyper["actuator_dof"],
            activation=activations[hyper["activation1"]],
        )
    )

    pot_mat = potential_fn.apply
    input_mat = input_fn.apply

    # Mirror exactly how training code constructs forward_model
    forward_model = delan.forward_model(
        params=params,
        key=None,
        pot_mat=pot_mat,
        input_mat=input_mat,
    )

    def predict(q):
        q = jnp.asarray(q, dtype=jnp.float32)
        batched_forward = jax.vmap(forward_model)
        _, tau_hat = batched_forward(q)  # forward_model returns (A, tau_hat)
        if tau_hat.ndim == 1:
            tau_hat = tau_hat[:, None]
        return np.asarray(tau_hat)

    return predict


def plot_results(csv_path, model_path):
    df, q, pressure_true = load_data(csv_path)

    predict = load_static_delan(model_path)
    pressure_pred = predict(q)
    pressure_pred = np.asarray(pressure_pred).reshape(-1, 1)

    time_axis = df["t_system[s]"].to_numpy()

    # Both are in the same scaled units (mBar/100); convert to kPa: x * 10
    # mBar/100 * 10 = mBar/10 = kPa  (1 mBar = 0.1 kPa)
    true_kpa = pressure_true[:, 0] * 10
    pred_kpa = pressure_pred[:, 0] * 10

    x_rel = q[:, 0]
    y_rel = q[:, 1]
    angle_deg = np.rad2deg(q[:, 2])

    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Time series ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title("Pressure: Measured vs Static DeLaN")
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


if __name__ == "__main__":
    basepath = os.path.dirname(__file__)

    csv_path = os.path.join(
        basepath,
        "../",
        "data",
        "experiments",
        "online_output20260428-132925.csv",
    )

    model_path = os.path.join(
        basepath,
        "..",
        "models",
        "static_models",
        "20260311-125118.jax",
    )

    plot_results(csv_path, model_path)