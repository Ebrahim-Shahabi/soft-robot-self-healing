import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from os import path

basepath = path.dirname(__file__)

# csv_path = path.join(
#     basepath,
#     "..",
#     "exp_setup_ws",
#     "data",
#     "tracker_online",
#     "robot2_damage1_healed",
#     "controller",
#     "online",
#     "online_output20260513-115034.csv"
# )

csv_path = path.join(
    basepath,
    "..",
    "exp_setup_ws",
    "data",
    "tracker_online",
    "robot2",
    "controller",
    "online",
    "online_output20260518-181226.csv"
    # "online_output20260316-130710.csv"
    # "online_output20260316-114121.csv"
    # "online_output20260312-143728_chirp.csv"
    # "online_output20260311-125118.csv"
    # "online_output20260420-120538.csv"
)


# -----------------------------
# Read CSV
# -----------------------------
df = pd.read_csv(csv_path)
df.columns = df.columns.str.strip()

print(df.head())
print(df.columns)

# -----------------------------
# Clean and compute values
# -----------------------------
df["FOUND_INITIAL"] = df["FOUND_INITIAL"].astype(str).str.strip()
df["FOUND_ENDPOINT"] = df["FOUND_ENDPOINT"].astype(str).str.strip()

df["ANGLE_VALID"] = (
    (df["FOUND_INITIAL"] == "True") &
    (df["FOUND_ENDPOINT"] == "True")
)

df["dx_calc"] = df["xe[m]"] - df["x0[m]"]
df["dy_calc"] = df["ye[m]"] - df["y0[m]"]

# -----------------------------
# Reference values from ROS ref topic
# ros2 topic pub --once /ref std_msgs/msg/Float32MultiArray "{data: [0.6414013, -180.3318, 134.6581]}"
# data = [angle(rad), dx, dy]
# -----------------------------
ref_angle_rad = 0.6414013
ref_angle_deg = np.rad2deg(ref_angle_rad)

ref_dx = -180.3318
ref_dy = 134.6581

# Endpoint reference reconstructed from base point + reference dx/dy
df["ref_xe"] = df["x0[m]"] + ref_dx
df["ref_ye"] = df["y0[m]"] + ref_dy

print("\nReference values:")
print(f"angle = {ref_angle_rad:.7f} rad = {ref_angle_deg:.4f} deg")
print(f"dx = {ref_dx:.4f}")
print(f"dy = {ref_dy:.4f}")

# -----------------------------
# Color definitions
# -----------------------------
endpoint_colors = {
    "True": "green",
    "False": "red",
}

endpoint_labels = {
    "True": "FOUND_ENDPOINT = True",
    "False": "FOUND_ENDPOINT = False",
}

angle_colors = {
    True: "green",
    False: "red",
}

angle_labels = {
    True: "Angle valid",
    False: "Angle invalid",
}

pair_colors = {
    ("True", "True"): "green",
    ("True", "False"): "blue",
    ("False", "True"): "orange",
    ("False", "False"): "red",
}

pair_labels = {
    ("True", "True"): "FOUND_INITIAL=True, FOUND_ENDPOINT=True",
    ("True", "False"): "FOUND_INITIAL=True, FOUND_ENDPOINT=False",
    ("False", "True"): "FOUND_INITIAL=False, FOUND_ENDPOINT=True",
    ("False", "False"): "FOUND_INITIAL=False, FOUND_ENDPOINT=False",
}

# -----------------------------
# 1) Plot difference as scatter
# -----------------------------
plt.figure(figsize=(8, 6))

for (fi, fe), color in pair_colors.items():
    subset = df[
        (df["FOUND_INITIAL"] == fi) &
        (df["FOUND_ENDPOINT"] == fe)
    ]

    plt.scatter(
        subset["dx_calc"],
        subset["dy_calc"],
        c=color,
        label=pair_labels[(fi, fe)],
        alpha=0.7
    )

plt.axvline(
    ref_dx,
    color="black",
    linestyle="--",
    label=f"ref dx = {ref_dx:.1f}"
)

plt.axhline(
    ref_dy,
    color="purple",
    linestyle="--",
    label=f"ref dy = {ref_dy:.1f}"
)

plt.xlabel("x difference xe - x0 [pix]")
plt.ylabel("y difference ye - y0 [pix]")
plt.title("Endpoint - Base Difference")
plt.legend()
plt.grid(True)

# -----------------------------
# 2) Plot endpoint vs time
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.suptitle("Endpoint vs Time")

for val, color in endpoint_colors.items():
    subset = df[df["FOUND_ENDPOINT"] == val]

    ax1.scatter(
        subset["t_system[s]"],
        subset["xe[m]"],
        c=color,
        label=endpoint_labels[val],
        alpha=0.7,
        s=12
    )

    ax2.scatter(
        subset["t_system[s]"],
        subset["ye[m]"],
        c=color,
        label=endpoint_labels[val],
        alpha=0.7,
        s=12
    )

ax1.plot(
    df["t_system[s]"],
    df["ref_xe"],
    color="black",
    linestyle="--",
    label="ref xe = x0 + ref dx"
)

ax2.plot(
    df["t_system[s]"],
    df["ref_ye"],
    color="purple",
    linestyle="--",
    label="ref ye = y0 + ref dy"
)

ax1.set_ylabel("xe [pix]")
ax2.set_ylabel("ye [pix]")
ax2.set_xlabel("time [s]")

ax1.grid(True)
ax2.grid(True)

ax1.legend()
ax2.legend()

# -----------------------------
# 3) Plot angle vs time
# -----------------------------
plt.figure(figsize=(10, 5))
plt.title("Angle vs Time")

for val, color in angle_colors.items():
    subset = df[df["ANGLE_VALID"] == val]

    plt.scatter(
        subset["t_system[s]"],
        np.rad2deg(subset["angle[rad]"]),
        c=color,
        label=angle_labels[val],
        alpha=0.7,
        s=12
    )

plt.axhline(
    ref_angle_deg,
    color="black",
    linestyle="--",
    label=f"ref angle = {ref_angle_deg:.1f} deg"
)

plt.xlabel("time [s]")
plt.ylabel("angle [deg]")
plt.legend()
plt.grid(True)

# -----------------------------
# 4) Plot pressure vs time
# -----------------------------
plt.figure(figsize=(10, 5))
plt.title("Pressure vs Time")

plt.plot(
    df["t_system[s]"].to_numpy(),
    df["pressure[mBar]"].to_numpy() / 10.0,
    label="pressure"
)

plt.xlabel("time [s]")
plt.ylabel("pressure [kPa]")
plt.legend()
plt.grid(True)

# -----------------------------
# 5) Plot pressure vs angle
# -----------------------------
plt.figure(figsize=(8, 6))
plt.title("Pressure vs Angle")

for val, color in angle_colors.items():
    subset = df[df["ANGLE_VALID"] == val]

    plt.scatter(
        subset["pressure[mBar]"] / 10.0,
        np.rad2deg(subset["angle[rad]"]),
        c=color,
        label=angle_labels[val],
        alpha=0.7,
        s=12
    )

plt.axhline(
    ref_angle_deg,
    color="black",
    linestyle="--",
    label=f"ref angle = {ref_angle_deg:.1f} deg"
)

plt.xlabel("pressure [kPa]")
plt.ylabel("angle [deg]")
plt.legend()
plt.grid(True)

plt.show()