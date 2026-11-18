import csv
import os
import pandas as pd
import matplotlib
# matplotlib.use("Agg")  # use this if you only want to save images
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from os import path
import pandas as pd

basepath = path.dirname(__file__)

# csv_path = path.join(
#     basepath,
#     "..",
#     "exp_setup_ws",
#     "data",
#     "tracker_online",
#     "robot2",
#     "controller",
#     "online",
#     "online_output20260518-181226.csv"
#     # "online_output20260316-130710.csv"
#     # "online_output20260316-114121.csv"
#     # "online_output20260312-143728_chirp.csv"
#     # "online_output20260311-125118.csv"
#     # "online_output20260420-120538.csv"
# )

csv_path = path.join(
    basepath,
    "online_output20260526-150423.csv"
)

df = pd.read_csv(csv_path)

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Show first rows (optional)
print(df.head())
print(df.columns)

# Clean whitespace inside FOUND columns too
df["FOUND_INITIAL"] = df["FOUND_INITIAL"].astype(str).str.strip()
df["FOUND_ENDPOINT"] = df["FOUND_ENDPOINT"].astype(str).str.strip()

# Create validity mask
df["ANGLE_VALID"] = (
    (df["FOUND_INITIAL"] == "True") &
    (df["FOUND_ENDPOINT"] == "True")
)

# Compute differences
df["dx_calc"] = df["xe[m]"] - df["x0[m]"]
df["dy_calc"] = df["ye[m]"] - df["y0[m]"]

# Remove rows where angle == 0 and pressure == 0
df = df[~((df["angle[rad]"] == 0))]

# Optional: reset index after filtering
df = df.reset_index(drop=True)

print(df.head())

# -----------------------------
# Color definitions
# -----------------------------
initial_colors = {
    "True": "green",
    "False": "red",
}
initial_labels = {
    "True": "AprilTag Detected",
    "False": "AprilTag Not Detected",
}

endpoint_colors = {
    "True": "green",
    "False": "red",
}
endpoint_labels = {
    "True": "AprilTag Detected",
    "False": "AprilTag Not Detected",
}

angle_colors = {
    True: "green",
    False: "red"
}

angle_labels = {
    True: "AprilTag Detected",
    False: "AprilTag Not Detected"
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

plt.xlabel("x difference (xe - x0) [pix]")
plt.ylabel("y difference (ye - y0) [pix]")
plt.title("Endpoint - Base Difference")
plt.legend()
plt.grid(True)

# -----------------------------
# 2) Plot base vs time
#    Color by FOUND_INITIAL only
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.suptitle('Base vs Time')

for val, color in initial_colors.items():
    subset = df[df["FOUND_INITIAL"] == val]

    ax1.scatter(
        subset["t_system[s]"],
        subset["x0[m]"],
        c=color,
        label=initial_labels[val],
        alpha=0.7,
        s=12
    )
    ax2.scatter(
        subset["t_system[s]"],
        subset["y0[m]"],
        c=color,
        label=initial_labels[val],
        alpha=0.7,
        s=12
    )

ax1.set_ylabel("x0 [pix]")
ax2.set_ylabel("y0 [pix]")
ax2.set_xlabel("time [s]")
ax1.grid(True)
ax2.grid(True)
ax1.legend()

# -----------------------------
# 3) Plot endpoint vs time
#    Color by FOUND_ENDPOINT only
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.suptitle('Endpoint vs Time')

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

ax1.set_ylabel("xe [pix]")
ax2.set_ylabel("ye [pix]")
ax2.set_xlabel("time [s]")
ax1.grid(True)
ax2.grid(True)
ax1.legend()

# -----------------------------
# 4) Plot endpoint-base in XY
#    Color by both flags
# -----------------------------
plt.figure(figsize=(8, 6))
plt.title("Endpoint - Base (XY)")

# Valid points: both detected
valid_subset = df[
    (df["FOUND_INITIAL"] == "True") &
    (df["FOUND_ENDPOINT"] == "True")
]

# Invalid points: one or both missing
invalid_subset = df[
    (df["FOUND_INITIAL"] != "True") |
    (df["FOUND_ENDPOINT"] != "True")
]

# Plot valid points in green
if not valid_subset.empty:
    plt.scatter(
        valid_subset["xe[m]"] - valid_subset["x0[m]"],
        valid_subset["ye[m]"] - valid_subset["y0[m]"],
        c="green",
        label="AprilTag Detected",
        alpha=0.7
    )

# Plot invalid points in red
if not invalid_subset.empty:
    plt.scatter(
        invalid_subset["xe[m]"] - invalid_subset["x0[m]"],
        invalid_subset["ye[m]"] - invalid_subset["y0[m]"],
        c="red",
        label="AprilTag Not Detected",
        alpha=0.7
    )

plt.xlabel("x difference (xe - x0) [pix]")
plt.ylabel("y difference (ye - y0) [pix]")
plt.legend()
plt.grid(True)

# -----------------------------
# 5) Plot angle vs time
#    Usually endpoint-related, so color by FOUND_ENDPOINT
# -----------------------------
plt.figure(figsize=(10,5))
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

plt.xlabel("time [s]")
plt.ylabel("angle [deg]")
plt.legend()
plt.grid(True)

# -----------------------------
# 6) Plot pressure vs time
#    Pressure is independent, so use a normal line
# -----------------------------
plt.figure(figsize=(10, 5))
plt.title("Pressure vs Time")
plt.plot(
    df["t_system[s]"].to_numpy(),
    df["pressure[mBar]"].to_numpy()/10,
    label="pressure"
)
plt.xlabel("time [s]")
plt.ylabel("pressure [kPa]")
plt.legend()
plt.grid(True)

# -----------------------------
# 7) Plot angle vs pressure
#    Angle is endpoint-related, so color by FOUND_ENDPOINT
# -----------------------------
plt.figure(figsize=(8, 6))
plt.title("Pressure vs Angle")

for val, color in angle_colors.items():
    subset = df[df["ANGLE_VALID"] == val]

    plt.scatter(
        subset["pressure[mBar]"]/10,
        np.rad2deg(subset["angle[rad]"]),
        c=color,
        label=angle_labels[val],
        alpha=0.7,
        s=12
    )

plt.xlabel("pressure [kPa]")
plt.ylabel("angle [deg]")
plt.legend()
plt.grid(True)



# plot endpoint velocity
plt.figure()
plt.subplot(1, 2, 1)
plt.plot(df["t_system[s]"],df["dx[m]"])
plt.subplot(1, 2, 2)
plt.plot(df["t_system[s]"],df["dy[m]"])



# -----------------------------
# 8) Angle vs pressure
#    Discard first 15s, then color every 80s segment differently
# -----------------------------
t = 0
df_cut = df[df["t_system[s]"] >= t].copy()

# Time relative to after discarded 15s
df_cut["t_after_cut[s]"] = df_cut["t_system[s]"] - t

# Segment index: 0 for 15-95s, 1 for 95-175s, etc.
df_cut["segment_80s"] = (df_cut["t_after_cut[s]"] // 80).astype(int)

plt.figure(figsize=(8, 6))
plt.title(f"Pressure vs Angle, colored every 80s after first {t}s")

for seg in sorted(df_cut["segment_80s"].unique()):
    subset = df_cut[df_cut["segment_80s"] == seg]

    plt.scatter(
        subset["pressure[mBar]"] / 10,
        np.rad2deg(subset["angle[rad]"]),
        s=12,
        alpha=0.7,
        label=f"{t + seg*80:.0f}s - {t + (seg+1)*80:.0f}s"
    )

plt.xlabel("pressure [kPa]")
plt.ylabel("angle [deg]")
plt.legend()
plt.grid(True)



plt.show()

# x = df["t_system[s]"].to_numpy()
# y = df["angle[rad]"].to_numpy()


# mask1 = (x > 5) & (x < 48)
# mask2 = (x > 112.5) & (x < 127.5)

# x_masked = x[mask1 | mask2]
# y_masked = y[mask1 | mask2]

# # Create DataFrame
# df = pd.DataFrame({
#     'x': x_masked,
#     'y': y_masked
# })

# # Save to CSV
# df.to_csv('new_trajectory.csv', index=False)