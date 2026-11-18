import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

CSV_FILES = {
    "subspace": "tracking_output20260623-203000_subspace_31_4.csv",
    "Scratch": "tracking_output20260623-204507_scratch.csv",
    "Fine Tuning": "tracking_output20260623-204253_full_finetuning.csv",
    "LoRA rank 1": "tracking_output20260623-202644_lora_31.4.csv",
}



# old 
# subspace": "tracking_output20260605-154119.csv
# LoRA rank 1": "tracking_output20260604-175149.csv
# Fine Tuning": "tracking_output20260604-163640.csv"
# Scratch": "tracking_output20260604-162256.csv


# scratch : tracking_output20260604-162256.csv
# one shot svd with w0 weights : tracking_output20260605-153310.csv, tracking_output20260605-160151.csv", # trained on weights from robot2 undamaged
# metaSVD file (another one): tracking_output20260604-150315
# oneshot with W2 weights and gains same as metaSVD : tracking_output20260605-154706.csv

if len(sys.argv) > 1:
    CSV_FILES = {f"Run {i+1}": file for i, file in enumerate(sys.argv[1:])}

data = {}

for name, file in CSV_FILES.items():
    if not os.path.exists(file):
        print(f"File not found: {file}")
        continue

    df = pd.read_csv(file, skipinitialspace=True)
    df.columns = df.columns.str.strip()

    data[name] = df
    print(f"Loaded {len(df)} rows from {file} as {name}")

if not data:
    raise FileNotFoundError("No CSV files were loaded.")


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Saved {filename}")


def first_df():
    return next(iter(data.values()))


def relative_time(df, time_col="t_system[s]"):
    """Rebuild a time axis that starts at 0 and accumulates each run's own
    per-sample dt, instead of using the raw (possibly offset) system
    timestamp. This is what keeps different runs aligned at t=0 instead of
    showing up shifted relative to each other."""
    t_raw = df[time_col].to_numpy()
    dt = np.diff(t_raw)
    t = np.concatenate(([0.0], np.cumsum(dt)))
    return t


T_MAX = 200.0  # don't plot anything past this relative time
 
# Build the synced time axis per run, then truncate both the time array and
# the dataframe itself to t < T_MAX, once, up front. Everything below just
# uses data/time_data as-is, no per-plot masking needed.
time_data = {}
for name, df in data.items():
    t = relative_time(df)
    keep = t > 0.1
    time_data[name] = t[keep]
    data[name] = df[keep].reset_index(drop=True)
 
ref_name = next(iter(data))
t_ref = time_data[ref_name]
 
 
# 1. Control output: u, u_ff, u_fb
plt.figure(figsize=(14, 6))
 
for name, df in data.items():
    if (name == 'subspace') or (name == 'LoRA rank 1'):
        t = time_data[name]
        # plt.plot(t, df["u[mBar]"]/10, label=f"{name} u total", linewidth=1.8)
        plt.plot(t, df["u_ff[mBar]"]/10, label=f"{name} u_ff", )
        plt.plot(t, df["u_fb[mBar]"]/10, label=f"{name} u_fb", linestyle="--")
    
plt.title("Control Output Components Comparison")
plt.xlabel("Time [s]")
plt.ylabel("Pressure [kPa]")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
#save_plot("01_control_output_comparison.png")
 
 
# 2. Angle tracking
plt.figure(figsize=(14, 6))
 
for name, df in data.items():
    t = time_data[name]
    if name == 'subspace':
        plt.plot(
            t,
            np.rad2deg(df["angle_current[rad]"]),
            label=f"{name}",
            linewidth=1.8,
        )
    elif name == 'LoRA rank 1':
        plt.plot(
            t,
            np.rad2deg(df["angle_current[rad]"]),
            label=f"{name}",
            linewidth=1.8,
        )
    else:
        mask = df["angle_current[rad]"] > 0
        plt.plot(
            t[mask],
            np.rad2deg(df["angle_current[rad]"][mask]),
            label=f"{name}",
            linewidth=1.8,
        )
plt.axhline(y=31.7, color="gray", linestyle=":", linewidth=1.4, label="Initial angle (31.7°)")

 
ref = first_df()
plt.plot(
    t_ref,
    np.rad2deg(ref["angle_ref[rad]"]),
    "k--",
    linewidth=2,
    label="Reference",
)
 
plt.title("Angle Tracking Comparison")
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
#save_plot("02_angle_tracking_comparison.png")
 
 
# 3. Angle error
plt.figure(figsize=(14, 6))
 
for name, df in data.items():
    t = time_data[name]
    plt.plot(
        t,
        np.rad2deg(df["angle_error[rad]"]),
        label=name,
        linewidth=1.8,
    )
 
plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
plt.title("Angle Error Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Error (deg)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
#save_plot("03_angle_error_comparison.png")
 
 
# 4. X tracking
plt.figure(figsize=(14, 6))
 
for name, df in data.items():
    t = time_data[name]
    plt.plot(
        t,
        df["x_current[m]"],
        label=f"{name}",
        linewidth=1.8,
    )
 
ref = first_df()
plt.plot(
    t_ref,
    ref["x_ref[m]"],
    "k--",
    linewidth=2,
    label="Reference",
)
 
plt.title("X Position Tracking Comparison")
plt.xlabel("Time (s)")
plt.ylabel("X Position (m)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
#save_plot("04_x_tracking_comparison.png")
 
 
# 5. X error
plt.figure(figsize=(14, 6))
 
for name, df in data.items():
    t = time_data[name]
    plt.plot(t, df["dx_error[m]"], label=name, linewidth=1.8)
 
plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
plt.title("X Position Error Comparison")
plt.xlabel("Time (s)")
plt.ylabel("dx Error (m)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
#save_plot("05_x_error_comparison.png")
 
 
# 6. Y tracking
plt.figure(figsize=(14, 6))
 
for name, df in data.items():
    t = time_data[name]
    plt.plot(
        t,
        df["y_current[m]"],
        label=f"{name}",
        linewidth=1.8,
    )
 
ref = first_df()
plt.plot(
    t_ref,
    ref["y_ref[m]"],
    "k--",
    linewidth=2,
    label="Reference",
)
 
plt.title("Y Position Tracking Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Y Position (m)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
#save_plot("06_y_tracking_comparison.png")
 
 
# 7. Y error
plt.figure(figsize=(14, 6))
 
for name, df in data.items():
    t = time_data[name]
    plt.plot(t, df["dy_error[m]"], label=name, linewidth=1.8)
 
plt.axhline(0, color="k", linestyle="--", linewidth=0.8)
plt.title("Y Position Error Comparison")
plt.xlabel("Time (s)")
plt.ylabel("dy Error (m)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
#save_plot("07_y_error_comparison.png")
 
 
# 8. Position error magnitude
plt.figure(figsize=(14, 6))
 
for name, df in data.items():
    t = time_data[name]
    pos_err_mag = np.sqrt(df["dx_error[m]"] ** 2 + df["dy_error[m]"] ** 2)
    plt.plot(t, pos_err_mag, label=name, linewidth=1.8)
 
plt.title("Position Error Magnitude Comparison")
plt.xlabel("Time (s)")
plt.ylabel("Position Error Magnitude (m)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
#save_plot("08_position_error_magnitude_comparison.png")
 
 
# 9. 2D trajectory
plt.figure(figsize=(8, 8))
 
for name, df in data.items():
    plt.plot(
        df["x_current[m]"],
        df["y_current[m]"],
        label=f"{name}",
        linewidth=2,
    )
 
ref = first_df()
plt.plot(
    ref["x_ref[m]"],
    ref["y_ref[m]"],
    "k--",
    linewidth=3,
    label="Reference",
)
 
plt.title("2D Trajectory Comparison")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
#save_plot("09_trajectory_comparison.png")
 
# 3b. Absolute angle tracking error
plt.figure(figsize=(14, 6))
 
for name, df in data.items():
    t = time_data[name]
 
    abs_angle_err_deg = np.abs(
        np.rad2deg(df["angle_ref[rad]"] - df["angle_current[rad]"])
    )
 
    plt.plot(
        t,
        abs_angle_err_deg,
        label=name,
        linewidth=1.8,
    )
 
plt.title("Absolute Angle Tracking Error")
plt.xlabel("Time (s)")
plt.ylabel("|Error| (deg)")
plt.grid(True, alpha=0.3)
plt.legend(fontsize=8)
 
#save_plot("03b_absolute_angle_tracking_error.png")
 
print(data.keys())

 

fig, axes = plt.subplots(2, 1, figsize=(12, 9))
fig.subplots_adjust(hspace=0.35)

colors = {"subspace": "#185FA5", "LoRA rank 1": "#D85A30"}
linestyles = {"subspace": "-", "LoRA rank 1": "--"}

# Panel 1: Angle tracking
ax1 = axes[0]
ax1.axhline(y=31.7, color="gray", linestyle=":", linewidth=1.4, label="Initial angle (31.7°)")
ax1.plot(t_ref, np.rad2deg(first_df()["angle_ref[rad]"]), "k--", linewidth=2, label="Reference")

for name, df in data.items():
    if name not in colors:
        continue
    t = time_data[name]
    ax1.plot(t, np.rad2deg(df["angle_current[rad]"]), color=colors[name], linestyle=linestyles[name], linewidth=1.8, label=name)

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Angle [deg]")
ax1.set_title("Angular reference tracking")
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)

# Panel 2: Control inputs
ax2 = axes[1]
for name, df in data.items():
    if name not in colors:
        continue
    t = time_data[name]
    ax2.plot(t, df["u_ff[mBar]"]/10, color=colors[name], linestyle=linestyles[name], linewidth=1.8, label=f"{name} — feedforward")
    ax2.plot(t, df["u_fb[mBar]"]/10, color=colors[name], linestyle=":", linewidth=1.4, label=f"{name} — feedback")

ax2.set_xlabel("Time [s]")
ax2.set_ylabel("Pressure [kPa]")
ax2.set_title("Control inputs: feedforward and feedback")
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()