from pathlib import Path
import re
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent

TRACKER_ROOT = (
    SCRIPT_DIR.parents[1] / "exp_setup_ws" / "data" / "tracker_online"
)

ROBOTS = {
    "robot1_damage1": TRACKER_ROOT / "robot1_damage1",
    "robot2_damage0": TRACKER_ROOT / "robot2_damage0",
}

# Choose which motion to compare.
# Examples: "rampUpwards", "rampDownwards", "Trapezoid"
RUN_NAME_FILTER = "rampUpwards"   # set to e.g. "rampUpwards", or leave None for latest video per robot

SAVE_FIGURES = True
OUTPUT_DIR = SCRIPT_DIR / "comparison_plots" / RUN_NAME_FILTER

def save_fig(fig, name):
    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        filepath = OUTPUT_DIR / f"{name}.png"
        fig.savefig(filepath, dpi=200, bbox_inches="tight")
        print(f"Saved: {filepath}")


# -----------------------------
# Helpers
# -----------------------------
def extract_stamp(filename: str) -> str:
    """
    Extracts 20260428-132216 from names like:
    video_output20260428-132644_rampDownwards.avi
    video_output_20260428-132216_rampUpwards.avi
    """
    m = re.search(r"output_?(\d{8}-\d{6})", filename)
    if not m:
        raise ValueError(f"Could not extract timestamp from {filename}")
    return m.group(1)


def find_video(robot_dir: Path) -> Path:
    videos = sorted((robot_dir / "video").glob("*.avi"))

    # Ignore files with "deg" in the name, e.g. 135deg_Trapezoid
    videos = [v for v in videos if "deg" not in v.name.lower()]

    if RUN_NAME_FILTER is not None:
        videos = [
            v for v in videos
            if RUN_NAME_FILTER.lower() in v.name.lower()
        ]

    if not videos:
        raise FileNotFoundError(
            f"No matching non-deg videos found in {robot_dir / 'video'} "
            f"for RUN_NAME_FILTER={RUN_NAME_FILTER}"
        )

    return videos[-1]

def find_csv(robot_dir: Path, stamp: str) -> Path:
    online_dir = robot_dir / "online"

    candidates = list(online_dir.glob(f"*{stamp}*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV containing {stamp} found in {online_dir}")

    return candidates[0]


def load_robot_df(label: str, robot_dir: Path) -> pd.DataFrame:
    video_path = find_video(robot_dir)
    stamp = extract_stamp(video_path.name)
    csv_path = find_csv(robot_dir, stamp)

    print(f"{label}")
    print(f"  video: {video_path}")
    print(f"  csv:   {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    df["FOUND_INITIAL"] = df["FOUND_INITIAL"].astype(str).str.strip()
    df["FOUND_ENDPOINT"] = df["FOUND_ENDPOINT"].astype(str).str.strip()

    df["ANGLE_VALID"] = (
        (df["FOUND_INITIAL"] == "True") &
        (df["FOUND_ENDPOINT"] == "True")
    )

    df["dx_calc"] = df["xe[m]"] - df["x0[m]"]
    df["dy_calc"] = df["ye[m]"] - df["y0[m]"]

    # Start both robots at t = 0 for easier comparison
    df["t_rel[s]"] = df["t_system[s]"] - df["t_system[s]"].iloc[0]

    df["robot"] = label
    df["video_file"] = video_path.name
    df["csv_file"] = csv_path.name

    return df


def scatter_by_robot(ax, dfs, x_col, y_col, title, xlabel, ylabel, valid_col=None):
    markers = {
        "robot1_damage1": "o",
        "robot2_damage0": "x",
    }

    for label, df in dfs.items():
        if valid_col is None:
            ax.scatter(
                df[x_col],
                df[y_col],
                s=14,
                alpha=0.7,
                marker=markers[label],
                label=label,
            )
        else:
            valid = df[df[valid_col]]
            invalid = df[~df[valid_col]]

            ax.scatter(
                valid[x_col],
                valid[y_col],
                s=14,
                alpha=0.7,
                marker=markers[label],
                label=f"{label} valid",
            )
            ax.scatter(
                invalid[x_col],
                invalid[y_col],
                s=14,
                alpha=0.35,
                marker=markers[label],
                label=f"{label} invalid",
            )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()


# -----------------------------
# Load data
# -----------------------------
dfs = {
    label: load_robot_df(label, robot_dir)
    for label, robot_dir in ROBOTS.items()
}

# -----------------------------
# 1) Endpoint - base difference
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
scatter_by_robot(
    ax,
    dfs,
    "dx_calc",
    "dy_calc",
    "Endpoint - Base Difference",
    "x difference xe - x0 [m]",
    "y difference ye - y0 [m]",
    valid_col="ANGLE_VALID",
)
save_fig(fig, "endpoint_base_difference")
# -----------------------------
# 2) Base vs time
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.suptitle("Base vs Time")

for label, df in dfs.items():
    ax1.scatter(df["t_rel[s]"], df["x0[m]"], s=12, alpha=0.7, label=label)
    ax2.scatter(df["t_rel[s]"], df["y0[m]"], s=12, alpha=0.7, label=label)

ax1.set_ylabel("x0 [m]")
ax2.set_ylabel("y0 [m]")
ax2.set_xlabel("time [s]")
ax1.grid(True)
ax2.grid(True)
ax1.legend()
save_fig(fig, "base_vs_time")
# -----------------------------
# 3) Endpoint vs time
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
fig.suptitle("Endpoint vs Time")

for label, df in dfs.items():
    ax1.scatter(df["t_rel[s]"], df["xe[m]"], s=12, alpha=0.7, label=label)
    ax2.scatter(df["t_rel[s]"], df["ye[m]"], s=12, alpha=0.7, label=label)

ax1.set_ylabel("xe [m]")
ax2.set_ylabel("ye [m]")
ax2.set_xlabel("time [s]")
ax1.grid(True)
ax2.grid(True)
ax1.legend()
save_fig(fig, "endpoint_vs_time")
# -----------------------------
# 4) Endpoint - base XY
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))
scatter_by_robot(
    ax,
    dfs,
    "dx_calc",
    "dy_calc",
    "Endpoint - Base XY",
    "xe - x0 [m]",
    "ye - y0 [m]",
    valid_col="ANGLE_VALID",
)
save_fig(fig, "endpoint_base_xy")
# -----------------------------
# 5) Angle vs time
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))

for label, df in dfs.items():
    valid = df[df["ANGLE_VALID"]]
    invalid = df[~df["ANGLE_VALID"]]

    ax.scatter(
        valid["t_rel[s]"],
        np.rad2deg(valid["angle[rad]"]),
        s=12,
        alpha=0.7,
        label=f"{label} valid",
    )
    ax.scatter(
        invalid["t_rel[s]"],
        np.rad2deg(invalid["angle[rad]"]),
        s=12,
        alpha=0.3,
        label=f"{label} invalid",
    )

ax.set_title("Angle vs Time")
ax.set_xlabel("time [s]")
ax.set_ylabel("angle [deg]")
ax.grid(True)
ax.legend()
save_fig(fig, "angle_vs_time")
# -----------------------------
# 6) Pressure vs time
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))

for label, df in dfs.items():
    ax.plot(
        df["t_rel[s]"],
        df["pressure[mBar]"] / 10,
        label=label,
    )

ax.set_title("Pressure vs Time")
ax.set_xlabel("time [s]")
ax.set_ylabel("pressure [kPa]")
ax.grid(True)
ax.legend()
save_fig(fig, "pressure_vs_time")
# -----------------------------
# 7) Pressure vs angle
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 6))

for label, df in dfs.items():
    valid = df[df["ANGLE_VALID"]]
    invalid = df[~df["ANGLE_VALID"]]

    ax.scatter(
        valid["pressure[mBar]"] / 10,
        np.rad2deg(valid["angle[rad]"]),
        s=12,
        alpha=0.7,
        label=f"{label} valid",
    )
    ax.scatter(
        invalid["pressure[mBar]"] / 10,
        np.rad2deg(invalid["angle[rad]"]),
        s=12,
        alpha=0.3,
        label=f"{label} invalid",
    )

ax.set_title("Pressure vs Angle")
ax.set_xlabel("pressure [kPa]")
ax.set_ylabel("angle [deg]")
ax.grid(True)
ax.legend()
save_fig(fig, "pressure_vs_angle")
# -----------------------------
# 8) dx and dy vs time
# -----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
fig.suptitle("Endpoint Velocity / Displacement Signals")

for label, df in dfs.items():
    ax1.plot(df["t_rel[s]"], df["dx[m]"], label=label)
    ax2.plot(df["t_rel[s]"], df["dy[m]"], label=label)

ax1.set_title("dx vs time")
ax1.set_xlabel("time [s]")
ax1.set_ylabel("dx [m]")
ax1.grid(True)
ax1.legend()

ax2.set_title("dy vs time")
ax2.set_xlabel("time [s]")
ax2.set_ylabel("dy [m]")
ax2.grid(True)
ax2.legend()
save_fig(fig, "dx_dy_vs_time")
plt.show()