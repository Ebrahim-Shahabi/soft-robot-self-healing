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

ROBOT_DIR = TRACKER_ROOT / "robot2_damage0"

NO_DAMAGE_VIDEO_DIR = ROBOT_DIR / "video"
DAMAGE_VIDEO_DIR = ROBOT_DIR / "video" / "damage1"

NO_DAMAGE_ONLINE_DIR = ROBOT_DIR / "online"
DAMAGE_ONLINE_DIR = ROBOT_DIR / "online"

# None = compare all matching clean motions
# Or set to e.g. "rampUpwards", "rampDownwards", "Trapezoid"
RUN_NAME_FILTER = "Trapezoid"

SAVE_FIGURES = True
OUTPUT_DIR = SCRIPT_DIR / "damage_comparison_plots" / RUN_NAME_FILTER


# -----------------------------
# Helpers
# -----------------------------
def extract_stamp(filename: str) -> str:
    m = re.search(r"output_?(\d{8}-\d{6})", filename)
    if not m:
        raise ValueError(f"Could not extract timestamp from {filename}")
    return m.group(1)


def extract_motion_name(filename: str) -> str:
    """
    Examples:
    video_output20260428-132644_rampDownwards.avi -> rampDownwards
    video_output_20260428-132216_rampUpwards.avi -> rampUpwards
    video_output20260428-132925_Trapezoid.avi -> Trapezoid
    """
    stem = Path(filename).stem
    m = re.search(r"output_?\d{8}-\d{6}_?(.*)", stem)

    if not m or not m.group(1):
        return "unknown"

    return m.group(1)


def get_video_runs(video_dir: Path):
    runs = {}

    if not video_dir.exists():
        print(f"Warning: video directory does not exist: {video_dir}")
        return runs

    for video_path in sorted(video_dir.glob("*.avi")):
        name_lower = video_path.name.lower()

        # Ignore files like:
        # video_output20260428-133400_135deg_Trapezoid.avi
        if "deg" in name_lower:
            print(f"Skipping because it contains 'deg': {video_path.name}")
            continue

        motion = extract_motion_name(video_path.name)

        if RUN_NAME_FILTER is not None:
            if RUN_NAME_FILTER.lower() not in motion.lower():
                continue

        # If multiple files have same motion, keep latest alphabetically.
        # Timestamp format makes this usually the newest run.
        runs[motion] = video_path

    return runs


def find_csv_for_video(video_path: Path, online_dir: Path) -> Path:
    stamp = extract_stamp(video_path.name)

    candidates = sorted(online_dir.glob(f"*{stamp}*.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"No CSV containing timestamp {stamp} found in {online_dir}"
        )

    return candidates[0]


def load_df(label: str, video_path: Path, online_dir: Path) -> pd.DataFrame:
    csv_path = find_csv_for_video(video_path, online_dir)

    print(f"\n{label}")
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

    # Align each run to start at t = 0
    df["t_rel[s]"] = df["t_system[s]"] - df["t_system[s]"].iloc[0]

    df["condition"] = label
    df["video_file"] = video_path.name
    df["csv_file"] = csv_path.name

    return df


def save_or_show(fig, name):
    if SAVE_FIGURES:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


def plot_one_motion(motion: str, dfs: dict):
    # -----------------------------
    # 1) Endpoint - base difference
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"{motion} - Endpoint Base Difference")

    for label, df in dfs.items():
        valid = df[df["ANGLE_VALID"]]
        invalid = df[~df["ANGLE_VALID"]]

        ax.scatter(
            valid["dx_calc"],
            valid["dy_calc"],
            s=14,
            alpha=0.7,
            label=f"{label} valid",
        )
        ax.scatter(
            invalid["dx_calc"],
            invalid["dy_calc"],
            s=14,
            alpha=0.25,
            label=f"{label} invalid",
        )

    ax.set_xlabel("xe - x0 [m]")
    ax.set_ylabel("ye - y0 [m]")
    ax.grid(True)
    ax.legend()
    save_or_show(fig, f"{motion}_endpoint_base_difference")

    # -----------------------------
    # 2) Base vs time
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig.suptitle(f"{motion} - Base vs Time")

    for label, df in dfs.items():
        ax1.scatter(df["t_rel[s]"], df["x0[m]"], s=12, alpha=0.7, label=label)
        ax2.scatter(df["t_rel[s]"], df["y0[m]"], s=12, alpha=0.7, label=label)

    ax1.set_ylabel("x0 [m]")
    ax2.set_ylabel("y0 [m]")
    ax2.set_xlabel("time [s]")
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    save_or_show(fig, f"{motion}_base_vs_time")

    # -----------------------------
    # 3) Endpoint vs time
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
    fig.suptitle(f"{motion} - Endpoint vs Time")

    for label, df in dfs.items():
        ax1.scatter(df["t_rel[s]"], df["xe[m]"], s=12, alpha=0.7, label=label)
        ax2.scatter(df["t_rel[s]"], df["ye[m]"], s=12, alpha=0.7, label=label)

    ax1.set_ylabel("xe [m]")
    ax2.set_ylabel("ye [m]")
    ax2.set_xlabel("time [s]")
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    save_or_show(fig, f"{motion}_endpoint_vs_time")

    # -----------------------------
    # 4) Angle vs time
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"{motion} - Angle vs Time")

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
            alpha=0.25,
            label=f"{label} invalid",
        )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("angle [deg]")
    ax.grid(True)
    ax.legend()
    save_or_show(fig, f"{motion}_angle_vs_time")

    # -----------------------------
    # 5) Pressure vs time
    # -----------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"{motion} - Pressure vs Time")

    for label, df in dfs.items():
        ax.plot(
            df["t_rel[s]"],
            df["pressure[mBar]"] / 10,
            label=label,
        )

    ax.set_xlabel("time [s]")
    ax.set_ylabel("pressure [kPa]")
    ax.grid(True)
    ax.legend()
    save_or_show(fig, f"{motion}_pressure_vs_time")

    # -----------------------------
    # 6) Pressure vs angle
    # -----------------------------
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.suptitle(f"{motion} - Pressure vs Angle")

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
            alpha=0.25,
            label=f"{label} invalid",
        )

    ax.set_xlabel("pressure [kPa]")
    ax.set_ylabel("angle [deg]")
    ax.grid(True)
    ax.legend()
    save_or_show(fig, f"{motion}_pressure_vs_angle")

    # -----------------------------
    # 7) dx and dy vs time
    # -----------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    fig.suptitle(f"{motion} - dx/dy vs Time")

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

    save_or_show(fig, f"{motion}_dx_dy_vs_time")


# -----------------------------
# Main
# -----------------------------
no_damage_runs = get_video_runs(NO_DAMAGE_VIDEO_DIR)
damage_runs = get_video_runs(DAMAGE_VIDEO_DIR)

common_motions = sorted(set(no_damage_runs.keys()) & set(damage_runs.keys()))

missing_in_damage = sorted(set(no_damage_runs.keys()) - set(damage_runs.keys()))
missing_in_no_damage = sorted(set(damage_runs.keys()) - set(no_damage_runs.keys()))

print("\nNo-damage motions found:")
for m in sorted(no_damage_runs.keys()):
    print(f"  {m}")

print("\nDamage motions found:")
for m in sorted(damage_runs.keys()):
    print(f"  {m}")

print("\nMatching motions:")
for m in common_motions:
    print(f"  {m}")

if missing_in_damage:
    print("\nSkipped because missing in damage1:")
    for m in missing_in_damage:
        print(f"  {m}")

if missing_in_no_damage:
    print("\nSkipped because missing in no_damage:")
    for m in missing_in_no_damage:
        print(f"  {m}")

if not common_motions:
    raise RuntimeError("No matching clean motions found between damage and no-damage videos.")

for motion in common_motions:
    print(f"\n--- Comparing motion: {motion} ---")

    dfs = {
        "no_damage": load_df(
            "no_damage",
            no_damage_runs[motion],
            NO_DAMAGE_ONLINE_DIR,
        ),
        "damage1": load_df(
            "damage1",
            damage_runs[motion],
            DAMAGE_ONLINE_DIR,
        ),
    }

    plot_one_motion(motion, dfs)

plt.show()