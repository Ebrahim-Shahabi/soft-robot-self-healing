from pathlib import Path
import re

import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import plots as plts

# -----------------------------
# File-name helpers
# -----------------------------
'''
Extracts the timestamp from a video filename.

The timestamp is expected in the format YYYYMMDD-HHMMSS and is located
after "output" or "output_".

@param[in] filename : name of the file.

@return stamp : extracted timestamp string.

@throws ValueError if no timestamp is found.
'''
def extract_stamp(filename: str) -> str:
    m = re.search(r"output_?(\d{8}-\d{6})", filename)
    if not m:
        raise ValueError(f"Could not extract timestamp from {filename}")
    return m.group(1)

'''
Extracts the motion name from a video filename.

Examples:
    video_output20260428-132644_rampDownwards.avi -> rampDownwards
    video_output20260428-133400_135deg_Trapezoid.avi -> 135deg_Trapezoid

@param[in] filename : name of the file.

@return motion : extracted motion name or "unknown".
'''
def extract_motion_name(filename: str) -> str:

    stem = Path(filename).stem
    m = re.search(r"output_?\d{8}-\d{6}_?(.*)", stem)

    if not m or not m.group(1):
        return "unknown"

    return m.group(1)

'''
Removes angle prefixes from motion names.

Examples:
    135deg_Trapezoid -> Trapezoid
    45deg_Trapezoid  -> Trapezoid

@param[in] motion : raw motion string.

@return cleaned_motion : motion without angle prefix.
'''
def strip_angle_from_motion(motion: str) -> str:

    return re.sub(r"^(45|135)deg_?", "", motion, flags=re.IGNORECASE)


# -----------------------------
# Video filtering helpers
# -----------------------------
'''
Determines whether a video matches a given angle filter.

@param[in] video_path : path to video file.
@param[in] angle      : filter mode:
                        None -> all videos
                        0    -> only non-deg videos
                        45   -> only 45deg videos
                        135  -> only 135deg videos

@return bool : True if match, False otherwise.
'''
def video_matches_angle(video_path: Path, angle: int | None = None) -> bool:
   
    name = video_path.name.lower()

    if angle is None:
        return True

    if angle == 0:
        return "deg" not in name

    if angle in (45, 135):
        return f"{angle}deg" in name

    raise ValueError("angle must be one of: None, 0, 45, 135")

'''
Determines whether a video matches a given motion name filter.

@param[in] video_path      : path to video file.
@param[in] run_name_filter : substring to match in motion name.
                            None -> accept all.

@return bool : True if match, False otherwise.
'''
def video_matches_run_name(
    video_path: Path,
    run_name_filter: str | None = None,
) -> bool:
   
    if run_name_filter is None:
        return True

    motion = extract_motion_name(video_path.name)
    clean_motion = strip_angle_from_motion(motion)

    return run_name_filter.lower() in clean_motion.lower()

'''
Collects all videos matching filters.

@param[in] video_dir       : directory with videos.
@param[in] angle           : angle filter.
@param[in] run_name_filter : motion filter.

@return videos : sorted list of matching video paths.
'''
def get_matching_videos(
    video_dir: Path,
    angle: int | None = 0,
    run_name_filter: str | None = None,
) -> list[Path]:

    if not video_dir.exists():
        print(f"Warning: video directory does not exist: {video_dir}")
        return []

    videos = []

    for video_path in sorted(video_dir.glob("*.avi")):
        if not video_matches_angle(video_path, angle):
            continue

        if not video_matches_run_name(video_path, run_name_filter):
            continue

        videos.append(video_path)

    return videos

'''
Returns latest video per motion.

@param[in] video_dir       : directory with videos.
@param[in] angle           : angle filter.
@param[in] run_name_filter : motion filter.

@return runs : dict mapping motion -> latest video path.
'''
def get_video_runs(
    video_dir: Path,
    angle: int | None = 0,
    run_name_filter: str | None = None,
) -> dict[str, Path]:

    runs = {}

    videos = get_matching_videos(video_dir, angle, run_name_filter)

    for video_path in videos:
        motion = extract_motion_name(video_path.name)
        clean_motion = strip_angle_from_motion(motion)
        if "trapezoid" in clean_motion.lower():
            clean_motion = "trapezoid"
        runs[clean_motion] = video_path

    return runs

'''
Returns latest matching video.

@param[in] video_dir       : directory with videos.
@param[in] angle           : angle filter.
@param[in] run_name_filter : motion filter.

@return video_path : latest matching video.

@throws FileNotFoundError if none found.
'''
def find_latest_video(
    video_dir: Path,
    angle: int | None = 0,
    run_name_filter: str | None = None,
) -> Path:

    videos = get_matching_videos(video_dir, angle, run_name_filter)

    if not videos:
        raise FileNotFoundError(
            f"No videos found in {video_dir} "
            f"for angle={angle}, run_name_filter={run_name_filter}"
        )

    return videos[-1]

'''
Returns latest video for a specific motion.

@param[in] video_dir   : directory with videos.
@param[in] motion_name : motion name to search for.
@param[in] angle       : angle filter.

@return video_path or None if not found.
'''
def find_latest_video_for_motion(
    video_dir: Path,
    motion_name: str,
    angle: int | None = 0,
) -> Path | None:

    videos = get_matching_videos(video_dir, angle, motion_name)

    if not videos:
        print(f"Warning: no video found for {motion_name}")
        return None

    return videos[-1]


# -----------------------------
# CSV + Data loading
# -----------------------------
'''
Finds CSV corresponding to a video.

@param[in] video_path : video file path.
@param[in] online_dir : directory containing CSV files.

@return csv_path : matching CSV file.

@throws FileNotFoundError if none found.
'''
def find_csv_for_video(video_path: Path, online_dir: Path) -> Path:

    stamp = extract_stamp(video_path.name)

    candidates = sorted(online_dir.glob(f"*{stamp}*.csv"))

    if not candidates:
        raise FileNotFoundError(
            f"No CSV containing {stamp} found in {online_dir}"
        )

    return candidates[0]

'''
Loads dataframe corresponding to a video.

@param[in] label      : dataset label.
@param[in] video_path : video file.
@param[in] online_dir : CSV directory.

@return df : processed dataframe.
'''
def load_df(label: str, video_path: Path, online_dir: Path, print_flags=False) -> pd.DataFrame:

    csv_path = find_csv_for_video(video_path, online_dir)

    if print_flags:
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

    df["t_rel[s]"] = df["t_system[s]"] - df["t_system[s]"].iloc[0]

    df["label"] = label

    return df


# -----------------------------
# Plotting
# -----------------------------
'''
Saves or displays a figure.

@param[in] fig        : matplotlib figure.
@param[in] name       : filename (without extension).
@param[in] output_dir : directory to save plots. If None, shows plot.
'''
def save_or_show(fig, name: str, output_dir: Path | None = None):

    if output_dir is None:
        plt.show()
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

'''
Generates plots for a given motion.

@param[in] motion     : motion name.
@param[in] dfs        : dict of label -> dataframe.
@param[in] output_dir : where to save plots.
'''
def plot_one_motion(
        
    motion: str,
    dfs: dict[str, pd.DataFrame],
    output_dir: Path | None = None,
):

    # Endpoint difference
    fig, ax = plt.subplots()
    for label, df in dfs.items():
        ax.scatter(df["dx_calc"], df["dy_calc"], s=10, label=label)
    ax.set_title(motion)
    ax.grid(True)
    ax.legend()
    save_or_show(fig, f"{motion}_endpoint", output_dir)

'''
Extracts state variables and target from a dataframe.

The function builds a feature matrix consisting of endpoint position and
bending angle, and a target vector consisting of pressure. It optionally
filters invalid angle measurements and converts angles to degrees.

States:
    - xe[m]
    - ye[m]
    - angle[rad] (or degrees)

Target:
    - pressure[mBar]

@param[in] df         : input dataframe containing robot data.
    should have the entry dx_calc and dy_calc which is obtained from load_df.
@param[in] use_degrees: if True, converts angle from radians to degrees.
@param[in] only_valid : if True, only keeps rows where ANGLE_VALID is True.

@return states : array of shape (N, 3) -> [xe, ye, angle]
@return target : array of shape (N,)   -> pressure
'''
def extract_states_targets(df, only_valid=False, use_degrees=False):

    # Optionally filter valid rows
    if only_valid:
        df = df[df["ANGLE_VALID"]].copy()
    else:
        df = df.copy()

    # Check if dx_calc / dy_calc exist
    if "dx_calc" not in df.columns:
        print("Warning: 'dx_calc' not found. Calculating dx_calc = xe[m] - x0[m]")
        df["dx_calc"] = df["xe[m]"] - df["x0[m]"]

    if "dy_calc" not in df.columns:
        print("Warning: 'dy_calc' not found. Calculating dy_calc = ye[m] - y0[m]")
        df["dy_calc"] = df["ye[m]"] - df["y0[m]"]

    # Extract states
    dx = df["dx_calc"].to_numpy()
    dy = df["dy_calc"].to_numpy()
    angle = df["angle[rad]"].to_numpy()

    if use_degrees:
        angle = np.rad2deg(angle)

    states = np.column_stack((dx, dy, angle))

    # Extract target
    target = df["pressure[mBar]"].to_numpy() / 100

    return states, target

def get_all_robot_dfs(ROBOTS, ANGLES, name_filter="trapezoid", plot_flag=False):
    all_robot_dirs = {}
    for robot_name, robot_dir in ROBOTS.items():
        all_robot_dirs[robot_name] = {}
        if robot_name == "robot1_damage1_healed":
            for angle in ANGLES[1:]:
                deg_key = f"{angle}deg"
                all_robot_dirs[robot_name][deg_key] = get_video_runs(
                    video_dir=robot_dir / "video",
                    angle=angle,
                    run_name_filter=name_filter,
                )
        else:
            for angle in ANGLES:
                deg_key = f"{angle}deg"
                all_robot_dirs[robot_name][deg_key] = get_video_runs(
                    video_dir=robot_dir / "video",
                    angle=angle,
                    run_name_filter=name_filter,
                )

    all_robot_dfs = {}
    for robot in all_robot_dirs.keys():
        all_robot_dfs[robot] = {}
        for angle in all_robot_dirs[robot]:
            if len(all_robot_dirs[robot][angle]) == 0:
                print(f"Warning: No data found for {robot} at {angle}")
                continue
            print()
            all_robot_dfs[robot][angle] = load_df(
                label=f"{robot}_{deg_key}_trapezoid",
                video_path=all_robot_dirs[robot][angle]["trapezoid"],
                online_dir= ROBOTS[robot] / "online",
            )

    # plot every plot and filter appropriate data
    for robot in all_robot_dfs.keys():
        for angle in all_robot_dfs[robot].keys():
            if len(all_robot_dfs[robot][angle]) == 0:
                continue
            # filtering initial noise
            all_robot_dfs[robot][angle] = all_robot_dfs[robot][angle][all_robot_dfs[robot][angle]["pressure[mBar]"] >= 25]
            # this is a hard rule. I noticed it during plotting of this robot
            if (robot == "robot1_damage1_healed") and (angle == "135deg"):
                all_robot_dfs[robot][angle] = all_robot_dfs[robot][angle][all_robot_dfs[robot][angle]["angle[rad]"] != 0]
            if plot_flag:
                plts.plot_one_motion(
                    robot_name=robot,
                    motion=f"{angle}_trapezoid",
                    dfs={"trapezoid": all_robot_dfs[robot][angle]},
                    save_flag=False,
                )
    return all_robot_dfs
