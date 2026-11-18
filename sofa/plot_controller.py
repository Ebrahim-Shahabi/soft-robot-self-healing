import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

columns = [
    "time_s", "u_MPa",
    "x0_deg", "x1_deg", "x2_deg", "x3_deg",
    "dx0_deg_s", "dx1_deg_s", "dx2_deg_s", "dx3_deg_s",
]

df = pd.read_csv("regulation_subspace_rank2.txt", sep=r"\s+", comment="#", names=columns)

time_ = df["time_s"].to_numpy()
angle_raw = df["x3_deg"].to_numpy()
dangle = df["dx3_deg_s"].to_numpy()

# --- Unwrap angle to remove ±180 wraparound discontinuities ---
angle_unwrapped = np.rad2deg(np.unwrap(np.deg2rad(angle_raw)))

# --- Option A: just unwrap (keeps continuity, but may stay negative) ---
angle = 180 + angle_unwrapped

# --- Option B: shift into [0, 360) range instead ---
# angle = angle_unwrapped % 360

# --- Option C: flip sign if your convention is just inverted ---
# angle = -angle_unwrapped
# ---------------------------------------------------------
# 2. Reference values
# ---------------------------------------------------------
angle_ref = 34.99      # deg
velocity_ref = 0.0     # deg/s

# ---------------------------------------------------------
# 3. Plot: angle vs reference
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(time_, angle, label="Angle (x3)")
plt.axhline(angle_ref, color="red", linestyle="--", label=f"Reference angle = {angle_ref}°")
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Bending Angle vs Reference")
plt.legend()
plt.grid(True, alpha=0.3)

# ---------------------------------------------------------
# 4. Plot: angular velocity vs reference
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(time_, dangle, color="orange", label="Angular velocity (dx3)")
plt.axhline(velocity_ref, color="red", linestyle="--", label=f"Reference velocity = {velocity_ref}°/s")
plt.xlabel("Time [s]")
plt.ylabel("Angular velocity [deg/s]")
plt.title("Angular Velocity vs Reference")
plt.legend()
plt.grid(True, alpha=0.3)

# ---------------------------------------------------------
# 5. Combined plot: angle (left axis) + velocity (right axis)
#    Both references line up at the same height on the plot.
# ---------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(11, 5.5))

color_angle = "tab:blue"
color_vel = "tab:orange"

ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Angle [deg]", color=color_angle)
ax1.plot(time_, angle, color=color_angle, label="Angle")
ax1.axhline(angle_ref, color=color_angle, linestyle="--", alpha=0.6, label=f"Angle ref = {angle_ref}°")
ax1.tick_params(axis="y", labelcolor=color_angle)

ax2 = ax1.twinx()
ax2.set_ylabel("Angular velocity [deg/s]", color=color_vel)
ax2.plot(time_, dangle, color=color_vel, label="Velocity")
ax2.axhline(velocity_ref, color=color_vel, linestyle="--", alpha=0.6, label=f"Velocity ref = {velocity_ref}°/s")
ax2.tick_params(axis="y", labelcolor=color_vel)

# Align the two reference lines at the same vertical position on the plot
y1_min, y1_max = ax1.get_ylim()
y2_min, y2_max = ax2.get_ylim()

# Fraction of the axis where the angle reference sits
frac = (angle_ref - y1_min) / (y1_max - y1_min)

# Set ax2 limits so that velocity_ref sits at the same fraction
span2 = y2_max - y2_min
new_y2_min = velocity_ref - frac * span2
new_y2_max = new_y2_min + span2
ax2.set_ylim(new_y2_min, new_y2_max)

# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best")

plt.title("Angle and Angular Velocity (references aligned)")
fig.tight_layout()

plt.show()