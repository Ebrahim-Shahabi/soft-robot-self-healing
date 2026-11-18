import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
import os
import numpy as np

# CSV_FILE = "tracking_output20260603-162623.csv"

CSV_FILE = "tracking_output20260604-150315.csv" # svd output
CSV_FILE = "tracking_output20260604-162256.csv" # scratch output
CSV_FILE = "tracking_output20260604-163640.csv" # fine tuning output
CSV_FILE = "tracking_output20260604-175149.csv" # Lora rank 4

if len(sys.argv) > 1:
    CSV_FILE = sys.argv[1]

if not os.path.exists(CSV_FILE):
    print(f"File not found: {CSV_FILE}")
    print("Usage: python plot_control_data.py <path_to_csv>")
    sys.exit(1)

df = pd.read_csv(CSV_FILE, skipinitialspace=True)
df.columns = df.columns.str.strip()

print(f"Loaded {len(df)} rows from '{CSV_FILE}'")
print(f"Columns: {list(df.columns)}\n")

t = df["t_system[s]"]

fig = plt.figure(figsize=(16, 14))
fig.suptitle("Control System Telemetry", fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.55, wspace=0.35)

# ── 1. Control output u and its components ──────────────────────────────────
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(t, df["u[mBar]"],    label="u (total)",   color="#1f77b4", linewidth=1.8)
ax1.plot(t, df["u_ff[mBar]"], label="u_ff (feed-forward)", color="#ff7f0e",
         linewidth=1.4, linestyle="--")
ax1.plot(t, df["u_fb[mBar]"], label="u_fb (feedback)",     color="#2ca02c",
         linewidth=1.4, linestyle=":")
ax1.set_ylabel("Pressure (mBar)")
ax1.set_title("Control Output")
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, alpha=0.3)

# ── 2. Angle: current vs reference ──────────────────────────────────────────
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(t, np.rad2deg(df["angle_current[rad]"]), label="current", color="#1f77b4", linewidth=1.8)
ax2.plot(t, np.rad2deg(df["angle_ref[rad]"]),     label="reference", color="#d62728",
         linewidth=1.4, linestyle="--")
ax2.set_ylabel("Angle (deg)")
ax2.set_title("Angle: Current vs Reference")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# ── 3. Angle error ───────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(t, np.rad2deg(df["angle_error[rad]"]), color="#9467bd", linewidth=1.6)
ax3.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax3.set_ylabel("Error (deg)")
ax3.set_title("Angle Error")
ax3.grid(True, alpha=0.3)

# ── 4. X position: current vs reference ─────────────────────────────────────
ax4 = fig.add_subplot(gs[2, 0])
ax4.plot(t, df["x_current[m]"], label="current",   color="#1f77b4", linewidth=1.8)
ax4.plot(t, df["x_ref[m]"],     label="reference", color="#d62728",
         linewidth=1.4, linestyle="--")
ax4.set_ylabel("X position (m)")
ax4.set_title("X Position: Current vs Reference")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

# ── 5. Y position: current vs reference ─────────────────────────────────────
ax5 = fig.add_subplot(gs[2, 1])
ax5.plot(t, df["y_current[m]"], label="current",   color="#1f77b4", linewidth=1.8)
ax5.plot(t, df["y_ref[m]"],     label="reference", color="#d62728",
         linewidth=1.4, linestyle="--")
ax5.set_ylabel("Y position (m)")
ax5.set_title("Y Position: Current vs Reference")
ax5.legend(fontsize=8)
ax5.grid(True, alpha=0.3)

# ── 6. Position errors dx / dy ───────────────────────────────────────────────
ax6 = fig.add_subplot(gs[3, 0])
ax6.plot(t, df["dx_error[m]"], label="dx error", color="#8c564b", linewidth=1.6)
ax6.plot(t, df["dy_error[m]"], label="dy error", color="#e377c2",
         linewidth=1.4, linestyle="--")
ax6.axhline(0, color="gray", linewidth=0.8, linestyle="--")
ax6.set_ylabel("Error (m)")
ax6.set_xlabel("Time (s)")
ax6.set_title("Position Errors (dx, dy)")
ax6.legend(fontsize=8)
ax6.grid(True, alpha=0.3)

# ── 7. 2-D trajectory: current vs reference ──────────────────────────────────
ax7 = fig.add_subplot(gs[3, 1])
ax7.plot(df["x_current[m]"], df["y_current[m]"],
         label="current", color="#1f77b4", linewidth=1.8, marker="o", markersize=4)
ax7.plot(df["x_ref[m]"],     df["y_ref[m]"],
         label="reference", color="#d62728", linewidth=1.4,
         linestyle="--", marker="x", markersize=4)
ax7.set_xlabel("X (m)")
ax7.set_ylabel("Y (m)")
ax7.set_title("2-D Trajectory")
ax7.legend(fontsize=8)
ax7.grid(True, alpha=0.3)

for ax in [ax1, ax2, ax3, ax4, ax5]:
    ax.set_xlabel("Time (s)")

# plt.savefig("control_data_plots.png", dpi=150, bbox_inches="tight")
print("Plot saved to control_data_plots.png")
# plt.show()


def save_fig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches="tight")
    print(f"Saved {name}")
    plt.show()


angle_cur_deg = np.rad2deg(df["angle_current[rad]"])
angle_ref_deg = np.rad2deg(df["angle_ref[rad]"])
angle_err_deg = np.rad2deg(df["angle_error[rad]"])
pos_err_mag = np.sqrt(df["dx_error[m]"]**2 + df["dy_error[m]"]**2)


# 1. Control output
plt.figure(figsize=(14, 6))
plt.plot(t, df["u[mBar]"], label="u total", linewidth=1.8)
plt.plot(t, df["u_ff[mBar]"], label="u_ff feed-forward", linestyle="--")
plt.plot(t, df["u_fb[mBar]"], label="u_fb feedback", linestyle=":")
plt.title("Control Output Components")
plt.xlabel("Time (s)")
plt.ylabel("Pressure (mBar)")
plt.grid(True, alpha=0.3)
plt.legend()
#save_fig("01_control_output.png")


# 2. Angle tracking
plt.figure(figsize=(14, 6))
plt.plot(t, angle_cur_deg, label="current", linewidth=1.8)
plt.plot(t, angle_ref_deg, label="reference", linestyle="--")
plt.title("Angle Tracking")
plt.xlabel("Time (s)")
plt.ylabel("Angle (deg)")
plt.grid(True, alpha=0.3)
plt.legend()
#save_fig("02_angle_tracking.png")


# 3. Angle error
plt.figure(figsize=(14, 6))
plt.plot(t, angle_err_deg, linewidth=1.8)
plt.axhline(0, linestyle="--", linewidth=0.8)
plt.title("Angle Error")
plt.xlabel("Time (s)")
plt.ylabel("Error (deg)")
plt.grid(True, alpha=0.3)
#save_fig("03_angle_error.png")


# 4. X tracking
plt.figure(figsize=(14, 6))
plt.plot(t, df["x_current[m]"], label="current", linewidth=1.8)
plt.plot(t, df["x_ref[m]"], label="reference", linestyle="--")
plt.title("X Position Tracking")
plt.xlabel("Time (s)")
plt.ylabel("X Position (m)")
plt.grid(True, alpha=0.3)
plt.legend()
#save_fig("04_x_tracking.png")


# 5. X error
plt.figure(figsize=(14, 6))
plt.plot(t, df["dx_error[m]"], linewidth=1.8)
plt.axhline(0, linestyle="--", linewidth=0.8)
plt.title("X Position Error")
plt.xlabel("Time (s)")
plt.ylabel("dx Error (m)")
plt.grid(True, alpha=0.3)
#save_fig("05_x_error.png")


# 6. Y tracking
plt.figure(figsize=(14, 6))
plt.plot(t, df["y_current[m]"], label="current", linewidth=1.8)
plt.plot(t, df["y_ref[m]"], label="reference", linestyle="--")
plt.title("Y Position Tracking")
plt.xlabel("Time (s)")
plt.ylabel("Y Position (m)")
plt.grid(True, alpha=0.3)
plt.legend()
#save_fig("06_y_tracking.png")


# 7. Y error
plt.figure(figsize=(14, 6))
plt.plot(t, df["dy_error[m]"], linewidth=1.8)
plt.axhline(0, linestyle="--", linewidth=0.8)
plt.title("Y Position Error")
plt.xlabel("Time (s)")
plt.ylabel("dy Error (m)")
plt.grid(True, alpha=0.3)
#save_fig("07_y_error.png")


plt.show()