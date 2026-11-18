#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

# ---- load data ----
data = np.loadtxt("input_test1.txt", comments="#")

time = data[:, 0]
pressure = data[:, 1]

data[:,2:10] = data[:,2:10]   # Convert first two angles from rad to deg
data[:,2:6] += 180
x0 = data[:, 2] 
x1 = data[:, 3] 
x2 = data[:, 4]
x3 = data[:, 5]

dx0 = data[:, 6]
dx1 = data[:, 7]
dx2 = data[:, 8]
dx3 = data[:, 9]

# Sort by time just in case
idx = np.argsort(time)
time = time[idx]
pressure = pressure[idx]
x0, x1, x2, x3 = x0[idx], x1[idx], x2[idx], x3[idx]
dx0, dx1, dx2, dx3 = dx0[idx], dx1[idx], dx2[idx], dx3[idx]

# ---- Plot 1: Pressure vs Tip Angle (x3) ----
plt.figure()
plt.plot(pressure, x3, marker="o", markersize=3, linewidth=1)
plt.xlabel("Pressure [MPa]")
plt.ylabel("Tip angle x3 [deg]")
plt.title("Pressure vs Tip Angle")
plt.grid(True)


plt.figure()
plt.plot(time, x3, linewidth=1.5, label="tip angle")
plt.plot(time, dx3, linewidth=1.5, label="angular velocity of tip")
plt.xlabel("Time [s]")
plt.ylabel("Tip angle x3 [deg] / Angular velocity of tip [deg/s]")
plt.axhline(y=25, color='k', linestyle='--', label='Desired angle for tip(25 deg)')
plt.title("Tip Angle and Angular Velocity vs Time")
plt.grid(True)
plt.legend()


fig, ax1 = plt.subplots()

ax1.set_xlabel('Time [s]')
ax1.set_ylabel('Tip angle [deg]', labelpad=15)

l1, = ax1.plot(time, x3, linewidth=1.5, label="tip angle", color='blue')
l2 = ax1.axhline(y=25, color='k', linestyle='--',
                 label='Desired angle for tip (25 deg)')

ax1.tick_params(axis='y')
ax1.grid(True)

ax2 = ax1.twinx()
ax2.set_ylabel('Angular velocity of tip [deg/s]', rotation=270, labelpad=15)

l3, = ax2.plot(time, dx3, linewidth=1.5,
               label="angular velocity of tip",
               color='orange', linestyle='--')

ax2.tick_params(axis='y')

# 🔑 combine legends from both axes
lines = [l1, l2, l3]
labels = [line.get_label() for line in lines]
ax1.legend(lines, labels, loc='best')

fig.tight_layout()
# plt.grid(True)
plt.show()

# ---- Plot 2: Time vs Tip Angle (x3) ----
plt.figure()
plt.plot(time, x3, linewidth=1.5, label="tip angle")
plt.xlabel("Time [s]")
plt.ylabel(r"$\theta$ [deg]")
plt.axhline(y=25, color='k', linestyle='--', label='Desired angle for tip(25 deg)')
plt.title("Tip Angle vs Time")
plt.grid(True)
plt.legend()

# ---- Plot 2: Time vs dTip Angle (x3) ----
plt.figure()
plt.plot(time, dx3, linewidth=1.5)
plt.xlabel("Time [s]")
plt.ylabel(r"$\omega$[deg/s]")
plt.title("Angular velocity of tip vs Time")
plt.grid(True)

# ---- Plot 2: Pressure vs Time ----
plt.figure()
plt.plot(time, pressure*1000, linewidth=1.5)
plt.xlabel("Time [s]")
plt.ylabel("Pressure [KPa]")
plt.title("Pressure vs Time")
plt.grid(True)



# ---- Plot 3: Angles vs Time (x0..x3) ----
plt.figure()
plt.plot(time, x0, linewidth=1.5, label="x0")
plt.plot(time, x1, linewidth=1.5, label="x1")
plt.plot(time, x2, linewidth=1.5, label="x2")
plt.plot(time, x3, linewidth=1.5, label="x3 (tip)")
plt.axhline(y=25, color='k', linestyle='--', label='Desired angle for x3(25 deg)')
plt.xlabel("Time [s]")
plt.ylabel("Angle [deg]")
plt.title("Angles vs Time")
plt.grid(True)
plt.legend()

# ---- Plot 4: Angular velocities vs Time (dx0..dx3) ----
plt.figure()
plt.plot(time, dx0, linewidth=1.5, label="dx0")
plt.plot(time, dx1, linewidth=1.5, label="dx1")
plt.plot(time, dx2, linewidth=1.5, label="dx2")
plt.plot(time, dx3, linewidth=1.5, label="dx3 (tip)")
plt.xlabel("Time [s]")
plt.ylabel("Angular velocity [deg/s]")
plt.title("Angular Velocities vs Time")
plt.grid(True)
plt.legend()

plt.show()
