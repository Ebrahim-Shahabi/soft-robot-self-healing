import random
from random import shuffle
import time
import numpy as np
# package for plot
import matplotlib.pyplot as plt

print("Loading Data:")
data = np.loadtxt("new_new_finger_io_const.txt", comments="#")

time_steps = data[:, 0]
input = data[:, 1]

x0 = np.zeros(data.shape[0])
dx0 = np.zeros(data.shape[0])
x1 = data[:, 2]
x2 = data[:, 3]
x3 = data[:, 4]
x4 = data[:, 5]

states = np.column_stack((x0, x1, x2, x3, x4))

dx1 = data[:, 6]
dx2 = data[:, 7]
dx3 = data[:, 8]
dx4 = data[:, 9]
states = np.column_stack((states, dx0,dx1, dx2, dx3, dx4))

n_data = states.shape[0]-1
print(f"Total data samples: {n_data}")


data = states[:-1,:].copy()
target = states[1:,:].copy()

temp = np.column_stack((data, input[:-1], target))
random.shuffle(temp) # shuffle along the first axis

# sanity check plot
# plt.figure()
# plt.plot(temp[:,4], temp[:,3], marker="o", markersize=3, linewidth=1)
# plt.xlabel("Input [MPa]")
# plt.ylabel("Tip angle x3 [deg]")        
# plt.show()

'''
3. Write data to file.
'''
file = open("new_new_data_const.txt", "w")
file.write(
    "# x0[deg] x1[deg] x2[deg] x3[deg] x4[deg]"
    " dx0[deg/s] dx1[deg/s] dx2[deg/s] dx3[deg/s] dx4[deg/s] u[MPa]"
    " xt0[deg] xt1[deg] xt2[deg] xt3[deg] xt4[deg]"
    " dxt0[deg/s] dxt1[deg/s] dxt2[deg/s] dxt3[deg/s] dxt4[deg/s]\n"
)

for t in temp:
    x0, x1, x2, x3, x4, dx0, dx1, dx2, dx3, dx4, u, xt0, xt1, xt2, xt3, xt4, dxt0, dxt1, dxt2, dxt3, dxt4 = t
    file.write(
            f"{x0:.6f} {x1:.6f} {x2:.6f} {x3:.6f} {x4:.6f} "
            f"{dx0:.6f} {dx1:.6f} {dx2:.6f} {dx3:.6f} {dx4:.6f} {u:.6f} "
            f"{xt0:.6f} {xt1:.6f} {xt2:.6f} {xt3:.6f} {xt4:.6f} "
            f"{dxt0:.6f} {dxt1:.6f} {dxt2:.6f} {dxt3:.6f} {dxt4:.6f}\n"
    )
file.flush()

