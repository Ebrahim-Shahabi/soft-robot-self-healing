import pandas as pd
from os import path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

basepath = path.dirname(__file__)

csv_path = path.join(
    basepath,
    "../",
    "svd_loss.csv"
)

csv1_path = path.join(
    basepath,
    "../",
    "no_svd_loss.csv"
)



df = pd.read_csv(csv_path)
df1 = pd.read_csv(csv1_path)

iterations = df["iterations"]
iterations[0] = 1
train_losses = df["train_loss"]
train_variance = df["train_variation"]
train_losses1 = df1["train_loss"]
train_variance1 = df1["train_variation"]
test_losses = df["test_loss"]
test_variance = df["test_variation"]
test_losses1 = df1["test_loss"]
test_variance1 = df1["test_variation"]

blue = '#4C78A8'
orange = '#F58518'

f, ax = plt.subplots(1, 1)


# Second curve (orange)
ax.plot(iterations, train_losses1, color=blue, label='Full Fine tuning')
r1_forward = [x - v for x, v in zip(train_losses1, train_variance1)]
r2_forward = [x + v for x, v in zip(train_losses1, train_variance1)]
ax.fill_between(iterations, r1_forward, r2_forward, color=blue, alpha=0.18)


# First curve (blue)
ax.plot(iterations, train_losses, color=orange, label='SVD')
r1_forward = [x - v for x, v in zip(train_losses, train_variance)]
r2_forward = [x + v for x, v in zip(train_losses, train_variance)]
ax.fill_between(iterations, r1_forward, r2_forward, color=orange, alpha=0.18)

ax.set_title("Train Loss")
ax.legend(loc='upper right')
ax.set_xlim(1, 6e+3 + 1)
ax.set_xlabel('epoch', fontsize=12)
ax.set_ylabel('loss', fontsize=12)

f, ax1 = plt.subplots(1, 1)
# Second curve (orange)
ax1.plot(iterations, test_losses1, color=blue, label='Full Fine tuning')
r1_forward = [x - v for x, v in zip(test_losses1, test_variance1)]
r2_forward = [x + v for x, v in zip(test_losses1, test_variance1)]
ax1.fill_between(iterations, r1_forward, r2_forward, color=blue, alpha=0.18)

# First curve (blue)
ax1.plot(iterations, test_losses, color=orange, label='SVD')
r1_forward = [x - v for x, v in zip(test_losses, test_variance)]
r2_forward = [x + v for x, v in zip(test_losses, test_variance)]
ax1.fill_between(iterations, r1_forward, r2_forward, color=orange, alpha=0.18)
ax1.set_title("Test Loss")

ax1.legend(loc='upper right')
ax1.set_xlim(1, 6e+3 + 1)
ax1.set_xlabel('epoch', fontsize=12)
ax1.set_ylabel('loss', fontsize=12)
# plt.show()






fig, (ax, ax1) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

# -------------------- Train Loss --------------------
# Second curve (orange)
ax.plot(iterations, train_losses1, color=blue, label='Full Fine tuning')
r1_forward = [x - v for x, v in zip(train_losses1, train_variance1)]
r2_forward = [x + v for x, v in zip(train_losses1, train_variance1)]
ax.fill_between(iterations, r1_forward, r2_forward, color=blue, alpha=0.18)

# First curve (blue)
ax.plot(iterations, train_losses, color=orange, label='SVD')
r1_forward = [x - v for x, v in zip(train_losses, train_variance)]
r2_forward = [x + v for x, v in zip(train_losses, train_variance)]
ax.fill_between(iterations, r1_forward, r2_forward, color=orange, alpha=0.18)

ax.set_xlim(1, 6e+3 + 1)
ax.set_ylabel('Train Loss', fontsize=12)
ax.legend(loc='upper right')
# -------------------- Test Loss --------------------
# Second curve (orange)
ax1.plot(iterations, test_losses1, color=blue, label='Full Fine tuning')
r1_forward = [x - v for x, v in zip(test_losses1, test_variance1)]
r2_forward = [x + v for x, v in zip(test_losses1, test_variance1)]
ax1.fill_between(iterations, r1_forward, r2_forward, color=blue, alpha=0.18)

# First curve (blue)
ax1.plot(iterations, test_losses, color=orange, label='SVD')
r1_forward = [x - v for x, v in zip(test_losses, test_variance)]
r2_forward = [x + v for x, v in zip(test_losses, test_variance)]
ax1.fill_between(iterations, r1_forward, r2_forward, color=orange, alpha=0.18)

ax1.set_xlim(1, 6e+3 + 1)
ax1.set_xlabel('epoch', fontsize=12)
ax1.set_ylabel('Test Loss', fontsize=12)

# One legend only
# handles, labels = ax.get_legend_handles_labels()
# fig.legend(handles, labels, loc='upper right')

plt.tight_layout()
plt.show()









# # test forward loss
# ax.plot(iterations, test_losses["forward_loss"], color=palette(2), label='test_forward_loss')
# t1_forward = list(map(lambda x: x[0] - x[1], zip(test_losses['forward_loss'], test_losses['forward_var'])))
# t2_forward = list(map(lambda x: x[0] + x[1], zip(test_losses['forward_loss'], test_losses['forward_var'])))
# ax.fill_between(iterations, t1_forward, t2_forward, color=palette(2), alpha=0.2)

# ax.legend(loc='upper right')
# ax.set_xlim(0, hyper['max_epoch']+1)
# # ax.set_ylim(0, max(train_losses["forward_loss"][10:]))
# ax.set_xlabel('epoch', fontsize=12)
# ax.set_ylabel('loss', fontsize=12)
# plt.show()
