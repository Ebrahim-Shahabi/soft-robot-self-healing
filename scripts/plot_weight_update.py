import json
import math
# package for plot
import matplotlib
# matplotlib.use("Agg")  # use this if you only want to save images
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



# -----------------------------
# Load tracked params from JSON
# -----------------------------
with open("tracked_params_temp1.json", "r") as f:
    tracked_params = json.load(f)


# -----------------------------
# Plot function
# -----------------------------
def plot_tracked_params(tracked_params):
    for group_name, layer_entries in tracked_params.items():
        n_layers = len(layer_entries)

        if n_layers == 0:
            continue

        # Decide subplot grid shape
        ncols = 2
        nrows = math.ceil(n_layers / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        fig.suptitle(f"{group_name} tracked values", fontsize=16)

        # Flatten axes for easy indexing
        if n_layers == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, entry in enumerate(layer_entries):
            layer_name, val1, val2, values = entry

            ax = axes[i]

            # Only plot if values exist and are non-empty
            if values is not None and len(values) > 0:
                ax.plot(values, linewidth=1)
            else:
                ax.text(
                    0.5, 0.5, "No values yet",
                    ha='center', va='center', fontsize=12
                )

            ax.set_title(layer_name)
            ax.set_xlabel("Step")
            ax.set_ylabel("Tracked value")
            ax.grid(True)

            # Optional: include the two integer metadata values
            ax.text(
                0.02, 0.95,
                f"meta: ({val1}, {val2})",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", alpha=0.2)
            )

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


# -----------------------------
# Run plotting
# -----------------------------
plot_tracked_params(tracked_params)