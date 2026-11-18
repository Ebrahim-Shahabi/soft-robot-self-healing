import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np
import LoRA as lora

def save_or_show(fig, robot_name: str, motion: str, name: str):
    OUTPUT_DIR = Path(__file__).resolve().parent / "plots" / robot_name / motion 
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Saving figure to {OUTPUT_DIR}")
    fig.savefig(OUTPUT_DIR / f"{name}.png", dpi=200, bbox_inches="tight")


def plot_one_motion(robot_name: str,motion: str, dfs: dict, save_flag: bool = False):
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
    if save_flag:
        save_or_show(fig, robot_name, motion, "endpoint_base_difference")

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
    if save_flag:
        save_or_show(fig, robot_name, motion, "base_vs_time")

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
    if save_flag:
        save_or_show(fig, robot_name, motion, "endpoint_vs_time")

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
    if save_flag:
        save_or_show(fig, robot_name, motion, "angle_vs_time")

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
    if save_flag:
        save_or_show(fig, robot_name, motion, "pressure_vs_time")

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
    if save_flag:
        save_or_show(fig, robot_name, motion, "pressure_vs_angle")

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

    if save_flag:
        save_or_show(fig, robot_name, motion, "dx_dy_vs_time")
        
    if not save_flag:
        plt.show()

def plot_boundaries(df, df_filtered, Q_MIN=None, Q_MAX=None):
    # plot filtered vs full to confirm
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(np.rad2deg(df["angle[rad]"].values), df["pressure[mBar]"].values/10,
            s=4, alpha=0.2, color="grey",  label="excluded")
    ax.scatter(np.rad2deg(df_filtered["angle[rad]"].values),
            df_filtered["pressure[mBar]"].values/10,
            s=4, alpha=0.5, color="#2563eb", label="training region")

    if Q_MIN is not None:
        ax.axvline(np.rad2deg(Q_MIN), color="red",    linestyle="--", label="Q_MIN")
    if Q_MAX is not None:
        ax.axvline(np.rad2deg(Q_MAX), color="orange",  linestyle="--", label="Q_MAX")

    ax.set_xlabel("Angle [deg]")
    ax.set_ylabel("Pressure [kPa]")
    ax.set_title(f"Robot 2 — 0° with boundaries")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_subspace_analysis(subspace: dict, save_path=None):
    S = subspace["S"]          # already truncated to rank r
    var = subspace["variance_explained"]   # full K components
    r = subspace["rank"]

    K = len(var)               # total number of tasks
    ks_full = np.arange(1, K + 1)   # for variance plot (all K)
    ks_trunc = np.arange(1, r + 1)  # for singular values (only top r)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Task Subspace SVD Analysis", fontsize=14, fontweight="bold")

    ax1.bar(ks_trunc, S, color="#2563eb", alpha=0.8)
    # ax1.axvline(r + 0.5, color="red", linestyle="--", label=f"rank={r}")
    ax1.set_xlabel("Component")
    from matplotlib.ticker import MaxNLocator
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_ylabel("Singular value")
    ax1.set_title("Singular Value Spectrum (kept components)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(ks_full, var * 100, "o-", color="#2563eb", linewidth=2)
    ax2.axhline(90, color="orange", linestyle="--", label="90%")
    ax2.axhline(95, color="red", linestyle="--", label="95%")
    # ax2.axvline(r + 0.5, color="grey", linestyle="--", label=f"rank={r}")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative variance explained (%)")
    ax2.set_title("Variance Explained")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 105)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

def _plot_lora_vs_subspace(results: dict, lora_ranks: list[int], save_path=None):
    fracs = np.array(results["fractions"]) * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0f1117")

    # SVD subspace
    ax.semilogy(fracs, results["subspace"], color="#38bdf8", marker="o",
                lw=2.5, markersize=7, label="SVD Subspace (ours)")

    # LoRA ranks — colour gradient from green to orange
    colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(lora_ranks)))
    for color, r in zip(colors, lora_ranks):
        key = f"lora_r{r}"
        ax.semilogy(fracs, results[key], color=color, marker="s",
                    lw=2, markersize=6, linestyle="--", label=f"LoRA r={r}")

    ax.set_xlabel("Training data used (%)", color="white", fontsize=12)
    ax.set_ylabel("Eval loss (log scale)",  color="white", fontsize=12)
    ax.set_title("SVD Subspace vs LoRA — Data & Parameter Efficiency",
                 color="white", fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.legend(framealpha=0.2, labelcolor="white", facecolor="#1e2030",
              edgecolor="#444", fontsize=9)
    ax.grid(True, alpha=0.15, color="white")
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{f:.0f}%" for f in fracs])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"[Plot] Saved to {save_path}")
    plt.show()


def plot_parameter_efficiency(
    w0_star:        dict,
    subspace:       dict,
    SubspaceParams,
    lora_ranks:     list[int],
    eval_losses:    dict,           # {"subspace": float, "lora_r1": float, ...}
    free_layer_names: list[str] | None = None,
    save_path:      Path | str | None = None,
):
    """
    Bar chart: trainable parameter count vs eval loss for each method.
    eval_losses: dict with keys matching results dict, values are scalar losses
                 (e.g. take the 100% data entry from run_lora_vs_subspace).
    """
    sp = SubspaceParams(subspace=subspace, w0_star=w0_star,
                        free_layer_names=free_layer_names)
    n_sub = int(sp.free_mask.sum()) + subspace["rank"]

    param_counts = {"SVD Subspace": n_sub}
    losses_plot  = {"SVD Subspace": eval_losses["subspace"]}

    for r in lora_ranks:
        lora = lora.LoRAParams(w0_star, rank=r)
        param_counts[f"LoRA r={r}"] = lora.n_trainable
        losses_plot[f"LoRA r={r}"]  = eval_losses[f"lora_r{r}"]

    labels  = list(param_counts.keys())
    counts  = [param_counts[l] for l in labels]
    losses  = [losses_plot[l]  for l in labels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0f1117")
    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1117")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#333")

    colors = ["#38bdf8"] + list(plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(lora_ranks))))
    bars1 = ax1.bar(labels, counts, color=colors, alpha=0.85)
    ax1.set_ylabel("Trainable parameters", color="white")
    ax1.set_title("Parameter Count", color="white", fontweight="bold")
    ax1.set_xticklabels(labels, rotation=20, ha="right", color="white")
    for bar, count in zip(bars1, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                 f"{count:,}", ha="center", va="bottom", color="white", fontsize=9)

    bars2 = ax2.bar(labels, losses, color=colors, alpha=0.85)
    ax2.set_ylabel("Eval loss (100% data)", color="white")
    ax2.set_title("Final Eval Loss", color="white", fontweight="bold")
    ax2.set_xticklabels(labels, rotation=20, ha="right", color="white")
    for bar, loss in zip(bars2, losses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                 f"{loss:.4f}", ha="center", va="bottom", color="white", fontsize=9)

    fig.suptitle("Parameter Efficiency: SVD Subspace vs LoRA",
                 color="white", fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    plt.show()