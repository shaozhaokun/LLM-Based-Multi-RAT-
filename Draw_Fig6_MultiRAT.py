import os
from contextlib import nullcontext

import numpy as np
import matplotlib.pyplot as plt

plot_styles = []
try:
    import scienceplots  # noqa: F401
    plot_styles = ["science", "ieee"]
    plt.style.use(plot_styles)
except Exception:
    pass


M = np.array([1, 2, 3], dtype=int)

proposed_urllc = np.array([0.0190, 0.0250, 0.0350], dtype=float)
proposed_embb = np.array([0.312886, 0.2150, 0.1579], dtype=float)

show_single_mga = False
single_mga_urllc = np.array([0.076, 0.0850, 0.097], dtype=float)
single_mga_embb = np.array([0.3650, 0.2700, 0.232], dtype=float)

multi_rat_urllc = np.array([0.0561, 0.0628, 0.07166], dtype=float)
multi_rat_embb = np.array([0.3522, 0.2465, 0.1860], dtype=float)

# size_yxlim = 14
# size_label = 9
# size_tick = 8
# size_yxlim = 16
# size_label = 11
# size_tick = 8
size_yxlim = 16
size_tick = 12
size_label = 11
LINEWIDTH = 1.5
MARKERSIZE = 7

urllc_color = "#0072B2"
embb_color = "#D55E00"

os.makedirs("Fig_journal", exist_ok=True)

with plt.style.context(plot_styles) if plot_styles else nullcontext():
    fig, ax1 = plt.subplots(figsize=(5.6, 3.8))
    ax2 = ax1.twinx()

    line1 = ax1.plot(
        M,
        proposed_urllc,
        color=urllc_color,
        linestyle="-",
        marker="^",
        markerfacecolor="none",
        markeredgecolor=urllc_color,
        markeredgewidth=1.0,
        linewidth=1.2,
        markersize=6,
        label="Proposed, delay-sensitive",
    )
    line2 = ax2.plot(
        M,
        proposed_embb,
        color=embb_color,
        linestyle="-",
        marker="o",
        markerfacecolor="none",
        markeredgecolor=embb_color,
        markeredgewidth=1.0,
        linewidth=1.2,
        markersize=5,
        label="Proposed, delay-tolerant",
    )
    line3 = ax1.plot(
        M,
        multi_rat_urllc,
        color=urllc_color,
        linestyle="--",
        marker="^",
        markerfacecolor="none",
        markeredgecolor=urllc_color,
        markeredgewidth=1.0,
        linewidth=1.2,
        markersize=6,
        label="Multi-Agent EC, delay-sensitive",
    )
    line4 = ax2.plot(
        M,
        multi_rat_embb,
        color=embb_color,
        linestyle="--",
        marker="o",
        markerfacecolor="none",
        markeredgecolor=embb_color,
        markeredgewidth=1.0,
        linewidth=1.2,
        markersize=5,
        label="Multi-Agent EC, delay-tolerant",
    )
    if show_single_mga:
        line5 = ax1.plot(
            M,
            single_mga_urllc,
            color=urllc_color,
            linestyle=":",
            marker="^",
            markerfacecolor="none",
            markeredgecolor=urllc_color,
            markeredgewidth=1.0,
            linewidth=1.2,
            markersize=6,
            label="Single-Agent MGA, delay-sensitive",
        )
        line6 = ax2.plot(
            M,
            single_mga_embb,
            color=embb_color,
            linestyle=":",
            marker="o",
            markerfacecolor="none",
            markeredgecolor=embb_color,
            markeredgewidth=1.0,
            linewidth=1.2,
            markersize=5,
            label="Single-Agent MGA, delay-tolerant",
        )
    else:
        line5 = []
        line6 = []

    ax1.set_xlabel("$N^e$", fontsize=size_yxlim)
    # ax1.set_ylabel("Outage ratio of delay-sensitive user", fontsize=size_yxlim, color=urllc_color)
    # ax2.set_ylabel("Average delay of delay-tolerant user", fontsize=size_yxlim, color=embb_color)
    ax1.set_ylabel("Outage ratio ($\phi=u$)", fontsize=size_yxlim, color=urllc_color)
    ax2.set_ylabel("Average delay ($\phi=e$)", fontsize=size_yxlim, color=embb_color)
    

    ax1.set_xlim(1, 3)
    ax1.set_xticks(M)
    ax1.set_ylim(0, 0.14)
    ax1.set_yticks(np.arange(0, 0.14 + 0.02, 0.02))

    ax2.set_ylim(0, 0.4)
    ax2.set_yticks(np.arange(0, 0.4 + 0.05, 0.05))

    ax1.tick_params(axis="x", labelsize=size_tick)
    ax1.tick_params(axis="y", labelsize=size_tick, colors=urllc_color)
    ax2.tick_params(axis="y", labelsize=size_tick, colors=embb_color)
    ax1.spines["left"].set_color(urllc_color)
    ax2.spines["right"].set_color(embb_color)
    ax1.grid(True, alpha=0.35)

    lines = line1 + line2 + line3 + line4 + line5 + line6
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, fontsize=size_label, loc="upper right")

    plt.tight_layout()
    plt.savefig("Fig_journal/Fig6_MultiRAT_.pdf", dpi=300)
