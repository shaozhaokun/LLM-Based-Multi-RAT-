import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "ieee"])
    style_context = plt.style.context("ieee")
except Exception:
    ieee_fallback = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    style_context = plt.style.context(ieee_fallback)


user_num = np.array([120, 180, 240, 300, 360], dtype=float)

average_cost = {
    "Proposed": np.array([0.104, 0.266, 0.463, 0.569, 0.703]),
    "Multi-Agent EC": np.array([0.164, 0.330, 0.528, 0.720, 0.876]),
    "Single-Agent MGA": np.array([0.212, 0.394, 0.598, 0.787, 0.964]),
}

execution_time = {
    "Proposed": np.array([173, 268, 356, 514, 671]),
    "Multi-Agent EC": np.array([247, 384, 571, 738, 957]),
    "Single-Agent MGA": np.array([536, 817, 1194, 1572, 2048]),
}

colors = {
    "Proposed": "#0072B2",
    "Multi-Agent EC": "#009E73",
    "Single-Agent MGA": "#E69F00",
}
markers = {
    "Proposed": "^",
    "Multi-Agent EC": "o",
    "Single-Agent MGA": "s",
}

SIZE_AXIS_LABEL = 12
SIZE_TICK = 9
SIZE_LEGEND = 7.5
LINEWIDTH = 1.3
MARKERSIZE = 4.5

os.makedirs("Fig_journal", exist_ok=True)

with style_context:
    fig, ax_cost = plt.subplots(figsize=(3.5, 3.0))
    ax_time = ax_cost.twinx()

    for name in average_cost:
        ax_cost.plot(
            user_num,
            average_cost[name],
            color=colors[name],
            linestyle="-",
            marker=markers[name],
            markerfacecolor="none",
            markeredgecolor=colors[name],
            markeredgewidth=0.9,
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            clip_on=False,
        )
        ax_time.plot(
            user_num,
            execution_time[name],
            color=colors[name],
            linestyle="--",
            linewidth=LINEWIDTH,
            clip_on=False,
        )

    ax_cost.set_xlabel(r"Number of users, $K_u+K_e$", fontsize=SIZE_AXIS_LABEL)
    ax_cost.set_ylabel("Average cost", fontsize=SIZE_AXIS_LABEL)
    ax_time.set_ylabel("Execution time (s)", fontsize=SIZE_AXIS_LABEL)

    ax_cost.set_xlim(user_num[0], user_num[-1])
    ax_cost.set_xticks(user_num)
    ax_cost.set_ylim(0.0, 1.0)
    ax_cost.set_yticks(np.arange(0.0, 1.0 + 0.001, 0.2))
    ax_time.set_ylim(0, 2200)
    ax_time.set_yticks(np.arange(0, 2200 + 1, 400))

    ax_cost.tick_params(axis="both", labelsize=SIZE_TICK)
    ax_time.tick_params(axis="y", labelsize=SIZE_TICK)
    ax_cost.grid(True, alpha=0.35)

    method_handles = [
        Line2D(
            [0],
            [0],
            color=colors[name],
            marker=markers[name],
            markerfacecolor="none",
            markeredgecolor=colors[name],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            label=name,
        )
        for name in average_cost
    ]
    metric_handles = [
        Line2D([0], [0], color="black", linestyle="-", linewidth=LINEWIDTH, label="Avg. cost"),
        Line2D([0], [0], color="black", linestyle="--", linewidth=LINEWIDTH, label="Time"),
    ]
    ax_cost.legend(
        handles=method_handles + metric_handles,
        fontsize=SIZE_LEGEND,
        loc="upper left",
        ncol=1,
    )

    fig.savefig(
        "Fig_journal/Fig_scalability_single.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
