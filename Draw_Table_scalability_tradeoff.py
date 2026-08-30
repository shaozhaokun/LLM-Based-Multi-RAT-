import os

import matplotlib.pyplot as plt
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


user_num = np.array([120, 180, 240, 300, 360])

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
SIZE_LEGEND = 8
LINEWIDTH = 1.4
MARKERSIZE = 5

os.makedirs("Fig_journal", exist_ok=True)

with style_context:
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    for name in average_cost:
        ax.plot(
            execution_time[name],
            average_cost[name],
            color=colors[name],
            linestyle="-",
            label=name,
            marker=markers[name],
            markerfacecolor="none",
            markeredgecolor=colors[name],
            markeredgewidth=0.9,
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )

    ax.set_xlabel("Execution time (s)", fontsize=SIZE_AXIS_LABEL)
    ax.set_ylabel("Average cost", fontsize=SIZE_AXIS_LABEL)
    ax.set_xlim(100, 2150)
    ax.set_xticks(np.arange(200, 2200 + 1, 400))
    ax.set_ylim(0.05, 1.0)
    ax.set_yticks(np.arange(0.1, 1.0 + 0.001, 0.1))
    ax.tick_params(axis="both", labelsize=SIZE_TICK)
    ax.legend(fontsize=SIZE_LEGEND, loc="upper left")
    ax.grid(True, alpha=0.35)

    fig.savefig(
        "Fig_journal/Fig_scalability_tradeoff.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
