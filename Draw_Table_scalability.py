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

SIZE_AXIS_LABEL = 14
SIZE_TICK = 10
SIZE_LEGEND = 9
LINEWIDTH = 1.5
MARKERSIZE = 5


def plot_panel(ax, data):
    for name, values in data.items():
        ax.plot(
            user_num,
            values,
            color=colors[name],
            linestyle="-",
            label=name,
            marker=markers[name],
            markerfacecolor="none",
            markeredgecolor=colors[name],
            markeredgewidth=1.0,
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            clip_on=False,
        )

    ax.set_xlabel(r"Number of users, $K_u+K_e$", fontsize=SIZE_AXIS_LABEL)
    ax.set_xlim(user_num[0], user_num[-1])
    ax.set_xticks(user_num)
    ax.tick_params(axis="both", labelsize=SIZE_TICK)
    ax.legend(fontsize=SIZE_LEGEND, loc="upper left")
    ax.grid(True, alpha=0.35)


os.makedirs("Fig_journal", exist_ok=True)

with style_context:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.16, 2.955),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.18, top=0.985, wspace=0.30)

    plot_panel(ax1, average_cost)
    ax1.set_ylabel("Average cost", fontsize=SIZE_AXIS_LABEL)
    ax1.set_ylim(0.0, 1.0)
    ax1.set_yticks(np.arange(0.0, 1.0 + 0.001, 0.1))

    plot_panel(ax2, execution_time)
    ax2.set_ylabel("Execution time (s)", fontsize=SIZE_AXIS_LABEL)
    ax2.set_ylim(0, 2200)
    ax2.set_yticks(np.arange(0, 2200 + 1, 400))

    fig.savefig(
        "Fig_journal/Fig_scalability.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
