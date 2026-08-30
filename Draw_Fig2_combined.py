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


user_num = np.arange(120, 360 + 60, 60)

embb_series = {
    "Proposed": np.array([0.1579, 0.188, 0.276, 0.351, 0.427]),
    "Multi-Agent EC": np.array([0.186, 0.2443, 0.355485, 0.458317, 0.579575]),
    "Single-Agent MGA": np.array([0.232, 0.311163, 0.431, 0.563873, 0.733068]),
    "Single-Agent MDE": np.array([0.173, 0.215683, 0.31822, 0.420408, 0.532089]),
}

urllc_series = {
    "Proposed": np.array([0.05, 0.1721, 0.305, 0.423, 0.493]),
    "Multi-Agent EC": np.array([0.07166, 0.208, 0.353333, 0.491333, 0.586667]),
    "Single-Agent MGA": np.array([0.09666, 0.225667, 0.372, 0.505333, 0.597778]),
    "Single-Agent MDE": np.array([0.138, 0.315556, 0.474167, 0.606, 0.692222]),
}

colors = {
    "Proposed": "#0072B2",
    "Multi-Agent EC": "#009E73",
    "Single-Agent MGA": "#E69F00",
    "Single-Agent MDE": "#D55E00",
}
markers = {
    "Proposed": "^",
    "Multi-Agent EC": "o",
    "Single-Agent MGA": "s",
    "Single-Agent MDE": "D",
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

    ax.set_xlabel("Number of users", fontsize=SIZE_AXIS_LABEL)
    ax.set_xlim(user_num[0], user_num[-1])
    ax.set_xticks(user_num)
    ax.set_yticks(np.arange(0.0, 0.9 + 0.001, 0.1))
    ax.set_ylim(0.0, 0.9)
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

    plot_panel(ax1, embb_series)
    ax1.set_ylabel(r"Average delay ($\phi=e$)", fontsize=SIZE_AXIS_LABEL)

    plot_panel(ax2, urllc_series)
    ax2.set_ylabel(r"Outage ratio ($\phi=u$)", fontsize=SIZE_AXIS_LABEL)

    fig.savefig(
        "Fig_journal/Fig2_eMBB_URLLC_combined.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
