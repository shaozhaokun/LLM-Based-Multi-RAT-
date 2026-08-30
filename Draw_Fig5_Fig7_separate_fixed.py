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


FIGSIZE = (3.5, 2.6)
AXES_RECT = [0.18, 0.16, 0.78, 0.80]  # left, bottom, width, height
SIZE_AXIS_LABEL = 11
SIZE_TICK = 8
SIZE_LEGEND = 7.5
LINEWIDTH = 1.2
MARKERSIZE = 4.5

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


def plot_series(ax, x, series):
    for name, values in series.items():
        ax.plot(
            x,
            values,
            color=colors[name],
            linestyle="-",
            label=name,
            marker=markers[name],
            markerfacecolor="none",
            markeredgecolor=colors[name],
            markeredgewidth=0.8,
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
            clip_on=False,
        )


def save_full_canvas(fig, output_path):
    # SciencePlots may set savefig.bbox='tight'. Disable it so both PDFs keep
    # exactly the same 3.5 x 2.6 inch page and the same axes rectangle.
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(output_path, dpi=300)


os.makedirs("Fig_journal", exist_ok=True)

with style_context:
    satellite_bandwidth = np.array([10, 20, 30, 40, 50], dtype=float)
    satellite_series = {
        "Proposed": np.array([15.10, 13.35, 12.56, 12.05, 11.72]) / 120,
        "Multi-Agent EC": np.array([23.80, 21.10, 19.70, 18.85, 18.35]) / 120,
        "Single-Agent MGA": np.array([31.90, 27.40, 25.449181, 24.15, 23.25]) / 120,
        "Single-Agent MDE": np.array([34.10, 29.20, 26.94548, 25.55, 24.58]) / 120,
    }

    fig5 = plt.figure(figsize=FIGSIZE)
    ax5 = fig5.add_axes(AXES_RECT)
    plot_series(ax5, satellite_bandwidth, satellite_series)

    ax5.set_ylabel("Average cost", fontsize=SIZE_AXIS_LABEL)
    ax5.set_xlabel("Satellite bandwidth (MHz)", fontsize=SIZE_AXIS_LABEL)
    ax5.set_xlim(10, 50)
    ax5.set_xticks(satellite_bandwidth)
    ax5.set_yticks(np.arange(0.08, 0.30 + 0.001, 0.02))
    ax5.set_ylim(0.08, 0.30)
    ax5.tick_params(axis="both", labelsize=SIZE_TICK)
    ax5.legend(fontsize=SIZE_LEGEND, loc="upper right")
    ax5.grid(True, alpha=0.35)

    save_full_canvas(fig5, "Fig_journal/Fig5_satellite_fixed_axes.pdf")
    plt.close(fig5)

    hybrid_x = np.arange(4)
    hybrid_series = {
        "Proposed": np.array([28.10, 12.56, 11.30, 7.20]) / 120,
        "Multi-Agent EC": np.array([30.490072, 19.70, 15.80, 10.60]) / 120,
        "Single-Agent MGA": np.array([33.014731, 25.44918, 17.420056, 13.80]) / 120,
        "Single-Agent MDE": np.array([33.912501, 26.94548, 24.80, 20.895606]) / 120,
    }

    fig7 = plt.figure(figsize=FIGSIZE)
    ax7 = fig7.add_axes(AXES_RECT)
    plot_series(ax7, hybrid_x, hybrid_series)

    ax7.set_ylabel("Average cost", fontsize=SIZE_AXIS_LABEL)
    ax7.set_xlabel("Hybrid user distribution case", fontsize=SIZE_AXIS_LABEL)
    ax7.set_xlim(0, 3)
    ax7.set_xticks(hybrid_x)
    ax7.set_xticklabels(["Case 1", "Case 2", "Case 3", "Case 4"])
    ax7.set_yticks(np.arange(0.04, 0.32 + 0.001, 0.04))
    ax7.set_ylim(0.04, 0.32)
    ax7.tick_params(axis="both", labelsize=SIZE_TICK)
    ax7.legend(fontsize=SIZE_LEGEND, loc="upper right")
    ax7.grid(True, alpha=0.35)

    save_full_canvas(fig7, "Fig_journal/Fig7_hybrid_fixed_axes.pdf")
    plt.close(fig7)
