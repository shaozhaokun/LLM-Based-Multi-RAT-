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


# Fig. 5 data: satellite bandwidth.
satellite_bandwidth = np.array([10, 20, 30, 40, 50], dtype=float)
satellite_series = {
    "Proposed": np.array([15.10, 13.35, 12.56, 12.05, 11.72]) / 120,
    "Multi-Agent EC": np.array([23.80, 21.10, 19.70, 18.85, 18.35]) / 120,
    "Single-Agent MGA": np.array([31.90, 27.40, 25.449181, 24.15, 23.25]) / 120,
    "Single-Agent MDE": np.array([34.10, 29.20, 26.94548, 25.55, 24.58]) / 120,
}

# Fig. 7 data: hybrid user distribution.
hybrid_labels = [
    r"$(30,\!30,\!0)$",
    r"$(20,\!20,\!20)$",
    r"$(10,\!10,\!40)$",
    r"$(0,\!0,\!60)$",
]
hybrid_x = np.arange(len(hybrid_labels))
hybrid_series = {
    "Proposed": np.array([28.10, 12.56, 11.30, 7.20]) / 120,
    "Multi-Agent EC": np.array([30.490072, 19.70, 15.80, 10.60]) / 120,
    "Single-Agent MGA": np.array([33.014731, 25.44918, 17.420056, 13.80]) / 120,
    "Single-Agent MDE": np.array([33.912501, 26.94548, 24.80, 20.895606]) / 120,
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

SIZE_AXIS_LABEL = 14.5
SIZE_TICK = 10.3
SIZE_LEGEND = 10
LINEWIDTH = 1.5
MARKERSIZE = 5

os.makedirs("Fig_journal", exist_ok=True)

with style_context:
    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(7.16, 3.0),
        gridspec_kw={"width_ratios": [1, 1]},
    )
    # Fixed geometry keeps both plotting areas exactly the same size.
    fig.subplots_adjust(left=0.085, right=0.99, bottom=0.18, top=0.985, wspace=0.28)

    for name, values in satellite_series.items():
        ax1.plot(
            satellite_bandwidth,
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

    ax1.set_ylabel("Average cost", fontsize=SIZE_AXIS_LABEL)
    ax1.set_xlabel("Satellite bandwidth (MHz)", fontsize=SIZE_AXIS_LABEL)
    ax1.set_xlim(satellite_bandwidth[0], satellite_bandwidth[-1])
    ax1.set_xticks(satellite_bandwidth)
    ax1.set_yticks(np.arange(0.08, 0.30 + 0.001, 0.02))
    ax1.set_ylim(0.08, 0.30)
    ax1.tick_params(axis="both", labelsize=SIZE_TICK)
    ax1.legend(fontsize=SIZE_LEGEND, loc="upper right")
    ax1.grid(True, alpha=0.35)

    for name, values in hybrid_series.items():
        ax2.plot(
            hybrid_x,
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

    ax2.set_ylabel("Average cost", fontsize=SIZE_AXIS_LABEL)
    ax2.set_xlabel(
        r"Hybrid user distribution $(K_1,K_2,K_3)$",
        fontsize=SIZE_AXIS_LABEL,
    )
    # Align the first and last category ticks with the two side spines.
    ax2.set_xlim(hybrid_x[0], hybrid_x[-1])
    ax2.set_xticks(hybrid_x)
    ax2.set_xticklabels(hybrid_labels)
    ax2.set_yticks(np.arange(0.04, 0.32 + 0.001, 0.04))
    ax2.set_ylim(0.04, 0.32)
    ax2.tick_params(axis="both", labelsize=SIZE_TICK)
    ax2.legend(fontsize=SIZE_LEGEND, loc="upper right")
    ax2.grid(True, alpha=0.35)

    fig.savefig(
        "Fig_journal/Fig5_Fig7_combined.pdf",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.02,
    )
