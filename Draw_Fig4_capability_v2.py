import os

import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
    style_context = plt.style.context("ieee")
except Exception:
    # Match the IEEE-style fallback used by the other drawing scripts.
    ieee_fallback = {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 0.8,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
    style_context = plt.style.context(ieee_fallback)


labels = ["4:8", "6:6", "8:4"]

# The 4:8 setting uses the 1:1 service-ratio data from Draw_Fig3_ratio.py.
proposed = np.array([12.56, 13.20, 13.95], dtype=float)
multi_agent = np.array([19.70, 20.80, 22.10], dtype=float)
single_ga = np.array([25.449181, 27.00, 28.70], dtype=float)
single_de = np.array([26.94548, 28.50, 30.30], dtype=float)

size_yxlim = 14
size_label = 9
size_tick = 10

x = np.arange(len(labels))
width = 0.18

# Color test based on the palette used by the RHS curves.
colors = {
    "proposed": "#83F8F6",
    "multi_agent": "#9DF7B5",
    "single_ga": "#F27A6D",
    "single_de": "#F2D897",
}

os.makedirs("Fig_journal", exist_ok=True)

with style_context:
    # fig, ax = plt.subplots(figsize=(5.3, 3.8))
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    ax.bar(
        x - width * 1.5,
        proposed / 120,
        width,
        label="Proposed",
        color=colors["proposed"],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="//",
    )
    ax.bar(
        x - width / 2,
        multi_agent / 120,
        width,
        label="Multi-Agent EC",
        color=colors["multi_agent"],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="\\\\",
    )
    ax.bar(
        x + width / 2,
        single_ga / 120,
        width,
        label="Single-Agent MGA",
        color=colors["single_ga"],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="xx",
    )
    ax.bar(
        x + width * 1.5,
        single_de / 120,
        width,
        label="Single-Agent MDE",
        color=colors["single_de"],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="..",
    )

    ax.set_ylabel("Average cost", fontsize=size_yxlim)
    ax.set_xlabel("Ratio between $Q^u$ and $Q^e$", fontsize=size_yxlim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", labelsize=size_tick)
    ax.set_yticks(np.arange(0.0, 0.36 + 0.001, 0.06))
    ax.set_ylim(0.0, 0.36)
    ax.legend(fontsize=size_label, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("Fig_journal/Fig4_capability_v2.pdf", dpi=300)
