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


labels = ["1:3", "1:1", "3:1"]
llm_de = np.array([16.036, 12.56, 37.2564], dtype=float)
multi = np.array([18.641903, 19.7, 47.202579], dtype=float)
all_de = np.array([19.256576, 26.94548, 61.598774], dtype=float)
all_ga = np.array([18.523646, 25.449181, 59.259615], dtype=float)

size_yxlim = 14
size_label = 9
size_tick = 10

x = np.arange(len(labels))
width = 0.18

# Keep the latest palette and hatch mapping consistent with Fig. 4 v2.
colors = {
    "proposed": "#83F8F6",
    "multi_agent": "#9DF7B5",
    "single_ga": "#F27A6D",
    "single_de": "#F2D897",
}

os.makedirs("Fig_journal", exist_ok=True)

with style_context:
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    ax.bar(
        x - width * 1.5,
        llm_de / 120,
        width,
        label="Proposed",
        color=colors["proposed"],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="//",
    )
    ax.bar(
        x - width / 2,
        multi / 120,
        width,
        label="Multi-Agent EC",
        color=colors["multi_agent"],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="\\\\",
    )
    ax.bar(
        x + width / 2,
        all_ga / 120,
        width,
        label="Single-Agent MGA",
        color=colors["single_ga"],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="xx",
    )
    ax.bar(
        x + width * 1.5,
        all_de / 120,
        width,
        label="Single-Agent MDE",
        color=colors["single_de"],
        edgecolor="#333333",
        linewidth=0.7,
        hatch="..",
    )

    ax.set_ylabel("Average cost", fontsize=size_yxlim)
    ax.set_xlabel("Ratio between $K_u$ and $K_e$", fontsize=size_yxlim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", labelsize=size_tick)
    ax.set_yticks(np.arange(0.0, 0.55 + 0.001, 0.05))
    ax.set_ylim(0.0, 0.55)
    ax.legend(fontsize=size_label, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("Fig_journal/Fig3_ratio_v2.pdf", dpi=300)
