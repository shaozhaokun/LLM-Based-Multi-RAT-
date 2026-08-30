import os

import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
except Exception:
    pass


labels = ["4:8", "6:6", "8:4"]

# The 4:8 setting uses the 1:1 service-ratio data from Draw_Fig3_ratio.py.
proposed = np.array([12.56, 13.20, 13.95], dtype=float)
multi_agent = np.array([19.70, 20.80, 22.10], dtype=float)
single_ga = np.array([25.449181, 27.00, 28.70], dtype=float)
single_de = np.array([26.94548, 28.50, 30.30], dtype=float)

# size_yxlim = 12
# size_label = 9
# size_tick = 8

# size_yxlim = 14
# size_label = 9
# size_tick = 8

size_yxlim = 16
size_label = 11
size_tick = 10


x = np.arange(len(labels))
width = 0.18

os.makedirs("Fig_journal", exist_ok=True)

with plt.style.context("ieee"):
    # fig, ax = plt.subplots(figsize=(3.5, 3.0))
    fig, ax = plt.subplots(figsize=(5.3, 3.8))

    ax.bar(x - width * 1.5, proposed/120, width, label="Proposed", color="#0072B2")
    ax.bar(x - width / 2, multi_agent/120, width, label="Multi-Agent EC", color="#009E73")
    ax.bar(x + width / 2, single_ga/120, width, label="Single-Agent MGA", color="#E69F00")
    ax.bar(x + width * 1.5, single_de/120, width, label="Single-Agent MDE", color="#D55E00")

    ax.set_ylabel("Average cost", fontsize=size_yxlim)
    ax.set_xlabel("Ratio between $f_u$ and $f_e$", fontsize=size_yxlim)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis="both", labelsize=size_tick)
    ax.set_yticks(np.arange(0.0, 0.35 + 0.001, 0.05))
    ax.set_ylim(0.0, 0.35)
    ax.legend(fontsize=size_label, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("Fig_journal/Fig4_capability.pdf", dpi=300)
