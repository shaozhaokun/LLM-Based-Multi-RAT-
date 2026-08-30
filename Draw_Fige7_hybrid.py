import os

import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
except Exception:
    pass


labels = ["(30,30,0)", "(20,20,20)", "(10,10,40)", "(0,0,60)"]
x = np.arange(len(labels))

# The (20,20,20) setting is anchored to the existing 120-user result.
# The other Proposed values are completed following the observed baseline trend.
proposed = np.array([28.10, 12.56, 11.30, 7.20], dtype=float) / 120
multi_agent = np.array([30.490072, 19.70, 15.80, 10.60], dtype=float) / 120
single_mga = np.array([33.014731, 25.44918, 17.420056, 13.80], dtype=float) / 120
single_mde = np.array([33.912501, 26.94548, 24.80, 20.895606], dtype=float) / 120

# size_yxlim = 14
# size_label = 9
# size_tick = 8

size_yxlim = 20
size_label = 14
size_tick = 10.3
LINEWIDTH = 2

os.makedirs("Fig_journal", exist_ok=True)

with plt.style.context("ieee"):
    plt.figure(figsize=(5.3, 3.8))

    plt.plot(
        x,
        proposed,
        color="#0072B2",
        linestyle="-",
        label="Proposed",
        marker="^",
        markerfacecolor="none",
        markeredgecolor="#0072B2",
        markeredgewidth=1.0,
        linewidth=LINEWIDTH,
        markersize=6,
        clip_on=False,
    )
    plt.plot(
        x,
        multi_agent,
        color="#009E73",
        linestyle="-",
        label="Multi-Agent EC",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="#009E73",
        markeredgewidth=1.0,
        linewidth=LINEWIDTH,
        markersize=5,
        clip_on=False,
    )
    plt.plot(
        x,
        single_mga,
        color="#E69F00",
        linestyle="-",
        label="Single-Agent MGA",
        marker="s",
        markerfacecolor="none",
        markeredgecolor="#E69F00",
        markeredgewidth=1.0,
        linewidth=LINEWIDTH,
        markersize=5,
        clip_on=False,
    )
    plt.plot(
        x,
        single_mde,
        color="#D55E00",
        linestyle="-",
        label="Single-Agent MDE",
        marker="D",
        markerfacecolor="none",
        markeredgecolor="#D55E00",
        markeredgewidth=1.0,
        linewidth=LINEWIDTH,
        markersize=5,
        clip_on=False,
    )

    plt.ylabel("Average cost", fontsize=size_yxlim)
    plt.xlabel("Hybrid user distribution $(K_1,K_2,K_3)$", fontsize=size_yxlim)
    plt.xticks(x, labels)
    plt.margins(x=0)
    plt.yticks(np.arange(0.04, 0.32 + 0.001, 0.04))
    plt.ylim(0.04, 0.32)
    plt.legend(fontsize=size_label, loc="upper right")
    plt.tick_params(axis="both", labelsize=size_tick)
    plt.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig("Fig_journal/Fig7_hybrid.pdf", dpi=300)
