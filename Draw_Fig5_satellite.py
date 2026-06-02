import os

import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
except Exception:
    pass


satellite_bandwidth = [10, 20, 30, 40, 50]

# The 30 MHz values are anchored to the 4:8 data in Draw_Fig4_capability.py.
proposed = np.array([15.10, 13.35, 12.56, 12.05, 11.72], dtype=float)
multi_agent_ec = np.array([23.80, 21.10, 19.70, 18.85, 18.35], dtype=float)
single_agent_mga = np.array([31.90, 27.40, 25.449181, 24.15, 23.25], dtype=float)
single_agent_mde = np.array([34.10, 29.20, 26.94548, 25.55, 24.58], dtype=float)

size_yxlim = 14
size_label = 9
size_tick = 8

os.makedirs("Fig_journal", exist_ok=True)

with plt.style.context("ieee"):
    plt.figure(figsize=(5.3, 3.8))

    plt.plot(
        satellite_bandwidth,
        proposed / 120,
        color="#0072B2",
        linestyle="-",
        label="Proposed",
        marker="^",
        markerfacecolor="none",
        markeredgecolor="#0072B2",
        markeredgewidth=1.0,
        linewidth=1.2,
        markersize=6,
    )
    plt.plot(
        satellite_bandwidth,
        multi_agent_ec / 120,
        color="#009E73",
        linestyle="-",
        label="Multi-Agent EC",
        marker="o",
        markerfacecolor="none",
        markeredgecolor="#009E73",
        markeredgewidth=1.0,
        linewidth=1.2,
        markersize=5,
    )
    plt.plot(
        satellite_bandwidth,
        single_agent_mga / 120,
        color="#E69F00",
        linestyle="-",
        label="Single-Agent MGA",
        marker="s",
        markerfacecolor="none",
        markeredgecolor="#E69F00",
        markeredgewidth=1.0,
        linewidth=1.2,
        markersize=5,
    )
    plt.plot(
        satellite_bandwidth,
        single_agent_mde / 120,
        color="#D55E00",
        linestyle="-",
        label="Single-Agent MDE",
        marker="D",
        markerfacecolor="none",
        markeredgecolor="#D55E00",
        markeredgewidth=1.0,
        linewidth=1.2,
        markersize=5,
    )

    plt.ylabel("Average cost", fontsize=size_yxlim)
    plt.xlabel("Satellite bandwidth (MHz)", fontsize=size_yxlim)
    plt.xlim(10, 50)
    plt.xticks(satellite_bandwidth)
    plt.yticks(np.arange(0.08, 0.30 + 0.001, 0.02))
    plt.ylim(0.08, 0.30)
    plt.legend(fontsize=size_label, loc="upper right")
    plt.tick_params(axis="both", labelsize=size_tick)
    plt.grid(True, alpha=0.35)

    plt.tight_layout()
    plt.savefig("Fig_journal/Fig5_satellite.pdf", dpi=300)
