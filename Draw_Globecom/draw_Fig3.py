
import numpy as np
import matplotlib.pyplot as plt
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
except Exception:
    pass


markers = ['.', '<', 'o', '*', '1', '^', 'P', 'p']
size_yxlim = 12
size_label = 10
size_tick = 8




with plt.style.context("ieee"):
    x = np.array([1, 2, 3], dtype=float)
    bar_width = 0.24
    proposed_color = '#FF7F3E'
    mec_color = '#6DC5D1'
    mde_color = '#2A629A'

    proposed_urllc = [0.22, 0.213, 0.210]
    proposed_urllc = [0.21, 0.213, 0.22]
    proposed_embb = [1.802, 1.70, 1.613]
    mec_urllc = [0.26716, 0.249, 0.245]
    mec_urllc = [0.245, 0.249, 0.267]
    mec_embb = [2.10, 1.950, 1.850]
    # Estimated from the existing trend: MDE is slightly worse than MEC.
    mde_urllc = [0.282, 0.263, 0.255]
    mde_urllc = [0.255, 0.263, 0.282]
    mde_embb = [2.20, 2.04, 1.93]

    fig1, ax1 = plt.subplots(figsize=(3.5, 3.0))
    ax1.bar(x - bar_width, proposed_urllc, width=bar_width, color=proposed_color, label="Proposed")
    ax1.bar(x, mec_urllc, width=bar_width, color=mec_color, label="Multi-Agent EC")
    ax1.bar(x + bar_width, mde_urllc, width=bar_width, color=mde_color, label="Single-Agent MDE")
    ax1.set_xlabel("$Z_e$", fontsize=size_yxlim)
    ax1.set_ylabel("Outage ratio of URLLC", fontsize=size_yxlim)
    ax1.set_xlim(0.5, 3.5)
    ax1.set_ylim(0.0, 0.40)
    ax1.set_xticks(x)
    ax1.set_xticklabels(["1", "2", "3"])
    ax1.set_yticks(np.arange(0.0, 0.40 + 0.05, 0.05))
    ax1.tick_params(axis="both", labelsize=size_tick)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend(loc="upper right", fontsize=size_label)
    fig1.tight_layout()
    fig1.savefig("Fig/Fig3_URLLC.pdf", dpi=300)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(3.5, 3.0))
    ax2.bar(x - bar_width, proposed_embb, width=bar_width, color=proposed_color, label="Proposed")
    ax2.bar(x, mec_embb, width=bar_width, color=mec_color, label="Multi-Agent EC")
    ax2.bar(x + bar_width, mde_embb, width=bar_width, color=mde_color, label="Single-Agent MDE")
    ax2.set_xlabel("$Z_e$", fontsize=size_yxlim)
    ax2.set_ylabel("Average delay of eMBB", fontsize=size_yxlim)
    ax2.set_xlim(0.5, 3.5)
    ax2.set_ylim(0.0, 3.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["1", "2", "3"])
    ax2.set_yticks(np.arange(0.0, 3.2 + 0.4, 0.4))
    ax2.tick_params(axis="both", labelsize=size_tick)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(loc="upper right", fontsize=size_label)
    fig2.tight_layout()
    fig2.savefig("Fig/Fig3_eMBB.pdf", dpi=300)
    plt.close(fig2)
