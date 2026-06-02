
import numpy as np
import matplotlib.pyplot as plt
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
except Exception:
    pass

    
# ### Fig 5 service ratio
size_yxlim = 12
size_label = 10
size_tick = 8

labels = ['1:3', '1:1', '3:1']

# Values follow the three-method trend from Fig. 2.
proposed = [1.0372, 1.0527, 1.06794]
mec = [1.051825, 1.06059, 1.07544]
mde = [1.0653, 1.07278, 1.08711]

# Bar width
bar_width = 0.22

# Set positions for bars
index = np.arange(len(labels))

# Plotting
with plt.style.context("ieee"):
    plt.figure(figsize=(3.5, 3.0))

    # Bar plots for each dataset
    plt.bar(index - bar_width, proposed, bar_width, label="Proposed", color="#FF7F3E")
    plt.bar(index, mec, bar_width, label="Multi-Agent EC", color="#6DC5D1")
    plt.bar(index + bar_width, mde, bar_width, label="Single-Agent MDE", color="#2A629A")

    # Adding labels and title
    plt.ylabel("Average cost", fontsize=size_yxlim)
    plt.xlabel("Ratio between $K_u$ and $K_e$", fontsize=size_yxlim)
    plt.tick_params(axis='both', labelsize=size_tick)  # 添加这行
    plt.xticks(index, labels)
    plt.legend(fontsize=size_label, loc='upper right')
    plt.yticks(np.arange(1.00, 1.12 + 0.02, 0.02))
    plt.ylim(1.00, 1.12)
    # plt.grid(True, axis='y')

    # Save the figure
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("Fig/Fig4.pdf", dpi=300)
