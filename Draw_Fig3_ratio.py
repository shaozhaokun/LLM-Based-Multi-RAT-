import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
except Exception:
    pass


labels = ['1:3', '1:1', '3:1']
llm_de = np.array([17.236, 12.56, 37.2564], dtype=float)
multi = np.array([18.641903, 19.7, 47.202579], dtype=float)
all_de = np.array([19.256576, 26.94548, 61.598774], dtype=float)
all_ga = np.array([18.523646, 25.449181, 59.259615], dtype=float)

size_yxlim = 12
size_label = 9
size_tick = 8

# Set width of bars and positions
x = np.arange(len(labels))
width = 0.18

fig, ax = plt.subplots(figsize=(3.5, 3.0))

# Plot bars
rects1 = ax.bar(x - width*1.5, llm_de / 120, width, label='Proposed', color="#0072B2")
rects2 = ax.bar(x - width/2, multi / 120, width, label='Multi-Agent EC', color="#009E73")
rects3 = ax.bar(x + width/2, all_ga / 120, width, label='Single-Agent MGA', color="#E69F00")
rects4 = ax.bar(x + width*1.5, all_de / 120, width, label='Single-Agent MDE', color="#D55E00")


# Add labels and title
ax.set_ylabel('Average Cost', fontsize=size_yxlim)
ax.set_xlabel('Ratio between $K_u$ and $K_e$', fontsize=size_yxlim)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.tick_params(axis='both', labelsize=size_tick)
ax.set_yticks(np.arange(0, 0.55 + 0.001, 0.05))
ax.set_ylim(0, 0.55)
ax.legend(fontsize=size_label, loc='upper left')
ax.grid(True, axis="y", alpha=0.3)

# Adjust layout and display
plt.tight_layout()
# plt.show()
plt.savefig("Fig_journal/Fig3_ratio.pdf", dpi=300)
