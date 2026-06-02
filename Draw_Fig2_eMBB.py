import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
except Exception:
    pass


# # ### Fig 1

markers = ['.', '<', 'o', '*', '1', '^', 'P', 'p']
size_yxlim = 12
size_label = 10
size_tick = 8





user_num_LLM = [0.1579, 0.188,0.276,0.351, 0.427]
user_num_multi_RAT = [0.186, 0.2443, 0.355485, 0.458317, 0.579575]
user_num_GA = [0.232 , 0.311163 ,0.431 ,0.563873, 0.733068 ]
user_num_DE = [0.173 , 0.215683 ,0.31822 ,0.420408 ,0.532089] 



user_num_xrange = np.arange(120, 360 + 60, 60)




with plt.style.context("ieee"):
    plt.figure(figsize=(5.3, 3.8))

    # plt.figure(figsize=(3.5, 3.0))
    # plt.plot(user_num_xrange, user_num_LLM, color="#0072B2", linestyle="-", label="Proposed", marker=markers[3])
    # plt.plot(user_num_xrange, user_num_multi_RAT, color="#009E73", linestyle="-", label="Multi-Agent EC", marker=markers[0])
    # plt.plot(user_num_xrange, user_num_GA, color="#C0392B", linestyle="-", label="Single-Agent MGA", marker=markers[1])
    # plt.plot(user_num_xrange, user_num_DE, color="#D55E00", linestyle="-", label="Single-Agent MDE", marker=markers[2])

    plt.plot(user_num_xrange, user_num_LLM, color="#0072B2", linestyle="-", label="Proposed",
             marker="^", markerfacecolor="none", markeredgecolor="#0072B2", markeredgewidth=1.0)
    plt.plot(user_num_xrange, user_num_multi_RAT, color="#009E73", linestyle="-", label="Multi-Agent EC",
             marker="o", markerfacecolor="none", markeredgecolor="#009E73", markeredgewidth=1.0)
    plt.plot(user_num_xrange, user_num_GA, color="#C0392B", linestyle="-", label="Single-Agent MGA",
             marker="s", markerfacecolor="none", markeredgecolor="#C0392B", markeredgewidth=1.0)
    plt.plot(user_num_xrange, user_num_DE, color="#D55E00", linestyle="-", label="Single-Agent MDE",
             marker="D", markerfacecolor="none", markeredgecolor="#D55E00", markeredgewidth=1.0)


    
    plt.ylabel("Average delay of delay-tolerant user",fontsize=size_yxlim)
    plt.xlabel("Number of users",fontsize=size_yxlim)
    plt.xlim(120, 360)
    plt.xticks(np.arange(120, 360 + 60, 60))
    bottom, top = plt.ylim()
    plt.yticks(np.arange(0, 0.9 + 0.1, 0.1))
    plt.ylim(0, 0.9)
    plt.legend(fontsize=size_label,loc='upper left')
    plt.tick_params(axis='both', labelsize=size_tick)  # 添加这行
    plt.grid()

    ax = plt.gca()
    axins = inset_axes(ax, width="30%", height="28%", loc="lower right", borderpad=1.0)
    axins.plot(user_num_xrange, user_num_LLM, color="#0072B2", linestyle="-",
               marker="^", markerfacecolor="none", markeredgecolor="#0072B2", markeredgewidth=1.0)
    axins.plot(user_num_xrange, user_num_multi_RAT, color="#009E73", linestyle="-",
               marker="o", markerfacecolor="none", markeredgecolor="#009E73", markeredgewidth=1.0)
    axins.plot(user_num_xrange, user_num_GA, color="#C0392B", linestyle="-",
               marker="s", markerfacecolor="none", markeredgecolor="#C0392B", markeredgewidth=1.0)
    axins.plot(user_num_xrange, user_num_DE, color="#D55E00", linestyle="-",
               marker="D", markerfacecolor="none", markeredgecolor="#D55E00", markeredgewidth=1.0)
    axins.set_xlim(115, 185)
    axins.set_ylim(0.14, 0.32)
    axins.set_xticks([120, 180])
    axins.set_yticks([0.16, 0.24, 0.32])
    axins.tick_params(axis="both", labelsize=6)
    axins.grid(True, alpha=0.3)
    # ax.add_patch(Ellipse((150, 0.23), width=55, height=0.05, fill=False, edgecolor="black", linewidth=0.45))
    ax.annotate(
        "",
        xy=(210, 0.2),
        xycoords="data",
        xytext=(260, 0.15),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="black", linewidth=0.9),
    )
    # plt.show()

    plt.tight_layout()
    plt.savefig("Fig_journal/Fig2_eMBB_2.pdf", dpi=300)
