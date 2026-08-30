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
# size_yxlim = 12
# size_label = 10
# size_tick = 8

size_yxlim = 20
size_label = 15
size_tick = 14





# user_num_LLM = [0.038, 0.1721,0.305,0.423,0.493]
user_num_LLM = [0.05, 0.1721,0.305,0.423,0.493]
user_num_multi_RAT = [0.07166, 0.208, 0.353333, 0.491333, 0.586667]
user_num_GA = [0.09666 , 0.225667 ,0.372 ,0.505333, 0.597778 ]
user_num_DE = [0.138, 0.315556,0.474167 ,0.606000 ,0.692222] 



user_num_xrange = np.arange(120, 360 + 60, 60)




with plt.style.context("ieee"):
    plt.figure(figsize=(5.3, 3.8))

    # plt.figure(figsize=(3.5, 3.0))
    # plt.plot(user_num_xrange, user_num_LLM, color="#0072B2",            label="Proposed", marker=markers[3])
    # plt.plot(user_num_xrange, user_num_multi_RAT, color="#009E73", label="Multi-Agent EC", marker=markers[0])
    # plt.plot(user_num_xrange, user_num_GA, color="#C0392B",    label="Single-Agent MGA", marker=markers[1])
    # plt.plot(user_num_xrange, user_num_DE, color="#D55E00",       label="Single-Agent MDE", marker=markers[2])
    
    plt.plot(user_num_xrange, user_num_LLM, color="#0072B2", linestyle="-", label="Proposed",
             marker="^", markerfacecolor="none", markeredgecolor="#0072B2", markeredgewidth=1.0)
    plt.plot(user_num_xrange, user_num_multi_RAT, color="#009E73", linestyle="-", label="Multi-Agent EC",
             marker="o", markerfacecolor="none", markeredgecolor="#009E73", markeredgewidth=1.0)
    plt.plot(user_num_xrange, user_num_GA, color="#C0392B", linestyle="-", label="Single-Agent MGA",
             marker="s", markerfacecolor="none", markeredgecolor="#C0392B", markeredgewidth=1.0)
    plt.plot(user_num_xrange, user_num_DE, color="#D55E00", linestyle="-", label="Single-Agent MDE",
             marker="D", markerfacecolor="none", markeredgecolor="#D55E00", markeredgewidth=1.0)


    
    plt.ylabel("Outage ratio of task (u,k)",fontsize=size_yxlim)
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
    axins.set_ylim(0.04, 0.34)
    axins.set_xticks([120, 180])
    axins.set_yticks([0.10, 0.20, 0.30])
    axins.tick_params(axis="both", labelsize=6)
    axins.grid(True, alpha=0.3)
    # ax.add_patch(Ellipse((150, 0.18), width=55, height=0.065, fill=False, edgecolor="black", linewidth=0.45))
    ax.annotate(
        "",
           xy=(210, 0.18),
        xycoords="data",
        xytext=(260, 0.2),
        textcoords="data",
        arrowprops=dict(arrowstyle="->", color="black", linewidth=0.8),
    )
    # plt.show()

    plt.tight_layout()
    plt.savefig("Fig_journal/Fig2_urllc_2.pdf", dpi=300)
