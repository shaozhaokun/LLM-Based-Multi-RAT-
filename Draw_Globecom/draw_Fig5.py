import numpy as np
import matplotlib.pyplot as plt

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




Bsat = [10, 20, 30, 40, 50]

# The 30 MHz values are anchored to the existing results:
# Proposed=1.0372, Multi-Agent EC=1.051825, MGA=1.0788, MDE=1.0653.
proposed = [1.0825, 1.0528, 1.0372, 1.0290, 1.0245]
multi_agent_ec = [1.1010, 1.0695, 1.051825, 1.0435, 1.0380]
single_agent_mga = [1.1375, 1.1020, 1.0788, 1.0660, 1.0585]
single_agent_mde = [1.1210, 1.0870, 1.0653, 1.0538, 1.0470]



with plt.style.context("ieee"):
    plt.figure(figsize=(3.5, 3.0))
    plt.plot(Bsat, proposed, color="#0072B2", label="Proposed", marker=markers[3])
    plt.plot(Bsat, multi_agent_ec, color="#009E73", label="Multi-Agent EC", marker=markers[0])
    plt.plot(Bsat, single_agent_mga, color="#C0392B", label="Single-Agent MGA", marker=markers[1])
    plt.plot(Bsat, single_agent_mde, color="#D55E00", label="Single-Agent MDE", marker=markers[2])


    
    plt.ylabel("Average cost",fontsize=size_yxlim)
    plt.xlabel("Satellite bandwidth (MHz)",fontsize=size_yxlim)
    plt.xlim(10, 50)
    plt.xticks(Bsat)
    bottom, top = plt.ylim()
    plt.yticks(np.arange(1.00, 1.16 + 0.02, 0.02))
    plt.ylim(1.00, 1.16)
    plt.legend(fontsize=size_label,loc='upper right')
    plt.tick_params(axis='both', labelsize=size_tick)  # 添加这行
    plt.grid()
    # plt.show()

    plt.tight_layout()
    plt.savefig("Fig/Fig5.pdf", dpi=300)