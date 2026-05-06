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





user_num_LLM = [1.0372, 1.0527,1.06794,1.0879, 1.1101]
user_num_multi_RAT = [1.051825, 1.06059, 1.07544, 1.09500, 1.1148]
user_num_GA = [1.0788 , 1.08585 ,1.10055 ,1.1198, 1.1406 ]
user_num_DE = [1.0653 , 1.07278 ,1.08711 ,1.1066 ,1.1271] 



user_num_xrange = np.arange(60, 180 + 30, 30)




with plt.style.context("ieee"):
    plt.figure(figsize=(3.5, 3.0))
    plt.plot(user_num_xrange, user_num_LLM, color="#0072B2",            label="Proposed", marker=markers[3])
    plt.plot(user_num_xrange, user_num_multi_RAT, color="#009E73", label="Multi-Agent EC", marker=markers[0])
    plt.plot(user_num_xrange, user_num_GA, color="#C0392B",    label="Single-Agent MGA", marker=markers[1])
    plt.plot(user_num_xrange, user_num_DE, color="#D55E00",       label="Single-Agent MDE", marker=markers[2])


    
    plt.ylabel("Average cost",fontsize=size_yxlim)
    plt.xlabel("Number of users",fontsize=size_yxlim)
    plt.xlim(60, 180)
    plt.xticks(np.arange(60, 180 + 30, 30))
    bottom, top = plt.ylim()
    plt.yticks(np.arange(1.02, 1.16 + 0.02, 0.02))
    plt.ylim(1.02, 1.16)
    plt.legend(fontsize=size_label,loc='upper left')
    plt.tick_params(axis='both', labelsize=size_tick)  # 添加这行
    plt.grid()
    # plt.show()

    plt.tight_layout()
    plt.savefig("Fig/Fig2.pdf", dpi=300)