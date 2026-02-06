
import numpy as np
import matplotlib.pyplot as plt

csv_path = "Result/fitness_generation_best.csv"
save_path = "Result/fitness_generation_best.png"  # 不想保存就改成 None
title = "Fitness Valueunder Fixed Outer Association"

# 注意：np.load 只能读 .npy/.npz；CSV/纯文本请用 np.loadtxt
data = np.loadtxt(csv_path, delimiter=",", dtype=float)
data = np.asarray(data, dtype=float)

plt.figure(figsize=(7, 5))

plt.plot(np.arange(30,300),data[30:300], label=title)



plt.title(title)
plt.xlabel("Inner Generation")
plt.ylabel("Fitness Value")
plt.ylim(20, 35)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
if save_path is not None:
    plt.savefig(save_path, dpi=200)
plt.show()