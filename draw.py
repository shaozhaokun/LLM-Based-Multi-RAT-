
import numpy as np
import matplotlib.pyplot as plt

save_path = "Result/fitness_generation_best.png"  # 不想保存就改成 None
title = "Fitness Valueunder Fixed Outer Association"

# 注意：np.load 只能读 .npy/.npz；CSV/纯文本请用 np.loadtxt
start_gen = 30
end_gen = 300

curves = []
for seed in range(10):
    csv_path = "Result/fitness_generation_best_seed{}.csv".format(seed)
    data = np.loadtxt(csv_path, delimiter=",", dtype=float)
    data = np.asarray(data, dtype=float).reshape(-1)
    end = min(end_gen, data.shape[0])
    curves.append(data[start_gen:end])

# 对齐长度（用最短的那条，避免不同 seed 长度不一致）
min_len = min(c.shape[0] for c in curves)
curves = np.vstack([c[:min_len] for c in curves])  # (num_seed, T)

mean_curve = np.mean(curves, axis=0)
std_curve = np.std(curves, axis=0)
x = np.arange(start_gen, start_gen + min_len)

plt.figure(figsize=(7, 5))
plt.plot(x, mean_curve, label=f"{title} (mean)")
# plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2, label="±1 std")



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