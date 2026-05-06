
import os
import re
import numpy as np
import matplotlib.pyplot as plt

# 画“所有结果”在同一张图上：
# - CSV: Result*/fitness_generation_best_seed{seed}.csv
# - Multi-agent: Result_MultiAgent/simulation{seed}_multi_agent_BestValue.npy（画 best value）
save_path = "Result/fitness_generation_best_compare_all_2.pdf"  # 不想保存就改成 None

start_gen = 50
end_gen = 500
num_seed = 10
show_std = False  # True: 画 ±1 std 阴影

# 若你想固定 y 轴范围，填 (ymin, ymax)；否则设为 None 自动缩放
Y_LIM = None

# 视觉参数
LINEWIDTH = 2.0
MARKERSIZE = 7
LEGEND_FONTSIZE = 10
LABEL_FONTSIZE = 11
TICK_FONTSIZE = 10

# 想要抽取/画出来的“代数点”（按原始数组下标=代数取值）
sample_gens = np.array([50, 100,150, 200, 250, 300], dtype=int)
save_points_dir = "Result/points"  # 保存抽样点数组的目录


def _load_curve(series_dir: str, seed: int) -> np.ndarray:
    """
    返回 1D curve：
    - 对 CSV 系列：load fitness_generation_best_seed{seed}.csv
    - 对 MultiAgent 系列：load simulation{seed}_multi_agent_BestValue.npy
    """
    if series_dir == "Result_MultiAgent":
        npy_path = os.path.join(series_dir, f"simulation{seed}_multi_agent_BestValue.npy")
        data = np.load(npy_path, allow_pickle=True)
        data = np.asarray(data, dtype=float).reshape(-1)
        return data
    else:
        csv_path = os.path.join(series_dir, f"fitness_generation_best_seed{seed}.csv")
        data = np.loadtxt(csv_path, delimiter=",", dtype=float)
        data = np.asarray(data, dtype=float).reshape(-1)
        return data


series = [
    {"dir": "Result", "label": "LLM+DE"},
    {"dir": "Result_DE", "label": "All_DE"},
    {"dir": "Result_GA", "label": "All_GA"},
    {"dir": "Result_random", "label": "Random"},
    {"dir": "Result_Closest", "label": "Closest"},
    {"dir": "Result_MultiAgent", "label": "Multi_Agent"},
]

plt.figure(figsize=(7, 5))
points_summary = []  # [(label, mean_points, std_points)]

for s in series:
    curves = []
    points_seed = []  # (num_seed, len(sample_gens))
    for seed in range(num_seed):
        data = _load_curve(s["dir"], seed)

        # 抽取离散点；越界则填 nan
        p = np.full(sample_gens.shape[0], np.nan, dtype=float)
        valid = (sample_gens >= 0) & (sample_gens < data.shape[0])
        p[valid] = data[sample_gens[valid]]
        points_seed.append(p)

        end = min(end_gen, data.shape[0])
        curves.append(data[start_gen:end])

    # 对齐长度（用最短的那条，避免不同 seed 长度不一致）
    min_len = min(c.shape[0] for c in curves)
    curves = np.vstack([c[:min_len] for c in curves])  # (num_seed, T)

    mean_curve = np.mean(curves, axis=0)
    std_curve = np.std(curves, axis=0)
    x = np.arange(start_gen, start_gen + min_len)

    # 保存/打印抽样点（按 seed 聚合后取均值/方差）
    os.makedirs(save_points_dir, exist_ok=True)
    points_seed = np.asarray(points_seed, dtype=float)  # (num_seed, P)
    mean_points = np.nanmean(points_seed, axis=0)
    std_points = np.nanstd(points_seed, axis=0)
    np.save(os.path.join(save_points_dir, f"points_seed_{s['dir']}.npy"), points_seed)
    np.save(os.path.join(save_points_dir, f"points_mean_{s['dir']}.npy"), mean_points)
    np.savetxt(os.path.join(save_points_dir, f"points_mean_{s['dir']}.csv"), mean_points.reshape(1, -1), delimiter=",")
    points_summary.append((s["label"], mean_points, std_points))
    print(f"{s['label']} sampled points @ {sample_gens.tolist()} = {mean_points}")

    plt.plot(x, mean_curve, label=s["label"], linewidth=LINEWIDTH)
    if show_std:
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)

# plt.title(title)
plt.xlabel("Inner Generation", fontsize=LABEL_FONTSIZE)
plt.ylabel("Cost Value", fontsize=LABEL_FONTSIZE)
if Y_LIM is not None:
    plt.ylim(*Y_LIM)
plt.xlim(50, 300)

# plt.xlim(0,300)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.tight_layout()
if save_path is not None:
    plt.savefig(save_path, dpi=200)
plt.show()


# 画离散点对比图（均值；show_std=True 时加误差棒）
plt.figure(figsize=(7, 4.5))
for label, mean_points, std_points in points_summary:
    if show_std:
        plt.errorbar(
            sample_gens,
            mean_points,
            yerr=std_points,
            marker="o",
            markersize=MARKERSIZE,
            capsize=4,
            linewidth=LINEWIDTH,
            label=label,
        )
    else:
        plt.plot(sample_gens, mean_points/30, marker="o", markersize=MARKERSIZE, linewidth=LINEWIDTH, label=label)
plt.xlabel("Inner Generation", fontsize=LABEL_FONTSIZE)
plt.ylabel("Cost Value", fontsize=LABEL_FONTSIZE)
plt.grid(True, alpha=0.3)
plt.xticks(fontsize=TICK_FONTSIZE)
plt.xlim(50, 300)
plt.yticks(fontsize=TICK_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)
plt.tight_layout()
plt.savefig("Result/fitness_generation_sample_points_compare_all.pdf", dpi=200)
plt.show()