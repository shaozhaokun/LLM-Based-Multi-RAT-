import os
import numpy as np
import matplotlib.pyplot as plt

# 尽量使用 scienceplots 的 ieee 风格；如果环境里没有也能跑
try:
    import scienceplots  # noqa: F401
    plt.style.use(["science", "ieee"])
except Exception:
    pass


# x 轴采样代数点（你之前抽取的 50/100/200/300/400/500）
marker_points = np.array([50, 100, 200, 300, 400, 500], dtype=int)

# 如果你已经运行过 draw.py，会生成这些 mean 点文件；这里优先读文件，避免手抄数值
points_dir = os.path.join("Result", "points")
mean_files = {
    "LLM+DE": os.path.join(points_dir, "points_mean_Result.npy"),
    "All_DE": os.path.join(points_dir, "points_mean_Result_DE.npy"),
    "All_GA": os.path.join(points_dir, "points_mean_Result_GA.npy"),
    "Random_DE": os.path.join(points_dir, "points_mean_Result_random.npy"),
}

data_series = {}
for name, path in mean_files.items():
    if os.path.exists(path):
        y = np.load(path).reshape(-1).astype(float)
        data_series[name] = y

# 如果文件不存在，就用这里的 fallback（可选）
if not data_series:
    data_series = {
        "LLM+DE": np.array([27.24988642, 26.83072281, 26.59879387, 26.53770505, 26.50936294, np.nan]),
        "All_DE": np.array([29.28715388, 28.45973597, 28.16973770, 28.11055304, 28.08331594, np.nan]),
        "All_GA": np.array([29.79175140, 29.35006086, 29.23621077, 29.20334357, 29.17295315, np.nan]),
        "Random_DE": np.array([28.96937572, 28.46337876, 28.31552999, 28.26957529, 28.25884540, np.nan]),
    }

markers = ["o", "s", "^", "D", "v", "P", "X"]
colors = {
    "LLM+DE": "#0072BD",
    "All_DE": "#7E2F8E",
    "All_GA": "#D95319",
    "Random_DE": "#A2142F",
}

size_yxlim = 14
size_label = 14
LINEWIDTH = 2.0
MARKERSIZE = 7

os.makedirs("Fig", exist_ok=True)

with plt.style.context(plt.rcParams):
    # 画布加大一点（更适合截图/论文排版时看清 marker 和文字）
    plt.figure(figsize=(7.2, 5.2))

    # 画线：如果最后一个点是 nan（例如取不到 500 代），matplotlib 会自动断开
    for idx, (name, y) in enumerate(data_series.items()):
        plt.plot(
            marker_points,
            y,
            color=colors.get(name, None),
            label=name,
            marker=markers[idx % len(markers)],
            linewidth=LINEWIDTH,
            markersize=MARKERSIZE,
        )

    plt.ylabel("Cost Value", fontsize=size_yxlim + 1)
    plt.xlabel("Number of generations", fontsize=size_yxlim + 1)
    plt.xlim(50, int(marker_points.max()))
    plt.xticks(np.arange(50, int(marker_points.max()) + 1, 100))

    # y 轴范围：按数据自动给一点边距（忽略 nan）
    all_y = np.hstack([np.asarray(v, dtype=float) for v in data_series.values()])
    all_y = all_y[np.isfinite(all_y)]
    if all_y.size:
        y_min = float(np.min(all_y))
        y_max = float(np.max(all_y)+1)
        pad = 0.03 * (y_max - y_min + 1e-9)
        plt.ylim(y_min - pad, y_max + pad)

    plt.legend(fontsize=size_label + 1, loc="upper right", markerscale=1.2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("Fig/Fig_sample_points.pdf", dpi=300, bbox_inches="tight")
    # plt.show()