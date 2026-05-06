import os
from typing import Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    import scienceplots  # noqa: F401

    _HAS_SCIENCEPLOTS = True
except Exception:
    _HAS_SCIENCEPLOTS = False


def _load_points() -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    返回：
    - x: sampled generations
    - data_series: {label: y}

    直接使用固定结果（按你指定的数据）。
    """
    marker_points = np.array([50, 100, 150, 200, 250, 300], dtype=int)

    # 直接使用你给定的结果
    data_series = {
        "LLM+DE": np.array(
            [63.02835594, 62.80734780, 62.60142694, 62.47760328, 62.30646012, 62.23274968],
            dtype=float,
        ),
        "All_DE": np.array(
            [67.15922399, 65.86172224, 65.05058127, 64.61023765, 64.27591154, 64.10266143],
            dtype=float,
        ),
        "All_GA": np.array(
            [65.26584207, 64.85024894, 64.68084578, 64.58241744, 64.49711595, 64.43215879],
            dtype=float,
        ),
        "Random": np.array(
            [71.14623139, 69.66186083, 69.06590281, 68.76637988, 68.59642324, 68.46858721],
            dtype=float,
        ),
        "Closest": np.array(
            [68.10604125, 67.23857399, 66.79130075, 66.55863746, 66.41145091, 66.30465433],
            dtype=float,
        ),
        "Multi_Agent": np.array(
            [64.32919325, 64.02902754, 63.72742207, 63.40718936, 63.25975706, 63.10957061],
            dtype=float,
        ),
    }
    return marker_points, data_series


def main() -> None:
    if _HAS_SCIENCEPLOTS:
        plt.style.use(["science", "ieee"])
        style_ctx = plt.style.context("ieee")
    else:
        # fallback：没有 scienceplots 时给一个接近 IEEE 的观感
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                "mathtext.fontset": "dejavuserif",
                "axes.labelsize": 14,
                "xtick.labelsize": 12,
                "ytick.labelsize": 12,
                "legend.fontsize": 14,
                "axes.linewidth": 0.8,
                "lines.linewidth": 2.0,
                "lines.markersize": 7,
            }
        )
        style_ctx = plt.style.context(plt.rcParams)

    marker_points, data_series = _load_points()

    colors = {
        "LLM+DE": "#0072BD",
        "All_DE": "#7E2F8E",
        "All_GA": "#D95319",
        "Random": "#A2142F",
        "Closest": "#77AC30",
        "Multi_Agent": "#4DBEEE",
    }
    markers = {
        "LLM+DE": "o",
        "All_DE": "s",
        "All_GA": "^",
        "Random": "D",
        "Closest": "v",
        "Multi_Agent": "P",
    }

    size_yxlim = 14
    size_label = 9
    LINEWIDTH = 1
    MARKERSIZE = 5

    os.makedirs("Fig", exist_ok=True)

    with style_ctx:
        plt.figure(figsize=(5.3, 3.8))

        order = ["LLM+DE", "Multi_Agent", "All_DE", "All_GA", "Closest", "Random"]
        for name in order:
            if name not in data_series:
                continue
            y = np.asarray(data_series[name], dtype=float).reshape(-1)
            plt.plot(
                marker_points,
                y/30,
                color=colors.get(name, None),
                label=name,
                marker=markers.get(name, "o"),
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
            )

        plt.ylabel("Average cost", fontsize=size_yxlim)
        plt.xlabel("Number of generations", fontsize=size_yxlim)
        plt.xlim(50, int(marker_points.max()))
        plt.xticks(np.arange(50, int(marker_points.max()) + 50, 50))

        # y 轴范围：按数据自动给一点边距
        all_y = np.hstack([np.asarray(v, dtype=float).reshape(-1) for v in data_series.values()])
        all_y = all_y[np.isfinite(all_y)]
        if all_y.size:
            y_min = float(60/30)
            y_max = float(75/30)
            pad = 0.03 * (y_max - y_min + 1e-9)
            plt.ylim(y_min - pad, y_max + pad)

        plt.legend(fontsize=size_label, loc="best")
        plt.grid()
        # plt.show()
        plt.savefig("Fig/Fig3_sampled_points_ieee.pdf", dpi=300)


if __name__ == "__main__":
    main()

