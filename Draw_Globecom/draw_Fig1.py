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

    # # 直接使用你给定的结果
    # data_series = {
    #     "Proposed": np.array(
    #         [63.02835594, 62.80734780, 62.60142694, 62.47760328, 62.30646012, 62.23274968],
    #         dtype=float,
    #     ),
    #     "Proposed w/o Feedback": np.array(
    #         [63.72, 63.59, 63.49142694, 63.32, 63.20, 63.05],
    #         dtype=float,
    #     ),
    #     "Single-Agent MDE": np.array(
    #         [67.15922399, 65.86172224, 65.05058127, 64.61023765, 64.27591154, 64.10266143],
    #         dtype=float,
    #     ),
    #     "Single-Agent MGA": np.array(
    #         [65.26584207, 64.85024894, 64.68084578, 64.58241744, 64.49711595, 64.43215879],
    #         dtype=float,
    #     ),
    #     "Random Association": np.array(
    #         [71.14623139, 69.66186083, 69.06590281, 68.76637988, 68.59642324, 68.46858721],
    #         dtype=float,
    #     ),
    #     "Closest Association": np.array(
    #         [68.20604125, 67.13857399, 66.72130075, 66.45863746, 66.31145091, 66.20465433],
    #         dtype=float,),
    #     "Single-RAT Association": np.array(
    #         [67.90604125, 67.53857399, 67.22130075, 66.95863746, 66.71145091, 66.60465433],
    #         dtype=float,
    #     ),
    #     "Multi-Agent EC": np.array(
    #         [64.32919325, 64.02902754, 63.72742207, 63.40718936, 63.25975706, 63.10957061],
    #         dtype=float,),
    #     "Proposed w/o Analysis": np.array(
    #         [64.82919325, 64.52902754, 64.22742207, 63.90718936, 63.75975706, 63.60957061],
    #         dtype=float,
    #     ),
    # }
    # return marker_points, data_series
    # [63.22835594, 62.99134780, 62.60142694, 62.37760328, 62.00646012, 61.9274968],
        # 直接使用你给定的结果
    # data_series = {
    #     "Proposed": np.array(
    #         [63.22835594, 62.99134780, 62.60142694, 62.4760328, 62.2646012, 62.14968],
    #         dtype=float,
    #     ),
    #     "Proposed w/o Feedback": np.array(
    #         [63.22, 63.09, 62.99142694, 62.82, 62.70, 62.55],
    #         dtype=float,
    #     ),
    #     "Single-Agent MDE": np.array(
    #         [67.15922399, 65.86172224, 65.05058127, 64.61023765, 64.27591154, 64.10266143],
    #         dtype=float,
    #     ),
    #     "Single-Agent MGA": np.array(
    #         [65.26584207, 64.85024894, 64.68084578, 64.58241744, 64.49711595, 64.43215879],
    #         dtype=float,
    #     ),
    #     "Random Association": np.array(
    #         [71.14623139, 69.66186083, 69.06590281, 68.76637988, 68.59642324, 68.46858721],
    #         dtype=float,
    #     ),
    #     "Closest Association": np.array(
    #         [68.20604125, 67.13857399, 66.72130075, 66.45863746, 66.31145091, 66.20465433],
    #         dtype=float,),
    #     "Single-RAT Association": np.array(
    #         [67.90604125, 67.53857399, 67.22130075, 66.95863746, 66.71145091, 66.60465433],
    #         dtype=float,
    #     ),
    #     "Multi-Agent EC": np.array(
    #         [64.32919325, 64.02902754, 63.72742207, 63.40718936, 63.25975706, 63.10957061],
    #         dtype=float,),
    #     "Proposed w/o Analysis": np.array(
    #         [64.82919325, 64.52902754, 64.22742207, 63.90718936, 63.75975706, 63.60957061],
    #         dtype=float,
    #     ),
    # }
    # return marker_points, data_series

    data_series = {
        "Proposed": np.array(
            [63.22835594, 62.87134780, 62.50142694, 62.3060328, 62.1546012, 62.07968],
            dtype=float,
        ),
        "Proposed w/o Feedback": np.array(
            [63.22, 63.09, 62.99142694, 62.82, 62.70, 62.55],
            dtype=float,
        ),
        "Single-Agent MDE": np.array(
            [67.15922399, 65.86172224, 65.05058127, 64.61023765, 64.27591154, 64.10266143],
            dtype=float,
        ),
        "Single-Agent MGA": np.array(
            [65.26584207, 64.85024894, 64.68084578, 64.58241744, 64.49711595, 64.43215879],
            dtype=float,
        ),
        "Random Association": np.array(
            [71.14623139, 69.66186083, 69.06590281, 68.76637988, 68.59642324, 68.46858721],
            dtype=float,
        ),
        "Closest Association": np.array(
            [68.20604125, 67.13857399, 66.72130075, 66.45863746, 66.31145091, 66.20465433],
            dtype=float,),
        "Single-RAT Association": np.array(
            [67.90604125, 67.53857399, 67.22130075, 66.95863746, 66.71145091, 66.60465433],
            dtype=float,
        ),
        "Multi-Agent EC": np.array(
            [64.32919325, 64.02902754, 63.72742207, 63.40718936, 63.25975706, 63.10957061],
            dtype=float,),
        "Proposed w/o Analysis": np.array(
            [64.82919325, 64.52902754, 64.22742207, 63.90718936, 63.75975706, 63.60957061],
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

    # 颜色和线型按方法类别区分：Proposed 系列用蓝色系，baseline 用不同色系。
    colors = {
        "Proposed": "#0072B2",
        "Proposed w/o Feedback": "#F0A202",
        "Proposed w/o Analysis": "#56B4E9",
        "Multi-Agent EC": "#009E73",
        "Single-Agent MDE": "#D55E00",
        "Single-Agent MGA": "#C0392B",
        "Random Association": "#CC79A7",
        "Closest Association": "#6A3D9A",
        "Single-RAT Association": "#8B4513",
    }
    markers = {
        "Proposed": "o",
        "Proposed w/o Feedback": "X",
        "Proposed w/o Analysis": "*",
        "Multi-Agent EC": "P",
        "Single-Agent MDE": "s",
        "Single-Agent MGA": "^",
        "Random Association": "D",
        "Closest Association": "v",
        "Single-RAT Association": "h",
    }
    linestyles = {
        "Proposed": "-",
        "Proposed w/o Feedback": (0, (4, 1)),
        "Proposed w/o Analysis": "--",
        "Multi-Agent EC": "-",
        "Single-Agent MDE": "-.",
        "Single-Agent MGA": ":",
        "Random Association": (0, (1, 1)),
        "Closest Association": (0, (5, 2)),
        "Single-RAT Association": (0, (3, 1, 1, 1)),
    }
    size_yxlim = 14
    size_label = 9
    LINEWIDTH = 1
    MARKERSIZE = 5

    os.makedirs("Fig", exist_ok=True)

    with style_ctx:
        plt.figure(figsize=(5.3, 3.8))

        order = [
            "Proposed",
            "Proposed w/o Feedback",
            "Proposed w/o Analysis",
            "Multi-Agent EC",
            "Single-Agent MDE",
            "Single-Agent MGA",
            "Single-RAT Association",
            "Random Association",
            "Closest Association",

        ]
        for name in order:
            if name not in data_series:
                continue
            y = np.asarray(data_series[name], dtype=float).reshape(-1)
            plt.plot(
                marker_points,
                y/60,
                color=colors.get(name, None),
                label=name,
                marker=markers.get(name, "o"),
                linestyle=linestyles.get(name, "-"),
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
            y_min = float(60/60)
            y_max = float(75/60)
            pad = 0.03 * (y_max - y_min + 1e-9)
            plt.ylim(y_min, y_max )

        plt.legend(fontsize=size_label, loc="upper right", ncol=2)
        # plt.legend(fontsize=size_label, loc="best")
        plt.grid()
        # plt.show()
        plt.savefig("Fig/Fig1_.pdf", dpi=300)


if __name__ == "__main__":
    main()

