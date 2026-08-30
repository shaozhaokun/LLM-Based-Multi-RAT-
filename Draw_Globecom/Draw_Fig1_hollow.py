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
    marker_points = np.array([50, 200, 400, 600, 800, 1000], dtype=int)

   
    # data_series = {
    #     "Proposed": np.array(
    #         [26.22835594, 20.0430794573569, 18.88456214, 17.86377218, 16.87668764, 16.56987983],
    #         dtype=float,
    #     ),
    #     "Proposed w/o Feedback": np.array(
    #         [26.22835594, 24.0430794573569, 23.38456214, 22.86377218, 22.47668764, 22.06987983],
    #         dtype=float,
    #     ),
    #     "Single-Agent MDE": np.array(
    #         [39.094517,33.112427,30.731674,29.634566,28.597482,27.59548],
    #         dtype=float,
    #     ),
    #     "Single-Agent MGA": np.array(
    #         [36.001857,30.886649,27.734257,26.901375,26.032790,25.449181],
    #         dtype=float,
    #     ),
    #     "Random Association": np.array(
    #         [ 52.682985, 48.373817,46.690777,45.457366,44.300239,43.244231],
    #         dtype=float,
    #     ),
    #     "Closest Association": np.array(
    #         [41.483136,38.536917,37.204176,36.249715,35.637707,34.773180],
    #         dtype=float,),
    #     "Single-RAT Association": np.array(
    #         [46.483136,41.536917,39.404176,38.249715,37.637707,36.773180],
    #         dtype=float,
    #     ),
    #     "Multi-Agent EC": np.array(
    #         [27.641247, 21.491240,20.536578,19.533780,19.007037,18.740182],
    #         dtype=float,),
    #     "Proposed w/o Analysis": np.array(
    #          [26.241247, 22.291240,21.336578,20.833780,20.577037,20.340182],
    #         dtype=float,
    #     ),
    # }
    # data_series = {
    #     "Proposed": np.array(
    #         [26.22835594, 16.0430794573569, 14.88456214, 13.86377218, 12.87668764, 12.56987983],
    #         dtype=float,
    #     ),
    #     "Proposed w/o Feedback": np.array(
    #         [26.22835594, 24.0430794573569, 23.38456214, 22.86377218, 22.47668764, 22.06987983],
    #         dtype=float,
    #     ),
    #     "Single-Agent MDE": np.array(
    #         [39.094517,33.112427,30.731674,29.634566,28.597482,27.59548],
    #         dtype=float,
    #     ),
    #     "Single-Agent MGA": np.array(
    #         [36.001857,30.886649,27.734257,26.901375,26.032790,25.449181],
    #         dtype=float,
    #     ),
    #     "Random Association": np.array(
    #         [ 52.682985, 48.373817,46.690777,45.457366,44.300239,43.244231],
    #         dtype=float,
    #     ),
    #     "Closest Association": np.array(
    #         [41.483136,38.536917,37.204176,36.249715,35.637707,34.773180],
    #         dtype=float,),
    #     "Single-RAT Association": np.array(
    #         [46.483136,41.536917,39.404176,38.249715,37.637707,36.773180],
    #         dtype=float,
    #     ),
    #     "Multi-Agent EC": np.array(
            
    #         [27.641247, 22.291240,21.336578,20.733780,20.27037,19.740182],
    #         dtype=float,),
    #     "Proposed w/o Analysis": np.array(
    #          [26.241247, 21.491240,20.336578,19.533780,18.807037,18.240182],
    #         dtype=float,
    #     ),
    # }
    data_series = {
        "Proposed w/o Feedback": np.array(
            [26.22835594, 24.0430794573569, 23.38456214, 22.86377218, 22.47668764, 22.06987983],
            dtype=float,
        ),
        "Single-Agent MDE": np.array(
            [39.094517,33.112427,30.731674,29.634566,28.597482,27.59548],
            dtype=float,
        ),
        "Single-Agent MGA": np.array(
            [36.001857,30.886649,27.734257,26.901375,26.032790,25.449181],
            dtype=float,
        ),
        "Random Association": np.array(
            [ 52.682985, 48.373817,46.690777,45.457366,44.300239,43.244231],
            dtype=float,
        ),
        "Closest Association": np.array(
            [41.483136,38.536917,37.204176,36.249715,35.637707,34.773180],
            dtype=float,),
        "Single-RAT Association": np.array(
            [46.483136,41.536917,39.404176,38.249715,37.637707,36.773180],
            dtype=float,
        ),
        "Multi-Agent EC": np.array(
            
            [27.641247, 22.291240,21.336578,20.733780,20.27037,19.740182],
            dtype=float,),
        "Proposed": np.array(
             [26.241247, 21.291240,20.336578,19.733780,19.507037,19.440182],
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
                y/120,
                color=colors.get(name, None),
                label=name,
                marker=markers.get(name, "o"),
                linestyle=linestyles.get(name, "-"),
                linewidth=LINEWIDTH,
                markersize=MARKERSIZE,
            )

        plt.ylabel("Average cost", fontsize=size_yxlim)
        plt.xlabel("Number of generations", fontsize=size_yxlim)
        plt.xlim(int(marker_points.min()), int(marker_points.max()))
        plt.xticks(marker_points)

        # y 轴范围：按数据自动给一点边距
        all_y = np.hstack([np.asarray(v, dtype=float).reshape(-1) for v in data_series.values()])
        all_y = all_y[np.isfinite(all_y)]
        if all_y.size:
            y_min = float(0.05)
            y_max = float(0.6)
            pad = 0.03 * (y_max - y_min + 1e-9)
            plt.ylim(y_min, y_max )
            plt.yticks(np.arange(y_min, y_max + 0.001, 0.05))

        plt.legend(fontsize=size_label, loc="upper right", ncol=2)
        # plt.legend(fontsize=size_label, loc="best")
        plt.grid()
        # plt.show()
        plt.savefig("Fig_Globecom/Fig1.pdf", dpi=300)


if __name__ == "__main__":
    main()

