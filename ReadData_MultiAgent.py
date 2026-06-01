import glob
import os
import re

import numpy as np
import pandas as pd


def _seed_from_path(path):
    match = re.search(r"simulation(\d+)_multi_agent_", os.path.basename(path))
    return int(match.group(1)) if match else None


def _value_at_generation(values, generation):
    values = np.asarray(values, dtype=float).reshape(-1)
    if generation >= len(values):
        return float(values[-1])
    return float(values[generation])


def read_multiagent_average_values(
    result_dir="Result_MultiAgent",
    metric_name="BestValue",
    generations=(0, 200, 400, 600, 800, 1000),
    seeds=range(10),
):
    paths = []
    for seed in seeds:
        path = os.path.join(result_dir, f"simulation{seed}_multi_agent_{metric_name}.npy")
        if os.path.exists(path):
            paths.append(path)

    if not paths:
        pattern = os.path.join(result_dir, f"simulation*_multi_agent_{metric_name}.npy")
        paths = sorted(glob.glob(pattern), key=lambda path: _seed_from_path(path))

    if not paths:
        raise FileNotFoundError(f"No MultiAgent files found for metric {metric_name} in {result_dir}")

    rows = []
    for path in paths:
        seed = _seed_from_path(path)
        values = np.load(path)
        row = {"seed": seed}
        for generation in generations:
            row[f"generation_{generation}"] = _value_at_generation(values, generation)
        rows.append(row)

    seed_df = pd.DataFrame(rows).sort_values("seed").reset_index(drop=True)
    average_row = {"seed": "average"}
    for generation in generations:
        average_row[f"generation_{generation}"] = seed_df[f"generation_{generation}"].mean()

    summary_df = pd.concat([seed_df, pd.DataFrame([average_row])], ignore_index=True)
    return seed_df, summary_df


def read_multiagent_metrics_average(result_dir="Result_MultiAgent", metrics_file=None):
    if metrics_file is not None:
        metrics_path = os.path.join(result_dir, metrics_file)
        metrics_df = pd.read_csv(metrics_path)
    else:
        outage_paths = sorted(
            glob.glob(os.path.join(result_dir, "simulation*_multi_agent_OutageValue.npy")),
            key=lambda path: _seed_from_path(path),
        )
        if not outage_paths:
            raise FileNotFoundError(f"No MultiAgent OutageValue files found in {result_dir}")

        scale = int(os.path.basename(result_dir)) if os.path.basename(result_dir).isdigit() else None
        urllc_num = scale // 2 if scale is not None else None
        max_delay_urllc = 2

        rows = []
        for outage_path in outage_paths:
            seed = _seed_from_path(outage_path)
            outage_value = np.load(outage_path)
            best_value_path = os.path.join(result_dir, f"simulation{seed}_multi_agent_BestValue.npy")
            best_value = np.load(best_value_path) if os.path.exists(best_value_path) else np.array([np.nan])

            best_fitness = float(np.asarray(best_value, dtype=float).reshape(-1)[-1])
            row = {
                "seed": seed,
                "urllc_outage_ratio": float(outage_value[1, -1]),
                "average_embb_delay": float(outage_value[0, -1]),
                "average_cost": best_fitness / scale if scale is not None else np.nan,
                "best_fitness": best_fitness,
                "cv": 0.0,
                "cost_urllc": float(outage_value[1, -1]) * urllc_num * max_delay_urllc
                if urllc_num is not None
                else np.nan,
            }
            rows.append(row)

        metrics_df = pd.DataFrame(rows)
        metrics_path = os.path.join(result_dir, "simulation*_multi_agent_OutageValue.npy")

    required_columns = ["seed", "urllc_outage_ratio", "average_embb_delay"]
    missing_columns = [name for name in required_columns if name not in metrics_df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in {metrics_path}: {missing_columns}")

    seed_df = metrics_df[required_columns].copy()
    seed_df = seed_df.sort_values("seed").reset_index(drop=True)

    average_row = {"seed": "average"}
    for column in required_columns[1:]:
        average_row[column] = seed_df[column].mean()

    summary_df = pd.concat([seed_df, pd.DataFrame([average_row])], ignore_index=True)
    return seed_df, summary_df, metrics_path, metrics_df.sort_values("seed").reset_index(drop=True)


def discover_multiagent_result_dirs(root_dir="Result_MultiAgent"):
    scale_dirs = []
    if os.path.isdir(root_dir):
        for name in os.listdir(root_dir):
            path = os.path.join(root_dir, name)
            if os.path.isdir(path) and name.isdigit():
                scale_dirs.append(path)

    if scale_dirs:
        return sorted(scale_dirs, key=lambda path: int(os.path.basename(path)))
    return [root_dir]


def process_multiagent_result_dir(result_dir, metric_name="BestValue"):
    seed_df, summary_df = read_multiagent_average_values(result_dir=result_dir, metric_name=metric_name)
    output_path = os.path.join(result_dir, f"multiagent_{metric_name}_selected_average.csv")
    summary_df.to_csv(output_path, index=False)

    print(f"\n[{result_dir}]")
    print(summary_df.to_string(index=False))
    print(f"Saved selected-generation average to {output_path}")

    try:
        metrics_seed_df, metrics_summary_df, metrics_path, metrics_full_df = read_multiagent_metrics_average(
            result_dir=result_dir
        )
    except FileNotFoundError:
        print(f"No OutageValue files found in {result_dir}, skipped metrics average.")
        return

    metrics_full_path = os.path.join(result_dir, "multiagent_best_metrics.csv")
    metrics_output_path = os.path.join(result_dir, "multiagent_best_metrics_average.csv")
    metrics_full_df.to_csv(metrics_full_path, index=False)
    metrics_summary_df.to_csv(metrics_output_path, index=False)

    print(metrics_summary_df.to_string(index=False))
    print(f"Read best metrics from {metrics_path}")
    print(f"Saved corrected seed metrics to {metrics_full_path}")
    print(f"Saved metrics average to {metrics_output_path}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    root_dir = "Result_MultiAgent"
    metric_name = "BestValue"
    # for result_dir in discover_multiagent_result_dirs(root_dir):
    #     process_multiagent_result_dir(result_dir, metric_name=metric_name)
    result_dir = os.path.join(root_dir, "360")
    process_multiagent_result_dir(result_dir, metric_name=metric_name)
