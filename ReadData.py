import glob
import os
import re

import numpy as np
import pandas as pd


def _seed_from_path(path):
    match = re.search(r"seed(\d+)\.csv$", os.path.basename(path))
    return int(match.group(1)) if match else None


def _value_at_generation(values, generation):
    values = np.asarray(values, dtype=float).reshape(-1)
    if generation >= len(values):
        return float(values[-1])
    return float(values[generation])


def read_average_values(result_dir="Result_Pure_GA/300", generations=(0, 200, 400, 600, 800, 1000)):
    pattern = os.path.join(result_dir, "fitness_generation_best_seed*.csv")
    paths = sorted(glob.glob(pattern), key=lambda path: _seed_from_path(path))
    if not paths:
        raise FileNotFoundError(f"No seed CSV files found: {pattern}")

    rows = []
    for path in paths:
        seed = _seed_from_path(path)
        values = np.loadtxt(path, delimiter=",")
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


def read_metrics_average(result_dir="Result_Pure_GA/300", metrics_file=None):
    if metrics_file is None:
        paths = sorted(glob.glob(os.path.join(result_dir, "*best_metrics.csv")))
        if not paths:
            raise FileNotFoundError(f"No best metrics CSV file found: {os.path.join(result_dir, '*best_metrics.csv')}")
        metrics_path = paths[0]
    else:
        metrics_path = os.path.join(result_dir, metrics_file)

    metrics_df = pd.read_csv(metrics_path)
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
    return seed_df, summary_df, metrics_path


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # result_dir = "Result_Pure_GA/180"
    result_dir = "Result_hybrid/hybrid1(30_30_0)/Result_Pure_GA"


    seed_df, summary_df = read_average_values(result_dir)
    output_path = os.path.join(result_dir, "fitness_generation_selected_average.csv")
    summary_df.to_csv(output_path, index=False)

    metrics_seed_df, metrics_summary_df, metrics_path = read_metrics_average(result_dir)
    metrics_output_path = os.path.join(result_dir, "best_metrics_average.csv")
    metrics_summary_df.to_csv(metrics_output_path, index=False)

    print(summary_df.to_string(index=False))
    print(f"Saved selected-generation average to {output_path}")
    print(metrics_summary_df.to_string(index=False))
    print(f"Read best metrics from {metrics_path}")
    print(f"Saved metrics average to {metrics_output_path}")
