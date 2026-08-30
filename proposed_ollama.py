"""Run the proposed inner EC with Ollama-generated outer associations.

Examples
--------
Run the complete two-round proposed method::

    python proposed_ollama.py --rounds 2 --model qwen3.5:35b

Run only the EC evaluation for an already generated outer iteration::

    python proposed_ollama.py --iteration 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from build_outer_from_offloading_decision import build_outer_from_offloading_decision
from main_improved import (
    MyproblemInner,
    _append_outer_iteration_metrics,
    _write_diagnostic_csvs,
    _write_fitness_result,
)
from ollama_outer_optimizer import (
    POOL_ROOT,
    SOLUTION_ROOT,
    configure_scale,
    count_initial_prompt_tokens,
    generate_outer_association,
)


BASE_DIR = Path(__file__).resolve().parent
SIXG_NUM = 2
WIFI_NUM = 4
SAT_NUM = 2
RAT_NUM = SIXG_NUM + WIFI_NUM + SAT_NUM
RAT_LIST = np.array([SIXG_NUM, WIFI_NUM, SAT_NUM, SAT_NUM])


def _scalar(value) -> float:
    return float(np.asarray(value).reshape(-1)[0])


def run_inner_ec(
    iteration: int,
    seed: int = 42,
    generations: int | None = None,
    population_size: int | None = None,
    users_per_service: int = 150,
) -> dict:
    """Evaluate one Ollama-generated outer association with the proposed inner EC."""
    configure_scale(users_per_service)
    if users_per_service <= 0 or users_per_service % 3 != 0:
        raise ValueError("users_per_service must be a positive multiple of 3")
    k_urllc = k_embb = int(users_per_service)
    region_size = users_per_service // 3
    num_list = [region_size] * 6
    scale = k_urllc + k_embb

    os.chdir(BASE_DIR)
    np.random.seed(seed)
    random.seed(seed)

    solution_dir = SOLUTION_ROOT / str(scale) / f"Outer_{iteration}"
    urllc_csv = solution_dir / "urllc_offloading_decision.csv"
    embb_csv = solution_dir / "embb_offloading_decision.csv"
    for path in (urllc_csv, embb_csv):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing outer decision {path}. Run ollama_outer_optimizer.py --iteration {iteration} first."
            )

    outer = build_outer_from_offloading_decision(
        str(urllc_csv),
        str(embb_csv),
        sixg_num=SIXG_NUM,
        wifi_num=WIFI_NUM,
        sat_num=SAT_NUM,
    )
    expected_shape = (scale, RAT_NUM + SAT_NUM)
    if outer.shape != expected_shape:
        raise ValueError(f"Outer shape {outer.shape} does not match {expected_shape}")
    np.savetxt(solution_dir / "outer_solution.csv", outer, delimiter=",", fmt="%d")

    task_paths = (
        BASE_DIR / "Data" / f"urllc_tasks_{k_urllc}.csv",
        BASE_DIR / "Data" / f"embb_tasks_{k_embb}.csv",
    )
    channel_u_path = BASE_DIR / "Channel" / f"channel_URLLC_{k_urllc}_{k_embb}.csv"
    channel_e_path = BASE_DIR / "Channel" / f"channel_eMBB_{k_urllc}_{k_embb}.csv"
    for path in (*task_paths, channel_u_path, channel_e_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing {k_urllc}+{k_embb} quantification input: {path}")
    channel_u = np.loadtxt(channel_u_path, delimiter=",", dtype=complex)
    channel_e = np.loadtxt(channel_e_path, delimiter=",", dtype=complex)
    if channel_u.shape != (k_urllc, 10):
        raise ValueError(f"Expected URLLC channel shape {(k_urllc, 10)}, got {channel_u.shape}")
    if channel_e.shape != (k_embb, 10):
        raise ValueError(f"Expected eMBB channel shape {(k_embb, 10)}, got {channel_e.shape}")
    channel = np.vstack((channel_u, channel_e))

    inner = MyproblemInner(
        k_urllc,
        k_embb,
        RAT_NUM,
        seed,
        outer,
        channel,
        num_list,
        RAT_LIST,
    )
    if generations is not None:
        if generations <= 0:
            raise ValueError("generations must be positive")
        inner.generation = int(generations)
    if population_size is not None:
        if population_size < 4:
            raise ValueError("population_size must be at least 4")
        inner.population_size = int(population_size)

    population_best, fitness_best, cv_best, cost_urllc_best, fitness_history = inner.run_origin()

    pool_dir = POOL_ROOT / str(scale) / f"Outer{iteration}"
    pool_dir.mkdir(parents=True, exist_ok=True)
    fitness_path = _write_fitness_result(
        str(pool_dir),
        iteration,
        fitness_best,
        cv_best,
        cost_urllc_best,
        fitness_history,
    )

    result_dir = BASE_DIR / "Result_Ollama_Proposed" / str(scale)
    metrics_path = _append_outer_iteration_metrics(
        str(result_dir),
        iteration,
        seed,
        inner.best_metrics,
    )
    *_, diagnostics = inner.evalVars(
        np.asarray(population_best, dtype=float).reshape(1, -1),
        return_details=True,
    )
    diagnostic_paths = _write_diagnostic_csvs(str(pool_dir), iteration, diagnostics)

    summary = {
        "outer_iteration": int(iteration),
        "users_per_service": int(users_per_service),
        "total_users": int(scale),
        "seed": int(seed),
        "generations": int(inner.generation),
        "population_size": int(inner.population_size),
        "fitness": _scalar(fitness_best),
        "cv": _scalar(cv_best),
        "cost_urllc": _scalar(cost_urllc_best),
        **{key: float(value) for key, value in inner.best_metrics.items()},
        "outer_solution": str(solution_dir / "outer_solution.csv"),
        "fitness_feedback": str(fitness_path),
        "metrics": str(metrics_path),
        "diagnostics": diagnostic_paths,
    }
    summary_path = pool_dir / f"proposed_summary_iteration{iteration}.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] outer iteration {iteration}: fitness={summary['fitness']}, cv={summary['cv']}")
    print(f"[OK] feedback: {pool_dir}")
    return summary


def run_rounds(args) -> None:
    for iteration in range(args.rounds):
        print(f"\n===== OUTER ITERATION {iteration}: OLLAMA =====")
        generate_outer_association(
            iteration=iteration,
            model=args.model,
            host=args.host,
            temperature=args.temperature,
            num_ctx=args.num_ctx,
            retries=args.retries,
            users_per_service=args.users_per_service,
            gain_decimals=args.gain_decimals,
            num_predict=args.num_predict,
        )
        print(f"\n===== OUTER ITERATION {iteration}: INNER EC =====")
        run_inner_ec(
            iteration=iteration,
            seed=args.seed,
            generations=args.generations,
            population_size=args.population_size,
            users_per_service=args.users_per_service,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--iteration", type=int, help="Evaluate one existing outer iteration")
    mode.add_argument("--rounds", type=int, help="Call Ollama and EC alternately for this many rounds")
    mode.add_argument(
        "--count-first-prompt-tokens",
        action="store_true",
        help="Evaluate only the initial Ollama prompt and print its exact input-token count",
    )
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen3.5:35b"))
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--num-predict", type=int, default=8192)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--users-per-service", type=int, default=150)
    parser.add_argument("--gain-decimals", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generations", type=int, default=None, help="Override the current proposed EC generation count")
    parser.add_argument("--population-size", type=int, default=None, help="Override the current proposed EC population size")
    args = parser.parse_args()

    if args.count_first_prompt_tokens:
        stats = count_initial_prompt_tokens(
            model=args.model,
            host=args.host,
            num_ctx=args.num_ctx,
            users_per_service=args.users_per_service,
            gain_decimals=args.gain_decimals,
        )
        print(json.dumps(stats, ensure_ascii=False, indent=2))
    elif args.rounds is not None:
        if args.rounds <= 0:
            parser.error("--rounds must be positive")
        run_rounds(args)
    else:
        run_inner_ec(
            iteration=args.iteration,
            seed=args.seed,
            generations=args.generations,
            population_size=args.population_size,
            users_per_service=args.users_per_service,
        )


if __name__ == "__main__":
    main()
