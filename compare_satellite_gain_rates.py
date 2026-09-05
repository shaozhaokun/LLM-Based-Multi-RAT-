"""Compare satellite uplink rates for two antenna-gain configurations.

The test keeps user locations and small-scale fading identical between cases.
It reports rates obtained with the standard complex-channel interpretation of
the Friis power gain and reconstructs the previous formula for reference.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.special import j1

from Position_channel_gen import RATDistanceCalculator


GAIN_CASES = (
    ("sat60_user50", 60.0, 50.0),
    ("sat40_user30", 40.0, 30.0),
)


def dbi_to_linear(gain_dbi: float) -> float:
    return 10.0 ** (gain_dbi / 10.0)


def friis_power_gain(
    calculator: RATDistanceCalculator,
    user_positions: np.ndarray,
    satellite_gain_linear: float,
    user_gain_linear: float,
) -> np.ndarray:
    """Return beta = G_user G_sat Gamma(theta) (c / 4 pi d f)^2."""
    sat_start = calculator.M1 + calculator.M2
    beta = np.empty((user_positions.shape[0], calculator.M3), dtype=float)

    for local_sat, sat_idx in enumerate(range(sat_start, calculator.RAT_num)):
        sat_position = calculator.RAT_positions[sat_idx]
        displacement = sat_position - user_positions
        distance = np.linalg.norm(displacement, axis=1)
        horizontal_distance = np.linalg.norm(displacement[:, :2], axis=1)
        elevation = np.arctan2(displacement[:, 2], horizontal_distance)

        argument = 20.0 * np.pi * np.cos(elevation)
        ratio = np.full_like(argument, 0.5)
        nonzero = np.abs(argument) > 1e-12
        ratio[nonzero] = j1(argument[nonzero]) / argument[nonzero]
        pattern_gain = 4.0 * np.abs(ratio) ** 2

        frequency_hz = calculator.f_up_sat[local_sat]
        free_space_gain = (
            calculator.c_light / (4.0 * np.pi * distance * frequency_hz)
        ) ** 2
        beta[:, local_sat] = (
            user_gain_linear
            * satellite_gain_linear
            * pattern_gain
            * free_space_gain
        )

    return beta


def rate_bps(channel: np.ndarray, bandwidth_hz: float, power_w: float) -> np.ndarray:
    noise_density_w_hz = 10.0 ** (-174.0 / 10.0) * 1e-3
    snr = np.abs(channel) ** 2 * power_w / (noise_density_w_hz * bandwidth_hz)
    return bandwidth_hz * np.log2(1.0 + snr)


def summarize(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urllc-users", type=int, default=90)
    parser.add_argument("--embb-users", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bandwidth-hz", type=float, default=6e5)
    parser.add_argument("--power-w", type=float, default=0.2)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("satellite_gain_rate_comparison.csv"),
    )
    args = parser.parse_args()

    rat_list = np.array([2, 4, 2, 2])
    calculator = RATDistanceCalculator(
        urllc_num=args.urllc_users,
        embb_num=args.embb_users,
        RAT_num=8,
        time_=args.seed,
        RAT_list=rat_list,
    )
    user_positions = calculator.generate_user_positions()

    # calculate_DistancesAndChennel draws fading from NumPy's global RNG.
    # Restoring this state makes the two antenna cases directly comparable.
    fading_rng_state = np.random.get_state()
    results = []
    summaries = {}
    sat_start = calculator.M1 + calculator.M2
    sat_stop = sat_start + calculator.M3

    for case_name, sat_gain_dbi, user_gain_dbi in GAIN_CASES:
        sat_gain = dbi_to_linear(sat_gain_dbi)
        user_gain = dbi_to_linear(user_gain_dbi)
        calculator.G_sat[:] = sat_gain
        calculator.g_sat[:] = user_gain
        np.random.set_state(fading_rng_state)
        _, complete_channel = calculator.calculate_DistancesAndChennel(user_positions)
        corrected_channel = complete_channel[:, sat_start:sat_stop]

        beta = friis_power_gain(calculator, user_positions, sat_gain, user_gain)
        # The production code now uses h = z * sqrt(beta) * sqrt(Gain).
        # Reconstruct the previous h = z * beta * sqrt(Gain) for comparison.
        current_channel = corrected_channel * np.sqrt(beta)

        current_rate = rate_bps(current_channel, args.bandwidth_hz, args.power_w)
        corrected_rate = rate_bps(corrected_channel, args.bandwidth_hz, args.power_w)
        summaries[case_name] = {
            "legacy": summarize(current_rate),
            "standard": summarize(corrected_rate),
        }

        for user_idx in range(user_positions.shape[0]):
            for sat_idx in range(calculator.M3):
                results.append(
                    {
                        "case": case_name,
                        "satellite_gain_dbi": sat_gain_dbi,
                        "user_gain_dbi": user_gain_dbi,
                        "user_index": user_idx,
                        "satellite_index": sat_idx,
                        "legacy_formula_rate_bps": current_rate[user_idx, sat_idx],
                        "standard_friis_rate_bps": corrected_rate[user_idx, sat_idx],
                        "legacy_formula_snr_db": 10.0
                        * np.log10(
                            max(
                                np.abs(current_channel[user_idx, sat_idx]) ** 2
                                * args.power_w
                                / (10.0 ** (-174.0 / 10.0) * 1e-3 * args.bandwidth_hz),
                                1e-300,
                            )
                        ),
                        "standard_friis_snr_db": 10.0
                        * np.log10(
                            max(
                                np.abs(corrected_channel[user_idx, sat_idx]) ** 2
                                * args.power_w
                                / (10.0 ** (-174.0 / 10.0) * 1e-3 * args.bandwidth_hz),
                                1e-300,
                            )
                        ),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(
        f"Users={user_positions.shape[0]}, satellites={calculator.M3}, "
        f"bandwidth={args.bandwidth_hz / 1e3:.0f} kHz, power={args.power_w:g} W"
    )
    print("Rates below average all user-satellite links.")
    print()
    for case_name, sat_gain_dbi, user_gain_dbi in GAIN_CASES:
        print(f"{case_name} (satellite={sat_gain_dbi:.0f} dBi, user={user_gain_dbi:.0f} dBi)")
        for model_name in ("legacy", "standard"):
            stats = summaries[case_name][model_name]
            print(
                f"  {model_name:9s}: mean={stats['mean'] / 1e6:.6f} Mbps, "
                f"median={stats['median'] / 1e6:.6f} Mbps, "
                f"min={stats['min'] / 1e6:.6f}, max={stats['max'] / 1e6:.6f} Mbps"
            )
    print()
    for model_name in ("legacy", "standard"):
        high = summaries[GAIN_CASES[0][0]][model_name]["mean"]
        low = summaries[GAIN_CASES[1][0]][model_name]["mean"]
        print(
            f"{model_name:9s} mean-rate ratio (60/50 divided by 40/30): "
            f"{high / low:.6g}x"
        )
    print(f"Detailed results: {args.output.resolve()}")


if __name__ == "__main__":
    main()
