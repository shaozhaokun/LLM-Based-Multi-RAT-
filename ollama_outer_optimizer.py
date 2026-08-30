"""Ollama-backed outer association optimizer for the 150+150 quantification case."""

from __future__ import annotations

import argparse
import ast
import io
import json
import math
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
SOLUTION_ROOT = BASE_DIR / "Solution_Ollama"
POOL_ROOT = BASE_DIR / "Pool_Ollama"
K_URLLC = 150
K_EMBB = 150
SCALE = K_URLLC + K_EMBB
BS_CODES = ("G1", "G2", "W1", "W2", "W3", "W4", "S1", "S2")


def configure_scale(users_per_service: int) -> None:
    """Configure an equal URLLC/eMBB scale with three equal user regions."""
    if users_per_service <= 0 or users_per_service % 3 != 0:
        raise ValueError("users_per_service must be a positive multiple of 3")
    global K_URLLC, K_EMBB, SCALE
    K_URLLC = int(users_per_service)
    K_EMBB = int(users_per_service)
    SCALE = K_URLLC + K_EMBB

SYSTEM_PROMPT = r"""You are the outer association optimizer for a multi-RAT mobile edge computing (MEC) system.

Your task is to generate an offloading association for __K_URLLC__ URLLC tasks and __K_EMBB__ eMBB tasks. Continuous communication-resource allocation is subsequently optimized by an inner evolutionary computation (EC) algorithm. Therefore, optimize only the discrete BS association decisions.

Optimization objective:
- Minimize the heterogeneous-task cost.
- eMBB cost increases linearly with end-to-end delay.
- A URLLC task incurs an outage penalty if its end-to-end delay exceeds its hard deadline.
- End-to-end delay includes UE-to-BS transmission, BS-to-cloud transmission, cloud queueing, and cloud computation.

BS codes:
- G1, G2: 6G BSs.
- W1, W2, W3, W4: WiFi BSs.
- S1, S2: satellite BSs.

For each service type, the candidate BS set is:
- Users 1--__R1_END__: G1, G2, W1, W2, W3, W4.
- Users __R2_START__--__R2_END__: S1, S2.
- Users __R3_START__--__K_URLLC__: G1, G2, W1, W2, W3, W4, S1, S2.

Each URLLC task must select exactly one candidate BS. Each eMBB task may select multiple candidate BSs, but at most one 6G BS, at most one WiFi BS, at most one satellite BS, and at least one BS in total.

Network configuration:
- 6G: 2 BSs, 50e6 Hz per BS, terrestrial BS-to-cloud rate 4e7 bit/s.
- WiFi: 4 BSs, 10e6 Hz per BS, terrestrial BS-to-cloud rate 2e7 bit/s.
- Satellite: 2 BSs, 20e6 Hz per satellite.
- UE uplink power: 0.2 W.
- URLLC satellite downlink power budget: 100 W.
- eMBB satellite downlink power budget: 20 W.
- Cloud computation: URLLC 5e9 cycles/s and eMBB 7e9 cycles/s.
- CPU cycles = 41.25 times input data size in bits.
- Cloud scheduling: Moore-Hodgson for URLLC and SPT for eMBB.

Compact data format:
- Rows follow users 1--__K_URLLC__; indices are omitted.
- E_TASK row: [input_data_size_bits].
- U_TASK row: [input_data_size_bits, hard_deadline_ms].
- E_UL_GAIN_DB and U_UL_GAIN_DB are uplink channel power gains in dB.
- Rows 1--__R1_END__ have columns [G1,G2,W1,W2,W3,W4].
- Rows __R2_START__--__R2_END__ have columns [S1,S2].
- Rows __R3_START__--__K_URLLC__ have columns [G1,G2,W1,W2,W3,W4,S1,S2].
- SAT_DOWN_GAIN_DB is [S1-to-cloud,S2-to-cloud] and is global.

<E_TASK>
__E_TASK__
</E_TASK>

<U_TASK>
__U_TASK__
</U_TASK>

<E_UL_GAIN_DB>
__E_UL_GAIN_DB__
</E_UL_GAIN_DB>

<U_UL_GAIN_DB>
__U_UL_GAIN_DB__
</U_UL_GAIN_DB>

<SAT_DOWN_GAIN_DB>
__SAT_DOWN_GAIN_DB__
</SAT_DOWN_GAIN_DB>

The inner EC algorithm optimizes continuous communication resources. Do not output bandwidth, power, or other continuous variables.

Return exactly two variables and no explanation:
urllc_offloading_decision = [...]
embb_offloading_decision = [...]

The URLLC list must contain exactly __K_URLLC__ BS-code strings. The eMBB list must contain exactly __K_EMBB__ lists of BS-code strings and obey all candidate and per-RAT constraints.
"""

INITIAL_USER_PROMPT = """Construct an initial high-quality association from the supplied task and channel data. Account for channel quality and load balancing rather than assigning every user to the same strongest RAT. Return only the two requested variables."""

FEEDBACK_USER_PROMPT = """The inner EC evaluation of the current association has been completed.

<CURRENT_ASSOCIATION>
{current_association}
</CURRENT_ASSOCIATION>

<EC_FEEDBACK>
{ec_feedback}
</EC_FEEDBACK>

Using the previously provided system state, task information, channel information, and association constraints, generate a new association expected to improve the overall fitness.

Give particular attention to URLLC tasks that miss or nearly miss their deadlines. However, do not simply assign every critical task to its individually strongest link; account for BS/RAT congestion, shared bandwidth, satellite-resource competition, queueing effects, and the provided EC feedback.

Return only:

urllc_offloading_decision = [...]
embb_offloading_decision = [...]
"""


def _power_gain_db(values: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.abs(values) ** 2, 1e-300))


def _format_rows(rows: Iterable[Iterable[float]], decimals: int = 3) -> str:
    lines = []
    for row in rows:
        vals = ",".join(f"{float(v):.{decimals}f}" for v in row)
        lines.append(f"[{vals}]")
    return "\n".join(lines)


def _candidate_columns(user_index: int) -> slice:
    region_size = K_URLLC // 3
    if user_index < region_size:
        return slice(0, 6)
    if user_index < 2 * region_size:
        return slice(6, 8)
    return slice(0, 8)


def build_quantification_system_prompt(gain_decimals: int = 1) -> str:
    if gain_decimals < 0:
        raise ValueError("gain_decimals must be nonnegative")
    embb_path = BASE_DIR / "Data" / f"embb_tasks_{K_EMBB}.csv"
    urllc_path = BASE_DIR / "Data" / f"urllc_tasks_{K_URLLC}.csv"
    channel_u_path = BASE_DIR / "Channel" / f"channel_URLLC_{K_URLLC}_{K_EMBB}.csv"
    channel_e_path = BASE_DIR / "Channel" / f"channel_eMBB_{K_URLLC}_{K_EMBB}.csv"
    for path in (embb_path, urllc_path, channel_u_path, channel_e_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing quantification input: {path}")

    embb = pd.read_csv(embb_path)
    urllc = pd.read_csv(urllc_path)
    channel_u = np.loadtxt(channel_u_path, delimiter=",", dtype=complex)
    channel_e = np.loadtxt(channel_e_path, delimiter=",", dtype=complex)
    if len(embb) != K_EMBB or len(urllc) != K_URLLC:
        raise ValueError(f"Expected {K_URLLC}+{K_EMBB} tasks, got {len(urllc)}+{len(embb)}")
    if channel_u.shape != (K_URLLC, 10):
        raise ValueError(f"Expected URLLC channel shape {(K_URLLC, 10)}, got {channel_u.shape}")
    if channel_e.shape != (K_EMBB, 10):
        raise ValueError(f"Expected eMBB channel shape {(K_EMBB, 10)}, got {channel_e.shape}")

    e_task = "\n".join(f"[{int(v)}]" for v in embb["Data Size (bits)"].to_numpy())
    u_task = "\n".join(
        f"[{int(bits)},{1000.0 * float(deadline):.4f}]"
        for bits, deadline in zip(urllc["Data Size (bits)"], urllc["Deadline (s)"])
    )

    u_gain = _power_gain_db(channel_u[:, :8])
    e_gain = _power_gain_db(channel_e[:, :8])
    u_rows = [u_gain[i, _candidate_columns(i)] for i in range(K_URLLC)]
    e_rows = [e_gain[i, _candidate_columns(i)] for i in range(K_EMBB)]
    sat_down_coeff = channel_u[0, 8:10]
    if not np.allclose(channel_u[:, 8:10], sat_down_coeff, rtol=1e-10, atol=1e-15):
        raise ValueError("URLLC satellite-to-cloud columns are not global/identical")
    if not np.allclose(channel_e[:, 8:10], sat_down_coeff, rtol=1e-10, atol=1e-15):
        raise ValueError("eMBB satellite-to-cloud columns differ from the URLLC global values")
    sat_down = _power_gain_db(sat_down_coeff).reshape(1, -1)

    region_size = K_URLLC // 3
    replacements = {
        "__K_URLLC__": str(K_URLLC),
        "__K_EMBB__": str(K_EMBB),
        "__R1_END__": str(region_size),
        "__R2_START__": str(region_size + 1),
        "__R2_END__": str(2 * region_size),
        "__R3_START__": str(2 * region_size + 1),
        "__E_TASK__": e_task,
        "__U_TASK__": u_task,
        "__E_UL_GAIN_DB__": _format_rows(e_rows, gain_decimals),
        "__U_UL_GAIN_DB__": _format_rows(u_rows, gain_decimals),
        "__SAT_DOWN_GAIN_DB__": _format_rows(sat_down, gain_decimals),
    }
    prompt = SYSTEM_PROMPT
    for token, value in replacements.items():
        prompt = prompt.replace(token, value)
    return prompt


def _canonical_to_code(value: str) -> str:
    text = str(value).strip().replace(" ", "")
    patterns = (
        (r"(?i)^(G|6G)_?([12])$", "G"),
        (r"(?i)^(W|WiFi|Wi-Fi)_?([1-4])$", "W"),
        (r"(?i)^(S|Sat|Satellite)_?([12])$", "S"),
    )
    for pattern, prefix in patterns:
        match = re.match(pattern, text)
        if match:
            return prefix + match.group(2)
    raise ValueError(f"Unknown BS code: {value!r}")


def _code_to_canonical(code: str) -> str:
    code = _canonical_to_code(code)
    if code.startswith("G"):
        return f"6G_{code[1:]}"
    if code.startswith("W"):
        return f"WiFi_{code[1:]}"
    return f"Satellite_{code[1:]}"


def _candidate_set(index: int) -> set[str]:
    region_size = K_URLLC // 3
    if index < region_size:
        return {"G1", "G2", "W1", "W2", "W3", "W4"}
    if index < 2 * region_size:
        return {"S1", "S2"}
    return set(BS_CODES)


def validate_decisions(urllc: list, embb: list) -> tuple[list[str], list[list[str]]]:
    if len(urllc) != K_URLLC or len(embb) != K_EMBB:
        raise ValueError(
            f"Expected {K_URLLC} URLLC and {K_EMBB} eMBB entries, "
            f"got {len(urllc)} and {len(embb)}"
        )

    clean_u: list[str] = []
    clean_e: list[list[str]] = []
    for i, value in enumerate(urllc):
        code = _canonical_to_code(value)
        if code not in _candidate_set(i):
            raise ValueError(f"URLLC user {i + 1}: {code} is outside its candidate set")
        clean_u.append(code)

    for i, values in enumerate(embb):
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError(f"eMBB user {i + 1}: selection must be a nonempty list")
        codes = list(dict.fromkeys(_canonical_to_code(v) for v in values))
        if any(code not in _candidate_set(i) for code in codes):
            raise ValueError(f"eMBB user {i + 1}: selection {codes} is outside its candidate set")
        groups = [sum(c.startswith(prefix) for c in codes) for prefix in ("G", "W", "S")]
        if any(count > 1 for count in groups):
            raise ValueError(f"eMBB user {i + 1}: at most one BS per RAT is allowed: {codes}")
        clean_e.append(codes)
    return clean_u, clean_e


def parse_model_response(text: str) -> tuple[list[str], list[list[str]]]:
    cleaned = re.sub(r"```(?:python)?|```", "", text, flags=re.IGNORECASE)
    u_match = re.search(
        r"urllc_offloading_decision\s*=\s*(\[.*?\])\s*(?=embb_offloading_decision\s*=)",
        cleaned,
        flags=re.DOTALL,
    )
    e_match = re.search(r"embb_offloading_decision\s*=\s*(\[.*\])\s*$", cleaned, flags=re.DOTALL)
    if not u_match or not e_match:
        raise ValueError("Response does not contain the two required assignments")
    return validate_decisions(ast.literal_eval(u_match.group(1)), ast.literal_eval(e_match.group(1)))


def format_decisions(urllc: list[str], embb: list[list[str]]) -> str:
    return (
        f"urllc_offloading_decision = {urllc!r}\n"
        f"embb_offloading_decision = {embb!r}"
    )


def save_decisions(iteration: int, urllc: list[str], embb: list[list[str]]) -> Path:
    output_dir = SOLUTION_ROOT / str(SCALE) / f"Outer_{iteration}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"User": np.arange(1, K_URLLC + 1), "Selected_BS": [_code_to_canonical(v) for v in urllc]}
    ).to_csv(output_dir / "urllc_offloading_decision.csv", index=False)
    pd.DataFrame(
        {
            "User": np.arange(1, K_EMBB + 1),
            "Selected_BS_List": [" + ".join(_code_to_canonical(v) for v in row) for row in embb],
        }
    ).to_csv(output_dir / "embb_offloading_decision.csv", index=False)
    (output_dir / "ollama_response.txt").write_text(format_decisions(urllc, embb), encoding="utf-8")
    return output_dir


def load_decisions(iteration: int) -> tuple[list[str], list[list[str]]]:
    directory = SOLUTION_ROOT / str(SCALE) / f"Outer_{iteration}"
    u_df = pd.read_csv(directory / "urllc_offloading_decision.csv")
    e_df = pd.read_csv(directory / "embb_offloading_decision.csv")
    urllc = [_canonical_to_code(v) for v in u_df["Selected_BS"]]
    selected_col = "Selected_BS_List" if "Selected_BS_List" in e_df else "Selected_Paths"
    embb = [
        [_canonical_to_code(part.strip()) for part in str(value).split("+") if part.strip()]
        for value in e_df[selected_col]
    ]
    return validate_decisions(urllc, embb)


def build_ec_feedback(previous_iteration: int, max_bad_users: int = 80) -> str:
    pool_dir = POOL_ROOT / str(SCALE) / f"Outer{previous_iteration}"
    fitness_path = pool_dir / f"best_fitness_iteration{previous_iteration}.json"
    rat_path = pool_dir / f"feedback_rat_utilization_iteration{previous_iteration}.csv"
    bad_path = pool_dir / f"feedback_bad_users_iteration{previous_iteration}.csv"
    selected_path = pool_dir / f"feedback_selected_link_rates_iteration{previous_iteration}.csv"
    for path in (fitness_path, rat_path, bad_path, selected_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing EC feedback for outer iteration {previous_iteration}: {path}")

    fitness = json.loads(fitness_path.read_text(encoding="utf-8"))
    rat = pd.read_csv(rat_path)
    bad = pd.read_csv(bad_path)
    selected = pd.read_csv(selected_path)

    if not bad.empty:
        bad = bad.copy()
        bad["service_user_id"] = np.where(
            bad["user_type"].str.upper() == "URLLC",
            bad["user_id"] + 1,
            bad["user_id"] - K_URLLC + 1,
        )
        # Always expose deadline-sensitive URLLC users before slow eMBB users.
        bad["_service_priority"] = np.where(
            bad["user_type"].str.upper() == "URLLC", 0, 1
        )
        bad = bad.sort_values(
            ["_service_priority", "violation", "total_delay"],
            ascending=[True, False, False],
        )
        bad = bad.head(max_bad_users)

    critical_global_ids = set(int(v) for v in bad.get("user_id", []))
    critical_links = selected[selected["user_id"].isin(critical_global_ids)].copy()
    if not critical_links.empty:
        critical_links["service_user_id"] = np.where(
            critical_links["user_type"].str.upper() == "URLLC",
            critical_links["user_id"] + 1,
            critical_links["user_id"] - K_URLLC + 1,
        )

    keep_rat = [
        "rat_name", "link_direction", "utilization_total", "utilization_urllc",
        "utilization_embb", "connected_urllc_count", "connected_embb_count",
    ]
    keep_bad = [
        "service_user_id", "user_type", "selection_reason", "communication_delay",
        "queue_delay", "processing_delay", "total_delay", "deadline", "violation",
        "bottleneck_reason",
    ]
    keep_links = [
        "service_user_id", "user_type", "rat_name", "uplink_rate",
        "downlink_rate", "effective_e2e_rate",
    ]

    def csv_text(frame: pd.DataFrame, columns: list[str]) -> str:
        if frame.empty:
            return "none"
        buffer = io.StringIO()
        frame[[c for c in columns if c in frame.columns]].to_csv(buffer, index=False)
        return buffer.getvalue().strip()

    return (
        "FITNESS_SUMMARY\n" + json.dumps(fitness, ensure_ascii=False) +
        "\n\nRAT_UTILIZATION\n" + csv_text(rat, keep_rat) +
        "\n\nCRITICAL_OR_SLOW_USERS (service_user_id is 1-based within its own service)\n" + csv_text(bad, keep_bad) +
        "\n\nSELECTED_LINK_RATES_FOR_LISTED_USERS\n" + csv_text(critical_links, keep_links)
    )


def build_outer_messages(iteration: int, gain_decimals: int = 1) -> list[dict]:
    """Build the exact chat history used by one outer iteration."""
    if iteration < 0:
        raise ValueError("iteration must be nonnegative")
    messages = [
        {"role": "system", "content": build_quantification_system_prompt(gain_decimals)}
    ]
    if iteration == 0:
        messages.append({"role": "user", "content": INITIAL_USER_PROMPT})
    else:
        previous_u, previous_e = load_decisions(iteration - 1)
        current = format_decisions(previous_u, previous_e)
        feedback = build_ec_feedback(iteration - 1)
        messages.extend(
            [
                {"role": "user", "content": INITIAL_USER_PROMPT},
                {"role": "assistant", "content": current},
                {
                    "role": "user",
                    "content": FEEDBACK_USER_PROMPT.format(
                        current_association=current,
                        ec_feedback=feedback,
                    ),
                },
            ]
        )
    return messages


def _ollama_chat_result(
    host: str,
    model: str,
    messages: list[dict],
    temperature: float,
    num_ctx: int,
    num_predict: int | None = None,
) -> dict:
    options = {"temperature": temperature, "num_ctx": num_ctx}
    if num_predict is not None:
        options["num_predict"] = num_predict
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": options,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        host.rstrip("/") + "/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise RuntimeError(
            f"Cannot call Ollama at {host}. Start Ollama and pull model {model!r}, or pass --host/--model."
        ) from exc
    return result


def _ollama_chat(host: str, model: str, messages: list[dict], temperature: float, num_ctx: int) -> str:
    return _ollama_chat_result(host, model, messages, temperature, num_ctx)["message"]["content"]


def count_initial_prompt_tokens(
    model: str,
    host: str = "http://127.0.0.1:11434",
    num_ctx: int = 32768,
    users_per_service: int = 150,
    gain_decimals: int = 1,
) -> dict:
    """Ask Ollama to evaluate only the first-round prompt and report its exact token count."""
    configure_scale(users_per_service)
    messages = build_outer_messages(0, gain_decimals)
    result = _ollama_chat_result(
        host=host,
        model=model,
        messages=messages,
        temperature=0.0,
        num_ctx=num_ctx,
        num_predict=1,
    )
    stats = {
        "model": result.get("model", model),
        "prompt_eval_count": int(result["prompt_eval_count"]),
        "generated_tokens": int(result.get("eval_count", 0)),
        "load_duration_seconds": float(result.get("load_duration", 0)) / 1e9,
        "prompt_eval_duration_seconds": float(result.get("prompt_eval_duration", 0)) / 1e9,
        "num_ctx": int(num_ctx),
        "users_per_service": int(users_per_service),
        "gain_decimals": int(gain_decimals),
    }
    log_dir = BASE_DIR / "Quantification" / "Ollama" / str(SCALE)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_path = log_dir / "token_count_iteration0.json"
    output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    stats["output_path"] = str(output_path)
    return stats


def generate_outer_association(
    iteration: int,
    model: str,
    host: str = "http://127.0.0.1:11434",
    temperature: float = 0.2,
    num_ctx: int = 32768,
    retries: int = 2,
    dry_run: bool = False,
    users_per_service: int = 150,
    gain_decimals: int = 1,
) -> Path:
    configure_scale(users_per_service)
    messages = build_outer_messages(iteration, gain_decimals)

    log_dir = BASE_DIR / "Quantification" / "Ollama" / str(SCALE)
    log_dir.mkdir(parents=True, exist_ok=True)
    request_path = log_dir / f"request_iteration{iteration}.json"
    request_path.write_text(
        json.dumps({"model": model, "host": host, "messages": messages}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if dry_run:
        return request_path

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        response_text = _ollama_chat(host, model, messages, temperature, num_ctx)
        (log_dir / f"response_iteration{iteration}_attempt{attempt}.txt").write_text(
            response_text, encoding="utf-8"
        )
        try:
            urllc, embb = parse_model_response(response_text)
            return save_decisions(iteration, urllc, embb)
        except (ValueError, SyntaxError) as exc:
            last_error = exc
            messages.extend(
                [
                    {"role": "assistant", "content": response_text},
                    {
                        "role": "user",
                        "content": (
                            f"Your output is invalid: {exc}. Correct it. Return only the two assignments, "
                            f"with exactly {K_URLLC} valid URLLC entries and {K_EMBB} valid eMBB entries."
                        ),
                    },
                ]
            )
    raise RuntimeError(f"Ollama failed to produce a valid association: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iteration", type=int, required=True, help="Outer iteration: 0 for initial, 1+ for feedback")
    parser.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen3.5:35b"))
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--users-per-service", type=int, default=150)
    parser.add_argument("--gain-decimals", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Build and save the request without calling Ollama")
    args = parser.parse_args()
    output = generate_outer_association(
        iteration=args.iteration,
        model=args.model,
        host=args.host,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
        retries=args.retries,
        dry_run=args.dry_run,
        users_per_service=args.users_per_service,
        gain_decimals=args.gain_decimals,
    )
    print(f"[OK] {output}")


if __name__ == "__main__":
    main()
