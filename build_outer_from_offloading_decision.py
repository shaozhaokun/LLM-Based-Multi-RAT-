import os
import re
from typing import List

import numpy as np
import pandas as pd


def _parse_bs_name_to_uplink_index(bs_name: str, sixg_num: int, wifi_num: int, sat_num: int) -> int:
    """
    Parse a BS name (e.g. 6G_1, WiFi_2, Satellite_1) into uplink index.

    Index convention (uplink):
      [0 .. sixg_num-1]                    -> 6G_1..6G_n
      [sixg_num .. sixg_num+wifi_num-1]    -> WiFi_1..WiFi_n
      [sixg_num+wifi_num .. +sat_num-1]    -> Satellite_1..Satellite_n
    """
    t = str(bs_name).strip()
    t = re.sub(r"\s+", "", t)
    t = t.replace("Wi-Fi", "WiFi").replace("wi-fi", "WiFi").replace("wifi", "WiFi")

    m = re.match(r"(?i)^6g_?(\d+)$", t)
    if m:
        i = int(m.group(1)) - 1
        if not (0 <= i < sixg_num):
            raise ValueError(f"6G index out of range: {bs_name} (sixg_num={sixg_num})")
        return i

    m = re.match(r"(?i)^wifi_?(\d+)$", t)
    if m:
        i = int(m.group(1)) - 1
        if not (0 <= i < wifi_num):
            raise ValueError(f"WiFi index out of range: {bs_name} (wifi_num={wifi_num})")
        return sixg_num + i

    # Satellite / Satellite_1 / Sat_1
    m = re.match(r"(?i)^(satellite|sat)_?(\d+)?$", t)
    if m:
        sid = m.group(2)
        i = int(sid) - 1 if sid is not None else 0
        if not (0 <= i < sat_num):
            raise ValueError(f"Satellite index out of range: {bs_name} (sat_num={sat_num})")
        return sixg_num + wifi_num + i

    raise ValueError(f"Unrecognized BS name: {bs_name}")


def _parse_selected_paths(cell: str) -> List[str]:
    """
    Parse eMBB selection cell (e.g. 'Selected_BS_List' or legacy 'Selected_Paths'), e.g.:
      - '6G_1 + WiFi_1'
      - 'sat1'
      - '6G_1 + WiFi_1 + sat1'
      - '6G:6G_1 + WiFi:WiFi_1' (legacy format with RAT prefix)

    Returns list of BS names: ['6G_1', 'WiFi_1', 'Satellite_1', ...]
    """
    s = str(cell)
    parts = [p.strip() for p in s.split("+")]
    out: List[str] = []
    for p in parts:
        if not p:
            continue
        # format: RAT:BS_NAME
        if ":" in p:
            _, bs = p.split(":", 1)
            bs = bs.strip()
            out.append(bs)
        else:
            # fallback: treat as BS name directly
            out.append(p.strip())
    return out


def _sort_tasks_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep task ordering stable across CSV variants.
    Prefer sorting by 'Task' (old) or 'User' (new). Otherwise keep file order.
    """
    if "Task" in df.columns:
        return df.sort_values("Task").reset_index(drop=True)
    if "User" in df.columns:
        return df.sort_values("User").reset_index(drop=True)
    return df.reset_index(drop=True)


def build_outer_from_offloading_decision(
    urllc_csv_path: str,
    embb_csv_path: str,
    sixg_num: int = 2,
    wifi_num: int = 4,
    sat_num: int = 2,
) -> np.ndarray:
    """
    Build full outer association matrix from offloading decision CSVs.

    Inputs:
      - URLLC: expects column ['Selected_BS', ...] (new CSV usually also has 'User')
      - eMBB : expects column ['Selected_BS_List', ...] (legacy: 'Selected_Paths')

    Notes:
      - For simplicity, (sixg_num, wifi_num, sat_num) have default values.
        If you want to change the RAT counts, you can directly edit the defaults above.

    Returns:
      outer_full: shape (K_total, M_uplink + M_sat_down)
        - uplink columns: [6G..., WiFi..., Sat_up...]
        - downlink satellite columns: copy of Sat_up columns (Sat_down = Sat_up)
    """
    urllc_df = _sort_tasks_df(pd.read_csv(urllc_csv_path))
    embb_df = _sort_tasks_df(pd.read_csv(embb_csv_path))

    if "Selected_BS" not in urllc_df.columns:
        raise ValueError(f"Missing column 'Selected_BS' in {urllc_csv_path}")

    # eMBB column name changed in the new CSV header
    if "Selected_BS_List" in embb_df.columns:
        embb_selected_col = "Selected_BS_List"
    elif "Selected_Paths" in embb_df.columns:
        embb_selected_col = "Selected_Paths"  # legacy
    else:
        raise ValueError(
            f"Missing column 'Selected_BS_List' (or legacy 'Selected_Paths') in {embb_csv_path}. "
            f"Columns={list(embb_df.columns)}"
        )

    k_urllc = int(len(urllc_df))
    k_embb = int(len(embb_df))
    k_total = k_urllc + k_embb

    m_uplink = int(sixg_num + wifi_num + sat_num)
    sat_uplink_start = int(sixg_num + wifi_num)

    outer_uplink = np.zeros((k_total, m_uplink), dtype=int)

    # URLLC: one-hot (single selected BS)
    for i in range(k_urllc):
        bs = urllc_df.loc[i, "Selected_BS"]
        idx = _parse_bs_name_to_uplink_index(bs, sixg_num, wifi_num, sat_num)
        outer_uplink[i, idx] = 1

    # eMBB: multi-hot (could include multiple satellites)
    for e in range(k_embb):
        row = k_urllc + e
        paths = embb_df.loc[e, embb_selected_col]
        bs_list = _parse_selected_paths(paths)
        if len(bs_list) == 0:
            # Try to include an identifier if present
            ident = embb_df.loc[e, "Task"] if "Task" in embb_df.columns else embb_df.loc[e, "User"] if "User" in embb_df.columns else e
            raise ValueError(f"Empty {embb_selected_col} at eMBB row={ident}: {paths}")
        for bs in bs_list:
            idx = _parse_bs_name_to_uplink_index(bs, sixg_num, wifi_num, sat_num)
            outer_uplink[row, idx] = 1

    # downlink satellite columns: copy uplink satellite columns
    sat_cols = outer_uplink[:, sat_uplink_start : sat_uplink_start + sat_num]
    outer_full = np.concatenate([outer_uplink, sat_cols], axis=1)
    return outer_full


def build_and_save_outer_solution_csv(
    outer_iteration: int = 0,
    urllc_csv_path: str | None = None,
    embb_csv_path: str | None = None,
    out_csv_path: str | None = None,
    sixg_num: int = 2,
    wifi_num: int = 4,
    sat_num: int = 2,
) -> np.ndarray:
    """
    Convenience wrapper: build outer matrix from offloading-decision CSVs and save to a CSV file.

    Output format matches your old `outer_association.csv` style: plain 0/1 matrix, comma-separated.
    """
    outer_dir = os.path.join("Solution", f"Outer_{outer_iteration}")
    if urllc_csv_path is None:
        urllc_csv_path = os.path.join(outer_dir, "urllc_offloading_decision.csv")
    if embb_csv_path is None:
        embb_csv_path = os.path.join(outer_dir, "embb_offloading_decision.csv")
    if out_csv_path is None:
        out_csv_path = os.path.join(outer_dir, "outer_solution.csv")

    outer = build_outer_from_offloading_decision(
        urllc_csv_path=urllc_csv_path,
        embb_csv_path=embb_csv_path,
        sixg_num=sixg_num,
        wifi_num=wifi_num,
        sat_num=sat_num,
    )
    os.makedirs(os.path.dirname(out_csv_path) or ".", exist_ok=True)
    np.savetxt(out_csv_path, outer, delimiter=",", fmt="%d")
    return outer


if __name__ == "__main__":
    # Make relative paths work no matter where you run from
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    os.chdir(_BASE_DIR)

    outer_iteration = 1
    outer = build_and_save_outer_solution_csv(
        outer_iteration=outer_iteration,
        # 数量想改就改这里（或直接改函数默认值）
        sixg_num=2,
        wifi_num=4,
        sat_num=2,
    )
    print(f"[OK] outer shape = {outer.shape}")
    print(f"[OK] saved: Solution/Outer_{outer_iteration}/outer_solution.csv")

