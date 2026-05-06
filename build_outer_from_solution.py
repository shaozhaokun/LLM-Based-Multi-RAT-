import argparse
import os
import re
from typing import List

import numpy as np
import pandas as pd


def _token_to_uplink_indices(
    token: str,
    sixg_num: int,
    wifi_num: int,
    sat_num: int,
) -> List[int]:
    """
    Parse a BS token into uplink RAT indices.

    Supported token formats (case-insensitive, whitespace-insensitive):
    - 6G_1, 6G1
    - WiFi_3, WiFi3, Wi-Fi_3
    - Satellite, Satellite_1, Sat_1
    """
    t = str(token).strip()
    t = re.sub(r"\s+", "", t)
    t = t.replace("Wi-Fi", "WiFi").replace("wi-fi", "WiFi").replace("wifi", "WiFi")

    m = re.match(r"(?i)^6g_?(\d+)$", t)
    if m:
        idx = int(m.group(1)) - 1
        if not (0 <= idx < sixg_num):
            raise ValueError(f"6G index out of range: {token}")
        return [idx]

    m = re.match(r"(?i)^wifi_?(\d+)$", t)
    if m:
        wi = int(m.group(1)) - 1
        if not (0 <= wi < wifi_num):
            raise ValueError(f"WiFi index out of range: {token}")
        return [sixg_num + wi]

    m = re.match(r"(?i)^(satellite|sat)_?(\d+)?$", t)
    if m:
        sid = m.group(2)
        sat_i = int(sid) - 1 if sid is not None else 0
        if not (0 <= sat_i < sat_num):
            raise ValueError(f"Satellite index out of range: {token}")
        return [sixg_num + wifi_num + sat_i]

    raise ValueError(f"Unrecognized BS token: {token}")


def build_outer_from_solution(
    urllc_csv_path: str,
    embb_csv_path: str,
    sixg_num: int,
    wifi_num: int,
    sat_num: int,
    *,
    require_exact_user_count: bool = True,
) -> np.ndarray:
    """
    Build full outer association matrix from solution CSVs.

    Returns:
      outer_full: shape (K_total, M_uplink + M_sat_down)
        - uplink columns: [6G..., WiFi..., Sat_up...]
        - downlink satellite columns: copy of Sat_up columns (Sat_down = Sat_up)
    """
    urllc_df = pd.read_csv(urllc_csv_path).sort_values("User").reset_index(drop=True)
    embb_df = pd.read_csv(embb_csv_path).sort_values("User").reset_index(drop=True)

    k_urllc = int(len(urllc_df))
    k_embb = int(len(embb_df))
    k_total = k_urllc + k_embb

    m_uplink = int(sixg_num + wifi_num + sat_num)
    sat_uplink_start = int(sixg_num + wifi_num)

    outer_uplink = np.zeros((k_total, m_uplink), dtype=int)

    # URLLC: one-hot
    if "Selected_BS" not in urllc_df.columns:
        raise ValueError(f"Missing column 'Selected_BS' in {urllc_csv_path}")
    for i in range(k_urllc):
        sel = urllc_df.loc[i, "Selected_BS"]
        idxs = _token_to_uplink_indices(sel, sixg_num, wifi_num, sat_num)
        if require_exact_user_count and len(idxs) != 1:
            raise ValueError(f"URLLC Selected_BS parse error: {sel}")
        outer_uplink[i, idxs[0]] = 1

    # eMBB: multi-hot
    if "Selected_BS_List" not in embb_df.columns:
        raise ValueError(f"Missing column 'Selected_BS_List' in {embb_csv_path}")
    for e in range(k_embb):
        row = k_urllc + e
        sel_list = str(embb_df.loc[e, "Selected_BS_List"])
        parts = [p.strip() for p in sel_list.split("+")]
        picks: List[int] = []
        for p in parts:
            if p == "":
                continue
            picks.extend(_token_to_uplink_indices(p, sixg_num, wifi_num, sat_num))
        picks = sorted(set(int(x) for x in picks))
        if require_exact_user_count and len(picks) == 0:
            raise ValueError(f"eMBB Selected_BS_List empty: {sel_list}")
        outer_uplink[row, picks] = 1

    # downlink satellite: copy uplink satellite columns
    sat_cols = outer_uplink[:, sat_uplink_start : sat_uplink_start + sat_num]
    outer_full = np.concatenate([outer_uplink, sat_cols], axis=1)
    return outer_full


def main() -> None:
    parser = argparse.ArgumentParser(description="Build outer association matrix from Solution/*.csv")
    parser.add_argument("--urllc_csv", default=os.path.join("Solution", "urllc_association_solution.csv"))
    parser.add_argument("--embb_csv", default=os.path.join("Solution", "embb_association_solution.csv"))
    parser.add_argument("--sixg_num", type=int, default=2)
    parser.add_argument("--wifi_num", type=int, default=4)
    parser.add_argument("--sat_num", type=int, default=1)
    parser.add_argument("--out_csv", default=os.path.join("Solution", "outer_association.csv"))
    parser.add_argument("--out_npy", default=os.path.join("Solution", "outer_association.npy"))
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    outer = build_outer_from_solution(
        urllc_csv_path=args.urllc_csv,
        embb_csv_path=args.embb_csv,
        sixg_num=args.sixg_num,
        wifi_num=args.wifi_num,
        sat_num=args.sat_num,
    )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    np.savetxt(args.out_csv, outer, delimiter=",", fmt="%d")
    np.save(args.out_npy, outer)

    k_total, m_total = outer.shape
    print(f"[OK] outer shape = ({k_total}, {m_total})")
    print(f"[OK] saved: {args.out_csv}")
    print(f"[OK] saved: {args.out_npy}")


if __name__ == "__main__":
    main()

