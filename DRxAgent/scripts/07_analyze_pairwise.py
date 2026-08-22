#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from graphdrx_agent_eval.common import read_jsonl
from graphdrx_agent_eval.judge import PAIRWISE_DIMENSIONS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    if not rows:
        raise SystemExit("No pairwise judgments found")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    flat = []
    for row in rows:
        j = row["judgment"]
        rec = {
            "case_id": row["case_id"],
            "config_a": row["pair"][0],
            "config_b": row["pair"][1],
            "disease": row["disease"],
            "display_order": row["display_order"],
            "left_configuration": row["left_configuration"],
            "right_configuration": row["right_configuration"],
            "mapped_preference": row["mapped_preference"],
            "rationale": j["rationale"],
            "critical_difference": j["critical_difference"],
        }
        left = row["left_configuration"]
        right = row["right_configuration"]
        for dim in PAIRWISE_DIMENSIONS:
            pref = j["dimension_preferences"][dim]
            rec[dim] = "tie" if pref == "tie" else (left if pref == "A" else right)
        flat.append(rec)
    df = pd.DataFrame(flat)
    df.to_csv(out / "pairwise_rows.csv", index=False)

    order_rows = []
    for (a, b, disease), group in df.groupby(["config_a", "config_b", "disease"]):
        if set(group["display_order"]) != {"AB", "BA"}:
            raise ValueError(f"Missing AB/BA order for {a}/{b}, {disease}")
        prefs = dict(zip(group["display_order"], group["mapped_preference"]))
        order_rows.append(
            {
                "config_a": a,
                "config_b": b,
                "disease": disease,
                "preference_ab": prefs["AB"],
                "preference_ba": prefs["BA"],
                "order_consistent": prefs["AB"] == prefs["BA"],
            }
        )
    order_df = pd.DataFrame(order_rows)
    order_df.to_csv(out / "pairwise_order_consistency_by_disease.csv", index=False)

    summary_rows = []
    for (a, b), group in df.groupby(["config_a", "config_b"]):
        counts = Counter(group["mapped_preference"])
        paired = order_df[(order_df["config_a"] == a) & (order_df["config_b"] == b)]
        summary_rows.append(
            {
                "config_a": a,
                "config_b": b,
                "n_presentations": int(len(group)),
                "a_preferences": int(counts[a]),
                "b_preferences": int(counts[b]),
                "ties": int(counts["tie"]),
                "a_preference_rate": float(counts[a] / len(group)),
                "b_preference_rate": float(counts[b] / len(group)),
                "tie_rate": float(counts["tie"] / len(group)),
                "order_consistency_rate": float(paired["order_consistent"].mean()),
            }
        )
    pd.DataFrame(summary_rows).to_csv(out / "pairwise_pair_summary.csv", index=False)

    left_selected = int((df["mapped_preference"] == df["left_configuration"]).sum())
    right_selected = int((df["mapped_preference"] == df["right_configuration"]).sum())
    ties = int((df["mapped_preference"] == "tie").sum())
    n_presentations = int(len(df))
    position_bias = pd.DataFrame([{
        "n_presentations": n_presentations,
        "left_selected": left_selected,
        "right_selected": right_selected,
        "ties": ties,
        "left_selection_rate": float(left_selected / n_presentations),
        "right_selection_rate": float(right_selected / n_presentations),
        "tie_rate": float(ties / n_presentations),
    }])
    position_bias.to_csv(out / "pairwise_position_bias.csv", index=False)

    dim_rows = []
    for (a, b), group in df.groupby(["config_a", "config_b"]):
        for dim in PAIRWISE_DIMENSIONS:
            counts = Counter(group[dim])
            dim_rows.append(
                {
                    "config_a": a,
                    "config_b": b,
                    "dimension": dim,
                    "a_preferences": int(counts[a]),
                    "b_preferences": int(counts[b]),
                    "ties": int(counts["tie"]),
                }
            )
    pd.DataFrame(dim_rows).to_csv(out / "pairwise_dimension_summary.csv", index=False)
    print("Pairwise analysis complete:", out)


if __name__ == "__main__":
    main()
