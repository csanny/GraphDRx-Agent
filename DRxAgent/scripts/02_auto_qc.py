#!/usr/bin/env python3
from __future__ import annotations

# Allow direct execution as `python scripts/<name>.py` without package installation.
import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from pathlib import Path

import pandas as pd

from graphdrx_agent_eval.common import read_jsonl
from graphdrx_agent_eval.qc import load_panel, qc_panel


CHECK_COLUMNS = (
    "json_valid",
    "input_grounding_ready",
    "identity_valid",
    "rank_preserved",
    "required_fields_complete",
    "evidence_ids_valid",
    "each_role_cites_evidence",
    "observed_inferred_separated",
    "prohibited_fields_absent",
    "qc_pass",
    "expert_direction_cites_disease_context",
    "expert_direction_cites_drug_action",
    "chair_direction_cites_disease_context",
    "chair_direction_cites_drug_action",
    "expert_direction_trace_present",
    "chair_direction_synthesis_present",
    "direction_grounding_complete",
    "expert_direction_label_consistent",
    "chair_direction_label_consistent",
    "direction_quality_pass",
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="normalized candidate packets JSONL")
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument(
        "--summary-output",
        default=None,
        help="optional per-configuration summary CSV; defaults to <output>.summary.csv",
    )
    args = ap.parse_args()

    packets = {p["case_id"]: p for p in read_jsonl(args.input)}
    rows = []
    for panel_path in sorted((Path(args.run_root) / "panels").glob("*/*.json")):
        panel = load_panel(panel_path)
        case_id = panel.get("case_id")
        if case_id not in packets:
            raise SystemExit(f"Panel case_id not found in normalized input: {case_id}")
        rows.append(qc_panel(packets[case_id], panel))

    if not rows:
        raise SystemExit("No panel outputs found")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)

    summary_path = Path(args.summary_output) if args.summary_output else out.with_name(out.stem + "_summary.csv")
    agg = {col: "mean" for col in CHECK_COLUMNS if col in df.columns}
    summary = df.groupby("configuration", as_index=False).agg(
        n_reports=("case_id", "count"),
        n_diseases=("disease", "nunique"),
        **{f"{col}_rate": (col, "mean") for col in CHECK_COLUMNS if col in df.columns},
    )
    summary = summary.sort_values(["qc_pass_rate", "configuration"], ascending=[False, True])
    summary.to_csv(summary_path, index=False)

    total = len(df)
    passed = int(df["qc_pass"].sum())
    print(f"Wrote {total} candidate-level QC rows to {out}")
    print(f"QC passed: {passed}/{total} ({passed / total:.1%})")
    print(f"Per-configuration QC summary: {summary_path}")
    failed = df.loc[~df["qc_pass"], ["configuration", "disease", "drug", "case_id"]]
    if not failed.empty:
        print("QC failures detected. Inspect candidate-level CSV before running the GPT judge.")
        print(failed.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
