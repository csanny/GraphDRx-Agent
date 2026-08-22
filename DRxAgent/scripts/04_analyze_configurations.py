#!/usr/bin/env python3
from __future__ import annotations

# Allow direct execution as `python scripts/<name>.py` without package installation.
import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

import pandas as pd

from graphdrx_agent_eval.analysis import analyze, build_case_metadata, flatten_judgments
from graphdrx_agent_eval.common import read_jsonl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge-output", required=True)
    ap.add_argument("--qc", required=True)
    ap.add_argument(
        "--input",
        required=True,
        help="Grounded candidate packet JSONL used for direct-development-relation sensitivity metadata",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    score_df = flatten_judgments(read_jsonl(args.judge_output))
    if score_df.empty:
        raise SystemExit("No judge scores found")
    qc_df = pd.read_csv(args.qc)
    packets = read_jsonl(args.input)
    case_metadata = build_case_metadata(packets)
    analyze(
        score_df,
        qc_df,
        args.output_dir,
        args.bootstrap,
        args.seed,
        case_metadata_df=case_metadata,
    )
    n_present = int(case_metadata["direct_development_relation_present"].sum())
    print(
        "Analysis complete:",
        args.output_dir,
        f"(direct relation present={n_present}, absent={len(case_metadata) - n_present})",
    )


if __name__ == "__main__":
    main()
