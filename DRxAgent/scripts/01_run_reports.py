#!/usr/bin/env python3
from __future__ import annotations

# Allow direct execution as `python scripts/<name>.py` without package installation.
import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse

from graphdrx_agent_eval.common import load_yaml, read_jsonl
from graphdrx_agent_eval.report_runner import all_configs, run_factorial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--models-config", default="config/models.yaml")
    ap.add_argument("--configs", nargs="*", default=None, help="e.g. OQG GGG; omit for all 27")
    ap.add_argument("--with-disease-summary", action="store_true")
    ap.add_argument("--disease", action="append", default=[], help="exact disease name; repeatable")
    ap.add_argument("--max-packets", type=int, default=None, help="smoke-test limit after filtering")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.models_config)
    models = cfg["models"]
    ollama = cfg["ollama"]
    configs = args.configs or all_configs(tuple(sorted(models)))
    for c in configs:
        if len(c) != 3 or any(x not in models for x in c):
            raise SystemExit(f"Invalid configuration: {c}")
    packets = read_jsonl(args.input)
    if args.disease:
        wanted = {x.strip().lower() for x in args.disease}
        packets = [p for p in packets if str(p.get("disease", "")).strip().lower() in wanted]
        missing = wanted - {str(p.get("disease", "")).strip().lower() for p in packets}
        if missing:
            raise SystemExit(f"Disease filter not found: {sorted(missing)}")
    if args.max_packets is not None:
        packets = packets[: args.max_packets]
    if not packets:
        raise SystemExit("No packets selected")
    run_factorial(packets, args.output_root, models, ollama, configs, args.with_disease_summary, args.dry_run)


if __name__ == "__main__":
    main()
