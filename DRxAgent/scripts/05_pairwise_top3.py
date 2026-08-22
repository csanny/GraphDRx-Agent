#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

from graphdrx_agent_eval.common import append_jsonl, read_jsonl
from graphdrx_agent_eval.judge import (
    JUDGE_POLICY_VERSION,
    JudgeClient,
    compact_candidate_for_judge,
    validate_pairwise_judgment,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top3", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--base-url")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--max-chars-per-section", type=int, default=3500)
    ap.add_argument("--validation-attempts", type=int, default=2)
    ap.add_argument(
        "--reasoning-effort",
        choices=["default", "none", "minimal", "low", "medium", "high", "xhigh"],
        default="medium",
    )
    ap.add_argument("--max-output-tokens", type=int, default=6000)
    ap.add_argument("--openai-max-retries", type=int, default=3)
    args = ap.parse_args()

    configs = [x.strip() for x in Path(args.top3).read_text(encoding="utf-8").splitlines() if x.strip()]
    if len(configs) != 3:
        raise SystemExit(f"Expected exactly 3 configurations in {args.top3}; found {configs}")
    packets = {p["case_id"]: p for p in read_jsonl(args.input)}
    by_config_disease = defaultdict(list)
    for config in configs:
        for panel_path in sorted((Path(args.run_root) / "panels" / config).glob("*.json")):
            panel = json.loads(panel_path.read_text(encoding="utf-8"))
            packet = packets[panel["case_id"]]
            by_config_disease[(config, packet["disease"])].append(
                compact_candidate_for_judge(packet, panel, max_chars_per_section=args.max_chars_per_section)
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    failure_out = out.with_name(out.stem + ".validation_failures.jsonl")
    existing_records = read_jsonl(out) if out.exists() else []
    incompatible = [
        {
            "case_id": record.get("case_id"),
            "judge_policy_version": record.get("judge_policy_version"),
            "judge_model": record.get("judge_model"),
        }
        for record in existing_records
        if record.get("judge_policy_version") != JUDGE_POLICY_VERSION
        or record.get("judge_model") != args.judge_model
    ]
    if incompatible:
        raise SystemExit(
            "Existing pairwise output was produced with a different judge policy/model. "
            "Move or delete it before starting this run. Examples: " + repr(incompatible[:5])
        )
    completed = {str(record.get("case_id")) for record in existing_records if record.get("case_id")}
    diseases = sorted({d for _, d in by_config_disease})
    client = JudgeClient(
        args.provider,
        args.judge_model,
        timeout=args.timeout,
        base_url=args.base_url,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        openai_max_retries=args.openai_max_retries,
    )

    for a, b in itertools.combinations(configs, 2):
        for disease in diseases:
            reports_a = sorted(by_config_disease[(a, disease)], key=lambda x: int(x["candidate_packet"].get("rank", 0)))
            reports_b = sorted(by_config_disease[(b, disease)], key=lambda x: int(x["candidate_packet"].get("rank", 0)))
            if len(reports_a) != 5 or len(reports_b) != 5:
                raise SystemExit(f"Missing top-5 reports for {a}/{b}, {disease}")
            for order, left, right, left_code, right_code in (
                ("AB", reports_a, reports_b, a, b),
                ("BA", reports_b, reports_a, b, a),
            ):
                batch_id = f"{a}_vs_{b}::{disease}::{order}"
                if batch_id in completed:
                    continue
                result = None
                validation_errors: list[str] = []
                previous = None
                attempt_log = []
                for attempt in range(1, args.validation_attempts + 1):
                    result, meta = client.evaluate_pairwise(
                        disease,
                        left,
                        right,
                        validation_errors if attempt > 1 else None,
                        previous if attempt > 1 else None,
                    )
                    validation_errors = validate_pairwise_judgment(result)
                    attempt_log.append({"attempt": attempt, "meta": meta, "validation_errors": validation_errors})
                    if not validation_errors:
                        break
                    previous = result
                if validation_errors:
                    append_jsonl(
                        failure_out,
                        {
                            "case_id": batch_id,
                            "validation_errors": validation_errors,
                            "invalid_judgment": result,
                            "attempt_log": attempt_log,
                        },
                    )
                    raise RuntimeError(
                        f"Pairwise judge output failed validation for {batch_id}: {validation_errors}. Details: {failure_out}"
                    )
                preferred = result.get("preferred")
                mapped = "tie" if preferred == "tie" else (left_code if preferred == "A" else right_code)
                append_jsonl(
                    out,
                    {
                        "case_id": batch_id,
                        "pair": [a, b],
                        "disease": disease,
                        "display_order": order,
                        "left_configuration": left_code,
                        "right_configuration": right_code,
                        "mapped_preference": mapped,
                        "judgment": result,
                        "judge_attempt_log": attempt_log,
                        "judge_validation_pass": True,
                        "judge_model": args.judge_model,
                        "judge_policy_version": JUDGE_POLICY_VERSION,
                    },
                )
                print("pairwise judged", batch_id, "->", mapped)


if __name__ == "__main__":
    main()
