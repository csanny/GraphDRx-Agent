#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _BootstrapPath

_PROJECT_ROOT = _BootstrapPath(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

import pandas as pd

from graphdrx_agent_eval.analysis import build_case_metadata
from graphdrx_agent_eval.common import append_jsonl, read_jsonl
from graphdrx_agent_eval.judge import (
    JUDGE_POLICY_VERSION,
    JudgeClient,
    compact_candidate_for_judge,
    judge_prompt,
    validate_judgment,
)


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_qc_pass_map(path: str | None) -> dict[tuple[str, str], bool]:
    if not path:
        return {}
    df = pd.read_csv(path)
    required = {"configuration", "case_id", "qc_pass"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"QC CSV missing columns: {sorted(missing)}")
    return {
        (str(row.configuration), str(row.case_id)): _as_bool(row.qc_pass)
        for row in df.itertuples(index=False)
    }


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()



def _manifest_path(path: str | Path | None) -> str | None:
    """Return a public-safe path label for run manifests."""
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.resolve().relative_to(_PROJECT_ROOT.resolve()))
    except (ValueError, OSError):
        return p.name

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--provider", choices=["openai", "ollama"], default="openai")
    ap.add_argument("--judge-model", required=True)
    ap.add_argument("--base-url")
    ap.add_argument("--mode", choices=["candidate", "disease"], default="disease")
    ap.add_argument("--qc", default=None, help="automatic_qc.csv; only hard-QC-passed outputs are judged")
    ap.add_argument("--allow-partial-disease-batches", action="store_true")
    ap.add_argument("--max-chars-per-section", type=int, default=6000)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--judge-validation-attempts", type=int, default=2)
    ap.add_argument(
        "--reasoning-effort",
        choices=["default", "none", "minimal", "low", "medium", "high", "xhigh"],
        default="medium",
    )
    ap.add_argument("--max-output-tokens", type=int, default=12000)
    ap.add_argument("--openai-max-retries", type=int, default=3)
    ap.add_argument("--only-configuration", default=None)
    ap.add_argument("--only-disease", default=None)
    ap.add_argument("--max-groups", type=int, default=0, help="0 means all eligible groups")
    ap.add_argument("--dry-run", action="store_true", help="build and audit prompts without calling the judge")
    args = ap.parse_args()

    input_path = Path(args.input)
    packet_rows = read_jsonl(input_path)
    packets = {p["case_id"]: p for p in packet_rows}
    case_metadata = build_case_metadata(packet_rows)
    case_meta_by_id = case_metadata.set_index("case_id").to_dict("index")
    qc_pass = load_qc_pass_map(args.qc)

    raw_groups = defaultdict(list)
    root = Path(args.run_root) / "panels"
    for panel_path in sorted(root.glob("*/*.json")):
        config = panel_path.parent.name
        if args.only_configuration and config != args.only_configuration:
            continue
        with panel_path.open("r", encoding="utf-8") as f:
            panel = json.load(f)
        case_id = panel["case_id"]
        if case_id not in packets:
            raise SystemExit(f"Panel case_id not found in input: {case_id}")
        packet = packets[case_id]
        if args.only_disease and packet["disease"] != args.only_disease:
            continue
        group_name = packet["disease"] if args.mode == "disease" else packet["case_id"]
        raw_groups[(config, group_name)].append(
            (
                case_id,
                compact_candidate_for_judge(
                    packet,
                    panel,
                    max_chars_per_section=args.max_chars_per_section,
                ),
            )
        )

    groups: dict[tuple[str, str], list[dict]] = {}
    skipped_qc = 0
    for key, records in raw_groups.items():
        config, group_name = key
        expected_n = 5 if args.mode == "disease" else 1
        if len(records) != expected_n:
            raise SystemExit(f"Expected {expected_n} report(s) for {config}::{group_name}, found {len(records)}")
        if not qc_pass:
            groups[key] = [item for _, item in records]
            continue
        passed = [item for case_id, item in records if qc_pass.get((config, case_id), False)]
        if args.mode == "disease" and not args.allow_partial_disease_batches and len(passed) != len(records):
            skipped_qc += 1
            continue
        if not passed:
            skipped_qc += 1
            continue
        groups[key] = passed

    if not groups:
        raise SystemExit("No QC-eligible report groups found for judging")

    ordered_groups = sorted(groups.items())
    if args.max_groups > 0:
        ordered_groups = ordered_groups[: args.max_groups]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    failure_out = out.with_name(out.stem + ".validation_failures.jsonl")
    manifest_out = out.with_name(out.stem + ".run_manifest.json")

    prompt_chars: list[int] = []
    group_records: list[dict] = []
    for (config, group_name), items in ordered_groups:
        sorted_items = sorted(items, key=lambda x: int(x["candidate_packet"].get("rank", 0)))
        prompt = judge_prompt(sorted_items)
        expected_case_ids = [str(x["case_id"]) for x in sorted_items]
        prompt_chars.append(len(prompt))
        group_records.append(
            {
                "batch_id": f"{config}::{group_name}",
                "configuration": config,
                "group": group_name,
                "n_candidates": len(sorted_items),
                "expected_case_ids": expected_case_ids,
                "prompt_chars": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "n_direct_relation_present": int(
                    sum(
                        bool(case_meta_by_id[case_id]["direct_development_relation_present"])
                        for case_id in expected_case_ids
                    )
                ),
            }
        )

    manifest = {
        "provider": args.provider,
        "judge_model": args.judge_model,
        "judge_policy_version": JUDGE_POLICY_VERSION,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "mode": args.mode,
        "max_chars_per_section": args.max_chars_per_section,
        "n_groups": len(ordered_groups),
        "skipped_qc_groups": skipped_qc,
        "input_path": _manifest_path(input_path),
        "input_sha256": _sha256_file(input_path),
        "run_root": _manifest_path(args.run_root),
        "qc_path": _manifest_path(args.qc),
        "qc_sha256": _sha256_file(Path(args.qc)) if args.qc else None,
        "prompt_chars": {
            "min": min(prompt_chars),
            "median": statistics.median(prompt_chars),
            "mean": statistics.mean(prompt_chars),
            "max": max(prompt_chars),
            "total": sum(prompt_chars),
        },
        "direct_development_relation_counts": {
            "present": int(case_metadata["direct_development_relation_present"].sum()),
            "absent": int((~case_metadata["direct_development_relation_present"]).sum()),
        },
        "filters": {
            "only_configuration": args.only_configuration,
            "only_disease": args.only_disease,
            "max_groups": args.max_groups,
        },
        "groups": group_records,
    }
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Judge groups ready: {len(ordered_groups)}")
    print(f"Run manifest: {manifest_out}")
    if skipped_qc:
        print(f"Skipped because of hard QC failure: {skipped_qc} groups")
    if args.dry_run:
        print("Dry run complete; no judge calls were made.")
        return

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
            "Existing judge output was produced with a different judge policy/model. "
            "Move or delete it before starting this run. Examples: " + repr(incompatible[:5])
        )
    completed = {str(record.get("case_id")) for record in existing_records if record.get("case_id")}
    client = JudgeClient(
        args.provider,
        args.judge_model,
        temperature=0,
        timeout=args.timeout,
        base_url=args.base_url,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        openai_max_retries=args.openai_max_retries,
    )

    for (config, group_name), items in ordered_groups:
        batch_id = f"{config}::{group_name}"
        if batch_id in completed:
            continue
        items = sorted(items, key=lambda x: int(x["candidate_packet"].get("rank", 0)))
        expected_case_ids = [str(x["case_id"]) for x in items]
        judgment = None
        meta_log = []
        validation_errors: list[str] = []
        previous = None
        for attempt in range(1, args.judge_validation_attempts + 1):
            judgment, meta = client.evaluate(
                items,
                expected_case_ids,
                validation_errors=validation_errors if attempt > 1 else None,
                previous_judgment=previous if attempt > 1 else None,
            )
            validation_errors = validate_judgment(judgment, expected_case_ids)
            meta_log.append({"attempt": attempt, "meta": meta, "validation_errors": validation_errors})
            if not validation_errors:
                break
            previous = judgment

        if validation_errors:
            append_jsonl(
                failure_out,
                {
                    "case_id": batch_id,
                    "configuration": config,
                    "group": group_name,
                    "expected_case_ids": expected_case_ids,
                    "validation_errors": validation_errors,
                    "invalid_judgment": judgment,
                    "judge_attempt_log": meta_log,
                },
            )
            raise RuntimeError(
                f"Judge output failed validation for {batch_id}: {validation_errors}. Details: {failure_out}"
            )

        append_jsonl(
            out,
            {
                "case_id": batch_id,
                "configuration": config,
                "group": group_name,
                "n_candidates": len(items),
                "judgment": judgment,
                "judge_attempt_log": meta_log,
                "judge_validation_pass": True,
                "judge_model": args.judge_model,
                "judge_policy_version": JUDGE_POLICY_VERSION,
            },
        )
        print("judged", batch_id)


if __name__ == "__main__":
    main()
