from __future__ import annotations

import itertools
import json
import time
from pathlib import Path
from typing import Any, Callable

from .common import (
    atomic_write_json,
    canonicalize_evidence_ids,
    sanitize_evidence_id_placeholders,
    sha256_obj,
)
from .direction import normalize_direction_fields
from .ollama_client import ModelResponseError, OllamaClient
from .prompts import chair_prompt, disease_summary_prompt, expert_prompt, repair_prompt, sponsor_prompt, visible_evidence_ids
from .schemas import validate_role_output
from .structured_schemas import DISEASE_SUMMARY_SCHEMA, schema_for_role


OUTPUT_SCHEMA_VERSION = "graphdrx-agent-report-v4-direction-relation-20260804"


def all_configs(codes: list[str] | tuple[str, ...] = ("G", "O", "Q")) -> list[str]:
    return ["".join(x) for x in itertools.product(codes, repeat=3)]


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _call_role(
    client: OllamaClient,
    role: str,
    model: str,
    prompt: str,
    packet: dict[str, Any],
    output_path: Path,
    repair_attempts: int,
) -> dict[str, Any]:
    if output_path.exists():
        cached_record = _load_json(output_path)
        cached_obj = normalize_direction_fields(role, cached_record.get(role) or {})
        valid_ids = visible_evidence_ids(packet, role)
        cached_obj, cached_id_normalizations = canonicalize_evidence_ids(cached_obj, valid_ids)
        cached_obj, cached_placeholder_removals = sanitize_evidence_id_placeholders(
            cached_obj, valid_ids
        )
        cache_valid = (
            cached_record.get("packet_hash") == packet.get("packet_hash")
            and cached_record.get("output_schema_version") == OUTPUT_SCHEMA_VERSION
            and not validate_role_output(role, cached_obj, packet)
        )
        if cache_valid:
            if cached_id_normalizations or cached_placeholder_removals:
                cached_record[role] = cached_obj
                cached_record.setdefault("cache_normalizations", []).append({
                    "evidence_id_normalizations": cached_id_normalizations,
                    "evidence_id_placeholders_removed": cached_placeholder_removals,
                })
                atomic_write_json(output_path, cached_record)
            return cached_obj
        stale_path = output_path.with_name(
            output_path.stem + f".stale_{time.time_ns()}" + output_path.suffix
        )
        output_path.replace(stale_path)

    obj: dict[str, Any] | None = None
    meta: dict[str, Any] = {}
    raw_text = ""
    errors: list[str] = []
    attempt_log: list[dict[str, Any]] = []
    max_attempts = 1 + repair_attempts
    use_repair_prompt = False

    for attempt in range(1, max_attempts + 1):
        current_prompt = (
            repair_prompt(
                role,
                raw_text,
                errors,
                visible_evidence_ids(packet, role),
                original_prompt=prompt,
            )
            if use_repair_prompt
            else prompt
        )
        try:
            obj, meta = client.generate_json(model, current_prompt, schema_for_role(role))
            obj = normalize_direction_fields(role, obj)
            valid_ids = visible_evidence_ids(packet, role)
            obj, evidence_id_normalizations = canonicalize_evidence_ids(obj, valid_ids)
            obj, evidence_id_placeholders_removed = sanitize_evidence_id_placeholders(
                obj, valid_ids
            )
            raw_text = str(meta.get("raw_text", ""))
            errors = validate_role_output(role, obj, packet)
            attempt_log.append({
                "attempt": attempt,
                "mode": "repair" if use_repair_prompt else "original",
                "generation_meta": {k: v for k, v in meta.items() if k != "raw_text"},
                "evidence_id_normalizations": evidence_id_normalizations,
                "evidence_id_placeholders_removed": evidence_id_placeholders_removed,
                "validation_errors": errors,
            })
            if not errors:
                break
            # A non-empty, parseable object with schema/identity errors gets one
            # structural repair call. Scientific content is not corrected to fit an answer.
            use_repair_prompt = True
        except ModelResponseError as exc:
            errors = [f"{type(exc).__name__}: {exc}"]
            attempt_log.append({
                "attempt": attempt,
                "mode": "original-retry",
                "errors": errors,
                "diagnostics": exc.diagnostics,
            })
            raw_text = str(exc.diagnostics.get("raw_final_text") or "")
            # Transport, empty-output, and parse failures repeat the full original prompt.
            use_repair_prompt = False
        except Exception as exc:
            errors = [f"{type(exc).__name__}: {exc}"]
            attempt_log.append({"attempt": attempt, "mode": "original-retry", "errors": errors})
            # JSON parsing/transport failures also retry the original prompt because
            # there may be no usable invalid text to repair.
            use_repair_prompt = False
    else:
        failure_path = output_path.with_suffix(".failed.json")
        atomic_write_json(
            failure_path,
            {
                "case_id": packet["case_id"],
                "role": role,
                "model": model,
                "errors": errors,
                "attempt_log": attempt_log,
            },
        )
        raise RuntimeError(
            f"{role} failed for {packet['case_id']} after {max_attempts} attempts: {errors}. "
            f"Diagnostics: {failure_path}"
        )

    record = {
        "case_id": packet["case_id"],
        "packet_hash": packet["packet_hash"],
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "role": role,
        "model": model,
        "attempts": len(attempt_log),
        "attempt_log": attempt_log,
        "validation_errors": errors,
        "generation_meta": {k: v for k, v in meta.items() if k != "raw_text"},
        role: obj,
    }
    atomic_write_json(output_path, record)
    return obj or {}

def run_factorial(
    packets: list[dict[str, Any]],
    output_root: str | Path,
    models: dict[str, str],
    ollama_settings: dict[str, Any],
    configs: list[str] | None = None,
    with_disease_summary: bool = False,
    dry_run: bool = False,
) -> None:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    configs = configs or all_configs(tuple(sorted(models)))
    repair_attempts = int(ollama_settings.get("structural_repair_attempts", 1))
    client = OllamaClient(ollama_settings.get("base_url", "http://localhost:11434"), ollama_settings)
    manifest_settings = {k: v for k, v in ollama_settings.items() if k != "base_url"}

    manifest = {
        "created_unix": time.time(),
        "models": models,
        "ollama_settings": manifest_settings,
        "configs": configs,
        "n_packets": len(packets),
        "input_hash": sha256_obj(packets),
        "cache_design": "Sponsor cached by S; Expert by SE; Chair by SEC",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "with_disease_summary": with_disease_summary,
        "dry_run": dry_run,
    }
    atomic_write_json(root / "run_manifest.json", manifest)
    if dry_run:
        return

    client.assert_models_available(models[c] for config in configs for c in config)

    # Deterministic upstream caching avoids scientifically redundant calls.
    sponsor_outputs: dict[tuple[str, str], dict[str, Any]] = {}
    expert_outputs: dict[tuple[str, str, str], dict[str, Any]] = {}

    sponsor_codes = sorted({c[0] for c in configs})
    # Group by the model currently generating to reduce Ollama model swapping.
    expert_pairs = sorted({c[:2] for c in configs}, key=lambda se: (se[1], se[0]))
    configs = sorted(configs, key=lambda config: (config[2], config[0], config[1]))

    for s in sponsor_codes:
        for packet in packets:
            path = root / "sponsor" / s / f"{packet['case_id']}.json"
            sponsor_outputs[(s, packet["case_id"])] = _call_role(
                client, "sponsor", models[s], sponsor_prompt(packet), packet, path, repair_attempts
            )

    for se in expert_pairs:
        s, e = se[0], se[1]
        for packet in packets:
            sponsor = sponsor_outputs[(s, packet["case_id"])]
            path = root / "expert" / se / f"{packet['case_id']}.json"
            expert_outputs[(s, e, packet["case_id"])] = _call_role(
                client, "expert", models[e], expert_prompt(packet, sponsor), packet, path, repair_attempts
            )

    for config in configs:
        s, e, c = config
        chair_reports_by_disease: dict[str, list[dict[str, Any]]] = {}
        for packet in packets:
            sponsor = sponsor_outputs[(s, packet["case_id"])]
            expert = expert_outputs[(s, e, packet["case_id"])]
            chair_path = root / "chair" / config / f"{packet['case_id']}.json"
            chair = _call_role(
                client, "chair", models[c], chair_prompt(packet, sponsor, expert), packet, chair_path, repair_attempts
            )
            panel_path = root / "panels" / config / f"{packet['case_id']}.json"
            atomic_write_json(
                panel_path,
                {
                    "case_id": packet["case_id"],
                    "disease": packet["disease"],
                    "drug": packet["drug"],
                    "graphdrx_rank": packet["rank"],
                    "retrieval_score": packet["retrieval_score"],
                    "packet_hash": packet["packet_hash"],
                    "output_schema_version": OUTPUT_SCHEMA_VERSION,
                    "configuration": config,
                    "sponsor": sponsor,
                    "expert": expert,
                    "chair": chair,
                },
            )
            chair_reports_by_disease.setdefault(packet["disease"], []).append(chair)

        if with_disease_summary:
            for disease, reports in chair_reports_by_disease.items():
                summary_path = root / "disease_summaries" / config / f"{disease.replace('/', '_')}.json"
                if summary_path.exists():
                    cached_summary = _load_json(summary_path)
                    if cached_summary.get("output_schema_version") == OUTPUT_SCHEMA_VERSION:
                        continue
                    stale_path = summary_path.with_name(
                        summary_path.stem + f".stale_{time.time_ns()}" + summary_path.suffix
                    )
                    summary_path.replace(stale_path)
                summary, meta = client.generate_json(
                    models[c],
                    disease_summary_prompt(disease, sorted(reports, key=lambda x: int(x["graphdrx_rank"]))),
                    DISEASE_SUMMARY_SCHEMA,
                )
                atomic_write_json(summary_path, {
                    "configuration": config,
                    "disease": disease,
                    "output_schema_version": OUTPUT_SCHEMA_VERSION,
                    "summary": summary,
                    "generation_meta": {k: v for k, v in meta.items() if k != "raw_text"},
                })
