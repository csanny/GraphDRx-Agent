from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from .common import collect_evidence_ids, parse_json_object


DIMENSIONS = (
    "evidence_groundedness_factuality",
    "mechanistic_consistency",
    "relevance_scientific_specificity",
    "uncertainty_calibration",
    "experimental_actionability",
)

ROLE_SCORE_KEYS = (
    "sponsor_quality",
    "expert_quality",
    "chair_quality",
    "chair_synthesis",
)

JUDGE_POLICY_VERSION = "2026-08-05-direct-development-context-v1"

DIRECT_DEVELOPMENT_RELATIONS = (
    "indication",
    "off_label_use",
    "tested_indication",
    "studied_for_treatment_of",
    "studied_for_marker_mechanism_of",
)

CRITICAL_FLAGS = (
    "unsupported_claim",
    "relation_overreach",
    "fabricated_evidence_id",
    "external_knowledge_use",
    "missingness_as_negative",
    "pass_through_synthesis",
    "missed_direction_conflict",
)

PAIRWISE_DIMENSIONS = (*DIMENSIONS, "chair_synthesis")

SECTION_ORDER = (
    "graphdrx",
    "relation_semantics",
    "disease_context",
    "drug_pharmacology",
    "physicochemical_properties",
    "pk_pd_exposure",
    "safety_developability",
    "path_relation_evidence",
    "drug_gene_mechanism",
    "drug_disease_development",
    "counter_evidence",
)


def _compact_value(value: Any, string_limit: int = 2400, list_limit: int = 30) -> Any:
    if isinstance(value, str):
        if len(value) <= string_limit:
            return value
        return value[:string_limit] + " ... [TRUNCATED]"
    if isinstance(value, list):
        compacted = [_compact_value(x, string_limit, list_limit) for x in value[:list_limit]]
        if len(value) > list_limit:
            compacted.append({"omitted_items": len(value) - list_limit})
        return compacted
    if isinstance(value, dict):
        return {str(k): _compact_value(v, string_limit, list_limit) for k, v in value.items()}
    return value


def _budget_items(items: list[dict[str, Any]], max_chars: int) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    used = 0
    for raw in items:
        item = _compact_value(raw)
        encoded = json.dumps(item, ensure_ascii=False)
        if selected and used + len(encoded) > max_chars:
            break
        if not selected and len(encoded) > max_chars:
            item = _compact_value(raw, string_limit=max(500, max_chars // 2), list_limit=12)
            encoded = json.dumps(item, ensure_ascii=False)
        selected.append(item)
        used += len(encoded)
    return selected, max(0, len(items) - len(selected))


def compact_candidate_for_judge(
    packet: dict[str, Any],
    panel: dict[str, Any],
    max_chars_per_section: int = 6000,
) -> dict[str, Any]:
    """Build a structured judge packet.

    Core records from every section are retained under a per-section budget. Every
    evidence record cited by any agent is also included, even when it falls outside
    the core-section budget. This lets the judge test grounding without arbitrary
    whole-document prefix truncation.
    """
    sections = packet.get("evidence_sections") or {}
    selected_sections: dict[str, list[dict[str, Any]]] = {}
    omitted_counts: dict[str, int] = {}
    item_by_id: dict[str, dict[str, Any]] = {}

    for name in SECTION_ORDER:
        raw_items = [x for x in sections.get(name, []) if isinstance(x, dict)]
        for item in raw_items:
            evidence_id = item.get("evidence_id")
            if evidence_id:
                item_by_id[str(evidence_id)] = item
        budget = max_chars_per_section
        if name in {"graphdrx", "relation_semantics", "physicochemical_properties", "pk_pd_exposure"}:
            budget = min(max_chars_per_section, 4000)
        selected, omitted = _budget_items(raw_items, budget)
        selected_sections[name] = selected
        omitted_counts[name] = omitted

    cited_ids = sorted(
        collect_evidence_ids(panel.get("sponsor"))
        | collect_evidence_ids(panel.get("expert"))
        | collect_evidence_ids(panel.get("chair"))
    )
    selected_ids = {
        str(item.get("evidence_id"))
        for section_items in selected_sections.values()
        for item in section_items
        if isinstance(item, dict) and item.get("evidence_id")
    }
    cited_records: list[dict[str, Any]] = []
    for evidence_id in cited_ids:
        if evidence_id in selected_ids:
            continue
        raw = item_by_id.get(evidence_id)
        if raw is None:
            raw = next(
                (
                    x
                    for x in packet.get("evidence_catalog", [])
                    if str(x.get("evidence_id")) == evidence_id
                ),
                {"evidence_id": evidence_id, "record_not_found": True},
            )
        cited_records.append(_compact_value(raw, string_limit=1400, list_limit=20))

    return {
        "case_id": packet["case_id"],
        "candidate_packet": {
            "case_id": packet["case_id"],
            "disease": packet["disease"],
            "drug": packet["drug"],
            "rank": packet["rank"],
            "retrieval_score": packet["retrieval_score"],
            "graphdrx": _compact_value(packet.get("graphdrx", {}), string_limit=2400),
            "evidence_sections": selected_sections,
            "agent_cited_evidence_records": cited_records,
            "agent_cited_evidence_ids": cited_ids,
            "omitted_evidence_counts": omitted_counts,
            "evidence_coverage": packet.get("evidence_coverage", {}),
        },
        "sponsor_output": _compact_value(panel["sponsor"], string_limit=4000, list_limit=40),
        "expert_output": _compact_value(panel["expert"], string_limit=4000, list_limit=40),
        "chair_output": _compact_value(panel["chair"], string_limit=4000, list_limit=40),
    }


def _valid_score(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and float(value).is_integer()
        and 1 <= int(value) <= 5
    )


def validate_judgment(judgment: dict[str, Any], expected_case_ids: list[str]) -> list[str]:
    errors: list[str] = []
    if not isinstance(judgment, dict):
        return ["judgment must be an object"]
    evaluations = judgment.get("candidate_evaluations")
    if not isinstance(evaluations, list):
        return ["candidate_evaluations must be a list"]
    if len(evaluations) != len(expected_case_ids):
        errors.append(
            f"expected {len(expected_case_ids)} candidate evaluations, found {len(evaluations)}"
        )
    case_ids = [str(item.get("case_id", "")) for item in evaluations if isinstance(item, dict)]
    if len(case_ids) != len(set(case_ids)):
        errors.append("duplicate case_id in candidate_evaluations")
    if set(case_ids) != set(expected_case_ids):
        errors.append(
            "case_id set mismatch: expected="
            + repr(sorted(expected_case_ids))
            + " found="
            + repr(sorted(case_ids))
        )
    for idx, item in enumerate(evaluations):
        if not isinstance(item, dict):
            errors.append(f"candidate_evaluations[{idx}] must be an object")
            continue
        scores = item.get("scores")
        if not isinstance(scores, dict):
            errors.append(f"{item.get('case_id', idx)}: scores must be an object")
            continue
        for dimension in DIMENSIONS:
            if dimension not in scores:
                errors.append(f"{item.get('case_id', idx)}: missing score {dimension}")
            elif not _valid_score(scores[dimension]):
                errors.append(
                    f"{item.get('case_id', idx)}: invalid score {dimension}={scores[dimension]!r}"
                )
        for key in ROLE_SCORE_KEYS:
            if key not in item:
                errors.append(f"{item.get('case_id', idx)}: missing {key}")
            elif not _valid_score(item[key]):
                errors.append(f"{item.get('case_id', idx)}: invalid {key}={item[key]!r}")
        flags = item.get("critical_flags")
        if not isinstance(flags, list):
            errors.append(f"{item.get('case_id', idx)}: critical_flags must be a list")
        else:
            invalid_flags = sorted({str(x) for x in flags} - set(CRITICAL_FLAGS))
            if invalid_flags:
                errors.append(f"{item.get('case_id', idx)}: invalid critical flags {invalid_flags}")
            if len(flags) != len(set(map(str, flags))):
                errors.append(f"{item.get('case_id', idx)}: duplicate critical flags")
        for key in ("strengths", "weaknesses"):
            if not isinstance(item.get(key), list):
                errors.append(f"{item.get('case_id', idx)}: {key} must be a list")
    if not isinstance(judgment.get("disease_level_comment"), str):
        errors.append("disease_level_comment must be a string")
    return errors


def validate_pairwise_judgment(judgment: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not isinstance(judgment, dict):
        return ["pairwise judgment must be an object"]
    if judgment.get("preferred") not in {"A", "B", "tie"}:
        errors.append("preferred must be A, B, or tie")
    if not isinstance(judgment.get("rationale"), str):
        errors.append("rationale must be a string")
    if not isinstance(judgment.get("critical_difference"), str):
        errors.append("critical_difference must be a string")
    prefs = judgment.get("dimension_preferences")
    if not isinstance(prefs, dict):
        errors.append("dimension_preferences must be an object")
    else:
        for key in PAIRWISE_DIMENSIONS:
            if prefs.get(key) not in {"A", "B", "tie"}:
                errors.append(f"invalid dimension preference for {key}: {prefs.get(key)!r}")
    return errors


def judgment_json_schema(expected_case_ids: list[str]) -> dict[str, Any]:
    score_properties = {key: {"type": "integer", "minimum": 1, "maximum": 5} for key in DIMENSIONS}
    evaluation_properties: dict[str, Any] = {
        "case_id": {"type": "string", "enum": expected_case_ids},
        "scores": {
            "type": "object",
            "properties": score_properties,
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
        **{key: {"type": "integer", "minimum": 1, "maximum": 5} for key in ROLE_SCORE_KEYS},
        "critical_flags": {
            "type": "array",
            "items": {"type": "string", "enum": list(CRITICAL_FLAGS)},
        },
        "strengths": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
        "weaknesses": {"type": "array", "items": {"type": "string"}, "maxItems": 4},
    }
    return {
        "type": "object",
        "properties": {
            "candidate_evaluations": {
                "type": "array",
                "minItems": len(expected_case_ids),
                "maxItems": len(expected_case_ids),
                "items": {
                    "type": "object",
                    "properties": evaluation_properties,
                    "required": [
                        "case_id",
                        "scores",
                        *ROLE_SCORE_KEYS,
                        "critical_flags",
                        "strengths",
                        "weaknesses",
                    ],
                    "additionalProperties": False,
                },
            },
            "disease_level_comment": {"type": "string"},
        },
        "required": ["candidate_evaluations", "disease_level_comment"],
        "additionalProperties": False,
    }


def pairwise_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "preferred": {"type": "string", "enum": ["A", "B", "tie"]},
            "rationale": {"type": "string"},
            "dimension_preferences": {
                "type": "object",
                "properties": {
                    key: {"type": "string", "enum": ["A", "B", "tie"]}
                    for key in PAIRWISE_DIMENSIONS
                },
                "required": list(PAIRWISE_DIMENSIONS),
                "additionalProperties": False,
            },
            "critical_difference": {"type": "string"},
        },
        "required": ["preferred", "rationale", "dimension_preferences", "critical_difference"],
        "additionalProperties": False,
    }


def judge_prompt(
    items: list[dict[str, Any]],
    validation_errors: list[str] | None = None,
    previous_judgment: dict[str, Any] | None = None,
) -> str:
    repair = ""
    if validation_errors:
        repair = f"""
The previous response failed semantic validation. Correct the listed errors while preserving the same evaluation intent.
Validation errors: {json.dumps(validation_errors, ensure_ascii=False)}
Previous response: {json.dumps(previous_judgment or {}, ensure_ascii=False)}
"""
    return f"""You are a blinded evaluator of drug-repurposing evidence-interpretation reports.

SCOPE
- Evaluate the quality of the generated reports, not whether the candidate is truly effective.
- Use only the supplied candidate evidence packet and generated outputs.
- Do not use external biomedical knowledge, literature, or familiarity with approved indications.
- Model names and configurations are intentionally hidden.
- GraphDRx rank is fixed context only. Do not reward, penalize, or reorder candidates based on rank.
- A medically plausible statement is not grounded unless it is supported by the supplied packet.
- Missing evidence is uncertainty, not negative evidence.

SCORING ANCHORS
Use integer scores from 1 to 5. Apply these anchors consistently:
- 5: excellent; all material claims are supported and correctly interpreted, with no material omission or contradiction.
- 4: strong; mostly correct and useful, with only minor overstatement, omission, or lack of specificity.
- 3: mixed/adequate; the core report is usable but has at least one material limitation, unsupported bridge, generic section, or unresolved inconsistency.
- 2: weak; major grounding, mechanism, calibration, or actionability problem substantially limits usefulness.
- 1: poor; central conclusion is fabricated, externally supplied, internally contradictory, or unsupported by the packet.

PRIMARY DIMENSIONS
1. evidence_groundedness_factuality
   Verify that material claims are traceable to supplied records and cited evidence IDs. Penalize unsupported facts, fabricated IDs, and external knowledge.
2. mechanistic_consistency
   Verify KG path structure, relation semantics, drug-action direction, disease-direction evidence, and the distinction between graph-edge direction and therapeutic direction. A correct categorical label does not excuse a contradictory narrative.

DIRECT DRUG-DISEASE DEVELOPMENT CONTEXT
- The packet may contain indication, off-label-use, tested-indication, or treatment-study relations for a candidate. These records establish only that a use or development relationship was reported.
- Do not award a high score merely because a direct development relation is present. It does not independently prove the proposed mechanism, therapeutic direction, efficacy, safety, or experimental validity.
- A tested-indication or study relation is not evidence of successful efficacy or approval.
- When KG and RAG records describe the same underlying drug-disease relation, treat them as duplicated provenance for one fact, not as independent corroborating evidence.
- Do not penalize a report for accurately mentioning development context. Penalize only when the report overextends that context into unsupported mechanistic, directional, efficacy, or certainty claims. Use relation_overreach and/or unsupported_claim when appropriate.
3. relevance_scientific_specificity
   Reward disease-drug-specific use of the supplied path and pharmacology. Penalize generic biomedical prose and explanations that could apply to unrelated candidates.
4. uncertainty_calibration
   Verify separation of observed evidence, inference, assumptions, missingness, counter-evidence, and true negative evidence. Conclusions must not exceed the packet.
5. experimental_actionability
   The experiment should follow from the proposed mechanism and specify a relevant model, intervention, comparator, target-engagement or mechanistic readout, disease-relevant outcome, and explicit supporting and rejecting outcomes. Penalize reversed outcome logic. Include safety or nonspecific-effect controls when materially relevant.

ROLE DIAGNOSTICS
- sponsor_quality: specific and testable hypothesis; observed evidence and inference are separated; no unsupported causal leap.
- expert_quality: independently detects relation and therapeutic-direction overreach; evaluates supplied pharmacology, PK/exposure, safety/developability, and evidence limitations; does not treat missingness as harm.
- chair_quality: produces a coherent, useful, evidence-grounded integrated report.
- chair_synthesis: explicitly adjudicates Sponsor claims as retained, weakened, or rejected; resolves disagreements and direction conflicts; preserves material Expert caveats rather than copying either role.

CRITICAL FLAGS
Use an empty list when none apply. Otherwise use only these exact labels:
- unsupported_claim: a material claim is not supported by the packet.
- relation_overreach: an association, PPI, ontology hierarchy, binding, direct development relation, or another non-mechanistic/nondirectional relation is treated as causal, therapeutically directional, efficacious, or mechanistically validating without support.
- fabricated_evidence_id: a cited ID is absent from the packet.
- external_knowledge_use: the report relies on facts not present in the packet.
- missingness_as_negative: unavailable data are treated as evidence of harm, failure, or absence.
- pass_through_synthesis: Chair mostly repeats Sponsor/Expert without meaningful adjudication.
- missed_direction_conflict: drug action and disease abnormality are compared incorrectly, a narrative contradicts the selected relation/status, or a material direction conflict is not resolved.

Return exactly one evaluation for every supplied case_id, with no duplicates. Keep strengths and weaknesses concise and evidence-focused.
{repair}
ITEMS
{json.dumps(items, ensure_ascii=False)}
"""


def pairwise_prompt(
    disease: str,
    items_a: list[dict[str, Any]],
    items_b: list[dict[str, Any]],
    validation_errors: list[str] | None = None,
    previous_judgment: dict[str, Any] | None = None,
) -> str:
    repair = ""
    if validation_errors:
        repair = f"""
The previous response failed semantic validation. Correct the listed errors while preserving the same comparison intent.
Validation errors: {json.dumps(validation_errors, ensure_ascii=False)}
Previous response: {json.dumps(previous_judgment or {}, ensure_ascii=False)}
"""
    return f"""You are a blinded evaluator comparing two multi-agent report systems for one disease.
Use only the supplied candidate evidence and reports. Do not use external literature or familiarity with the candidates. Candidate ranks are fixed and must not be changed.

Compare System A and System B on evidence groundedness/factuality, mechanistic consistency, scientific specificity, uncertainty calibration, experimental actionability, and Chair synthesis. Judge the five-candidate set as a whole. Prefer a tie when differences are not material. Do not infer system identity from writing style.

Direct indication, off-label-use, tested-indication, or treatment-study records are development context only. They do not independently prove mechanism, therapeutic direction, efficacy, safety, or experimental validity. Do not prefer a system merely because it repeats such a relation. Treat equivalent KG and RAG records describing the same relation as duplicated provenance for one underlying fact, not independent corroboration. Reward systems that use development context accurately while preserving mechanistic and uncertainty discipline.
{repair}
Disease: {disease}
System A:
{json.dumps(items_a, ensure_ascii=False)}

System B:
{json.dumps(items_b, ensure_ascii=False)}
"""


class JudgeClient:
    def __init__(
        self,
        provider: str,
        model: str,
        temperature: float = 0,
        timeout: int = 900,
        base_url: str | None = None,
        reasoning_effort: str = "medium",
        max_output_tokens: int = 12000,
        openai_max_retries: int = 3,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.base_url = base_url
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.openai_max_retries = openai_max_retries

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _openai_structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        from openai import OpenAI

        started = time.time()
        client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=self.base_url or os.environ.get("OPENAI_BASE_URL"),
            timeout=self.timeout,
            max_retries=self.openai_max_retries,
        )
        request: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": prompt}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                },
                "verbosity": "low",
            },
            "max_output_tokens": self.max_output_tokens,
            "store": False,
        }
        if self.reasoning_effort != "default":
            request["reasoning"] = {"effort": self.reasoning_effort}

        response = client.responses.create(**request)
        status = getattr(response, "status", None)
        if status != "completed":
            incomplete = getattr(response, "incomplete_details", None)
            raise RuntimeError(
                f"OpenAI response not completed: status={status!r}, incomplete_details={incomplete!r}"
            )
        text = response.output_text or ""
        if not text.strip():
            raise RuntimeError("OpenAI response completed without output_text")
        usage = getattr(response, "usage", None)
        meta = {
            "provider": "openai_responses",
            "response_id": getattr(response, "id", None),
            "response_model": getattr(response, "model", None),
            "response_status": status,
            "latency_seconds": round(time.time() - started, 3),
            "reasoning_effort": self.reasoning_effort,
            "max_output_tokens": self.max_output_tokens,
            "prompt_sha256": self._hash_text(prompt),
            "prompt_chars": len(prompt),
            "output_chars": len(text),
            "usage": usage.model_dump() if usage and hasattr(usage, "model_dump") else None,
        }
        return parse_json_object(text), meta

    def _ollama_json(self, prompt: str) -> tuple[dict[str, Any], dict[str, Any]]:
        import requests

        started = time.time()
        url = (self.base_url or "http://localhost:11434").rstrip("/") + "/api/generate"
        response = requests.post(
            url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": self.temperature, "seed": 42},
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        text = body.get("response", "")
        return parse_json_object(text), {
            "provider": "ollama",
            "latency_seconds": round(time.time() - started, 3),
            "eval_count": body.get("eval_count"),
            "prompt_sha256": self._hash_text(prompt),
            "prompt_chars": len(prompt),
            "output_chars": len(text),
        }

    def evaluate(
        self,
        items: list[dict[str, Any]],
        expected_case_ids: list[str],
        validation_errors: list[str] | None = None,
        previous_judgment: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = judge_prompt(items, validation_errors, previous_judgment)
        if self.provider == "openai":
            return self._openai_structured(
                prompt,
                judgment_json_schema(expected_case_ids),
                "graphdrx_pointwise_judgment",
            )
        if self.provider == "ollama":
            return self._ollama_json(prompt)
        raise ValueError(f"Unsupported judge provider: {self.provider}")

    def evaluate_pairwise(
        self,
        disease: str,
        items_a: list[dict[str, Any]],
        items_b: list[dict[str, Any]],
        validation_errors: list[str] | None = None,
        previous_judgment: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        prompt = pairwise_prompt(
            disease,
            items_a,
            items_b,
            validation_errors,
            previous_judgment,
        )
        if self.provider == "openai":
            return self._openai_structured(
                prompt,
                pairwise_json_schema(),
                "graphdrx_pairwise_judgment",
            )
        if self.provider == "ollama":
            return self._ollama_json(prompt)
        raise ValueError(f"Unsupported judge provider: {self.provider}")
