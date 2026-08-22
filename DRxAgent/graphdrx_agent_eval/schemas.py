from __future__ import annotations

from collections import Counter
from typing import Any

from .common import collect_evidence_ids, sha256_obj, slugify
from .direction import DIRECTION_RELATION_TO_STATUS
from .evidence_builder import PACKET_SCHEMA_VERSION


REQUIRED_PACKET_FIELDS = (
    "packet_schema_version", "case_id", "disease", "therapeutic_area", "drug", "rank",
    "retrieval_score", "graphdrx", "evidence_sections", "evidence_catalog", "evidence_coverage",
)

SPONSOR_REQUIRED = (
    "disease",
    "drug",
    "graphdrx_rank",
    "kg_grounded_hypothesis",
    "observed_evidence",
    "inferred_mechanism",
    "path_and_relation_interpretation",
    "supporting_evidence_ids",
    "key_assumptions",
    "alternative_explanations",
    "remaining_uncertainties",
)

EXPERT_REQUIRED = (
    "disease",
    "drug",
    "graphdrx_rank",
    "overall_appraisal",
    "mechanistic_plausibility",
    "kg_relation_consistency",
    "pharmacological_action_direction",
    "physicochemical_pk_pd_exposure",
    "safety_developability",
    "evidence_limitations",
    "sponsor_claims_supported",
    "sponsor_claims_challenged",
    "proposed_validation_experiment",
)

CHAIR_REQUIRED = (
    "disease",
    "drug",
    "graphdrx_rank",
    "retrieval_score",
    "integrated_mechanistic_interpretation",
    "kg_path_interpretation",
    "therapeutic_direction_synthesis",
    "supporting_evidence",
    "claim_adjudication",
    "key_assumptions",
    "remaining_evidence_gaps",
    "pk_exposure_considerations",
    "safety_considerations",
    "validation_experiment",
    "research_oriented_interpretation",
)

PROHIBITED_KEYS = {
    "final_decision",
    "decision",
    "triage",
    "advance_hold_no_go",
    "recommended_rank",
    "reranked_position",
    "clinical_success_probability",
    "approval_probability",
}


def normalize_packet(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate/canonicalize grounded packets without rebuilding or replacing their evidence."""
    if raw.get("packet_schema_version") == PACKET_SCHEMA_VERSION:
        packet = dict(raw)
        packet["case_id"] = str(packet.get("case_id") or "")
        packet["disease"] = str(packet.get("disease") or "")
        packet["drug"] = str(packet.get("drug") or "")
        packet["therapeutic_area"] = str(packet.get("therapeutic_area") or "unknown")
        packet["rank"] = int(packet.get("rank") or 0)
        packet["retrieval_score"] = float(packet.get("retrieval_score") or 0.0)
        packet["packet_hash"] = sha256_obj({k: v for k, v in packet.items() if k != "packet_hash"})
        return packet

    # Refuse legacy information-poor packets rather than silently treating them as grounded evidence.
    disease = raw.get("disease") or raw.get("disease_name")
    drug = raw.get("drug") or raw.get("drug_name")
    case_id = raw.get("case_id") or f"{slugify(str(disease))}__{slugify(str(drug))}"
    return {
        "packet_schema_version": str(raw.get("packet_schema_version") or "legacy_unaccepted"),
        "case_id": str(case_id),
        "disease": str(disease or ""),
        "therapeutic_area": str(raw.get("therapeutic_area") or raw.get("area") or "unknown"),
        "drug": str(drug or ""),
        "rank": int(raw.get("rank") or 0),
        "retrieval_score": float(raw.get("retrieval_score") or raw.get("score") or 0.0),
        "graphdrx": raw.get("graphdrx") or {},
        "evidence_sections": raw.get("evidence_sections") or {},
        "evidence_catalog": raw.get("evidence_catalog") or [],
        "evidence_coverage": raw.get("evidence_coverage") or {"core_ready": False},
        "entity_resolution": raw.get("entity_resolution") or {},
        "metadata": raw.get("metadata") or {},
        "packet_hash": sha256_obj(raw),
    }


def validate_packets(packets: list[dict[str, Any]], expected_diseases: int = 15, expected_topk: int = 5) -> list[str]:
    errors: list[str] = []
    seen_cases: set[str] = set()
    by_disease: Counter[str] = Counter()
    ranks: dict[str, list[int]] = {}
    for idx, p in enumerate(packets, 1):
        for field in REQUIRED_PACKET_FIELDS:
            if field not in p or p[field] in (None, "", []):
                errors.append(f"row {idx}: missing {field}")
        cid = str(p.get("case_id", ""))
        if cid in seen_cases:
            errors.append(f"duplicate case_id: {cid}")
        seen_cases.add(cid)
        disease = str(p.get("disease", ""))
        by_disease[disease] += 1
        ranks.setdefault(disease, []).append(int(p.get("rank", 0)))

        if p.get("packet_schema_version") != PACKET_SCHEMA_VERSION:
            errors.append(
                f"{cid}: packet_schema_version must be {PACKET_SCHEMA_VERSION}; "
                f"found {p.get('packet_schema_version')}"
            )
        if not bool((p.get("evidence_coverage") or {}).get("core_ready")):
            errors.append(f"{cid}: core grounding evidence is not ready")
        catalog = p.get("evidence_catalog") or []
        if not catalog:
            errors.append(f"{cid}: empty evidence_catalog")
        ids = [str(x.get("evidence_id", "")) for x in catalog if isinstance(x, dict)]
        if any(not x for x in ids):
            errors.append(f"{cid}: evidence item missing evidence_id")
        if len(ids) != len(set(ids)):
            errors.append(f"{cid}: duplicate evidence IDs")
        sections = p.get("evidence_sections") or {}
        for required_section in ("graphdrx", "disease_context", "drug_pharmacology"):
            if not sections.get(required_section):
                errors.append(f"{cid}: required evidence section empty: {required_section}")

    if len(by_disease) != expected_diseases:
        errors.append(f"expected {expected_diseases} unique diseases, found {len(by_disease)}")
    for disease, n in sorted(by_disease.items()):
        if n != expected_topk:
            errors.append(f"{disease}: expected {expected_topk} candidates, found {n}")
        if sorted(ranks[disease]) != list(range(1, expected_topk + 1)):
            errors.append(f"{disease}: ranks must be 1..{expected_topk}; found {sorted(ranks[disease])}")
    return errors



def validate_role_output(role: str, obj: dict[str, Any], packet: dict[str, Any]) -> list[str]:
    required = {"sponsor": SPONSOR_REQUIRED, "expert": EXPERT_REQUIRED, "chair": CHAIR_REQUIRED}[role]
    errors: list[str] = []
    for key in required:
        if key not in obj:
            errors.append(f"missing key: {key}")
    if str(obj.get("disease", "")).strip().lower() != packet["disease"].strip().lower():
        errors.append("disease identity mismatch")
    if str(obj.get("drug", "")).strip().lower() != packet["drug"].strip().lower():
        errors.append("drug identity mismatch")
    try:
        if int(obj.get("graphdrx_rank")) != int(packet["rank"]):
            errors.append("GraphDRx rank changed")
    except Exception:
        errors.append("invalid graphdrx_rank")
    prohibited = sorted(PROHIBITED_KEYS.intersection(obj.keys()))
    if prohibited:
        errors.append("prohibited keys: " + ", ".join(prohibited))

    from .prompts import visible_evidence_ids
    valid_ids = set(visible_evidence_ids(packet, role))
    cited_ids = collect_evidence_ids(obj)
    invalid_ids = sorted(cited_ids - valid_ids)
    if invalid_ids:
        errors.append("invalid evidence IDs: " + ", ".join(invalid_ids))
    # A report that never cites supplied evidence is not a valid grounded role output.
    if role in {"sponsor", "expert", "chair"} and not cited_ids:
        errors.append("no evidence IDs cited")

    # Direction-trace fields are structurally required, but their scientific
    # completeness is evaluated as a QC/judge outcome rather than used as a
    # fatal generation gate. Otherwise a model is selectively regenerated
    # until it conforms to a desired scientific interpretation, which biases
    # factorial model comparison.
    if role == "expert":
        direction = obj.get("pharmacological_action_direction") or {}
        required_direction_keys = (
            "disease_direction_statement",
            "disease_direction_evidence_ids",
            "drug_action_statement",
            "drug_action_evidence_ids",
            "direction_comparison",
            "direction_relation",
            "edge_direction_support",
            "therapeutic_direction_consistency",
        )
        for key in required_direction_keys:
            if key not in direction:
                errors.append(f"pharmacological_action_direction missing: {key}")
        # Empty narrative trace fields are scientific-quality outcomes, not fatal
        # generation errors. Across thousands of calls, a model may occasionally
        # return an empty comparison/statement even though the structured object,
        # identity, evidence citations, and categorical direction are otherwise
        # valid. Such omissions are retained and scored by deterministic QC and
        # the blinded judge rather than selectively regenerated.
        relation = str(direction.get("direction_relation") or "")
        expected_status = DIRECTION_RELATION_TO_STATUS.get(relation)
        if expected_status is None:
            errors.append("invalid expert direction_relation")
        elif direction.get("therapeutic_direction_consistency") != expected_status:
            errors.append("expert direction status does not match direction_relation")

    if role == "chair":
        synthesis = obj.get("therapeutic_direction_synthesis") or {}
        required_synthesis_keys = (
            "disease_direction_statement",
            "disease_direction_evidence_ids",
            "drug_action_statement",
            "drug_action_evidence_ids",
            "direction_comparison",
            "direction_relation",
            "status",
            "assessment",
            "evidence_ids",
        )
        for key in required_synthesis_keys:
            if key not in synthesis:
                errors.append(f"therapeutic_direction_synthesis missing: {key}")
        # As for the Expert, empty narrative synthesis fields are preserved as
        # measurable quality deficiencies instead of triggering selective
        # regeneration. Enum validity and deterministic relation-status
        # consistency remain hard requirements.
        relation = str(synthesis.get("direction_relation") or "")
        expected_status = DIRECTION_RELATION_TO_STATUS.get(relation)
        if expected_status is None:
            errors.append("invalid chair direction_relation")
        elif synthesis.get("status") != expected_status:
            errors.append("chair direction status does not match direction_relation")

    return errors
