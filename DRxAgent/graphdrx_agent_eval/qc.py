from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import collect_evidence_ids
from .direction import DIRECTION_RELATION_TO_STATUS
from .prompts import visible_evidence_ids
from .schemas import CHAIR_REQUIRED, EXPERT_REQUIRED, PROHIBITED_KEYS, SPONSOR_REQUIRED



def _missing(obj: dict[str, Any], required: tuple[str, ...]) -> list[str]:
    """Check required key presence without treating a meaningful empty list as absent.

    Empty lists are valid for fields such as sponsor_claims_supported when the
    Expert supports none of the Sponsor claims. Content adequacy is measured by
    separate QC/judge fields rather than misclassified as a schema failure.
    """
    return [
        k for k in required
        if k not in obj or obj[k] is None or obj[k] == "" or obj[k] == {}
    ]


def _statement_is_not_assessable(value: Any) -> bool:
    text = str(value or "").strip().lower().replace("_", " ")
    return "not assessable" in text or "not provided" in text


def qc_panel(packet: dict[str, Any], panel: dict[str, Any]) -> dict[str, Any]:
    sponsor = panel.get("sponsor") or {}
    expert = panel.get("expert") or {}
    chair = panel.get("chair") or {}
    role_ids = {
        "sponsor": collect_evidence_ids(sponsor),
        "expert": collect_evidence_ids(expert),
        "chair": collect_evidence_ids(chair),
    }
    valid_ids_by_role = {
        role: set(visible_evidence_ids(packet, role)) for role in ("sponsor", "expert", "chair")
    }
    valid_ids = set().union(*valid_ids_by_role.values())
    cited_ids = set().union(*role_ids.values())
    invalid_ids_by_role = {
        role: sorted(role_ids[role] - valid_ids_by_role[role]) for role in role_ids
    }
    invalid_ids = sorted(set().union(*[set(x) for x in invalid_ids_by_role.values()]))
    prohibited = sorted(
        set(sponsor).intersection(PROHIBITED_KEYS)
        | set(expert).intersection(PROHIBITED_KEYS)
        | set(chair).intersection(PROHIBITED_KEYS)
    )
    rank_ok = all(int(x.get("graphdrx_rank", -1)) == int(packet["rank"]) for x in (sponsor, expert, chair))
    identity_ok = all(
        str(x.get("disease", "")).lower() == packet["disease"].lower()
        and str(x.get("drug", "")).lower() == packet["drug"].lower()
        for x in (sponsor, expert, chair)
    )
    missing_s = _missing(sponsor, SPONSOR_REQUIRED)
    missing_e = _missing(expert, EXPERT_REQUIRED)
    missing_c = _missing(chair, CHAIR_REQUIRED)
    observed_inferred_separated = "observed_evidence" in sponsor and "inferred_mechanism" in sponsor
    each_role_cites_evidence = all(bool(ids) for ids in role_ids.values())
    claim_adjudication = chair.get("claim_adjudication") or []
    chair_claim_adjudication_present = bool(claim_adjudication) and all(
        isinstance(item, dict)
        and item.get("status") in {"retained", "weakened", "rejected"}
        and bool(item.get("rationale"))
        for item in claim_adjudication
    )
    expert_direction = expert.get("pharmacological_action_direction") or {}
    expert_direction_fields_present = bool(
        expert_direction.get("edge_direction_support")
    ) and bool(
        expert_direction.get("therapeutic_direction_consistency")
    )
    expert_relation = str(expert_direction.get("direction_relation") or "")
    expert_status = str(expert_direction.get("therapeutic_direction_consistency") or "")
    expert_direction_trace_present = all(
        bool(expert_direction.get(key))
        for key in (
            "disease_direction_statement",
            "drug_action_statement",
            "direction_comparison",
            "direction_relation",
        )
    ) and (
        bool(expert_direction.get("disease_direction_evidence_ids"))
        or _statement_is_not_assessable(expert_direction.get("disease_direction_statement"))
    ) and (
        bool(expert_direction.get("drug_action_evidence_ids"))
        or _statement_is_not_assessable(expert_direction.get("drug_action_statement"))
    )

    chair_direction = chair.get("therapeutic_direction_synthesis") or {}
    chair_relation = str(chair_direction.get("direction_relation") or "")
    chair_status = str(chair_direction.get("status") or "")
    chair_direction_synthesis_present = all(
        bool(chair_direction.get(key))
        for key in (
            "disease_direction_statement",
            "drug_action_statement",
            "direction_comparison",
            "direction_relation",
            "status",
            "assessment",
        )
    ) and (
        bool(chair_direction.get("disease_direction_evidence_ids"))
        or _statement_is_not_assessable(chair_direction.get("disease_direction_statement"))
    ) and (
        bool(chair_direction.get("drug_action_evidence_ids"))
        or _statement_is_not_assessable(chair_direction.get("drug_action_statement"))
    )

    catalog_category = {
        str(item.get("evidence_id")): str(item.get("category"))
        for item in (packet.get("evidence_catalog") or [])
        if isinstance(item, dict) and item.get("evidence_id")
    }
    disease_categories = {"disease_context"}
    drug_action_categories = {
        "kg_relation", "drug_gene_mechanism", "drug_pharmacology", "relation_semantics"
    }

    expert_disease_ids = {
        str(x) for x in (expert_direction.get("disease_direction_evidence_ids") or [])
    }
    expert_drug_ids = {
        str(x) for x in (expert_direction.get("drug_action_evidence_ids") or [])
    }
    chair_disease_ids = {
        str(x) for x in (chair_direction.get("disease_direction_evidence_ids") or [])
    }
    chair_drug_ids = {
        str(x) for x in (chair_direction.get("drug_action_evidence_ids") or [])
    }

    expert_direction_cites_disease_context = (
        _statement_is_not_assessable(expert_direction.get("disease_direction_statement"))
        or any(catalog_category.get(x) in disease_categories for x in expert_disease_ids)
    )
    expert_direction_cites_drug_action = (
        _statement_is_not_assessable(expert_direction.get("drug_action_statement"))
        or any(catalog_category.get(x) in drug_action_categories for x in expert_drug_ids)
    )
    chair_direction_cites_disease_context = (
        _statement_is_not_assessable(chair_direction.get("disease_direction_statement"))
        or any(catalog_category.get(x) in disease_categories for x in chair_disease_ids)
    )
    chair_direction_cites_drug_action = (
        _statement_is_not_assessable(chair_direction.get("drug_action_statement"))
        or any(catalog_category.get(x) in drug_action_categories for x in chair_drug_ids)
    )
    direction_grounding_complete = all((
        expert_direction_cites_disease_context,
        expert_direction_cites_drug_action,
        chair_direction_cites_disease_context,
        chair_direction_cites_drug_action,
    ))
    expert_direction_label_consistent = (
        DIRECTION_RELATION_TO_STATUS.get(expert_relation) == expert_status
    )
    chair_direction_label_consistent = (
        DIRECTION_RELATION_TO_STATUS.get(chair_relation) == chair_status
    )

    input_grounding_ready = bool((packet.get("evidence_coverage") or {}).get("core_ready"))
    checks = {
        "json_valid": True,
        "input_grounding_ready": input_grounding_ready,
        "identity_valid": identity_ok,
        "rank_preserved": rank_ok,
        "required_fields_complete": not (missing_s or missing_e or missing_c),
        "evidence_ids_valid": not invalid_ids,
        "each_role_cites_evidence": each_role_cites_evidence,
        "observed_inferred_separated": observed_inferred_separated,
        "expert_direction_fields_present": expert_direction_fields_present,
        "chair_claim_adjudication_present": chair_claim_adjudication_present,
        "prohibited_fields_absent": not prohibited,
    }
    quality_flags = {
        "expert_direction_trace_present": expert_direction_trace_present,
        "chair_direction_synthesis_present": chair_direction_synthesis_present,
        "expert_direction_cites_disease_context": expert_direction_cites_disease_context,
        "expert_direction_cites_drug_action": expert_direction_cites_drug_action,
        "chair_direction_cites_disease_context": chair_direction_cites_disease_context,
        "chair_direction_cites_drug_action": chair_direction_cites_drug_action,
        "direction_grounding_complete": direction_grounding_complete,
        "expert_direction_label_consistent": expert_direction_label_consistent,
        "chair_direction_label_consistent": chair_direction_label_consistent,
    }
    return {
        "case_id": packet["case_id"],
        "disease": packet["disease"],
        "drug": packet["drug"],
        "configuration": panel["configuration"],
        **checks,
        **quality_flags,
        "qc_pass": all(checks.values()),
        "direction_quality_pass": all((
            expert_direction_trace_present,
            chair_direction_synthesis_present,
            direction_grounding_complete,
            expert_direction_label_consistent,
            chair_direction_label_consistent,
        )),
        "n_valid_evidence_items": len(valid_ids),
        "n_sponsor_citations": len(role_ids["sponsor"]),
        "n_expert_citations": len(role_ids["expert"]),
        "n_chair_citations": len(role_ids["chair"]),
        "invalid_evidence_ids": invalid_ids,
        "invalid_sponsor_evidence_ids": invalid_ids_by_role["sponsor"],
        "invalid_expert_evidence_ids": invalid_ids_by_role["expert"],
        "invalid_chair_evidence_ids": invalid_ids_by_role["chair"],
        "missing_sponsor_fields": missing_s,
        "missing_expert_fields": missing_e,
        "missing_chair_fields": missing_c,
        "prohibited_fields": prohibited,
    }


def load_panel(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
