from __future__ import annotations

from typing import Any


STR = {"type": "string"}
STR_LIST = {"type": "array", "items": STR}

PREFLIGHT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["ok"]}
    },
    "required": ["status"],
    "additionalProperties": False,
}


def _object(properties: dict[str, Any], required: list[str] | tuple[str, ...] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(required or properties.keys()),
        "additionalProperties": False,
    }


SPONSOR_SCHEMA = _object({
    "disease": STR,
    "drug": STR,
    "graphdrx_rank": {"type": "integer"},
    "kg_grounded_hypothesis": STR,
    "observed_evidence": {
        "type": "array",
        "items": _object({"claim": STR, "evidence_ids": STR_LIST}),
    },
    "inferred_mechanism": {
        "type": "array",
        "items": _object({
            "claim": STR,
            "based_on_evidence_ids": STR_LIST,
            "assumption": STR,
        }),
    },
    "path_and_relation_interpretation": STR,
    "supporting_evidence_ids": STR_LIST,
    "key_assumptions": STR_LIST,
    "alternative_explanations": STR_LIST,
    "remaining_uncertainties": STR_LIST,
})


EXPERT_SCHEMA = _object({
    "disease": STR,
    "drug": STR,
    "graphdrx_rank": {"type": "integer"},
    "overall_appraisal": STR,
    "mechanistic_plausibility": _object({
        "assessment": STR,
        "evidence_ids": STR_LIST,
        "limitations": STR_LIST,
    }),
    "kg_relation_consistency": _object({
        "assessment": STR,
        "evidence_ids": STR_LIST,
        "overreach_flags": STR_LIST,
    }),
    "pharmacological_action_direction": _object({
        "disease_direction_statement": STR,
        "disease_direction_evidence_ids": STR_LIST,
        "drug_action_statement": STR,
        "drug_action_evidence_ids": STR_LIST,
        "direction_comparison": STR,
        "direction_relation": {
            "type": "string",
            "enum": ["counteracts", "reinforces", "ambiguous", "not_assessable"],
        },
        "assessment": STR,
        "edge_direction_support": {
            "type": "string",
            "enum": ["supported", "uncertain", "conflicted", "not_assessable"],
        },
        "evidence_ids": STR_LIST,
    }),
    "physicochemical_pk_pd_exposure": _object({
        "assessment": STR,
        "data_availability": {"type": "string", "enum": ["available", "limited", "not_provided"]},
        "evidence_ids": STR_LIST,
    }),
    "safety_developability": _object({
        "assessment": STR,
        "data_availability": {"type": "string", "enum": ["available", "limited", "not_provided"]},
        "evidence_ids": STR_LIST,
    }),
    "evidence_limitations": STR_LIST,
    "sponsor_claims_supported": STR_LIST,
    "sponsor_claims_challenged": STR_LIST,
    "proposed_validation_experiment": _object({
        "model": STR,
        "intervention": STR,
        "comparator": STR,
        "readouts": STR_LIST,
        "supporting_outcome": STR,
        "rejecting_outcome": STR,
        "rationale_evidence_ids": STR_LIST,
    }),
})


CHAIR_SCHEMA = _object({
    "disease": STR,
    "drug": STR,
    "graphdrx_rank": {"type": "integer"},
    "retrieval_score": {"type": "number"},
    "integrated_mechanistic_interpretation": _object({
        "assessment": STR,
        "evidence_ids": STR_LIST,
    }),
    "kg_path_interpretation": _object({
        "assessment": STR,
        "evidence_ids": STR_LIST,
    }),
    "therapeutic_direction_synthesis": _object({
        "disease_direction_statement": STR,
        "disease_direction_evidence_ids": STR_LIST,
        "drug_action_statement": STR,
        "drug_action_evidence_ids": STR_LIST,
        "direction_comparison": STR,
        "direction_relation": {
            "type": "string",
            "enum": ["counteracts", "reinforces", "ambiguous", "not_assessable"],
        },
        "assessment": STR,
        "evidence_ids": STR_LIST,
    }),
    "supporting_evidence": {
        "type": "array",
        "items": _object({"statement": STR, "evidence_ids": STR_LIST}),
    },
    "claim_adjudication": {
        "type": "array",
        "minItems": 1,
        "items": _object({
            "sponsor_claim": STR,
            "status": {
                "type": "string",
                "enum": ["retained", "weakened", "rejected"],
            },
            "rationale": STR,
            "evidence_ids": STR_LIST,
        }),
    },
    "key_assumptions": STR_LIST,
    "remaining_evidence_gaps": STR_LIST,
    "pk_exposure_considerations": _object({
        "assessment": STR,
        "evidence_ids": STR_LIST,
    }),
    "safety_considerations": _object({
        "assessment": STR,
        "evidence_ids": STR_LIST,
    }),
    "validation_experiment": _object({
        "model": STR,
        "intervention": STR,
        "comparator": STR,
        "readouts": STR_LIST,
        "supporting_outcome": STR,
        "rejecting_outcome": STR,
        "rationale_evidence_ids": STR_LIST,
    }),
    "research_oriented_interpretation": STR,
})


DISEASE_SUMMARY_SCHEMA = _object({
    "disease": STR,
    "recurring_mechanistic_patterns": STR_LIST,
    "common_evidence_limitations": STR_LIST,
    "cross_candidate_experimental_priorities": STR_LIST,
})


def schema_for_role(role: str) -> dict[str, Any]:
    return {
        "sponsor": SPONSOR_SCHEMA,
        "expert": EXPERT_SCHEMA,
        "chair": CHAIR_SCHEMA,
    }[role]
