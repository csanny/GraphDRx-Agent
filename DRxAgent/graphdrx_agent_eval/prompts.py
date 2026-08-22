from __future__ import annotations

import json
from typing import Any


COMMON_RULES = """
Mandatory grounding rules:
1. Use only the supplied evidence packet and prior-agent output. General model knowledge is not admissible evidence.
2. Do not introduce drug properties, disease biology, clinical outcomes, safety findings, or mechanisms absent from the packet.
3. When information is absent, write "not provided" or "not assessable". Missing information is uncertainty, not negative evidence.
4. Preserve the original GraphDRx rank and retrieval score. Do not rerank candidates.
5. Do not assign Advance/Hold/No-go, approval, rejection, clinical-success probability, or any regulatory decision.
6. Distinguish directly supplied observations from interpretations and experimentally testable hypotheses.
7. Association, PPI, general interaction, ontology, enzyme, binding, carrier, transporter, and other nondirectional relations are not by themselves causal, inhibitory, activating, or therapeutic evidence.
8. Never convert a nondirectional or ambiguous relation into inhibition, activation, agonism, antagonism, upregulation, or downregulation unless that direction is explicitly supplied in the evidence packet.
9. A GraphDRx path may represent a therapeutic, neutral, or adverse relationship. Do not presume therapeutic benefit from retrieval rank or path existence.
10. Cite only evidence IDs listed in the packet. Never invent, alter, or combine evidence IDs.
11. If a substantive assessment has no supporting evidence ID, state that it is not assessable rather than relying on background knowledge.
12. Return one JSON object only, without markdown fences.
""".strip()


_DIRECTION_CUES = (
    "excess", "high level", "elevated", "increased", "increase", "overproduction",
    "hyperactive", "activation", "activated", "upregulat", "gain-of-function",
    "deficien", "low level", "reduced", "decreased", "decrease", "loss-of-function",
    "hypoactive", "suppression", "suppressed", "downregulat",
    "agonist", "antagonist", "inhibitor", "inhibition", "inhibit", "activator",
    "activate", "block", "stimulat", "corticosteroid", "glucocorticoid",
)


def _budget_items(items: list[dict[str, Any]], max_chars: int) -> tuple[list[dict[str, Any]], int]:
    selected: list[dict[str, Any]] = []
    used = 0
    for item in items:
        size = len(json.dumps(item, ensure_ascii=False))
        if selected and used + size > max_chars:
            break
        if not selected and size > max_chars:
            compact = dict(item)
            for key in ("value", "text", "assessment"):
                if key in compact and isinstance(compact[key], str):
                    compact[key] = compact[key][:max_chars] + " ... [TRUNCATED FOR PROMPT]"
            selected.append(compact)
            used += len(json.dumps(compact, ensure_ascii=False))
            break
        selected.append(item)
        used += size
    return selected, max(0, len(items) - len(selected))


def _catalog_index(packet: dict[str, Any], allowed_ids: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "evidence_id": item.get("evidence_id"),
            "category": item.get("category"),
            "source": item.get("source"),
        }
        for item in packet.get("evidence_catalog", [])
        if str(item.get("evidence_id")) in allowed_ids
    ]


def _item_text(item: dict[str, Any]) -> str:
    return json.dumps(item, ensure_ascii=False).lower()


def _compact_focus_item(item: dict[str, Any], max_chars: int = 2500) -> dict[str, Any]:
    compact = dict(item)
    for key in ("value", "text", "assessment"):
        if key in compact and isinstance(compact[key], str) and len(compact[key]) > max_chars:
            compact[key] = compact[key][:max_chars] + " ... [TRUNCATED FOR DIRECTIONAL FOCUS]"
    return compact


def _directional_evidence_focus(packet: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Surface explicit directional evidence without deriving a conclusion.

    This is an evidence-selection aid only. It does not label a candidate as
    supported or conflicted.
    """
    sections = packet.get("evidence_sections") or {}

    disease_items: list[dict[str, Any]] = []
    for item in sections.get("disease_context", []):
        text = _item_text(item)
        if any(cue in text for cue in _DIRECTION_CUES):
            disease_items.append(_compact_focus_item(item))
        if len(disease_items) >= 4:
            break

    drug_items: list[dict[str, Any]] = []
    directional_relations = {
        "activation", "inhibition", "agonist", "antagonist", "upregulation",
        "downregulation", "stimulation", "suppression",
    }

    # Exact KG/mechanism actions are highest priority.
    for section_name in ("path_relation_evidence", "drug_gene_mechanism"):
        for item in sections.get(section_name, []):
            relation = str(item.get("relation") or item.get("edge_relation") or "").lower()
            action_type = str((item.get("properties") or {}).get("Action_Type") or "").lower()
            text = _item_text(item)
            if (
                relation in directional_relations
                or action_type in directional_relations
                or any(cue in text for cue in _DIRECTION_CUES)
            ):
                drug_items.append(_compact_focus_item(item))
            if len(drug_items) >= 3:
                break
        if len(drug_items) >= 3:
            break

    # Prefer concise pharmacology fields over large category/ATC lists.
    pharmacology_items = list(sections.get("drug_pharmacology", []))
    preferred_tokens = (
        "mechanism", "pharmacodynamic", "description", "indication class",
        "action", "agonist", "antagonist",
    )
    pharmacology_items.sort(
        key=lambda item: (
            0 if any(token in str(item.get("field", "")).lower() for token in preferred_tokens) else 1,
            str(item.get("evidence_id", "")),
        )
    )
    for item in pharmacology_items:
        text = _item_text(item)
        if any(cue in text for cue in _DIRECTION_CUES):
            drug_items.append(_compact_focus_item(item))
        if len(drug_items) >= 7:
            break

    return {
        "disease_direction_evidence": disease_items,
        "drug_action_evidence": drug_items,
    }


def compact_packet(packet: dict[str, Any], role: str) -> dict[str, Any]:
    sections = packet.get("evidence_sections") or {}
    common = {
        "case_id": packet["case_id"],
        "disease": packet["disease"],
        "therapeutic_area": packet["therapeutic_area"],
        "drug": packet["drug"],
        "graphdrx_rank": packet["rank"],
        "retrieval_score": packet["retrieval_score"],
        "graphdrx": packet.get("graphdrx") or {},
        "evidence_coverage": packet.get("evidence_coverage") or {},
    }
    if role == "sponsor":
        budgets = {
            "graphdrx": 6000,
            "relation_semantics": 4000,
            "disease_context": 14000,
            "drug_pharmacology": 14000,
            "path_relation_evidence": 8000,
            "drug_gene_mechanism": 9000,
            "drug_disease_development": 6000,
        }
    else:
        budgets = {
            "graphdrx": 6000,
            "relation_semantics": 4000,
            "disease_context": 14000,
            "drug_pharmacology": 14000,
            "physicochemical_properties": 6000,
            "pk_pd_exposure": 5000,
            "safety_developability": 7000,
            "path_relation_evidence": 8000,
            "drug_gene_mechanism": 9000,
            "drug_disease_development": 6000,
            "counter_evidence": 7000,
        }

    selected_sections: dict[str, list[dict[str, Any]]] = {}
    omitted: dict[str, int] = {}
    allowed_ids: set[str] = set()
    for name, budget in budgets.items():
        selected, n_omitted = _budget_items(list(sections.get(name, [])), budget)
        selected_sections[name] = selected
        omitted[name] = n_omitted
        for item in selected:
            if isinstance(item, dict) and item.get("evidence_id"):
                allowed_ids.add(str(item["evidence_id"]))

    directional_focus = _directional_evidence_focus(packet)
    for items in directional_focus.values():
        for item in items:
            if item.get("evidence_id"):
                allowed_ids.add(str(item["evidence_id"]))

    common["directional_evidence_focus"] = directional_focus
    common["evidence_sections"] = selected_sections
    common["prompt_omitted_evidence_counts"] = omitted
    common["valid_evidence_id_index"] = _catalog_index(packet, allowed_ids)
    return common


def visible_evidence_ids(packet: dict[str, Any], role: str) -> list[str]:
    compact = compact_packet(packet, role)
    return [str(x["evidence_id"]) for x in compact.get("valid_evidence_id_index", [])]


def sponsor_prompt(packet: dict[str, Any]) -> str:
    schema = {
        "disease": packet["disease"],
        "drug": packet["drug"],
        "graphdrx_rank": packet["rank"],
        "kg_grounded_hypothesis": "string",
        "observed_evidence": [{"claim": "string", "evidence_ids": ["ID"]}],
        "inferred_mechanism": [{"claim": "string", "based_on_evidence_ids": ["ID"], "assumption": "string"}],
        "path_and_relation_interpretation": "string",
        "supporting_evidence_ids": ["ID"],
        "key_assumptions": ["string"],
        "alternative_explanations": ["string"],
        "remaining_uncertainties": ["string"],
    }
    return f"""You are the Sponsor Agent in a drug-repurposing evidence-interpretation workflow.
Formulate a specific, testable drug-target-disease hypothesis from the supplied GraphDRx, KG, and RAG evidence. You are not making a development decision.

The observed_evidence field must contain only statements directly present in supplied evidence. The inferred_mechanism field may connect supplied facts, but each claim must identify its assumptions and supporting evidence IDs.

Do not presume that the retrieved path is beneficial. If supplied disease and drug evidence suggest an adverse or directionally conflicting relationship, frame the hypothesis neutrally as an effect to be tested rather than as a therapeutic mechanism.

{COMMON_RULES}

Grounded candidate packet:
{json.dumps(compact_packet(packet, 'sponsor'), ensure_ascii=False, indent=2)}

Required JSON schema example:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def expert_prompt(packet: dict[str, Any], sponsor: dict[str, Any]) -> str:
    schema = {
        "disease": packet["disease"],
        "drug": packet["drug"],
        "graphdrx_rank": packet["rank"],
        "overall_appraisal": "string",
        "mechanistic_plausibility": {"assessment": "string", "evidence_ids": ["ID"], "limitations": ["string"]},
        "kg_relation_consistency": {"assessment": "string", "evidence_ids": ["ID"], "overreach_flags": ["string"]},
        "pharmacological_action_direction": {
            "disease_direction_statement": "string",
            "disease_direction_evidence_ids": ["ID"],
            "drug_action_statement": "string",
            "drug_action_evidence_ids": ["ID"],
            "direction_comparison": "string",
            "direction_relation": "counteracts|reinforces|ambiguous|not_assessable",
            "assessment": "string",
            "edge_direction_support": "supported|uncertain|conflicted|not_assessable",
            "evidence_ids": ["ID"],
        },
        "physicochemical_pk_pd_exposure": {
            "assessment": "string",
            "data_availability": "available|limited|not_provided",
            "evidence_ids": ["ID"],
        },
        "safety_developability": {
            "assessment": "string",
            "data_availability": "available|limited|not_provided",
            "evidence_ids": ["ID"],
        },
        "evidence_limitations": ["string"],
        "sponsor_claims_supported": ["string"],
        "sponsor_claims_challenged": ["string"],
        "proposed_validation_experiment": {
            "model": "string",
            "intervention": "string",
            "comparator": "string",
            "readouts": ["string"],
            "supporting_outcome": "string",
            "rejecting_outcome": "string",
            "rationale_evidence_ids": ["ID"],
        },
    }
    return f"""You are the Scientific Expert Agent. Independently appraise the Sponsor's hypothesis from mechanistic, pharmacological, physicochemical, PK/PD, exposure, safety, and experimental perspectives. The purpose is constructive scientific review, not mandatory opposition and not a development decision.

Evaluate direction in this mandatory order:
1. disease_direction_statement: state the disease-defining directional abnormality using supplied disease-context evidence. Cite at least one disease-context evidence ID, or write not assessable.
2. drug_action_statement: state the explicit drug action using supplied KG/RAG evidence. Cite at least one drug-action evidence ID, or write not assessable.
3. direction_comparison: state exactly one relationship between the explicit drug action and the disease abnormality.
4. direction_relation: choose exactly one of:
   - counteracts: the drug action plausibly opposes the disease-defining abnormality;
   - reinforces: the drug action plausibly strengthens the disease-defining abnormality;
   - ambiguous: both directions are supplied, but their relationship cannot be resolved;
   - not_assessable: the explicit disease direction or explicit drug action is absent.
5. Only then assign edge_direction_support. Do not output a therapeutic status label; code derives it deterministically from direction_relation.

Do not call a relation reinforces merely because evidence is indirect or insufficient. Weak pathway support is an evidence-strength limitation, not an adverse therapeutic direction. Likewise, do not call a relation counteracts merely because the candidate is already used clinically. Base direction_relation only on the supplied disease direction and explicit drug action.

If the packet explicitly shows that the drug action and disease abnormality point in the same adverse direction, use reinforces. For example, if supplied disease evidence identifies excess glucocorticoid exposure as disease-defining and supplied drug evidence identifies glucocorticoid-receptor agonism or activation, use reinforces unless the packet itself supplies a countervailing mechanism.

For edge_direction_support, assess only whether the exact drug-target action is explicitly supported. Nondirectional relations such as enzyme, binding, association, interaction, PPI, carrier, transporter, or ontology relations must not be rewritten as inhibition, activation, agonism, or antagonism.

When physicochemical, PK/exposure, safety, withdrawal, or warning data are supplied, interpret at least one actual supplied value or flag in each available section. Do not merely say that data are available. Avoid unsupported thresholds.

If direction_relation is reinforces, frame the validation experiment as a falsification or risk-characterization experiment that can distinguish worsening from a countervailing mechanism. Do not presume amelioration.

For each evidence-based assessment, cite packet evidence IDs. Do not use background pharmacology or safety knowledge that is absent from the packet.

{COMMON_RULES}

Grounded candidate packet:
{json.dumps(compact_packet(packet, 'expert'), ensure_ascii=False, indent=2)}

Sponsor output:
{json.dumps(sponsor, ensure_ascii=False, indent=2)}

Required JSON schema example:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def chair_prompt(packet: dict[str, Any], sponsor: dict[str, Any], expert: dict[str, Any]) -> str:
    schema = {
        "disease": packet["disease"],
        "drug": packet["drug"],
        "graphdrx_rank": packet["rank"],
        "retrieval_score": packet["retrieval_score"],
        "integrated_mechanistic_interpretation": {"assessment": "string", "evidence_ids": ["ID"]},
        "kg_path_interpretation": {"assessment": "string", "evidence_ids": ["ID"]},
        "therapeutic_direction_synthesis": {
            "disease_direction_statement": "string",
            "disease_direction_evidence_ids": ["ID"],
            "drug_action_statement": "string",
            "drug_action_evidence_ids": ["ID"],
            "direction_comparison": "string",
            "direction_relation": "counteracts|reinforces|ambiguous|not_assessable",
            "assessment": "string",
            "evidence_ids": ["ID"],
        },
        "supporting_evidence": [{"statement": "string", "evidence_ids": ["ID"]}],
        "claim_adjudication": [{
            "sponsor_claim": "string",
            "status": "retained|weakened|rejected",
            "rationale": "string",
            "evidence_ids": ["ID"],
        }],
        "key_assumptions": ["string"],
        "remaining_evidence_gaps": ["string"],
        "pk_exposure_considerations": {"assessment": "string", "evidence_ids": ["ID"]},
        "safety_considerations": {"assessment": "string", "evidence_ids": ["ID"]},
        "validation_experiment": {
            "model": "string",
            "intervention": "string",
            "comparator": "string",
            "readouts": ["string"],
            "supporting_outcome": "string",
            "rejecting_outcome": "string",
            "rationale_evidence_ids": ["ID"],
        },
        "research_oriented_interpretation": "string",
    }
    return f"""You are the Chair Agent. Synthesize the Sponsor hypothesis and Scientific Expert appraisal into a concise, evidence-grounded drug-repurposing report. Resolve disagreements explicitly and preserve material uncertainty. Do not merely copy either agent.

First complete therapeutic_direction_synthesis in this order:
1. restate the disease-defining direction and cite disease-context evidence;
2. restate the explicit drug action and cite drug-action evidence;
3. compare the two directions;
4. choose direction_relation as counteracts, reinforces, ambiguous, or not_assessable.

Use counteracts only when the explicit drug action plausibly opposes the disease abnormality. Use reinforces only when it plausibly strengthens the disease abnormality. Use ambiguous for unresolved relationships. Use not_assessable when an explicit disease direction or drug action is absent. Do not output a status label; code derives it deterministically from direction_relation.

Indirectness, association-only paths, or weak causal support must be described as evidence limitations. They do not by themselves change counteracts into reinforces. If the Chair departs from the Expert's direction_relation, explain the disagreement using additional supplied packet evidence.

If supplied evidence shows that the drug action plausibly reinforces a disease-defining abnormality, use reinforces; the final report must not call the candidate a therapeutic avenue, imply likely benefit, or soften the conflict to generic uncertainty. Corresponding Sponsor benefit claims must be weakened or rejected.

For every material Sponsor claim, populate claim_adjudication with retained, weakened, or rejected and explain why.

Do not convert a nondirectional or ambiguous relation such as enzyme, binding, association, interaction, PPI, carrier, transporter, or ontology into inhibition, activation, agonism, antagonism, upregulation, or downregulation unless that exact direction is explicitly supplied in the evidence packet. If an upstream agent introduces such a direction without evidence, weaken or reject that claim.

Any factual or mechanistic statement retained in the final report must cite evidence IDs from the packet. If PK, exposure, or safety evidence is absent, state that it is not assessable; when those data are supplied, interpret at least one actual value or flag without inventing thresholds.

If direction_relation is reinforces, the validation experiment must test worsening versus a countervailing mechanism rather than assume amelioration. A generic pathway change alone is not sufficient.

{COMMON_RULES}

Grounded candidate packet:
{json.dumps(compact_packet(packet, 'chair'), ensure_ascii=False, indent=2)}

Sponsor output:
{json.dumps(sponsor, ensure_ascii=False, indent=2)}

Scientific Expert output:
{json.dumps(expert, ensure_ascii=False, indent=2)}

Required JSON schema example:
{json.dumps(schema, ensure_ascii=False, indent=2)}
"""


def repair_prompt(
    role: str,
    invalid_text: str,
    errors: list[str],
    valid_evidence_ids: list[str] | None = None,
    original_prompt: str | None = None,
) -> str:
    return f"""The previous {role} output failed deterministic validation.
Return one complete corrected JSON object only, using the same grounded evidence packet and prior-agent outputs reproduced below.

Correction rules:
1. Correct only the reported validation errors and any directly dependent fields. Preserve all other supported content.
2. Do not invent scientific facts or evidence IDs. Use only evidence explicitly supplied in the original task.
3. Re-evaluate a direction status when the validation error shows that its required evidence trace is missing or inconsistent.
4. For therapeutic direction, report direction_relation only:
   - counteracts when the drug action opposes the disease abnormality;
   - reinforces when the drug action strengthens the disease abnormality;
   - ambiguous when both are supplied but their relationship is unresolved;
   - not_assessable when explicit disease direction or explicit drug action is absent.
   The code derives the status label deterministically.
5. A target, enzyme, binding, association, interaction, PPI, carrier, transporter, or ontology relation is not an explicit activating or inhibitory action unless the packet states that direction.
6. If direction_relation is not_assessable, cite the supplied evidence showing the available disease context and/or the nondirectional or missing-action basis; do not manufacture a directional citation.
7. Keep disease, drug, GraphDRx rank, retrieval score, and all prohibited-decision constraints unchanged.

Validation errors:
{json.dumps(errors, ensure_ascii=False)}

Valid evidence IDs:
{json.dumps(valid_evidence_ids or [], ensure_ascii=False)}

Original grounded task:
{original_prompt or ''}

Previous invalid output:
{invalid_text}
"""


def disease_summary_prompt(disease: str, chair_reports: list[dict[str, Any]]) -> str:
    return f"""You are the Chair Agent. Summarize the five preserved-rank candidate reports for {disease}.
Identify recurring mechanistic patterns and common evidence limitations. Use only the supplied reports. Do not rerank, select a winner, assign triage, or predict clinical success.
Return JSON with keys: disease, recurring_mechanistic_patterns, common_evidence_limitations, cross_candidate_experimental_priorities.
Reports:
{json.dumps(chair_reports, ensure_ascii=False, indent=2)}
"""
