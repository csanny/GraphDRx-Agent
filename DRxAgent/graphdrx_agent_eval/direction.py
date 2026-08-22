from __future__ import annotations

from typing import Any


DIRECTION_RELATION_TO_STATUS = {
    "counteracts": "supported",
    "reinforces": "conflicted",
    "ambiguous": "uncertain",
    "not_assessable": "not_assessable",
}


def derive_direction_status(relation: str) -> str:
    try:
        return DIRECTION_RELATION_TO_STATUS[str(relation)]
    except KeyError as exc:
        raise ValueError(
            "direction_relation must be one of: "
            + ", ".join(DIRECTION_RELATION_TO_STATUS)
        ) from exc


def normalize_direction_fields(role: str, obj: dict[str, Any]) -> dict[str, Any]:
    """Add deterministic status labels from the model-selected direction relation.

    The model chooses the substantive relation. Code maps that relation to the
    reporting label so a report cannot state 'counteracts' while labeling the
    result 'conflicted'.
    """
    if role == "expert":
        block = obj.get("pharmacological_action_direction")
        if isinstance(block, dict) and block.get("direction_relation"):
            block["therapeutic_direction_consistency"] = derive_direction_status(
                str(block["direction_relation"])
            )
    elif role == "chair":
        block = obj.get("therapeutic_direction_synthesis")
        if isinstance(block, dict) and block.get("direction_relation"):
            block["status"] = derive_direction_status(str(block["direction_relation"]))
    return obj
