from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_") or "item"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(obj, dict):
                raise ValueError(f"Expected object at {path}:{line_no}")
            rows.append(obj)
    return rows


def append_jsonl(path: str | Path, obj: dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def atomic_write_json(path: str | Path, obj: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=p.parent, delete=False) as tmp:
        json.dump(obj, tmp, ensure_ascii=False, indent=2)
        tmp.write("\n")
        temp_name = tmp.name
    os.replace(temp_name, p)


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with Path(path).open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj or {}


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise ValueError("Model output is not a JSON object")
    return obj


def existing_case_ids(paths: Iterable[str | Path]) -> set[str]:
    ids: set[str] = set()
    for path in paths:
        p = Path(path)
        if not p.exists():
            continue
        for row in read_jsonl(p):
            cid = row.get("case_id")
            if cid:
                ids.add(str(cid))
    return ids


_EVIDENCE_ID_SCALAR_KEYS = {"evidence_id", "path_id"}
def _is_evidence_id_list_key(key: str) -> bool:
    """Return True for every list-valued evidence-ID field.

    Role schemas may add fields such as disease_direction_evidence_ids or
    drug_action_evidence_ids. Matching the suffix prevents future fields from
    bypassing validation, normalization, or judge packet construction.
    """
    return key == "evidence_ids" or key.endswith("_evidence_ids")


def collect_evidence_ids(obj: Any) -> set[str]:
    """Recursively collect IDs from all evidence-ID fields.

    Non-list values in a *_evidence_ids field are collected as strings so that
    malformed values are exposed to validity checks rather than silently
    ignored.
    """
    found: set[str] = set()
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _EVIDENCE_ID_SCALAR_KEYS:
                if isinstance(value, str):
                    found.add(value)
                elif value is not None:
                    found.add(str(value))
            elif _is_evidence_id_list_key(key):
                if isinstance(value, list):
                    found.update(str(item) for item in value)
                elif value is not None:
                    found.add(str(value))
            else:
                found.update(collect_evidence_ids(value))
    elif isinstance(obj, list):
        for value in obj:
            found.update(collect_evidence_ids(value))
    return found


def _evidence_id_signature(value: str) -> tuple[object, ...]:
    """Return a formatting-insensitive evidence-ID signature.

    Letter groups are compared case-insensitively and numeric groups as
    integers, so RELSEM02 and RELSEM002 have the same signature. This does
    not collapse different letter/number sequences.
    """
    tokens = re.findall(r"[A-Za-z]+|\d+", str(value))
    return tuple(int(token) if token.isdigit() else token.upper() for token in tokens)


def canonicalize_evidence_ids(
    obj: Any, valid_ids: Iterable[str]
) -> tuple[Any, list[dict[str, str]]]:
    """Canonicalize unambiguous formatting variants of evidence IDs.

    Only values stored in evidence-ID fields are considered. Exact matches
    are preserved. A non-exact value is rewritten only when its
    letter/number signature maps to exactly one valid packet evidence ID.
    Semantically different or ambiguous IDs remain unchanged and are later
    rejected by normal validation.
    """
    valid = [str(value) for value in valid_ids]
    valid_set = set(valid)
    by_signature: dict[tuple[object, ...], list[str]] = {}
    for value in valid:
        by_signature.setdefault(_evidence_id_signature(value), []).append(value)

    replacements: list[dict[str, str]] = []

    def resolve(value: Any) -> Any:
        if not isinstance(value, str) or value in valid_set:
            return value
        matches = list(dict.fromkeys(by_signature.get(_evidence_id_signature(value), [])))
        if len(matches) == 1:
            replacements.append({"from": value, "to": matches[0]})
            return matches[0]
        return value

    def visit(value: Any) -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, child in value.items():
                if key in _EVIDENCE_ID_SCALAR_KEYS:
                    out[key] = resolve(child)
                elif _is_evidence_id_list_key(key) and isinstance(child, list):
                    out[key] = [resolve(item) for item in child]
                else:
                    out[key] = visit(child)
            return out
        if isinstance(value, list):
            return [visit(item) for item in value]
        return value

    normalized = visit(obj)
    unique_replacements = list({(x["from"], x["to"]): x for x in replacements}.values())
    return normalized, unique_replacements


_EVIDENCE_ID_PLACEHOLDER_TOKENS = {
    "",
    "n a",
    "na",
    "none",
    "not applicable",
    "not assessable",
    "not available",
    "not provided",
    "no evidence",
    "no evidence id",
    "unknown",
}


def _normalize_evidence_placeholder(value: str) -> str:
    normalized = str(value).strip().lower()
    normalized = re.sub(r"[\s_\-/]+", " ", normalized)
    return normalized.strip()


def sanitize_evidence_id_placeholders(
    obj: Any, valid_ids: Iterable[str] = ()
) -> tuple[Any, list[dict[str, str]]]:
    """Remove exact natural-language null sentinels from evidence-ID arrays.

    Models occasionally place values such as ``not assessable`` inside a
    ``*_evidence_ids`` array. Those values express missingness, not evidence,
    and therefore have the deterministic JSON representation ``[]``. Only a
    small exact allowlist of null sentinels is removed. Arbitrary invalid IDs
    remain untouched and are rejected by normal validation.
    """
    valid_set = {str(value) for value in valid_ids}
    removals: list[dict[str, str]] = []

    def is_placeholder(value: Any) -> bool:
        return (
            isinstance(value, str)
            and value not in valid_set
            and _normalize_evidence_placeholder(value) in _EVIDENCE_ID_PLACEHOLDER_TOKENS
        )

    def visit(value: Any, path: str = "") -> Any:
        if isinstance(value, dict):
            out: dict[str, Any] = {}
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                if _is_evidence_id_list_key(key):
                    if isinstance(child, list):
                        cleaned: list[Any] = []
                        for item in child:
                            if is_placeholder(item):
                                removals.append({
                                    "field": child_path,
                                    "value": str(item),
                                    "action": "removed_null_sentinel",
                                })
                            else:
                                cleaned.append(item)
                        out[key] = cleaned
                    elif is_placeholder(child):
                        removals.append({
                            "field": child_path,
                            "value": str(child),
                            "action": "replaced_null_sentinel_with_empty_list",
                        })
                        out[key] = []
                    else:
                        out[key] = visit(child, child_path)
                else:
                    out[key] = visit(child, child_path)
            return out
        if isinstance(value, list):
            return [visit(item, f"{path}[{index}]") for index, item in enumerate(value)]
        return value

    normalized = visit(obj)
    unique = list({(x["field"], x["value"], x["action"]): x for x in removals}.values())
    return normalized, unique
