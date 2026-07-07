"""Extract attribute names from an uploaded structured representation.

The user uploads either:
  * an **OSDU record/instance** - e.g. a Well/Wellbore JSON with a ``data`` object
    (like the files in ``matched_json/``); we take every attribute key, whether
    its value is filled or ``null``; or
  * a **JSON Schema** - a definition with ``properties`` (OSDU schema service
    style, where the real attributes live under ``properties.data``); we take the
    declared property names; or
  * a **plain list** of attribute-name strings (JSON array or newline text).

Each extracted attribute is mapped to a friendly label when it is a recognised
OSDU field (see :func:`query_generation.osdu_context.friendly_label`) so the
generated query stays consistent with - and scoreable like - the hand-written
queries. Everything else keeps its raw field name.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from query_generation.osdu_context import friendly_label

# Top-level OSDU envelope / plumbing keys that are never document-answerable
# attributes. Only relevant when a record has no ``data`` object and we fall back
# to reading root keys.
_SYSTEM_KEYS = {
    "kind", "source", "acl", "type", "version", "tags", "meta", "ancestry",
    "modifyuser", "modifytime", "createtime", "createuser", "authority",
    "namespace", "legal", "id",
}


@dataclass
class ExtractedAttribute:
    """One attribute pulled from an uploaded schema/record."""

    raw_name: str                 # field name as it appears in the source
    label: str                    # friendly label if recognised, else raw_name
    is_null: Optional[bool] = None  # True/False for a record; None for a schema


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _load(source: Union[str, Dict[str, Any], list]) -> Any:
    """Accept a path, a parsed object, or a list; return a parsed object."""
    if isinstance(source, (dict, list)):
        return source
    if not isinstance(source, str):
        raise TypeError(f"unsupported source type: {type(source)!r}")
    if os.path.exists(source):
        with open(source, "r", encoding="utf-8") as f:
            text = f.read()
    else:
        text = source  # treat as raw content
    text_stripped = text.strip()
    if text_stripped.startswith("{") or text_stripped.startswith("["):
        return json.loads(text_stripped)
    # Fall back: newline/comma separated attribute names.
    return [ln.strip() for ln in text_stripped.replace(",", "\n").splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Schema vs record vs list
# ---------------------------------------------------------------------------


def _schema_property_names(node: Any) -> List[str]:
    """Recursively collect declared property names from a JSON-Schema node."""
    names: List[str] = []
    if not isinstance(node, dict):
        return names
    props = node.get("properties")
    if isinstance(props, dict):
        names.extend(props.keys())
    for combiner in ("allOf", "anyOf", "oneOf"):
        for sub in node.get(combiner, []) or []:
            names.extend(_schema_property_names(sub))
    return names


def _from_schema(obj: Dict[str, Any]) -> List[ExtractedAttribute]:
    props = obj.get("properties")
    names: List[str] = []
    if isinstance(props, dict) and isinstance(props.get("data"), dict):
        # OSDU schema: real attributes live under properties.data(.allOf[]).
        names = _schema_property_names(props["data"])
    if not names and isinstance(props, dict):
        names = [k for k in props.keys() if k.lower() not in _SYSTEM_KEYS]
    return [ExtractedAttribute(raw_name=n, label=friendly_label(n), is_null=None) for n in names]


def _from_record(data: Dict[str, Any]) -> List[ExtractedAttribute]:
    out: List[ExtractedAttribute] = []
    for key, value in data.items():
        if key.lower() in _SYSTEM_KEYS:
            continue
        is_null = value is None or (isinstance(value, (list, dict, str)) and len(value) == 0)
        out.append(ExtractedAttribute(raw_name=key, label=friendly_label(key), is_null=is_null))
    return out


def _from_list(items: list) -> List[ExtractedAttribute]:
    out: List[ExtractedAttribute] = []
    for item in items:
        name = str(item).strip()
        if name:
            out.append(ExtractedAttribute(raw_name=name, label=friendly_label(name), is_null=None))
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_attributes(
    source: Union[str, Dict[str, Any], list],
    *,
    include_null: bool = True,
) -> List[ExtractedAttribute]:
    """Return the attributes declared/present in ``source``.

    ``source`` may be a file path, raw JSON/text, a parsed dict, or a list of
    names. Deduplicated by final label (first occurrence wins), order preserved.
    When ``include_null`` is False, record attributes whose value is null/empty
    are dropped (schema/list attributes are always kept - they have no value).
    """
    obj = _load(source)

    if isinstance(obj, list):
        attrs = _from_list(obj)
    elif isinstance(obj, dict):
        if isinstance(obj.get("properties"), dict) or "$schema" in obj:
            attrs = _from_schema(obj)
        elif isinstance(obj.get("data"), dict):
            attrs = _from_record(obj["data"])
        else:
            attrs = _from_record(obj)
    else:
        raise ValueError("could not interpret source as a schema, record, or list")

    if not include_null:
        attrs = [a for a in attrs if a.is_null is not True]

    # Deduplicate by label (case-insensitive), keeping first occurrence.
    seen: set[str] = set()
    unique: List[ExtractedAttribute] = []
    for a in attrs:
        key = a.label.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append(a)
    return unique
