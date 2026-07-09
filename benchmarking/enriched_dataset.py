"""Shared helpers for benchmarking against the human-verified enriched gold set.

The standard benchmarks score against the full OSDU records in ``matched_json/``
via :func:`rag_evaluation.extract_ground_truth`. This module swaps that gold
standard for the curated, flat files in ``enriched_ground_truth/`` produced by the
Ground Truth Enrichment tool. Each enriched file holds only the attributes a human
verified for that document, so we:

  * pair every ``enriched_ground_truth/<id>.json`` with its ``datalab-output``
    extraction (unstructured corpus) - :func:`discover_enriched`,
  * run only the queries whose attribute the enriched file actually populated
    (per document) - :func:`queries_for_enriched`, and
  * read the ground-truth value/tokens straight from the flat enriched record -
    :func:`extract_enriched_ground_truth` (drop-in for the ``gt_extractor`` hook
    on :class:`rag.inference.batch_runner.BatchRunner`).

No MTEB retrieval gold set is involved; only end-to-end accuracy is scored.
"""

from __future__ import annotations

import glob
import importlib.util
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

import evaluate_all  # noqa: E402  (repo root already on sys.path via common)
import rag_evaluation  # noqa: E402

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ENRICHED_DIR = os.path.join("enriched_ground_truth")
DOCS_DIR = evaluate_all.DEFAULT_DOCS_DIR
OUTPUT_ROOT = os.path.join("output_table", "benchmarking_enriched")


# ---------------------------------------------------------------------------
# OSDU attribute mapping (reused from the query-generation package)
#
# ``rag/query_generation/__init__.py`` runs heavy package-level imports, so we
# load the pure ``osdu_context`` module straight from its file to reuse the field
# mappings without triggering that package init.
# ---------------------------------------------------------------------------


def _load_osdu_context():
    name = "_enriched_osdu_context"
    path = os.path.join(_REPO_ROOT, "rag", "query_generation", "osdu_context.py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Register before exec so the module's frozen dataclass can resolve its own
    # __module__ during class creation.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_osdu = _load_osdu_context()
normalize_attribute = _osdu.normalize_attribute
OSDU_FIELD_TO_LABEL: Dict[str, str] = _osdu.OSDU_FIELD_TO_LABEL

# Canonical OSDU field key for each core attribute (mirrors
# ``webapp/backend/enrichment.CORE_CANONICAL_KEYS``; kept here because that module
# pulls in the web backend engine on import).
CORE_CANONICAL_KEYS: Dict[str, str] = {
    "facility name": "FacilityName",
    "lat+long": "SpatialLocation.Wgs84Coordinates",
    "openv id": "OperatingEnvironmentID",
    "facilityevent spud date": "FacilityEvents.SpudDate",
    "country name": "CountryName",
    "basin name": "GeoContexts.BasinID",
    "field name": "FieldName",
    "default vertical measurement id": "DefaultVerticalMeasurementID",
    "current operator id": "CurrentOperatorID",
    "marker names": "MarkerNames",
    "vertical measurement total depth": "VerticalMeasurements.TotalDepthMD",
    "completion date": "FacilityEvents.CompletionDate",
}

# Generated-query labels (in queries.txt) carry a trailing "*".
_GENERATED_MARKER = "*"

# Labels whose value is a date string.
_DATE_LABELS = {"facilityevent spud date", "completion date"}


# ---------------------------------------------------------------------------
# Field resolution
# ---------------------------------------------------------------------------


def _is_empty(value: Any) -> bool:
    return value is None or (isinstance(value, (list, dict, str)) and len(value) == 0)


def field_key_for_label(label: str, data: Dict[str, Any]) -> Optional[str]:
    """Return the enriched-record key that holds ``label``'s value, or None.

    Generated labels (``WellborePhase*``) map to their raw OSDU field name; core
    labels resolve through the canonical key and, failing that, a reverse lookup
    over ``OSDU_FIELD_TO_LABEL`` so alternate real keys (e.g. ``WellboreSituation``
    for ``openv ID`` or ``CurrentOperator`` for ``current operator ID``) still
    match. Only keys present with a non-empty value are returned.
    """
    raw = (label or "").strip()

    # Generated attribute label -> raw OSDU field name (drop the trailing marker).
    if raw.endswith(_GENERATED_MARKER):
        key = raw.rstrip(_GENERATED_MARKER).strip()
        return key if key in data and not _is_empty(data.get(key)) else None

    norm = normalize_attribute(raw)
    candidates: List[str] = []

    canonical = CORE_CANONICAL_KEYS.get(norm)
    if canonical:
        candidates.append(canonical)

    for key in data:
        friendly = OSDU_FIELD_TO_LABEL.get(key.strip().lower())
        if friendly and normalize_attribute(friendly) == norm and key not in candidates:
            candidates.append(key)

    for key in candidates:
        if key in data and not _is_empty(data.get(key)):
            return key
    return None


# ---------------------------------------------------------------------------
# Ground-truth extraction (drop-in gt_extractor for BatchRunner)
# ---------------------------------------------------------------------------


def _coordinate_tokens(value: str) -> List[str]:
    """Truncated lat/lon tokens from a "lat, lon" string (empty if unparseable)."""
    nums = re.findall(r"-?\d+(?:\.\d+)?", value or "")
    if len(nums) < 2:
        return []
    try:
        return [rag_evaluation._trunc(float(nums[0])), rag_evaluation._trunc(float(nums[1]))]
    except ValueError:
        return []


def _tokens_for(label: str, value: str) -> List[str]:
    norm = normalize_attribute(label)
    text = str(value).strip()
    if not text:
        return []
    if norm in rag_evaluation.COORDINATE_LABELS:
        return _coordinate_tokens(text)
    if norm == "marker names":
        return [t.strip() for t in text.split(",") if t.strip()]
    # Date and free-text labels: the value is used verbatim (dates in the enriched
    # files are already normalized to YYYY-MM-DD).
    return [text]


def extract_enriched_ground_truth(label: str, enriched_json: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve ``{"value", "tokens"}`` for ``label`` from a flat enriched record.

    Returns a null value when the enriched file does not populate the attribute;
    :func:`rag.inference.batch_runner.summarize` then excludes it from accuracy.
    """
    data = rag_evaluation._data_root(enriched_json)
    key = field_key_for_label(label, data)
    if key is None:
        return {"value": None, "tokens": []}
    value = data.get(key)
    if _is_empty(value):
        return {"value": None, "tokens": []}
    value = str(value).strip()
    return {"value": value, "tokens": _tokens_for(label, value)}


def resolve_well_name_enriched(enriched_json: Dict[str, Any], default: str = "the well") -> str:
    """The document's well/facility name from the enriched record (or ``default``)."""
    data = rag_evaluation._data_root(enriched_json)
    return (
        data.get("FacilityName")
        or data.get("VirtualProperties.DefaultName")
        or default
    )


# ---------------------------------------------------------------------------
# Per-document query selection
# ---------------------------------------------------------------------------


def queries_for_enriched(
    data: Dict[str, Any], all_queries: List[Dict[str, str]]
) -> List[Dict[str, str]]:
    """Keep only queries whose attribute the enriched record actually populated."""
    return [q for q in all_queries if field_key_for_label(q["label"], data) is not None]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def _datalab_index(docs_dir: str) -> Dict[str, str]:
    """Map doc id -> datalab-output path for every extraction file."""
    index: Dict[str, str] = {}
    for path in sorted(glob.glob(os.path.join(docs_dir, "*.json"))):
        doc_id = evaluate_all.derive_doc_id(path)
        index.setdefault(doc_id, path)
    return index


def discover_enriched(
    only: Optional[List[str]] = None,
    limit: Optional[int] = None,
    *,
    enriched_dir: str = ENRICHED_DIR,
    docs_dir: str = DOCS_DIR,
    queries: Optional[List[Dict[str, str]]] = None,
) -> List[Dict[str, Any]]:
    """Discover enriched docs paired with their datalab extraction and queries.

    Each descriptor has ``id``, ``input`` (datalab path), ``ground_truth`` (the
    enriched file) and ``queries`` (the per-document filtered query list). Docs
    without a datalab extraction are skipped with a warning.
    """
    datalab = _datalab_index(docs_dir)
    all_queries = queries if queries is not None else rag_evaluation.parse_queries_file(
        evaluate_all.DEFAULT_QUERIES
    )

    docs: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for gt_path in sorted(glob.glob(os.path.join(enriched_dir, "*.json"))):
        doc_id = os.path.splitext(os.path.basename(gt_path))[0]
        if only and doc_id not in only:
            continue
        input_path = datalab.get(doc_id)
        if not input_path:
            skipped.append(doc_id)
            continue
        with open(gt_path, "r", encoding="utf-8") as f:
            data = rag_evaluation._data_root(json.load(f))
        docs.append({
            "id": doc_id,
            "input": input_path,
            "ground_truth": gt_path,
            "queries": queries_for_enriched(data, all_queries),
        })

    if limit is not None:
        docs = docs[:limit]
    print(f"Discovered {len(docs)} enriched doc(s)"
          + (f", skipped {len(skipped)} without datalab output: {', '.join(skipped)}"
             if skipped else ""))
    return docs


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def build_manifest(docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reproducibility record: per-doc query counts and label -> field mappings."""
    entries = []
    for doc in docs:
        with open(doc["ground_truth"], "r", encoding="utf-8") as f:
            data = rag_evaluation._data_root(json.load(f))
        mappings = {}
        for q in doc["queries"]:
            key = field_key_for_label(q["label"], data)
            if key is not None:
                mappings[q["label"]] = key
        entries.append({
            "id": doc["id"],
            "input": doc["input"],
            "ground_truth": doc["ground_truth"],
            "num_queries": len(doc["queries"]),
            "label_to_field": mappings,
        })
    return {
        "dataset": "enriched_ground_truth",
        "docs": entries,
        "total_docs": len(entries),
        "total_query_tasks": sum(e["num_queries"] for e in entries),
    }
