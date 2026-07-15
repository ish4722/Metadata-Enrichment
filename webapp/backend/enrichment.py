"""Metadata Enrichment engine for the RAG Workbench.

Runs the existing RAG pipeline across every OSDU attribute query for a document,
fills / validates a structured JSON and streams per-attribute progress as NDJSON
so the frontend can drive a live process timeline and a human-in-the-loop review.

Reuses the cached components from :mod:`engine` (embedding backend, Chroma store,
cross-encoder reranker, LLM) and the PDF bbox mapper from :mod:`highlights`. The
chunking strategy, vector backend, context mode and models all come straight from
:class:`rag.config.RagConfig`; only the embedding model, retrieval mode, hybrid
alpha and reranker toggle are overridable from the UI (same rule as the Explorer).
"""

from __future__ import annotations

import copy
import datetime
import os
import queue
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import rag_evaluation

from engine import ENGINE, _chunk_payload
from highlights import highlights_for_matches
from rag.inference.context_builder import build_context
from rag.inference.hierarchical_answer import answer_with_hierarchical_context
from rag.inference.prompt import inject_facility_name, make_prompt, parse_generator_json
from rag.inference.reranker import get_reranker, rerank
from rag.inference import verification as V

# The query-generation package uses top-level ``from query_generation...`` imports,
# so its parent (``<repo>/rag``) must be importable (mirrors from_schema.py).
_RAG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "rag"
)
if _RAG_DIR not in sys.path:
    sys.path.insert(0, _RAG_DIR)

from query_generation.generator import (  # noqa: E402
    GeneratedQuery,
    _extract_json,
    _normalize_keywords,
    make_label,
)
from query_generation.osdu_context import normalize_attribute  # noqa: E402
from query_generation.prompt import build_generation_prompt  # noqa: E402
from query_generation.queries_file import write_generated_query  # noqa: E402
from query_generation.schema_ingest import ExtractedAttribute, extract_attributes  # noqa: E402

# Confidence at/above this is a "confident" enrichment; below it is "low confidence".
CONFIDENCE_THRESHOLD = 0.75

# For quick demos we only enrich the first N attributes of the record instead of
# generating a query for every single field. ``None`` (or <= 0) means no limit.
DEMO_MAX_ATTRIBUTES = 7

# Query generation is one independent (network-bound) LLM call per attribute, so we
# fan them out across a small thread pool instead of calling them one at a time.
QUERY_GEN_WORKERS = 8

# The answer step (retrieve -> rerank -> LLM answer) is likewise fanned out. The
# LLM answer call is the (network-bound) bottleneck and runs concurrently; the
# Chroma reads and the torch reranker are NOT thread-safe, so they are serialized
# with the locks below (both are fast, so this barely costs anything).
ANSWER_WORKERS = 8
_RETRIEVE_LOCK = threading.Lock()  # guards Chroma reads (retrieve + context fetch)
_RERANK_LOCK = threading.Lock()    # guards the shared torch cross-encoder

# Serializes appends to the shared queries file when caching generated queries.
_QUERIES_WRITE_LOCK = threading.Lock()

# Trailing marker that distinguishes an auto-generated query label from a
# hand-written (core) one in the queries file.
_GENERATED_MARKER = "*"

# The ground-truth tool always treats the top N queries in queries.txt as the
# "core" attributes and guarantees a saved value for each.
CORE_QUERY_COUNT = 12

# Canonical OSDU field key to use for a core attribute when the uploaded record
# has no matching field of its own (keyed by the normalized core label). Keeps
# the saved ground truth consistent across documents.
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

# Labels whose answers/ground truth are dates (normalized to YYYY-MM-DD).
_DATE_LABELS = {"facilityevent spud date", "completion date"}

_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
    "%d.%m.%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
    "%B %d %Y", "%b %d %Y",
]

_DATE_PATTERNS = (
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"|[0-9]{1,2}[/.\-][0-9]{1,2}[/.\-][0-9]{2,4}"
    r"|[0-9]{1,2}\s+[A-Za-z]{3,9},?\s+[0-9]{4}"
    r"|[A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4}"
)

_DATE_CANDIDATE = re.compile(_DATE_PATTERNS)

# Matches when the WHOLE answer is essentially a single date (allowing trailing
# punctuation) so any date-valued answer can be normalized to YYYY-MM-DD even
# when the field label does not contain the word "date".
_DATE_FULL = re.compile(r"^[\s,]*(?:" + _DATE_PATTERNS + r")[\s,.]*$")


def _is_date_label(label: str) -> bool:
    """A field is a date field when it is a known date label or its name
    contains "date" (covers flat keys like ``FacilityEvents.SpudDate``)."""
    key = (label or "").strip().lower()
    return key in _DATE_LABELS or "date" in key


# ── conversion / normalization ────────────────────────────────────────────────
def _normalize_date(text: str) -> Optional[str]:
    """Best-effort parse of a date-like string to ``YYYY-MM-DD`` (or ``None``)."""
    if not text:
        return None
    candidates = _DATE_CANDIDATE.findall(text)
    trials = [c.strip().strip(",") for c in candidates] + [text.strip().strip(",")]
    for cand in trials:
        for fmt in _DATE_FORMATS:
            try:
                return datetime.datetime.strptime(cand, fmt).strftime("%Y-%m-%d")
            except ValueError:
                continue
    return None


def normalize_value(label: str, text: str) -> Optional[str]:
    """Return a standardized string for an answer (lat/long, dates) or trimmed text."""
    if not text:
        return None
    key = (label or "").strip().lower()
    if key in rag_evaluation.COORDINATE_LABELS:
        coords = rag_evaluation.parse_coordinates(text)
        if not coords:
            coords = rag_evaluation.parse_grid_coordinates(text)
        if coords:
            lat, lon = coords[0]
            return f"{lat:.5f}, {lon:.5f}"
        return text.strip()
    if _is_date_label(label):
        return _normalize_date(text) or text.strip()
    # Any answer that is essentially a bare date gets normalized to YYYY-MM-DD.
    if _DATE_FULL.match(text.strip()):
        norm = _normalize_date(text)
        if norm:
            return norm
    return text.strip()


def matches_ground_truth(
    label: str,
    gt_value: Any,
    gt_tokens: List[str],
    answer: str,
) -> bool:
    """Compare an answer to ground truth using the label-appropriate comparator."""
    if not answer:
        return False
    key = (label or "").strip().lower()

    if key in rag_evaluation.COORDINATE_LABELS:
        is_match, _conf = rag_evaluation.coordinate_verdict(str(gt_value), answer)
        return bool(is_match)

    if _is_date_label(label):
        gt_norm = _normalize_date(str(gt_value))
        ans_norm = _normalize_date(answer)
        if gt_norm and ans_norm:
            try:
                gd = datetime.datetime.strptime(gt_norm, "%Y-%m-%d")
                ad = datetime.datetime.strptime(ans_norm, "%Y-%m-%d")
                return abs((gd - ad).days) <= 1
            except ValueError:
                return gt_norm == ans_norm
        # fall through to a token match when dates could not be parsed

    return rag_evaluation.answer_is_correct(answer, gt_tokens)


# ── category classification ───────────────────────────────────────────────────
def classify(
    *,
    available: bool,
    answer: str,
    has_gt: bool,
    matches_gt: Optional[bool],
    band: Optional[str],
) -> str:
    """Bucket an attribute result into a dashboard row category.

    Uses the Chain-of-Verification confidence ``band`` ("high" | "medium" |
    "low") rather than a raw threshold so the coloring matches the CoVe score
    shown in the UI.
    """
    if not available or not answer.strip():
        return "not_found"
    if has_gt:
        return "validated_correct" if matches_gt else "validated_incorrect"
    # Null ground truth -> this is an enrichment we discovered from the document.
    if band == "high":
        return "enriched_confident"
    if band == "medium":
        return "enriched_medium"
    return "enriched_low_conf"


# ── query cache (reuse queries already written to the queries file) ───────────
def _load_query_cache(queries_path: Optional[str]) -> Dict[str, Tuple[str, str]]:
    """Map ``normalize_attribute(label) -> (question, keywords)`` from a queries file.

    The normalization strips the generated ``*`` / numeric variant suffix, so a
    hand-written ``facility name`` and a generated ``facility name*`` collapse to
    the same key (first occurrence wins). Returns an empty map when the file is
    missing or unreadable.
    """
    cache: Dict[str, Tuple[str, str]] = {}
    if not queries_path or not os.path.isfile(queries_path):
        return cache
    try:
        parsed = rag_evaluation.parse_queries_file(queries_path)
    except Exception:  # noqa: BLE001
        return cache
    for q in parsed:
        key = normalize_attribute(q.get("label", ""))
        if key and key not in cache:
            cache[key] = (q.get("question", ""), q.get("keywords", ""))
    return cache


def _core_labels(queries_path: Optional[str]) -> List[str]:
    """Return the core labels: the top ``CORE_QUERY_COUNT`` queries in the file.

    These are the hand-written attributes at the top of ``queries.txt`` (the ones
    without the generated ``*`` marker); the ground-truth tool always treats them
    as core and guarantees a row + saved value for each.
    """
    if not queries_path or not os.path.isfile(queries_path):
        return []
    try:
        parsed = rag_evaluation.parse_queries_file(queries_path)
    except Exception:  # noqa: BLE001
        return []
    labels: List[str] = []
    seen: set[str] = set()
    for q in parsed:
        label = (q.get("label") or "").strip()
        if not label:
            continue
        key = label.lower()
        if key not in seen:
            seen.add(key)
            labels.append(label)
        if len(labels) >= CORE_QUERY_COUNT:
            break
    return labels


def _merge_core_attributes(
    attrs: List[ExtractedAttribute],
    queries_path: Optional[str],
    record_data: Dict[str, Any],
) -> List[ExtractedAttribute]:
    """Guarantee the top-12 core attributes are present, then keep extra fields.

    For every core label (top of ``queries.txt``) we reuse the matching record
    field when one exists (so we keep its real key + value); otherwise we inject a
    placeholder attribute keyed by a canonical OSDU field name with a null value,
    so the pipeline still answers it from the PDF. Core attributes come first (in
    queries order), followed by any non-core record fields.
    """
    core_lbls = _core_labels(queries_path)
    if not core_lbls:
        return attrs

    # Map normalized-core-label -> the record attribute that matches it (if any).
    existing_by_core: Dict[str, ExtractedAttribute] = {}
    extras: List[ExtractedAttribute] = []
    core_norms = {normalize_attribute(l) for l in core_lbls}
    for a in attrs:
        norm = normalize_attribute(a.label or a.raw_name)
        if norm in core_norms and norm not in existing_by_core:
            existing_by_core[norm] = a
        else:
            extras.append(a)

    merged: List[ExtractedAttribute] = []
    for label in core_lbls:
        norm = normalize_attribute(label)
        existing = existing_by_core.get(norm)
        if existing is not None:
            # Keep the record's real key/value but pin the label to the core one.
            merged.append(ExtractedAttribute(
                raw_name=existing.raw_name, label=label, is_null=existing.is_null
            ))
        else:
            raw_name = CORE_CANONICAL_KEYS.get(norm, label)
            is_null = _is_empty(record_data.get(raw_name))
            # The value may live nested (e.g. FacilityEvents[], GeoContexts[]); let
            # the RAG pipeline's extractor decide whether ground truth exists.
            if is_null and label.strip().lower() in rag_evaluation.GROUND_TRUTH_EXTRACTORS:
                resolved = rag_evaluation.extract_ground_truth(label, record_data)
                if resolved.get("value") is not None:
                    is_null = False
            merged.append(ExtractedAttribute(raw_name=raw_name, label=label, is_null=is_null))

    return merged + extras


def _append_generated_query(
    queries_path: str, attribute: str, query: str, keywords: str
) -> None:
    """Append a freshly generated query to the shared queries file (idempotent)."""
    gq = GeneratedQuery(
        attribute=attribute,
        label=make_label(attribute),
        query=query,
        keywords=keywords,
    )
    with _QUERIES_WRITE_LOCK:
        try:
            write_generated_query(gq, path=queries_path, overwrite=False)
        except FileExistsError:
            pass
        except Exception:  # noqa: BLE001
            pass


# ── name injection (generation only; retrieval stays generic) ─────────────────
def _generation_question(query: str, label: str, well_name: Optional[str]) -> str:
    """Name-injected question for the LLM answer step (retrieval stays generic).

    The well/facility name is injected into the generation question so the LLM
    answers about the specific well, EXCEPT for the ``facility name`` attribute
    itself (injecting the name there would give away the answer). Retrieval uses
    the raw, name-free query so the search stays about "this well" in general.
    """
    name = well_name if well_name and well_name.strip().lower() != "the well" else ""
    if (label or "").strip().lower() == "facility name":
        name = ""
    if not name:
        return query
    # Replace explicit placeholders ({well} / [facility name]) then the generic
    # "this well" / "the well" subject with the concrete name.
    q = rag_evaluation.inject_name(query, name)
    return inject_facility_name(q, name)


# ── query generation (one retrieval query per record field) ───────────────────
def _generate_query(llm: Any, attribute: str) -> Tuple[str, str, Dict[str, Any]]:
    """Generate a retrieval query + keywords for one attribute via the LLM.

    Falls back to the attribute name itself when no LLM is available or the reply
    cannot be parsed. Returns ``(query, keywords, usage)``.
    """
    if llm is None:
        return attribute, "", {}
    prompt = build_generation_prompt(attribute)
    raw, usage = rag_evaluation.invoke_llm_with_usage(llm, prompt)
    try:
        data = _extract_json(raw)
    except Exception:  # noqa: BLE001
        return attribute, "", usage or {}
    query = str(data.get("query") or "").strip() or attribute
    keywords = _normalize_keywords(data.get("keywords"))
    return query, keywords, usage or {}


# ── write enriched values back into the uploaded record (exact keys) ──────────
def build_enriched_record(
    record: Dict[str, Any],
    results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Return a deep copy of ``record`` with only the chosen fields rewritten.

    ``results`` is a list of ``{"raw_name", "value", "auto_write"}``. A field is
    written at its exact key ``data[raw_name]`` only when ``auto_write`` is true
    and a value was found. Every other key/value is left byte-for-byte unchanged.
    """
    out = copy.deepcopy(record)
    data = rag_evaluation._data_root(out)
    for r in results:
        if r.get("auto_write") and r.get("value") is not None and r.get("raw_name"):
            data[r["raw_name"]] = r["value"]
    return out


def _empty_summary() -> Dict[str, int]:
    return {
        "total": 0,
        # Dashboard headline metrics (see requirement 5).
        "validated_correct": 0,   # a) answer matches the ground truth
        "high_conf_update": 0,    # b) model updated/filled with HIGH confidence
        "medium_conf_fill": 0,    # c) model filled/updated with MEDIUM confidence
        "enriched": 0,            # d) ground truth null and the model gave an answer
        # Supporting detail counts.
        "validated_incorrect": 0,
        "low_conf": 0,
        "not_found": 0,
    }


def _bump_summary(summary: Dict[str, int], event: Dict[str, Any]) -> None:
    """Increment the (non-exclusive) dashboard metrics for one attribute event."""
    summary["total"] += 1
    has_gt = bool(event.get("has_gt"))
    matches_gt = event.get("matches_gt")
    available = bool(event.get("available"))
    value = event.get("value")
    band = event.get("band")
    has_answer = available and value is not None and str(value).strip() != ""
    is_update = bool(event.get("was_null")) or (
        str(value) != str(event.get("original_value"))
    )

    if has_gt and matches_gt:
        summary["validated_correct"] += 1
    elif has_gt and matches_gt is False:
        summary["validated_incorrect"] += 1

    if has_answer and is_update:
        if band == "high":
            summary["high_conf_update"] += 1
        elif band == "medium":
            summary["medium_conf_fill"] += 1
        elif band == "low":
            summary["low_conf"] += 1

    if not has_gt and has_answer:
        summary["enriched"] += 1


def _is_empty(value: Any) -> bool:
    """A record field counts as null/empty when it is None or an empty container."""
    return value is None or (isinstance(value, (list, dict, str)) and len(value) == 0)


# Numeric vertical-measurement labels whose ground truth is a bare number: we
# append the record's unit of measure so a reviewer knows the GT unit (e.g. "m")
# when the document may report the value in a different unit (e.g. ft).
_VM_UNIT_MATCHERS = {
    "vertical measurement total depth": "_is_total_depth_md_measurement",
    "verticalmeasurementid kb": "_is_kb_vertical_measurement",
}


def _vm_unit_for_label(label: str, record_data: Dict[str, Any]) -> Optional[str]:
    """Return the unit of measure for a vertical-measurement label, or ``None``."""
    matcher_name = _VM_UNIT_MATCHERS.get((label or "").strip().lower())
    matcher = getattr(rag_evaluation, matcher_name, None) if matcher_name else None
    if matcher is None:
        return None
    try:
        for vm in rag_evaluation._vertical_measurements(record_data):
            if matcher(vm):
                unit = rag_evaluation._strip_reference_id(
                    vm.get("VerticalMeasurementUnitOfMeasureID")
                )
                return unit or None
    except Exception:  # noqa: BLE001
        return None
    return None


# ── Chain-of-Verification (CoVe) confidence for one attribute answer ──────────
def _noop_emit(_event: Dict[str, Any]) -> None:
    """Default event sink used when a caller does not want streamed sub-events."""


def _cove_verify(
    *,
    store: Any,
    cfg: Any,
    doc_id: str,
    reranker: Any,
    llm: Any,
    question: str,
    answer_text: str,
    verbalized_conf: Optional[float],
    emit: Callable[[Dict[str, Any]], None] = _noop_emit,
    index: Optional[int] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the same CoVe pass the Explorer uses for one answer, streaming steps.

    Plans indirect corroboration checks, retrieves a related fact per check via
    the shared pipeline, lets the LLM judge each domain-logic rule, aggregates
    the verdicts into a 0-1 confidence + band, and (when the band is low) asks
    for a grounded corrected value. Never edits the answer itself. Returns
    ``{confidence, band, checks, suggestion, cost_usd}``. Chroma reads and the
    torch reranker are serialized via the module locks (not thread-safe).

    ``emit`` is called with per-step events (planned checks, each check verdict)
    keyed by ``index`` so the UI can visualise the verification live.
    """
    total_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

    def _acc(usage: Dict[str, Any]) -> None:
        if usage:
            total_usage["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            total_usage["output_tokens"] += int(usage.get("output_tokens", 0) or 0)

    def _one_pass(candidate: str, rnd: int) -> Dict[str, Any]:
        """Plan -> execute -> aggregate one verification pass for ``candidate``."""
        # 1. Plan the indirect checks.
        raw_plan, usage = rag_evaluation.invoke_llm_with_usage(
            llm, V.build_plan_prompt(question, candidate, None)
        )
        _acc(usage)
        checks = V.parse_plan(raw_plan, cfg.verify_num_questions)
        emit({"stage": "attribute", "status": "cove_plan", "index": index, "label": label,
              "round": rnd, "answer": candidate,
              "checks": [{"question": c["question"], "rule": c["rule"],
                          "relevance": c["relevance"]} for c in checks]})

        # 2. Execute each check (retrieve related fact -> LLM judges the rule).
        results: List[Dict[str, Any]] = []
        seen_context: List[Dict[str, Any]] = []
        seen_keys: set = set()
        for i, chk in enumerate(checks):
            emit({"stage": "attribute", "status": "cove_check", "index": index, "label": label,
                  "round": rnd, "i": i, "total": len(checks), "question": chk["question"],
                  "relevance": chk["relevance"]})
            with _RETRIEVE_LOCK:
                matches = ENGINE._retrieve(store, cfg, doc_id, chk["question"])
            if reranker is not None and matches:
                with _RERANK_LOCK:
                    matches = rerank(chk["question"], matches, cfg.reranker_model,
                                     top_k=cfg.rerank_top_k)
            elif matches:
                matches = matches[: cfg.rerank_top_k]
            with _RETRIEVE_LOCK:
                corpus = ENGINE._corpus_for_context(store, cfg, matches)
            ctx_items, _ = build_context(matches, corpus, context_mode=cfg.context_mode)
            for item in ctx_items:
                key = (item.get("source"), item.get("page"), (item.get("text") or "")[:80])
                if key not in seen_keys:
                    seen_keys.add(key)
                    seen_context.append(item)
            raw_judge, usage = rag_evaluation.invoke_llm_with_usage(
                llm, V.build_judge_prompt(chk, ctx_items, candidate)
            )
            _acc(usage)
            result = {**chk, **V.parse_judge(raw_judge)}
            results.append(result)
            emit({"stage": "attribute", "status": "cove_check_done", "index": index,
                  "label": label, "round": rnd, "i": i, "total": len(checks),
                  "check": result})

        # 3. Aggregate into a single confidence score + band.
        agg = V.aggregate(results, verbalized_conf)
        return {"agg": agg, "results": results, "seen_context": seen_context}

    # Verify the model's answer; while the band is LOW, accept the grounded
    # correction the verifier proposes and RE-VERIFY it, recomputing confidence
    # for the new value. Capped at ``max_verify_rounds`` correction rounds so a
    # persistently-low answer cannot loop forever.
    max_rounds = int(getattr(cfg, "max_verify_rounds", 3) or 3)
    current_answer = answer_text
    auto_corrected = False
    suggestion: Optional[Dict[str, Any]] = None

    passed = _one_pass(current_answer, 0)
    agg, results, seen_context = passed["agg"], passed["results"], passed["seen_context"]

    corrections = 0
    while agg["band"] == V.SUGGEST_BAND and corrections < max_rounds:
        # Propose a corrected answer grounded in the excerpts we just verified.
        proposal = None
        try:
            raw_sug, usage = rag_evaluation.invoke_llm_with_usage(
                llm, V.build_suggestion_prompt(question, current_answer, results,
                                               seen_context[:12])
            )
            _acc(usage)
            proposal = V.parse_suggestion(raw_sug)
        except Exception:  # noqa: BLE001
            proposal = None
        if not proposal or not proposal.get("suggested_answer"):
            break
        new_answer = str(proposal["suggested_answer"]).strip()
        # Stop if the model keeps suggesting the same value (no progress).
        if new_answer.lower() == current_answer.strip().lower():
            suggestion = proposal
            break

        # Accept the recommendation and re-verify it.
        corrections += 1
        current_answer = new_answer
        auto_corrected = True
        suggestion = proposal
        emit({"stage": "attribute", "status": "cove_retry", "index": index, "label": label,
              "round": corrections, "answer": current_answer,
              "reason": proposal.get("reason", "")})
        passed = _one_pass(current_answer, corrections)
        agg, results, seen_context = passed["agg"], passed["results"], passed["seen_context"]

    cost = rag_evaluation.token_cost_usd(
        total_usage.get("input_tokens", 0), total_usage.get("output_tokens", 0)
    )
    return {
        "confidence": agg["confidence"],
        "band": agg["band"],
        "checks": results,
        "suggestion": suggestion,
        "final_answer": current_answer,
        "auto_corrected": auto_corrected,
        "rounds": corrections,
        "cost_usd": cost,
    }


# ── main streamed operation ───────────────────────────────────────────────────
def run_enrichment(
    *,
    doc_id: str,
    datalab_path: str,
    record: Optional[Dict[str, Any]],
    overrides: Optional[Dict[str, Any]] = None,
    max_attributes: Optional[int] = DEMO_MAX_ATTRIBUTES,
    persist_queries: bool = False,
    queries_path: Optional[str] = None,
    guaranteed_core: bool = False,
    ground_truth_record: Optional[Dict[str, Any]] = None,
) -> Iterator[Dict[str, Any]]:
    """Embed the PDF, generate a query per record field, answer/validate each,
    and stream events; fills null fields and overwrites prefilled fields on a
    high-confidence disagreement, writing back into the record's exact keys.

    ``max_attributes`` caps how many record fields are turned into queries (top
    N by document order) so a demo stays fast; ``None`` or <= 0 means no cap.
    """
    if not record:
        yield {"stage": "error", "message": "No structured JSON provided to enrich."}
        return

    # 1. Build (embed-once). ENGINE.build manages the heavy-work lock internally.
    build_failed = False
    for ev in ENGINE.build(doc_id, datalab_path, overrides):
        yield {"stage": "build", "event": ev}
        if ev.get("stage") == "error":
            build_failed = True
    if build_failed:
        return

    cfg = ENGINE.make_config(overrides)
    record_data = rag_evaluation._data_root(record)
    # The record being enriched comes from ``new_checking_truth`` (may have nulls
    # / stale values). Validation compares the model's answers against a separate
    # ``enriched_ground_truth`` record when one is supplied; otherwise the input
    # record doubles as its own ground truth (legacy behavior).
    gt_source = ground_truth_record if isinstance(ground_truth_record, dict) else record
    gt_root = rag_evaluation._data_root(gt_source)
    attrs: List[ExtractedAttribute] = extract_attributes(record)
    if guaranteed_core:
        # Always include the top-12 core attributes (injecting canonical-keyed
        # placeholders for any not present in the uploaded record).
        attrs = _merge_core_attributes(attrs, queries_path, record_data)
    total_attrs = len(attrs)
    if max_attributes is not None and max_attributes > 0:
        attrs = attrs[:max_attributes]
    well_name = rag_evaluation.resolve_well_name(record)

    # When caching, reuse queries already in the queries file and only generate
    # (and persist) the ones we have never seen before.
    query_cache = _load_query_cache(queries_path) if persist_queries else {}

    yield {
        "stage": "queries",
        "status": "loaded",
        "count": len(attrs),
        "total_available": total_attrs,
        "limited": len(attrs) < total_attrs,
        "labels": [a.label for a in attrs],
        "core_labels": _core_labels(queries_path) if persist_queries else [],
    }

    results: List[Dict[str, Any]] = []
    summary = _empty_summary()

    with ENGINE._lock:  # serialize torch / chroma access (single-user local tool)
        store = ENGINE.get_store(cfg)
        if not store.has_document(doc_id):
            yield {"stage": "error",
                   "message": f"{doc_id} is not embedded for '{cfg.embedding_model}'."}
            return

        llm = ENGINE.get_llm(cfg)
        reranker = get_reranker(cfg.reranker_model) if cfg.use_reranker else None

        # 2. Generate one retrieval query per field, fanned out across a thread pool
        #    (each call is an independent, network-bound LLM request). When caching
        #    is on, reuse any query already in the queries file and only generate
        #    (and persist) attributes we have never seen before.
        gen_cost = 0.0
        results_by_idx: Dict[int, Tuple[str, str]] = {}
        yield {"stage": "generate_queries", "status": "start", "count": len(attrs)}
        completed = 0

        to_generate: List[int] = []
        for i, attr in enumerate(attrs):
            attr_name = attr.label or attr.raw_name
            cached = query_cache.get(normalize_attribute(attr_name)) if persist_queries else None
            if cached is not None:
                results_by_idx[i] = cached
                completed += 1
                yield {"stage": "generate_queries", "status": "progress",
                       "index": i, "completed": completed, "total": len(attrs),
                       "label": attr.label, "query": cached[0], "source": "cache"}
            else:
                to_generate.append(i)

        if to_generate:
            workers = max(1, min(QUERY_GEN_WORKERS, len(to_generate)))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                future_to_idx = {
                    pool.submit(_generate_query, llm, attrs[i].label or attrs[i].raw_name): i
                    for i in to_generate
                }
                for fut in as_completed(future_to_idx):
                    i = future_to_idx[fut]
                    attr = attrs[i]
                    query, keywords, usage = fut.result()
                    results_by_idx[i] = (query, keywords)
                    gen_cost += rag_evaluation.token_cost_usd(
                        usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                    ) if usage else 0.0
                    if persist_queries and queries_path:
                        _append_generated_query(
                            queries_path, attr.label or attr.raw_name, query, keywords
                        )
                    completed += 1
                    yield {"stage": "generate_queries", "status": "progress",
                           "index": i, "completed": completed, "total": len(attrs),
                           "label": attr.label, "query": query, "source": "generated"}
        # Restore original field order so downstream indices stay stable.
        specs: List[Tuple[ExtractedAttribute, str, str]] = [
            (attrs[i], results_by_idx[i][0], results_by_idx[i][1])
            for i in range(len(attrs))
        ]
        yield {"stage": "generate_queries", "status": "done", "cost_usd": gen_cost}

        # 3. Answer / validate each field, fanned out across a thread pool. The
        #    (network-bound) LLM answer call runs concurrently; Chroma reads and the
        #    torch reranker are serialized via module-level locks (they are fast and
        #    not thread-safe). Returns the fully-built "done" event for one field.
        def _process_attribute(
            idx: int, attr: ExtractedAttribute, query: str, keywords: str,
            emit: Callable[[Dict[str, Any]], None] = _noop_emit,
        ) -> Dict[str, Any]:
            label = attr.label
            raw_name = attr.raw_name
            was_null = bool(attr.is_null) if attr.is_null is not None else _is_empty(
                record_data.get(raw_name)
            )
            existing_value = record_data.get(raw_name)

            # Known OSDU labels (facility name, spud/completion date, basin name,
            # lat+long, total depth, ...) store their ground truth in nested or
            # object-typed fields (FacilityEvents[], GeoContexts[],
            # VerticalMeasurements[], Wgs84Coordinates geometry), so the raw
            # top-level value is often absent or an object that renders as
            # "[object Object]". Always prefer the RAG pipeline's extractor here so
            # these core attributes are shown + validated exactly as the pipeline
            # sees them (clean coordinate/date/name strings).
            gt_key = (label or "").strip().lower()
            if gt_key in rag_evaluation.GROUND_TRUTH_EXTRACTORS:
                resolved = rag_evaluation.extract_ground_truth(label, record)
                if resolved.get("value") is not None:
                    existing_value = resolved.get("value")
                    was_null = False
                    # Append the unit of measure to numeric vertical-measurement GTs
                    # (ground-truth tool only) so the reviewer knows the GT unit;
                    # validation below still compares against the raw number.
                    if guaranteed_core:
                        unit = _vm_unit_for_label(label, record_data)
                        if unit and unit.lower() not in str(existing_value).lower():
                            existing_value = f"{existing_value} {unit}"

            # Ground-truth tool: retrieval stays generic ("this well"); the well
            # name is injected only into the generation question. The regular
            # enrichment tool keeps its original placeholder-based behavior.
            if guaranteed_core:
                retrieval_query = f"{query} {keywords}".strip() if keywords else query
                question = _generation_question(query, label, well_name)
            else:
                question = rag_evaluation.inject_name(query, well_name) if well_name else query
                retrieval_query = f"{question} {keywords}".strip() if keywords else question

            # Step 1 - retrieval.
            emit({"stage": "attribute", "status": "retrieve", "index": idx,
                  "label": label, "query": retrieval_query})
            with _RETRIEVE_LOCK:
                retrieved = ENGINE._retrieve(store, cfg, doc_id, retrieval_query)
            retrieved_payload = [_chunk_payload(c, s) for c, s in retrieved]
            emit({"stage": "attribute", "status": "retrieved", "index": idx,
                  "label": label, "count": len(retrieved), "chunks": retrieved_payload})

            # Step 2 - reranking (cross-encoder shortlist).
            matches = retrieved
            if cfg.use_reranker and matches and reranker is not None:
                emit({"stage": "attribute", "status": "rerank", "index": idx, "label": label})
                with _RERANK_LOCK:
                    matches = rerank(retrieval_query, matches, cfg.reranker_model,
                                     top_k=cfg.rerank_top_k)
            elif matches:
                matches = matches[: cfg.rerank_top_k]
            reranked_payload = [_chunk_payload(c, s) for c, s in matches]
            emit({"stage": "attribute", "status": "reranked", "index": idx,
                  "label": label, "count": len(matches), "chunks": reranked_payload})

            with _RETRIEVE_LOCK:
                corpus = ENGINE._corpus_for_context(store, cfg, matches)
            context_items, _raw_text = build_context(matches, corpus,
                                                      context_mode=cfg.context_mode)

            # Step 3 - LLM answer.
            emit({"stage": "attribute", "status": "generate", "index": idx, "label": label})
            raw_answer = ""
            usage: Dict[str, Any] = {}
            if llm is not None and matches:
                if cfg.context_mode == "hierarchical":
                    # Hierarchical answering interleaves Chroma reads with LLM calls,
                    # so serialize the whole thing to keep the store access safe.
                    with _RETRIEVE_LOCK:
                        result = answer_with_hierarchical_context(
                            llm, question, matches, store, doc_id, cfg,
                            rag_evaluation.invoke_llm_with_usage,
                        )
                    raw_answer = result.answer
                    usage = result.usage
                else:
                    raw_answer, usage = rag_evaluation.invoke_llm_with_usage(
                        llm, make_prompt(question, context_items)
                    )

            parsed = parse_generator_json(raw_answer)
            answer_text = parsed["answer"]
            available = parsed["available"]
            verbalized_conf = parsed["confidence"]
            source_pages = parsed["source_pages"]
            emit({"stage": "attribute", "status": "answered", "index": idx, "label": label,
                  "answer": answer_text, "available": available,
                  "verbalized_confidence": verbalized_conf})

            # Ground truth for validation: the separate enriched_ground_truth
            # record when provided (``gt_source``), else the input record. Known
            # OSDU labels use the pipeline extractor (nested/object fields);
            # unknown fields read the raw key. ``was_null`` / ``existing_value``
            # above always reflect the INPUT record being enriched.
            gt_value: Any = None
            gt_tokens: List[str] = []
            if gt_key in rag_evaluation.GROUND_TRUTH_EXTRACTORS:
                g = rag_evaluation.extract_ground_truth(label, gt_source)
                if g.get("value") is not None:
                    gt_value = g.get("value")
                    gt_tokens = g.get("tokens") or [str(gt_value)]
            # Fall back to a direct key lookup (handles the flat dotted-key format
            # of new_checking_truth / enriched_ground_truth, where the extractor's
            # nested paths do not apply).
            if gt_value is None and gt_root is not None:
                raw_gt = gt_root.get(raw_name)
                if not _is_empty(raw_gt):
                    gt_value = raw_gt
                    gt_tokens = [str(raw_gt)]
            has_gt = gt_value is not None
            matches_gt: Optional[bool] = None
            if has_gt and available and answer_text.strip():
                matches_gt = matches_ground_truth(label, gt_value, gt_tokens, answer_text)

            # Chain-of-Verification confidence (same method as the RAG Explorer):
            # plan indirect checks, corroborate related facts + domain rules, and
            # aggregate into a 0-1 confidence + band. Overrides the model's own
            # verbalized confidence when it runs.
            confidence = verbalized_conf
            band: Optional[str] = None
            cove: Optional[Dict[str, Any]] = None
            if cfg.verify_enabled and llm is not None and available and answer_text.strip():
                emit({"stage": "attribute", "status": "verify", "index": idx, "label": label})
                try:
                    cove = _cove_verify(
                        store=store, cfg=cfg, doc_id=doc_id, reranker=reranker,
                        llm=llm, question=question, answer_text=answer_text,
                        verbalized_conf=verbalized_conf, emit=emit, index=idx, label=label,
                    )
                    confidence = cove["confidence"]
                    band = cove["band"]
                except Exception:  # noqa: BLE001
                    cove = None

            # Requirement 3: when the CoVe confidence was LOW, the verifier has
            # already looped (up to cfg.max_verify_rounds) accepting its own
            # grounded recommendation and re-scoring it. Adopt its final answer;
            # ``confidence``/``band`` above already reflect that re-verification.
            auto_corrected = False
            suggestion = cove.get("suggestion") if cove else None
            if cove and cove.get("auto_corrected") and cove.get("final_answer"):
                answer_text = str(cove["final_answer"]).strip()
                auto_corrected = True
                if has_gt and answer_text:
                    matches_gt = matches_ground_truth(label, gt_value, gt_tokens, answer_text)

            normalized = normalize_value(label, answer_text) if available else None

            proposed = None
            if available and answer_text.strip():
                proposed = normalized or answer_text.strip()

            category = classify(available=available, answer=answer_text,
                                has_gt=has_gt, matches_gt=matches_gt, band=band)

            # Auto write-back: fill nulls; take an auto-accepted low-confidence
            # correction; and overwrite a prefilled field on a high-confidence
            # disagreement. Human verdicts/edits are applied client-side.
            auto_write = False
            if proposed is not None:
                if was_null or auto_corrected:
                    auto_write = True
                elif matches_gt is False and band == "high":
                    auto_write = True

            chunks_payload = reranked_payload
            try:
                pages = highlights_for_matches(datalab_path, matches)
            except Exception:  # noqa: BLE001
                pages = []

            gen_cost = rag_evaluation.token_cost_usd(
                usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            ) if usage else 0.0
            attr_cost = gen_cost + (cove["cost_usd"] if cove else 0.0)

            event = {
                "stage": "attribute", "status": "done", "index": idx, "label": label,
                "raw_name": raw_name,
                "question": question,
                "answer": answer_text,
                "available": available,
                "confidence": confidence,
                "verbalized_confidence": verbalized_conf,
                "band": band,
                "cove_checks": (cove["checks"] if cove else []),
                "cove_suggestion": suggestion,
                "cove_rounds": (cove.get("rounds", 0) if cove else 0),
                "auto_corrected": auto_corrected,
                "normalized": normalized,
                "value": proposed,
                "source_pages": source_pages,
                "original_value": existing_value,
                "ground_truth": gt_value,
                "was_null": was_null,
                "has_gt": has_gt,
                "matches_gt": matches_gt,
                "auto_write": auto_write,
                "category": category,
                "retrieval_query": retrieval_query,
                "chunks": chunks_payload,
                "retrieved_chunks": retrieved_payload,
                "reranked_chunks": reranked_payload,
                "pages": pages,
                "usage": usage,
                "cost_usd": attr_cost,
            }
            return {
                "idx": idx,
                "event": event,
                "result": {"raw_name": raw_name, "value": proposed, "auto_write": auto_write},
                "category": category,
            }

        # Announce every attribute up front so the table populates immediately,
        # then fan the answering out across the pool.
        for idx, (attr, query, keywords) in enumerate(specs):
            if guaranteed_core:
                question = _generation_question(query, attr.label, well_name)
            else:
                question = rag_evaluation.inject_name(query, well_name) if well_name else query
            yield {"stage": "attribute", "status": "start", "index": idx,
                   "label": attr.label, "raw_name": attr.raw_name, "question": question}

        yield {"stage": "answers", "status": "start", "count": len(specs)}
        answer_workers = max(1, min(ANSWER_WORKERS, len(specs))) if specs else 1
        results_by_idx: Dict[int, Dict[str, Any]] = {}

        # Each worker streams its per-step events (retrieve -> rerank -> generate
        # -> verify/CoVe) onto a thread-safe queue so the UI can visualise every
        # query's full pipeline live; the main thread drains the queue and yields.
        ev_q: "queue.Queue[Dict[str, Any]]" = queue.Queue()

        def _worker(i: int, spec: Tuple[ExtractedAttribute, str, str]) -> None:
            try:
                out = _process_attribute(i, spec[0], spec[1], spec[2], ev_q.put)
                ev_q.put({"_final": out})
            except Exception as exc:  # noqa: BLE001
                ev_q.put({"_final": {
                    "idx": i,
                    "event": {"stage": "attribute", "status": "done", "index": i,
                              "label": spec[0].label, "raw_name": spec[0].raw_name,
                              "available": False, "value": None, "category": "not_found",
                              "error": str(exc)},
                    "result": {"raw_name": spec[0].raw_name, "value": None, "auto_write": False},
                    "category": "not_found",
                }})

        with ThreadPoolExecutor(max_workers=answer_workers) as pool:
            for i, spec in enumerate(specs):
                pool.submit(_worker, i, spec)
            finals = 0
            while finals < len(specs):
                item = ev_q.get()
                final = item.get("_final")
                if final is not None:
                    results_by_idx[final["idx"]] = final["result"]
                    _bump_summary(summary, final["event"])
                    yield final["event"]
                    finals += 1
                else:
                    yield item

        # Assemble the write-back list in original field order (idempotent anyway,
        # since each entry targets a distinct key).
        results = [results_by_idx[i] for i in range(len(specs)) if i in results_by_idx]

    enriched_json = build_enriched_record(record, results)
    yield {"stage": "complete", "enriched_json": enriched_json, "summary": summary}
