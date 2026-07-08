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
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterator, List, Optional, Tuple

import rag_evaluation

from engine import ENGINE, _chunk_payload
from highlights import highlights_for_matches
from rag.inference.context_builder import build_context
from rag.inference.hierarchical_answer import answer_with_hierarchical_context
from rag.inference.prompt import make_prompt, parse_generator_json
from rag.inference.reranker import get_reranker, rerank

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
DEMO_MAX_ATTRIBUTES = None

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

_DATE_CANDIDATE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"|[0-9]{1,2}[/.\-][0-9]{1,2}[/.\-][0-9]{2,4}"
    r"|[0-9]{1,2}\s+[A-Za-z]{3,9},?\s+[0-9]{4}"
    r"|[A-Za-z]{3,9}\s+[0-9]{1,2},?\s+[0-9]{4}"
)


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
    if key in _DATE_LABELS:
        return _normalize_date(text) or text.strip()
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

    if key in _DATE_LABELS:
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
    confidence: Optional[float],
) -> str:
    """Bucket an attribute result into a dashboard category."""
    if not available or not answer.strip():
        return "not_found"
    if has_gt:
        return "validated_correct" if matches_gt else "validated_incorrect"
    # Null ground truth -> this is an enrichment we discovered from the document.
    if confidence is not None and confidence >= CONFIDENCE_THRESHOLD:
        return "enriched_confident"
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
        "validated_correct": 0,
        "validated_incorrect": 0,
        "enriched_confident": 0,
        "enriched_low_conf": 0,
        "not_found": 0,
    }


def _is_empty(value: Any) -> bool:
    """A record field counts as null/empty when it is None or an empty container."""
    return value is None or (isinstance(value, (list, dict, str)) and len(value) == 0)


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
            idx: int, attr: ExtractedAttribute, query: str, keywords: str
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

            question = rag_evaluation.inject_name(query, well_name) if well_name else query
            retrieval_query = f"{question} {keywords}".strip() if keywords else question

            with _RETRIEVE_LOCK:
                matches = ENGINE._retrieve(store, cfg, doc_id, retrieval_query)

            if cfg.use_reranker and matches and reranker is not None:
                with _RERANK_LOCK:
                    matches = rerank(retrieval_query, matches, cfg.reranker_model,
                                     top_k=cfg.rerank_top_k)
            elif matches:
                matches = matches[: cfg.rerank_top_k]

            with _RETRIEVE_LOCK:
                corpus = ENGINE._corpus_for_context(store, cfg, matches)
            context_items, _raw_text = build_context(matches, corpus,
                                                      context_mode=cfg.context_mode)

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
            confidence = parsed["confidence"]
            source_pages = parsed["source_pages"]

            normalized = normalize_value(label, answer_text) if available else None

            # Validate prefilled fields against their existing value (the record's
            # own value acts as the ground truth). Known OSDU labels use the
            # coordinate/date comparators; unknown fields use a token containment.
            has_gt = not was_null
            gt_value: Any = None
            gt_tokens: List[str] = []
            matches_gt: Optional[bool] = None
            if has_gt:
                key = (label or "").strip().lower()
                if key in rag_evaluation.GROUND_TRUTH_EXTRACTORS:
                    g = rag_evaluation.extract_ground_truth(label, record)
                    gt_value = g.get("value") if g.get("value") is not None else existing_value
                    gt_tokens = g.get("tokens") or [str(existing_value)]
                else:
                    gt_value = existing_value
                    gt_tokens = [str(existing_value)]
                if available and answer_text.strip():
                    matches_gt = matches_ground_truth(label, gt_value, gt_tokens, answer_text)

            category = classify(available=available, answer=answer_text,
                                has_gt=has_gt, matches_gt=matches_gt,
                                confidence=confidence)

            proposed = None
            if available and answer_text.strip():
                proposed = normalized or answer_text.strip()

            # Auto write-back: fill nulls; overwrite a prefilled field only when we
            # disagree with high confidence. Human verdicts are applied client-side.
            auto_write = False
            if proposed is not None:
                if was_null:
                    auto_write = True
                elif (matches_gt is False and confidence is not None
                      and confidence >= CONFIDENCE_THRESHOLD):
                    auto_write = True

            chunks_payload = [_chunk_payload(c, s) for c, s in matches]
            try:
                pages = highlights_for_matches(datalab_path, matches)
            except Exception:  # noqa: BLE001
                pages = []

            attr_cost = rag_evaluation.token_cost_usd(
                usage.get("input_tokens", 0), usage.get("output_tokens", 0)
            ) if usage else 0.0

            event = {
                "stage": "attribute", "status": "done", "index": idx, "label": label,
                "raw_name": raw_name,
                "question": question,
                "answer": answer_text,
                "available": available,
                "confidence": confidence,
                "normalized": normalized,
                "value": proposed,
                "source_pages": source_pages,
                "original_value": existing_value,
                "ground_truth": existing_value,
                "was_null": was_null,
                "has_gt": has_gt,
                "matches_gt": matches_gt,
                "auto_write": auto_write,
                "category": category,
                "chunks": chunks_payload,
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
            question = rag_evaluation.inject_name(query, well_name) if well_name else query
            yield {"stage": "attribute", "status": "start", "index": idx,
                   "label": attr.label, "raw_name": attr.raw_name, "question": question}

        yield {"stage": "answers", "status": "start", "count": len(specs)}
        answer_workers = max(1, min(ANSWER_WORKERS, len(specs))) if specs else 1
        results_by_idx: Dict[int, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=answer_workers) as pool:
            fut_to_idx = {
                pool.submit(_process_attribute, i, spec[0], spec[1], spec[2]): i
                for i, spec in enumerate(specs)
            }
            for fut in as_completed(fut_to_idx):
                out = fut.result()
                results_by_idx[out["idx"]] = out["result"]
                summary[out["category"]] += 1
                summary["total"] += 1
                yield out["event"]

        # Assemble the write-back list in original field order (idempotent anyway,
        # since each entry targets a distinct key).
        results = [results_by_idx[i] for i in range(len(specs)) if i in results_by_idx]

    enriched_json = build_enriched_record(record, results)
    yield {"stage": "complete", "enriched_json": enriched_json, "summary": summary}
