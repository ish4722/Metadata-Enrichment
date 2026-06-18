"""
Reusable evaluation utilities for the datalab RAG pipelines.

This module is pipeline-agnostic: it provides the metrics machinery (per-step
timing, peak memory, token usage, cost, and accuracy) plus a generic
``run_evaluation`` driver. Each pipeline (datalab_rag_pipeline.py,
datalab_rag_pipeline2.py, datalab_rag_pipeline3.py, ...) wires up a tiny adapter
describing its own build steps and how it retrieves / answers, then calls
``run_evaluation``.

Accuracy is measured by comparing the pipeline output against ground-truth
values extracted from a "matched" JSON document (the structured record that the
PDF was linked to).

Token pricing (USD per 1,000 tokens) can be overridden via environment
variables ``LLM_INPUT_PRICE_PER_1K`` and ``LLM_OUTPUT_PRICE_PER_1K``. The
defaults are rough estimates for a small GPT-class model and should be treated
as approximate.
"""

from __future__ import annotations

import datetime
import json
import os
import pickle
import re
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

DEFAULT_CACHE_ROOT = "index_cache"
TIMINGS_FILENAME = "build_timings.txt"

LLM_INPUT_PRICE_PER_1K = float(os.getenv("LLM_INPUT_PRICE_PER_1K", "0.00015"))
LLM_OUTPUT_PRICE_PER_1K = float(os.getenv("LLM_OUTPUT_PRICE_PER_1K", "0.00060"))


# ---------------------------------------------------------------------------
# Metrics primitives
# ---------------------------------------------------------------------------


@dataclass
class StepMetrics:
    step: str
    cost_usd: float = 0.0
    tokens: int = 0
    accuracy: Optional[float] = None  # fraction in [0, 1]; None means N/A
    time_s: float = 0.0
    peak_mem_mb: float = 0.0


class StepProfiler:
    """Context manager that measures wall-clock time and peak memory of a step."""

    def __init__(self) -> None:
        self.elapsed = 0.0
        self.peak_mem_mb = 0.0
        self._start = 0.0

    def __enter__(self) -> "StepProfiler":
        tracemalloc.start()
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.elapsed = time.perf_counter() - self._start
        _current, peak = tracemalloc.get_traced_memory()
        self.peak_mem_mb = peak / (1024 * 1024)
        tracemalloc.stop()


# ---------------------------------------------------------------------------
# Queries file parsing
# ---------------------------------------------------------------------------


def parse_queries_file(path: str) -> List[Dict[str, str]]:
    """
    Parse a queries file shaped like:

        <label>:
        <question>

        <optional description>

        Keywords: ...

    Returns a list of dicts with keys: label, question, keywords, retrieval_query.
    Blocks are separated by a label line (a non-"Keywords:" line ending in ':').
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = [line.rstrip("\n") for line in f]

    queries: List[Dict[str, str]] = []
    current: Optional[Dict[str, str]] = None
    body_lines: List[str] = []

    def flush() -> None:
        if current is None:
            return
        question = ""
        keywords = ""
        for line in body_lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("keywords:"):
                keywords = stripped.split(":", 1)[1].strip()
            elif not question:
                question = stripped
        current["question"] = question or current["label"]
        current["keywords"] = keywords
        retrieval = current["question"]
        if keywords:
            retrieval = f"{retrieval} {keywords}"
        current["retrieval_query"] = retrieval
        queries.append(current)

    for line in raw_lines:
        stripped = line.strip()
        is_label = (
            stripped.endswith(":")
            and not stripped.lower().startswith("keywords:")
            and len(stripped) > 1
        )
        if is_label:
            flush()
            current = {"label": stripped[:-1].strip()}
            body_lines = []
        elif current is not None:
            body_lines.append(line)

    flush()
    return queries


# ---------------------------------------------------------------------------
# Ground-truth extraction from a matched JSON record
# ---------------------------------------------------------------------------


def _data_root(matched_json: Dict[str, Any]) -> Dict[str, Any]:
    """Return the inner 'data' object of a matched JSON, or the object itself."""
    if isinstance(matched_json, dict) and isinstance(matched_json.get("data"), dict):
        return matched_json["data"]
    return matched_json if isinstance(matched_json, dict) else {}


def _trunc(value: float, decimals: int = 2) -> str:
    """Truncate (not round) a float to a fixed number of decimals as a string."""
    text = f"{value:.6f}"
    int_part, _, dec_part = text.partition(".")
    return f"{int_part}.{dec_part[:decimals]}"


def _strip_reference_id(value: Optional[str]) -> Optional[str]:
    """Turn 'partition:reference-data--OperatingEnvironment:Onshore:' into 'Onshore'."""
    if not value or not isinstance(value, str):
        return value
    parts = [p for p in value.split(":") if p]
    return parts[-1] if parts else value


def gt_facility_name(data: Dict[str, Any]) -> Dict[str, Any]:
    name = data.get("FacilityName") or data.get("WellName")
    tokens = [name] if name else []
    return {"value": name, "tokens": tokens}


def gt_operating_environment(data: Dict[str, Any]) -> Dict[str, Any]:
    value = _strip_reference_id(data.get("OperatingEnvironmentID")) or data.get("WellboreSituation")
    tokens = [value] if value else []
    return {"value": value, "tokens": tokens}


def gt_spud_date(data: Dict[str, Any]) -> Dict[str, Any]:
    for event in data.get("FacilityEvents", []) or []:
        event_type = (event.get("FacilityEventTypeID") or "").lower()
        if "spud" in event_type:
            raw = event.get("EffectiveDateTime") or ""
            date_only = raw.split("T")[0] if raw else ""
            tokens = [date_only] if date_only else []
            return {"value": date_only or raw, "tokens": tokens}
    return {"value": None, "tokens": []}


def gt_lat_long(data: Dict[str, Any]) -> Dict[str, Any]:
    spatial = data.get("SpatialLocation.Wgs84Coordinates") or {}
    geometries = spatial.get("geometries", []) if isinstance(spatial, dict) else []
    if geometries:
        coords = geometries[0].get("coordinates")
        if coords and len(coords) >= 2:
            lon, lat = float(coords[0]), float(coords[1])
            value = f"{lat}, {lon}"
            tokens = [_trunc(lon), _trunc(lat)]
            return {"value": value, "tokens": tokens}
    return {"value": None, "tokens": []}


# Maps normalized query labels (lower-cased) to ground-truth extractor functions.
GROUND_TRUTH_EXTRACTORS: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {
    "facility name": gt_facility_name,
    "openv id": gt_operating_environment,
    "facilityevent spud date": gt_spud_date,
    "lat+long": gt_lat_long,
}


def extract_ground_truth(label: str, matched_json: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the ground-truth value/tokens for a query label."""
    data = _data_root(matched_json)
    key = label.strip().lower()
    extractor = GROUND_TRUTH_EXTRACTORS.get(key)
    if extractor is None:
        return {"value": None, "tokens": []}
    return extractor(data)


def _normalize_for_match(text: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def answer_is_correct(answer: str, tokens: List[str]) -> bool:
    """An answer is correct when every ground-truth token appears within it."""
    if not tokens:
        return False
    normalized_answer = _normalize_for_match(answer)
    return all(_normalize_for_match(tok) in normalized_answer for tok in tokens if tok)


# ---------------------------------------------------------------------------
# Token usage + cost
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """Rough fallback token estimate (~4 characters per token)."""
    return max(1, len(text) // 4)


def extract_token_usage(response: Any, prompt: str, answer: str) -> Dict[str, int]:
    """
    Pull input/output token counts from a LangChain LLM response, falling back
    to a character-based estimate when the provider does not return usage data.
    """
    input_tokens = output_tokens = 0

    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)

    if not (input_tokens or output_tokens):
        meta = getattr(response, "response_metadata", {}) or {}
        token_usage = meta.get("token_usage") or meta.get("usage") or {}
        input_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(token_usage.get("completion_tokens", 0) or 0)

    if not (input_tokens or output_tokens):
        input_tokens = _estimate_tokens(prompt)
        output_tokens = _estimate_tokens(answer)

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
    }


def token_cost_usd(input_tokens: int, output_tokens: int) -> float:
    """Compute the approximate USD cost for the given token counts."""
    return (
        input_tokens / 1000.0 * LLM_INPUT_PRICE_PER_1K
        + output_tokens / 1000.0 * LLM_OUTPUT_PRICE_PER_1K
    )


def invoke_llm_with_usage(llm: Any, prompt: str) -> Tuple[str, Dict[str, int]]:
    """
    Invoke a LangChain-style LLM with a fully-built prompt and return
    ``(answer_text, token_usage)``. Pipelines build their own prompt (each has a
    different prompt format) and delegate the call here so token accounting is
    centralized.
    """
    response = llm.invoke(prompt)
    answer = str(response.content) if hasattr(response, "content") else str(response)
    usage = extract_token_usage(response, prompt, answer)
    return answer, usage


def make_judge_prompt(question: str, ground_truth: str, answer: str) -> str:
    """Prompt asking an LLM to decide if an answer semantically matches the truth."""
    return (
        "You are a strict evaluator for a retrieval system. Decide whether the "
        "SYSTEM ANSWER correctly conveys the EXPECTED VALUE.\n\n"
        f"Question: {question}\n"
        f"Expected value (ground truth): {ground_truth}\n"
        f"System answer: {answer}\n\n"
        "Treat the answer as CORRECT if it states the same value as the expected "
        "value, allowing for differences in formatting, units, date formats, "
        "coordinate notation (e.g. decimal degrees vs degrees-minutes-seconds), "
        "abbreviations, synonyms, or extra surrounding text. Treat it as INCORRECT "
        "if it gives a different value, contradicts the expected value, refers to a "
        "different entity, or says the information is unavailable.\n"
        "Respond with exactly one word: YES if it is correct, or NO if it is not."
    )


def llm_judge_match(
    llm: Any,
    question: str,
    ground_truth: str,
    answer: str,
) -> Tuple[bool, Dict[str, int]]:
    """
    Use an LLM to judge whether ``answer`` semantically matches ``ground_truth``.
    Returns ``(is_match, token_usage)``.
    """
    prompt = make_judge_prompt(question, ground_truth, answer)
    verdict_text, usage = invoke_llm_with_usage(llm, prompt)
    normalized = verdict_text.strip().lower()
    is_match = normalized.startswith("yes") or re.match(r"^\W*yes\b", normalized) is not None
    return is_match, usage


# ---------------------------------------------------------------------------
# Table rendering
# ---------------------------------------------------------------------------


def render_metrics_table(metrics: List[StepMetrics]) -> str:
    """Render the per-step metrics as an aligned ASCII table."""
    headers = ["Step", "Cost ($)", "Tokens", "Accuracy", "Time (s)", "Memory (MB)"]

    def fmt_accuracy(acc: Optional[float]) -> str:
        return f"{acc * 100:.1f}%" if acc is not None else "N/A"

    rows = [
        [
            m.step,
            f"{m.cost_usd:.6f}",
            str(m.tokens),
            fmt_accuracy(m.accuracy),
            f"{m.time_s:.4f}",
            f"{m.peak_mem_mb:.2f}",
        ]
        for m in metrics
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    separator = "-+-".join("-" * w for w in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in rows)
    return "\n".join(lines)


def configure_utf8_stdout() -> None:
    """
    Make console output robust to non-ASCII characters in LLM answers on
    platforms whose default encoding (e.g. Windows cp1252) cannot encode them.
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


# ---------------------------------------------------------------------------
# Index caching + build-timing log
#
# Each backend ("tfidf", "jina", ...) gets its own folder under the cache root.
# Inside it we store one pickle per (PDF, chunk_size, overlap) plus a
# build_timings.txt log (JSON-lines) that records the per-step build timings.
# When a cached pickle is reused, the build timings are read back from the log
# instead of being re-measured.
# ---------------------------------------------------------------------------


def safe_cache_stem(input_path: str, chunk_size: int, overlap: int) -> str:
    """Build a filesystem-safe key for a given input PDF + chunking params."""
    base = os.path.basename(input_path)
    if base.lower().endswith(".json"):
        base = base[:-5]
    base = re.sub(r"\s+", "_", base.strip())
    base = re.sub(r"[^A-Za-z0-9._()-]", "_", base)
    return f"{base}__cs{chunk_size}_ov{overlap}"


def cache_index_path(cache_root: str, backend_name: str, stem: str) -> str:
    return os.path.join(cache_root, backend_name, stem + ".pkl")


def _timings_file(cache_root: str, backend_name: str) -> str:
    return os.path.join(cache_root, backend_name, TIMINGS_FILENAME)


def default_save_index(path: str, ctx: Any) -> None:
    """Persist the build context (a picklable object, usually a dict) to disk."""
    with open(path, "wb") as f:
        pickle.dump(ctx, f)


def default_load_index(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def read_build_timings(cache_root: str, backend_name: str, stem: str) -> Optional[List[StepMetrics]]:
    """Return the recorded build StepMetrics for a cache key, or None if absent."""
    path = _timings_file(cache_root, backend_name)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("key") == stem:
                return [StepMetrics(**step) for step in rec.get("steps", [])]
    return None


def write_build_timings(
    cache_root: str,
    backend_name: str,
    stem: str,
    pdf: str,
    build_metrics: List[StepMetrics],
    backend_mode: Optional[str] = None,
) -> None:
    """Append/replace the build-timing record for a cache key (JSON-lines txt)."""
    path = _timings_file(cache_root, backend_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    records: List[Dict[str, Any]] = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("key") != stem:
                    records.append(rec)

    records.append(
        {
            "key": stem,
            "pdf": pdf,
            "backend": backend_name,
            "backend_mode": backend_mode,
            "built_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "total_build_time_s": round(sum(m.time_s for m in build_metrics), 6),
            "steps": [asdict(m) for m in build_metrics],
        }
    )

    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Generic evaluation driver
# ---------------------------------------------------------------------------

# Type aliases for the adapter callables a pipeline must supply.
BuildStep = Tuple[str, Callable[[Any], Any]]
# retrieve_fn(ctx, query, top_k) -> (matches, retrieved_text)
RetrieveFn = Callable[[Any, str, int], Tuple[Any, str]]
# answer_fn(ctx, question, matches) -> (answer, usage_dict) or None when no LLM
AnswerFn = Callable[[Any, str, Any], Optional[Tuple[str, Dict[str, int]]]]
# judge_fn(question, ground_truth, answer) -> (is_match, usage_dict)
JudgeFn = Callable[[str, str, str], Tuple[bool, Dict[str, int]]]


def run_evaluation(
    *,
    input_path: str,
    queries_path: str,
    ground_truth_path: str,
    top_k: int,
    build_steps: List[BuildStep],
    retrieve_fn: RetrieveFn,
    answer_fn: Optional[AnswerFn] = None,
    judge_fn: Optional[JudgeFn] = None,
    metrics_output: Optional[str] = None,
    pipeline_name: str = "RAG pipeline",
    backend_name: str = "tfidf",
    chunk_size: int = 800,
    overlap: int = 200,
    cache_root: Optional[str] = DEFAULT_CACHE_ROOT,
    use_cache: bool = True,
    force_rebuild: bool = False,
    save_index_fn: Callable[[str, Any], None] = default_save_index,
    load_index_fn: Callable[[str], Any] = default_load_index,
    backend_mode_fn: Optional[Callable[[Any], str]] = None,
) -> List[StepMetrics]:
    """
    Run a full evaluation and print a per-step metrics table.

    Parameters
    ----------
    build_steps:
        A chained list of ``(name, fn)``. The first ``fn`` receives
        ``input_path``; each subsequent ``fn`` receives the previous return
        value. The final return value is the ``ctx`` passed to ``retrieve_fn`` /
        ``answer_fn``. Each step is profiled (time + peak memory).
    retrieve_fn:
        ``(ctx, query, top_k) -> (matches, retrieved_text)``. ``matches`` is
        opaque and forwarded to ``answer_fn``; ``retrieved_text`` is used to
        measure retrieval accuracy against ground truth.
    answer_fn:
        Optional ``(ctx, question, matches) -> (answer, usage)`` where ``usage``
        has ``input_tokens`` / ``output_tokens`` / ``total_tokens``. Return
        ``None`` (or pass ``answer_fn=None``) when no LLM is available.

    Index caching
    -------------
    When ``cache_root`` is set, the built ``ctx`` is pickled to
    ``<cache_root>/<backend_name>/<pdf>__cs<size>_ov<overlap>.pkl`` and the build
    timings are logged to ``<cache_root>/<backend_name>/build_timings.txt``. On a
    later run for the same PDF + params, the pickle is reused (no rebuild) and the
    build timings are read back from the log. Pass ``force_rebuild=True`` to
    ignore an existing cache, or ``use_cache=False`` to disable caching entirely.
    """
    print(f"=== Evaluating {pipeline_name} ===")
    print(f"Loading queries from {queries_path}")
    queries = parse_queries_file(queries_path)
    print(f"  -> {len(queries)} queries parsed")

    print(f"Loading ground-truth JSON from {ground_truth_path}")
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        matched_json = json.load(f)

    metrics: List[StepMetrics] = []

    # --- Build phase: reuse cached index when available, else build + cache ---
    stem = safe_cache_stem(input_path, chunk_size, overlap)
    index_path = cache_index_path(cache_root, backend_name, stem) if cache_root else None
    loaded_from_cache = False
    ctx: Any

    if index_path and use_cache and not force_rebuild and os.path.exists(index_path):
        print(f"Reusing cached index: {index_path}")
        ctx = load_index_fn(index_path)
        loaded_from_cache = True
        recorded = read_build_timings(cache_root, backend_name, stem)
        if recorded:
            metrics.extend(recorded)
            total_build = sum(m.time_s for m in recorded)
            print(f"  Build timings loaded from log ({total_build:.4f}s total, not re-measured).")
        else:
            metrics.append(StepMetrics("Build (cached; timings not in log)"))
            print("  No build timings found in log for this index.")
    else:
        ctx = input_path
        for name, fn in build_steps:
            with StepProfiler() as prof:
                ctx = fn(ctx)
            metrics.append(StepMetrics(name, time_s=prof.elapsed, peak_mem_mb=prof.peak_mem_mb))
            size_note = f" ({len(ctx)} items)" if isinstance(ctx, (list, tuple)) else ""
            print(f"  {name}: {prof.elapsed:.4f}s{size_note}")

        if index_path:
            try:
                os.makedirs(os.path.dirname(index_path), exist_ok=True)
                save_index_fn(index_path, ctx)
                backend_mode = backend_mode_fn(ctx) if backend_mode_fn else None
                write_build_timings(
                    cache_root, backend_name, stem,
                    os.path.basename(input_path), list(metrics), backend_mode,
                )
                print(f"Cached index -> {index_path}")
                print(f"Logged build timings -> {_timings_file(cache_root, backend_name)}")
            except Exception as exc:  # pragma: no cover - caching is best-effort
                print(f"[warn] Could not cache index ({exc}); continuing without cache.")

    # --- Query-phase steps (aggregated across all queries) ---
    retrieval_time = 0.0
    retrieval_peak = 0.0
    retrieval_correct = 0

    llm_time = 0.0
    llm_peak = 0.0
    llm_total_tokens = 0
    llm_cost = 0.0
    llm_correct = 0
    llm_evaluated = 0

    judge_time = 0.0
    judge_peak = 0.0
    judge_total_tokens = 0
    judge_cost = 0.0
    judge_used = 0

    per_query_rows: List[Dict[str, Any]] = []

    for q in queries:
        label = q["label"]
        gt = extract_ground_truth(label, matched_json)
        tokens = gt["tokens"]

        with StepProfiler() as prof:
            matches, retrieved_text = retrieve_fn(ctx, q["retrieval_query"], top_k)
        retrieval_time += prof.elapsed
        retrieval_peak = max(retrieval_peak, prof.peak_mem_mb)

        retrieval_hit = answer_is_correct(retrieved_text, tokens)
        if retrieval_hit:
            retrieval_correct += 1

        answer = ""
        usage = {"total_tokens": 0, "input_tokens": 0, "output_tokens": 0}
        llm_hit: Optional[bool] = None
        token_hit: Optional[bool] = None
        judge_hit: Optional[bool] = None
        if answer_fn is not None:
            with StepProfiler() as prof:
                result = answer_fn(ctx, q["question"], matches)
            if result is not None:
                answer, usage = result
                llm_time += prof.elapsed
                llm_peak = max(llm_peak, prof.peak_mem_mb)
                llm_total_tokens += usage.get("total_tokens", 0)
                llm_cost += token_cost_usd(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
                token_hit = answer_is_correct(answer, tokens)

                # Optional LLM judge: confirm a semantic match (handles format
                # differences the literal token check misses).
                if judge_fn is not None and gt["value"]:
                    with StepProfiler() as prof:
                        judge_hit, judge_usage = judge_fn(q["question"], str(gt["value"]), answer)
                    judge_time += prof.elapsed
                    judge_peak = max(judge_peak, prof.peak_mem_mb)
                    judge_total_tokens += judge_usage.get("total_tokens", 0)
                    judge_cost += token_cost_usd(judge_usage.get("input_tokens", 0), judge_usage.get("output_tokens", 0))
                    judge_used += 1
                    llm_hit = judge_hit
                else:
                    llm_hit = token_hit

                llm_evaluated += 1
                if llm_hit:
                    llm_correct += 1

        per_query_rows.append(
            {
                "label": label,
                "question": q["question"],
                "ground_truth": gt["value"],
                "token_match": token_hit,
                "judge_match": judge_hit,
                "retrieval_hit": retrieval_hit,
                "llm_hit": llm_hit,
                "answer": answer,
                "tokens": usage.get("total_tokens", 0),
            }
        )

    num_queries = len(queries) or 1
    retrieval_step_no = len(metrics) + 1
    metrics.append(
        StepMetrics(
            f"{retrieval_step_no}. Retrieval",
            accuracy=retrieval_correct / num_queries,
            time_s=retrieval_time,
            peak_mem_mb=retrieval_peak,
        )
    )

    llm_step_no = retrieval_step_no + 1
    if llm_evaluated:
        metrics.append(
            StepMetrics(
                f"{llm_step_no}. LLM generation",
                cost_usd=llm_cost,
                tokens=llm_total_tokens,
                accuracy=llm_correct / llm_evaluated,
                time_s=llm_time,
                peak_mem_mb=llm_peak,
            )
        )
    else:
        metrics.append(StepMetrics(f"{llm_step_no}. LLM generation (unavailable)"))

    if judge_used:
        metrics.append(
            StepMetrics(
                f"{llm_step_no + 1}. LLM verify (judge)",
                cost_usd=judge_cost,
                tokens=judge_total_tokens,
                accuracy=llm_correct / llm_evaluated if llm_evaluated else None,
                time_s=judge_time,
                peak_mem_mb=judge_peak,
            )
        )

    total = StepMetrics(
        "TOTAL",
        cost_usd=sum(m.cost_usd for m in metrics),
        tokens=sum(m.tokens for m in metrics),
        accuracy=metrics[-1].accuracy if metrics[-1].accuracy is not None else metrics[-2].accuracy,
        time_s=sum(m.time_s for m in metrics),
        peak_mem_mb=max((m.peak_mem_mb for m in metrics), default=0.0),
    )

    # --- Per-query detail ---
    print("\nPer-query results:")
    for row in per_query_rows:
        llm_status = "-" if row["llm_hit"] is None else ("MATCH" if row["llm_hit"] else "miss")
        retr_status = "MATCH" if row["retrieval_hit"] else "miss"
        print(f"\n  [{row['label']}] {row['question']}")
        print(f"    ground truth : {row['ground_truth']}")
        print(f"    retrieval    : {retr_status}")
        if row["judge_match"] is not None:
            token_status = "match" if row["token_match"] else "no-match"
            print(f"    llm (judge)  : {llm_status}  (literal token check: {token_status})")
        else:
            print(f"    llm          : {llm_status}")
        if row["answer"]:
            print(f"    answer       : {row['answer'][:200].strip()}")

    # --- Metrics table ---
    all_rows = metrics + [total]
    print("\nPipeline metrics by step:")
    print(render_metrics_table(all_rows))
    print(
        f"\nPricing used: input ${LLM_INPUT_PRICE_PER_1K}/1K tokens, "
        f"output ${LLM_OUTPUT_PRICE_PER_1K}/1K tokens "
        f"(override with LLM_INPUT_PRICE_PER_1K / LLM_OUTPUT_PRICE_PER_1K)."
    )

    if metrics_output:
        payload = {
            "pipeline": pipeline_name,
            "backend": backend_name,
            "input": input_path,
            "queries": queries_path,
            "ground_truth": ground_truth_path,
            "index_path": index_path,
            "loaded_from_cache": loaded_from_cache,
            "steps": [asdict(m) for m in all_rows],
            "per_query": per_query_rows,
        }
        with open(metrics_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics to {metrics_output}")

    return all_rows
