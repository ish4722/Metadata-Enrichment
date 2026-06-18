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

import json
import os
import re
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

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
# Generic evaluation driver
# ---------------------------------------------------------------------------

# Type aliases for the adapter callables a pipeline must supply.
BuildStep = Tuple[str, Callable[[Any], Any]]
# retrieve_fn(ctx, query, top_k) -> (matches, retrieved_text)
RetrieveFn = Callable[[Any, str, int], Tuple[Any, str]]
# answer_fn(ctx, question, matches) -> (answer, usage_dict) or None when no LLM
AnswerFn = Callable[[Any, str, Any], Optional[Tuple[str, Dict[str, int]]]]


def run_evaluation(
    *,
    input_path: str,
    queries_path: str,
    ground_truth_path: str,
    top_k: int,
    build_steps: List[BuildStep],
    retrieve_fn: RetrieveFn,
    answer_fn: Optional[AnswerFn] = None,
    metrics_output: Optional[str] = None,
    pipeline_name: str = "RAG pipeline",
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
    """
    print(f"=== Evaluating {pipeline_name} ===")
    print(f"Loading queries from {queries_path}")
    queries = parse_queries_file(queries_path)
    print(f"  -> {len(queries)} queries parsed")

    print(f"Loading ground-truth JSON from {ground_truth_path}")
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        matched_json = json.load(f)

    metrics: List[StepMetrics] = []

    # --- Build-phase steps (run once, chained) ---
    ctx: Any = input_path
    for name, fn in build_steps:
        with StepProfiler() as prof:
            ctx = fn(ctx)
        metrics.append(StepMetrics(name, time_s=prof.elapsed, peak_mem_mb=prof.peak_mem_mb))
        size_note = f" ({len(ctx)} items)" if isinstance(ctx, (list, tuple)) else ""
        print(f"  {name}: {prof.elapsed:.4f}s{size_note}")

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
        if answer_fn is not None:
            with StepProfiler() as prof:
                result = answer_fn(ctx, q["question"], matches)
            if result is not None:
                answer, usage = result
                llm_time += prof.elapsed
                llm_peak = max(llm_peak, prof.peak_mem_mb)
                llm_total_tokens += usage.get("total_tokens", 0)
                llm_cost += token_cost_usd(usage.get("input_tokens", 0), usage.get("output_tokens", 0))
                llm_hit = answer_is_correct(answer, tokens)
                llm_evaluated += 1
                if llm_hit:
                    llm_correct += 1

        per_query_rows.append(
            {
                "label": label,
                "question": q["question"],
                "ground_truth": gt["value"],
                "retrieval_hit": retrieval_hit,
                "llm_hit": llm_hit,
                "answer": answer,
                "tokens": usage.get("total_tokens", 0),
            }
        )

    num_queries = len(queries) or 1
    retrieval_step_no = len(build_steps) + 1
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
            "input": input_path,
            "queries": queries_path,
            "ground_truth": ground_truth_path,
            "steps": [asdict(m) for m in all_rows],
            "per_query": per_query_rows,
        }
        with open(metrics_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics to {metrics_output}")

    return all_rows
