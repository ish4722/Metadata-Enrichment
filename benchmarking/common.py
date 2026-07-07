"""Shared helpers for the benchmarking scripts.

Wraps the existing evaluation engines so every benchmark runs the pipeline the
same way and reports the same shape of numbers:

  * :func:`run_end_to_end`      -> BatchRunner build + run + summarize
    (LLM / token / retrieval accuracy, per-label, tokens, time, escalation).
  * :func:`run_retrieval_metrics` -> score_matryoshka_retrieval
    (pre/post-rerank MRR, MAP, Hit@k, Recall@k, nDCG@k, Distractor@k).
  * :func:`recompute_costs`     -> per-model USD cost from stored token counts,
    using ``benchmarking.model_specs`` (the pipeline itself bakes in one fixed
    price, so cross-model cost must be recomputed here).

Nothing is re-embedded when a Chroma collection already exists; BatchRunner's
``build_all`` skips docs that are already embedded (embed-once design).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches  # noqa: E402

apply_ssl_patches()

import evaluate_all  # noqa: E402
import rag_evaluation  # noqa: E402
from rag.config import RagConfig  # noqa: E402
from rag.evaluation.retrieval_metrics import score_matryoshka_retrieval  # noqa: E402
from rag.inference.batch_runner import BatchRunner, summarize  # noqa: E402

from benchmarking import model_specs  # noqa: E402

OUTPUT_ROOT = os.path.join("output_table", "benchmarking")
DEFAULT_GOLDSET = os.path.join("output_table", "retrieval_goldset.json")


# ---------------------------------------------------------------------------
# Config construction (baseline pipeline settings locked)
# ---------------------------------------------------------------------------


def build_config(
    *,
    embedding_model: Optional[str] = None,
    openai_model: Optional[str] = None,
    dimensions: Optional[int] = None,
    jina_late_chunking: Optional[bool] = None,
    use_reranker: Optional[bool] = None,
    llm_model: Optional[str] = None,
    retrieval_mode: str = "hybrid",
    hybrid_alpha: float = 0.85,
    context_mode: str = "hierarchical",
    hierarchical_max_depth: int = 5,
    chunk_strategy: str = "section",
) -> RagConfig:
    """A RagConfig with baseline pipeline settings locked; overrides applied.

    Everything except the explicitly-varied knob is fixed to the baseline batch
    run so comparisons are apples-to-apples.
    """
    cfg = RagConfig.from_env()
    if embedding_model is not None:
        cfg.embedding_model = embedding_model
    if openai_model is not None:
        cfg.openai_embedding_model = openai_model
    cfg.openai_embedding_dimensions = dimensions
    if jina_late_chunking is not None:
        cfg.jina_api_late_chunking = jina_late_chunking
    if use_reranker is not None:
        cfg.use_reranker = use_reranker
    if llm_model is not None:
        cfg.llm_model = llm_model
    cfg.retrieval_mode = retrieval_mode
    cfg.hybrid_alpha = hybrid_alpha
    cfg.context_mode = context_mode
    cfg.hierarchical_max_depth = hierarchical_max_depth
    cfg.chunk_strategy = chunk_strategy
    return cfg


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def discover(
    only: Optional[List[str]] = None,
    limit: Optional[int] = None,
    docs_dir: str = evaluate_all.DEFAULT_DOCS_DIR,
    matched_dir: str = evaluate_all.DEFAULT_MATCHED_DIR,
) -> List[Dict[str, str]]:
    """Discover (doc, ground-truth) pairs, optionally restricted / truncated."""
    matched, skipped = evaluate_all.discover_docs(docs_dir, matched_dir, only=only)
    if limit is not None:
        matched = matched[:limit]
    print(f"Discovered {len(matched)} doc(s)"
          + (f", skipped {len(skipped)} without ground truth" if skipped else ""))
    return matched


def load_queries(path: str = evaluate_all.DEFAULT_QUERIES) -> List[Dict[str, str]]:
    return rag_evaluation.parse_queries_file(path)


# ---------------------------------------------------------------------------
# Chroma collection inventory (to decide reuse vs build)
# ---------------------------------------------------------------------------


def collection_vector_count(cfg: RagConfig) -> int:
    """Number of vectors already stored for ``cfg``'s collection (0 if absent).

    Uses the pipeline's own ``get_chroma_client`` so the client settings match
    the ones BatchRunner will use: Chroma caches one client per path and raises
    "instance already exists ... with different settings" if a second client is
    opened for the same path with different settings.
    """
    try:
        from rag.indexing.chroma_store import get_chroma_client

        client = get_chroma_client(cfg.chroma_dir)
        for col in client.list_collections():
            if col.name == cfg.chroma_collection:
                return col.count()
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not inspect Chroma store: {exc}")
    return 0


# ---------------------------------------------------------------------------
# Run wrappers
# ---------------------------------------------------------------------------


def run_end_to_end(
    cfg: RagConfig,
    matched: List[Dict[str, str]],
    queries: List[Dict[str, str]],
    *,
    use_judge: bool = True,
    inject_name: bool = True,
    build: bool = True,
) -> Dict[str, Any]:
    """Build (reuse) + run + summarize one configuration end-to-end."""
    runner = BatchRunner(cfg, use_judge=use_judge, inject_name=inject_name)
    t_build = time.perf_counter()
    if build:
        runner.build_all(matched)
    build_s = time.perf_counter() - t_build

    t_run = time.perf_counter()
    results = runner.run(matched, queries)
    run_s = time.perf_counter() - t_run

    summary = summarize(results, embed_tokens=runner.embed_tokens,
                        embed_cost_usd=runner.embed_cost_usd,
                        llm_meta=runner.llm_usage_snapshot())
    summary["wall_time_s"] = {"build": build_s, "inference": run_s}
    return summary


def run_retrieval_metrics(
    cfg: RagConfig,
    goldset_path: str = DEFAULT_GOLDSET,
    queries_path: str = evaluate_all.DEFAULT_QUERIES,
) -> Dict[str, Any]:
    """MTEB-style retrieval metrics (pre + post rerank) for one config."""
    return score_matryoshka_retrieval(cfg, goldset_path, queries_path=queries_path)


# ---------------------------------------------------------------------------
# Cost recompute (per-model pricing)
# ---------------------------------------------------------------------------


def recompute_costs(
    summary: Dict[str, Any],
    *,
    llm_model: str,
    embed_spec_model: Optional[str] = None,
) -> Dict[str, Optional[float]]:
    """Recompute generation/judge/embedding cost with real per-model prices.

    ``summary["tokens"]`` holds the raw counts; ``llm_model`` prices generation
    and judge; ``embed_spec_model`` (a key in ``model_specs.EMBEDDING_SPECS``)
    prices the embedding input tokens (None => not metered, e.g. self-hosted).
    """
    tok = summary.get("tokens", {})
    gen_cost = model_specs.llm_cost(llm_model, tok.get("gen_input", 0), tok.get("gen_output", 0))
    judge_cost = model_specs.llm_cost(llm_model, tok.get("judge_input", 0), tok.get("judge_output", 0))
    embed = (
        model_specs.embed_cost(embed_spec_model, tok.get("embed_input", 0))
        if embed_spec_model else None
    )
    parts = [c for c in (gen_cost, judge_cost, embed) if c is not None]
    total = sum(parts) if parts else None
    return {
        "llm_model": llm_model,
        "embed_spec_model": embed_spec_model,
        "generation": gen_cost,
        "judge": judge_cost,
        "embedding": embed,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    print(f"  saved -> {path}")


def save_report(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  saved -> {path}")


def md_table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
    return "\n".join([line, sep, body])


def pct(v: Optional[float]) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "N/A"


def f3(v: Optional[float]) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"


def usd(v: Optional[float], decimals: int = 4) -> str:
    return f"${v:.{decimals}f}" if isinstance(v, (int, float)) else "N/A"


def num(v: Optional[float]) -> str:
    return f"{int(v):,}" if isinstance(v, (int, float)) else "N/A"


# ---------------------------------------------------------------------------
# OpenAI usage/cost callback metadata (surfaced from summary["llm_meta"])
# ---------------------------------------------------------------------------

# Column headers for the shared "LLM call metadata" table.
LLM_META_HEADERS = [
    "Requests", "Prompt tok", "Cached tok", "Cache hit",
    "Reason tok", "Compl tok", "Callback $",
]


def callback_cost(meta: Dict[str, Any], llm_model: str) -> Optional[float]:
    """USD from the callback's token counts using per-model list prices.

    Mirrors ``LLMUsageTracker.cost``: cached prompt tokens are billed at 25% of
    the input price (OpenAI prompt-cache discount). Returns None when the model
    price is unknown or no usage was captured.
    """
    spec = model_specs.LLM_SPECS.get(llm_model)
    if not spec or not meta:
        return None
    inp = float(spec["input_per_1m"])
    out = float(spec["output_per_1m"])
    prompt = int(meta.get("prompt_tokens", 0) or 0)
    cached = int(meta.get("cached_prompt_tokens", 0) or 0)
    completion = int(meta.get("completion_tokens", 0) or 0)
    billable = max(prompt - cached, 0)
    return (
        billable / 1_000_000.0 * inp
        + cached / 1_000_000.0 * inp * 0.25
        + completion / 1_000_000.0 * out
    )


def llm_meta_cells(summary: Dict[str, Any], llm_model: str) -> List[str]:
    """Formatted metadata cells (aligned with :data:`LLM_META_HEADERS`)."""
    meta = summary.get("llm_meta", {}) or {}
    return [
        num(meta.get("successful_requests")),
        num(meta.get("prompt_tokens")),
        num(meta.get("cached_prompt_tokens")),
        pct(meta.get("cache_hit_rate")),
        num(meta.get("reasoning_tokens")),
        num(meta.get("completion_tokens")),
        usd(callback_cost(meta, llm_model)),
    ]


def llm_meta_table(entries: List[Any], first_col: str = "Config") -> str:
    """Markdown metadata table for ``entries`` = [(name, summary, llm_model), ...]."""
    rows = [[name, *llm_meta_cells(summary, llm_model)] for name, summary, llm_model in entries]
    return md_table([first_col, *LLM_META_HEADERS], rows)
