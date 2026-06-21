"""Step 6 - Validation / evaluation.

Wires the modular pipeline into the existing pipeline-agnostic
``rag_evaluation.run_evaluation`` driver (kept untouched in the repo root) to
produce per-step metrics (time, memory, tokens, cost, accuracy).
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

from rag.config import RagConfig
from rag.indexing.chunking import build_useful_chunks
from rag.indexing.build_pipeline import make_backend
from rag.indexing.loader import load_extracted_json
from rag.indexing.merging import merge_across_pages, merge_blocks_by_section
from rag.indexing.vector_store import build_vector_store
from rag.inference.context_builder import build_context
from rag.inference.llm import get_llm
from rag.inference.prompt import make_prompt
from rag.inference.reranker import rerank
from rag.inference.retrieval import retrieve_top_chunks

# The shared evaluation driver lives in the repo root (one level above rag/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import rag_evaluation  # noqa: E402


def evaluate(
    config: RagConfig,
    input_path: str,
    queries_path: str,
    ground_truth_path: str,
    metrics_output: Optional[str] = None,
    cache_root: Optional[str] = rag_evaluation.DEFAULT_CACHE_ROOT,
    use_cache: bool = True,
    force_rebuild: bool = False,
    use_judge: bool = True,
) -> List[Any]:
    """Run the full evaluation using the current config's component choices.

    Returns the list of per-step :class:`rag_evaluation.StepMetrics` (including
    the final ``TOTAL`` row) so callers can reuse the totals programmatically.
    """
    llm = get_llm(config)
    if llm is None:
        print("LLM unavailable: accuracy will be measured against retrieved context only.")

    # ── build phase (chained, each step profiled by the driver) ──────────────
    build_steps: List[Any] = [
        ("1. Load JSON", load_extracted_json),
    ]
    if config.merge_sections:
        build_steps.append(("2. Merge (section)", merge_blocks_by_section))
        if config.merge_across_pages:
            build_steps.append(("3. Merge (cross-page)", merge_across_pages))
    build_steps.append(
        (f"{len(build_steps) + 1}. Chunking",
         lambda c: build_useful_chunks(c, chunk_size=config.chunk_size, overlap=config.overlap))
    )

    def build_index(useful) -> Dict[str, Any]:
        backend = make_backend(config)
        store = build_vector_store(useful, backend, metric=config.faiss_metric)
        return {"store": store, "chunks": useful}

    build_steps.append((f"{len(build_steps) + 1}. Build index", build_index))

    # ── query phase adapters ─────────────────────────────────────────────────
    def retrieve_fn(ctx: Dict[str, Any], query: str, k: int):
        matches = retrieve_top_chunks(
            query, ctx["store"], top_k=k,
            retrieval_mode=config.retrieval_mode, hybrid_alpha=config.hybrid_alpha,
        )
        if config.use_reranker and matches:
            matches = rerank(query, matches, config.reranker_model, top_k=config.rerank_top_k)
        context_items, raw_text = build_context(matches, ctx["chunks"], context_mode=config.context_mode)
        return context_items, raw_text

    def answer_fn(ctx: Dict[str, Any], question: str, context_items):
        if llm is None or not context_items:
            return None
        prompt = make_prompt(question, context_items)
        return rag_evaluation.invoke_llm_with_usage(llm, prompt)

    def judge_fn(question: str, ground_truth: str, answer: str):
        return rag_evaluation.llm_judge_match(llm, question, ground_truth, answer)

    def backend_mode_fn(ctx: Dict[str, Any]) -> str:
        try:
            return ctx["store"].backend.backend_name
        except Exception:
            return "unknown"

    return rag_evaluation.run_evaluation(
        input_path=input_path,
        queries_path=queries_path,
        ground_truth_path=ground_truth_path,
        top_k=config.top_k,
        build_steps=build_steps,
        retrieve_fn=retrieve_fn,
        answer_fn=answer_fn if llm is not None else None,
        judge_fn=judge_fn if (llm is not None and use_judge) else None,
        metrics_output=metrics_output,
        pipeline_name=f"rag ({config.embedding_model}, {config.retrieval_mode}, "
                      f"rerank={config.use_reranker}, {config.context_mode})",
        backend_name=config.backend_tag(),
        chunk_size=config.chunk_size,
        overlap=config.overlap,
        cache_root=cache_root,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
        backend_mode_fn=backend_mode_fn,
    )
