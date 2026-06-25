"""Folder 2 orchestrator: query  ->  answer.

Pipeline: embed query -> retrieve -> (rerank) -> build context -> prompt -> LLM answer.
Each step lives in its own module; this file wires them together using the
hyperparameters in :class:`rag.config.RagConfig`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from rag.common.schema import Chunk
from rag.config import RagConfig
from rag.indexing.vector_store import VectorStore, load_store
from rag.inference.context_builder import build_context, save_pages_sent_to_llm
from rag.inference.llm import answer_with_llm, get_llm
from rag.inference.prompt import make_prompt
from rag.inference.reranker import rerank
from rag.inference.retrieval import retrieve_top_chunks


def load_vector_store(config: RagConfig) -> Any:
    """Load the store named in config, matching ``config.vector_backend``.

    * "chroma" - open the persistent collection holding the precomputed vectors
                 (no re-embedding; serves the embed-once store built earlier).
    * "faiss"  - load ``index.faiss`` + ``meta.pkl`` from ``resolved_index_dir``.
    """
    if config.vector_backend == "chroma":
        from rag.indexing.build_pipeline import make_backend
        from rag.indexing.chroma_store import get_or_create_store

        print(
            f"Loading Chroma collection '{config.chroma_collection}' "
            f"from {config.chroma_dir} ..."
        )
        backend = make_backend(config)
        return get_or_create_store(
            config.chroma_dir, config.chroma_collection, backend, metric=config.faiss_metric
        )

    print(f"Loading vector store from {config.resolved_index_dir} ...")
    return load_store(config.resolved_index_dir)


def _retrieve_chroma(
    query: str,
    store: Any,
    config: RagConfig,
    section_filter: Optional[str],
    page_range: Optional[Tuple[int, int]],
) -> List[Tuple[Chunk, float]]:
    """Retrieve from a Chroma collection (dense or hybrid) with optional filters."""
    if config.retrieval_mode == "hybrid":
        return store.query_hybrid(
            query,
            top_k=config.top_k,
            hybrid_alpha=config.hybrid_alpha,
            section_filter=section_filter,
            page_range=page_range,
        )

    where: Optional[Dict[str, Any]] = None
    if page_range:
        lo, hi = page_range
        where = {"$and": [{"page": {"$gte": int(lo)}}, {"page": {"$lte": int(hi)}}]}

    matches = store.query(query, top_k=config.top_k, where=where)

    if section_filter:
        filtered = [
            (c, s) for c, s in matches
            if section_filter.lower() in (c.section_path or "").lower()
        ]
        if filtered:
            matches = filtered
        else:
            print("[warn] Section filter matched 0 chunks - ignoring filter.")
    return matches


def _context_corpus(
    store: Any,
    matches: List[Tuple[Chunk, float]],
    config: RagConfig,
) -> List[Chunk]:
    """Return the chunk corpus ``build_context`` needs for the chosen mode.

    For FAISS the whole corpus lives in ``store.chunks``. For Chroma we fetch
    just the chunks on the matched pages when full-page context is requested.
    """
    if config.vector_backend != "chroma":
        return store.chunks
    if config.context_mode == "full_page":
        keys = {(c.source, c.page) for c, _ in matches}
        return store.fetch_chunks_for_pages(keys)
    return [c for c, _ in matches]


def retrieve_and_build_context(
    query: str,
    store: Any,
    config: RagConfig,
    section_filter: Optional[str] = None,
    page_range: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Tuple[Chunk, float]], List[Dict[str, Any]], str]:
    """Run retrieval (+ optional rerank) and assemble context items."""
    if config.vector_backend == "chroma":
        matches = _retrieve_chroma(query, store, config, section_filter, page_range)
    else:
        matches = retrieve_top_chunks(
            query,
            store,
            top_k=config.top_k,
            retrieval_mode=config.retrieval_mode,
            hybrid_alpha=config.hybrid_alpha,
            section_filter=section_filter,
            page_range=page_range,
        )
    if config.use_reranker and matches:
        matches = rerank(query, matches, config.reranker_model, top_k=config.rerank_top_k)

    all_chunks = _context_corpus(store, matches, config)
    context_items, raw_text = build_context(matches, all_chunks, context_mode=config.context_mode)
    return matches, context_items, raw_text


def _print_matches(matches: List[Tuple[Chunk, float]]) -> None:
    """Print every retrieved chunk in full: header metadata + chunk text."""
    print("=" * 100)
    print(f"RETRIEVED CHUNKS ({len(matches)})")
    print("=" * 100)
    for idx, (chunk, score) in enumerate(matches, 1):
        print(
            f"\n[{idx}] score={score:.4f}  source={chunk.source}  page={chunk.page}  "
            f"block_type={chunk.block_type}"
        )
        print(f"     section: {chunk.section_path or '(none)'}")
        print("-" * 100)
        print(chunk.text.strip())
        print("-" * 100)
    print("=" * 100)


def answer_query(
    query: str,
    store: VectorStore,
    config: RagConfig,
    llm: Any = None,
    chat_history: Optional[List[Dict[str, str]]] = None,
    section_filter: Optional[str] = None,
    page_range: Optional[Tuple[int, int]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """End-to-end answer for a single query. Returns matches/context/answer."""
    matches, context_items, _raw = retrieve_and_build_context(
        query, store, config, section_filter=section_filter, page_range=page_range
    )

    if not matches:
        if verbose:
            print("No matching chunks found.")
        return {"matches": [], "context_items": [], "answer": None}

    if verbose:
        print(f"\nMatched {len(matches)} chunks -> {len(context_items)} context block(s).")
        _print_matches(matches)

    if config.save_pages_sent_to_llm:
        save_pages_sent_to_llm(context_items, query, f"{config.resolved_index_dir}/queries")

    answer: Optional[str] = None
    if llm is not None:
        if verbose:
            print("\nGenerating answer with LLM ...")
        prompt = make_prompt(query, context_items, chat_history=chat_history)
        answer = answer_with_llm(llm, prompt)
        if verbose:
            print("\nLLM answer:\n", answer)
    elif verbose:
        print("\n[info] LLM not available - showing retrieved context only.")

    return {"matches": matches, "context_items": context_items, "answer": answer}


def run_query(config: RagConfig, query: str, **kwargs) -> Dict[str, Any]:
    """Convenience: load store + LLM, answer one query."""
    store = load_vector_store(config)
    llm = get_llm(config)
    return answer_query(query, store, config, llm=llm, **kwargs)


def interactive_loop(config: RagConfig) -> None:
    """Interactive REPL with persistent chat history."""
    store = load_vector_store(config)
    llm = get_llm(config)
    chat_history: List[Dict[str, str]] = []

    print("Interactive mode - type 'exit' or press Enter on a blank line to quit.\n")
    while True:
        query = input("Query: ").strip()
        if not query or query.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        result = answer_query(query, store, config, llm=llm, chat_history=chat_history)
        if result["answer"]:
            chat_history.append({"query": query, "answer": result["answer"]})
        print()
