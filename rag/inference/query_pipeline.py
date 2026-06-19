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


def load_vector_store(config: RagConfig) -> VectorStore:
    """Load the FAISS store from the directory named in config."""
    print(f"Loading vector store from {config.index_dir} ...")
    return load_store(config.index_dir)


def retrieve_and_build_context(
    query: str,
    store: VectorStore,
    config: RagConfig,
    section_filter: Optional[str] = None,
    page_range: Optional[Tuple[int, int]] = None,
) -> Tuple[List[Tuple[Chunk, float]], List[Dict[str, Any]], str]:
    """Run retrieval (+ optional rerank) and assemble context items."""
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

    context_items, raw_text = build_context(matches, store.chunks, context_mode=config.context_mode)
    return matches, context_items, raw_text


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
        for idx, (chunk, score) in enumerate(matches, 1):
            print(
                f"  [{idx}] score={score:.4f} source={chunk.source} "
                f"page={chunk.page} section={chunk.section_path or chunk.block_type}"
            )

    if config.save_pages_sent_to_llm:
        save_pages_sent_to_llm(context_items, query, f"{config.index_dir}/queries")

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
