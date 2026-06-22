"""Folder 1 orchestrator: extraction JSON  ->  FAISS vector store.

Pipeline: load JSON -> merge (section, optional cross-page) -> chunk -> embed -> store.
Every step lives in its own module; this file just wires them together using the
hyperparameters in :class:`rag.config.RagConfig`.
"""

from __future__ import annotations

from typing import List, Optional

from rag.common.io_utils import save_chunks_jsonl
from rag.common.schema import Chunk
from rag.config import RagConfig
from rag.indexing.chunking import build_useful_chunks
from rag.indexing.embeddings import EmbeddingBackend
from rag.indexing.loader import load_extracted_json
from rag.indexing.merging import merge_across_pages, merge_blocks_by_section
from rag.indexing.vector_store import VectorStore, build_vector_store, save_store


def chunks_from_json(input_path: str, config: RagConfig) -> List[Chunk]:
    """Run only the text-processing part (load -> merge -> chunk)."""
    base = load_extracted_json(input_path)
    print(f"  {len(base)} raw text blocks extracted")

    if config.merge_sections:
        merged = merge_blocks_by_section(base)
        print(f"  {len(merged)} chunks after within-page section merge")

        if config.merge_across_pages:
            merged = merge_across_pages(merged)
            print(f"  {len(merged)} chunks after cross-page merge")
    else:
        merged = base
        print("  naive mode: skipping section merge")

    useful = build_useful_chunks(merged, chunk_size=config.chunk_size, overlap=config.overlap)
    print(f"  {len(useful)} final retrieval chunks")
    return useful


def make_backend(config: RagConfig) -> EmbeddingBackend:
    """Construct the embedding backend selected in config."""
    preferred = config.resolved_model_path or config.resolved_embedding_model
    return EmbeddingBackend(
        preferred_model=preferred,
        openai_model=config.openai_embedding_model,
        openai_base_url=config.openai_embedding_base_url or config.llm_base_url,
        openai_partition_id=config.partition_id,
    )


def build(
    input_path: str,
    config: RagConfig,
    chunks_output: Optional[str] = None,
) -> VectorStore:
    """Full build: produce chunks, embed them and persist the FAISS store."""
    print(f"Loading extracted JSON from {input_path}")
    useful = chunks_from_json(input_path, config)

    if chunks_output:
        print(f"Saving chunks -> {chunks_output}")
        save_chunks_jsonl(useful, chunks_output)

    backend = make_backend(config)
    store = build_vector_store(useful, backend, metric=config.faiss_metric)
    save_store(store, config.resolved_index_dir)
    return store
