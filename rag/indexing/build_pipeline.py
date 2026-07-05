"""Folder 1 orchestrator: extraction JSON  ->  persisted vector store.

Pipeline: load JSON -> chunk (section-aware or legacy) -> embed once -> store.

Two storage backends (``config.vector_backend``):
  * "chroma" - persistent ChromaDB collection holding our PRECOMPUTED vectors.
               A document is embedded exactly once; reusing it is a no-op.
  * "faiss"  - the legacy in-memory FAISS index serialised to meta.pkl.
"""

from __future__ import annotations

import os
import re
from typing import List, Optional

from rag.common.io_utils import save_chunks_jsonl
from rag.common.schema import Chunk
from rag.config import RagConfig
from rag.indexing.chunking import build_useful_chunks
from rag.indexing.embeddings import EmbeddingBackend
from rag.indexing.loader import load_extracted_json
from rag.indexing.merging import merge_across_pages, merge_blocks_by_section
from rag.indexing.section_chunking import build_section_chunks
from rag.indexing.vector_store import VectorStore, build_vector_store, save_store


def derive_doc_id(input_path: str) -> str:
    """Turn a datalab filename into its doc id (e.g. 'PR2610')."""
    stem = os.path.basename(input_path)
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    stem = re.sub(r"^datalab-output-", "", stem, flags=re.IGNORECASE).strip()
    match = re.search(r"PR\d+", stem)
    return match.group(0) if match else stem


def chunks_from_json(input_path: str, config: RagConfig) -> List[Chunk]:
    """Run the text-processing part (load -> chunk) per the configured strategy."""
    base = load_extracted_json(input_path)
    print(f"  {len(base)} content blocks extracted")

    if config.chunk_strategy == "section":
        useful = build_section_chunks(base, chunk_size=config.chunk_size, overlap=config.overlap)
        print(f"  {len(useful)} section-aware chunks")
        return useful

    # legacy strategy: section merge -> cross-page merge -> sliding window
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
        openai_dimensions=config.openai_embedding_dimensions,
        jina_api_url=config.jina_api_url,
        jina_api_late_chunking=config.jina_api_late_chunking,
        jina_api_window_tokens=config.jina_api_window_tokens,
    )


def build(
    input_path: str,
    config: RagConfig,
    chunks_output: Optional[str] = None,
    backend: Optional[EmbeddingBackend] = None,
):
    """Full build: chunk, embed ONCE, and persist to the configured store.

    Returns the store object (a :class:`ChromaVectorStore` or :class:`VectorStore`).
    Pass a shared ``backend`` to avoid reloading the embedding model per doc.
    """
    print(f"Loading extracted JSON from {input_path}")
    useful = chunks_from_json(input_path, config)

    if chunks_output:
        print(f"Saving chunks -> {chunks_output}")
        save_chunks_jsonl(useful, chunks_output)

    backend = backend or make_backend(config)

    if config.vector_backend == "chroma":
        return _build_chroma(input_path, useful, backend, config)

    store = build_vector_store(
        useful, backend, metric=config.faiss_metric,
        build_bm25=(config.retrieval_mode == "hybrid"),
    )
    save_store(store, config.resolved_index_dir)
    return store


def _build_chroma(input_path: str, chunks: List[Chunk], backend: EmbeddingBackend, config: RagConfig):
    """Embed chunks once and upsert into Chroma (embed-once)."""
    from rag.indexing.chroma_store import embed_corpus, get_or_create_store

    doc_id = derive_doc_id(input_path)
    store = get_or_create_store(
        config.chroma_dir, config.chroma_collection, backend, metric=config.faiss_metric
    )

    if store.has_document(doc_id):
        print(f"  [embed-once] {doc_id} already in collection '{config.chroma_collection}' - skipping embedding")
        return store

    texts = [c.text for c in chunks]
    if not texts:
        raise RuntimeError(
            "No chunks produced - check the input layout (expected datalab page "
            "nodes under 'children' or 'result.json.children')."
        )
    print(f"Embedding {len(texts)} chunks with [{backend.backend_name}] ...")
    embeddings = embed_corpus(backend, texts)
    added = store.add_document(doc_id, chunks, embeddings)
    print(f"  [chroma] added {added} vectors for {doc_id} -> collection '{config.chroma_collection}'")
    return store
