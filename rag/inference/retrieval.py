"""Step 1 - Retrieval.

Embeds the query, scores chunks with the FAISS dense index and (optionally) BM25,
applies metadata pre-filters (section / page range) in Python, and fuses scores.

Modes (from config):
  * ``retrieval_mode="dense"``  - dense cosine only.
  * ``retrieval_mode="hybrid"`` - normalized dense + BM25 blended by ``hybrid_alpha``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from rag.common.schema import Chunk
from rag.indexing.vector_store import VectorStore


def _normalize(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    mn, mx = finite.min(), finite.max()
    if mx == mn:
        return np.zeros_like(arr)
    out = (arr - mn) / (mx - mn)
    out[~np.isfinite(out)] = 0.0
    return out


def _metadata_mask(
    chunks: List[Chunk],
    section_filter: Optional[str],
    page_range: Optional[Tuple[int, int]],
) -> np.ndarray:
    mask = np.ones(len(chunks), dtype=bool)
    if section_filter:
        mask &= np.array([section_filter.lower() in c.section_path.lower() for c in chunks])
    if page_range:
        lo, hi = page_range
        mask &= np.array([lo <= c.page <= hi for c in chunks])
    return mask


def retrieve_top_chunks(
    query: str,
    store: VectorStore,
    top_k: int = 10,
    retrieval_mode: str = "dense",
    hybrid_alpha: float = 0.5,
    section_filter: Optional[str] = None,
    page_range: Optional[Tuple[int, int]] = None,
) -> List[Tuple[Chunk, float]]:
    """Return the top-k ``(chunk, score)`` pairs for a query."""
    chunks = store.chunks
    if not chunks:
        return []

    mask = _metadata_mask(chunks, section_filter, page_range)
    active_idx = np.where(mask)[0]
    if active_idx.size == 0:
        print("[warn] Metadata filter matched 0 chunks - ignoring filter.")
        active_idx = np.arange(len(chunks))

    query_vec = store.embed_query(query)
    dense_all = store.dense_scores(query_vec)
    dense_active = dense_all[active_idx]

    bm25_all = store.bm25_scores(query) if retrieval_mode == "hybrid" else None

    if retrieval_mode == "hybrid" and bm25_all is not None:
        combined = hybrid_alpha * _normalize(dense_active) + (1 - hybrid_alpha) * _normalize(bm25_all[active_idx])
    else:
        combined = dense_active

    order = np.argsort(combined)[::-1][:top_k]
    results = [
        (chunks[active_idx[i]], float(combined[i]))
        for i in order
        if np.isfinite(combined[i]) and combined[i] > 0
    ]
    return results
