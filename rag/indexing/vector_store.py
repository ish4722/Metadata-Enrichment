"""Step 5 - FAISS vector store.

    >>>>>  SWAP THE VECTOR DATABASE HERE  <<<<<

Wraps a FAISS dense index plus an optional BM25 sparse index and the fitted
:class:`EmbeddingBackend` (needed to embed queries at inference time).

Persistence: the whole :class:`VectorStore` is picklable - the FAISS index is
(de)serialised to/from bytes in ``__getstate__`` / ``__setstate__`` so a single
``meta.pkl`` round-trips everything. :func:`save_store` / :func:`load_store`
write/read ``<index_dir>/index.faiss`` + ``<index_dir>/meta.pkl`` for a
human-inspectable on-disk layout.

FAISS has no native metadata filtering, so retrieval scores the full index and
filters by section/page in Python (see ``rag.inference.retrieval``).
"""

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import numpy as np

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from rag.common.schema import Chunk
from rag.indexing.embeddings import EmbeddingBackend

INDEX_FILENAME = "index.faiss"
META_FILENAME = "meta.pkl"


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


@dataclass
class VectorStore:
    chunks: List[Chunk]
    backend: EmbeddingBackend
    metric: str                 # "cosine" | "l2"
    dim: int
    faiss_index: Any            # faiss.Index (serialised on pickle)
    bm25: Any = None            # BM25Okapi or None

    # ── pickle support (FAISS index is not natively picklable) ───────────────
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        if HAS_FAISS and self.faiss_index is not None:
            state["faiss_index"] = faiss.serialize_index(self.faiss_index)
        return state

    def __setstate__(self, state: dict) -> None:
        idx = state.get("faiss_index")
        if HAS_FAISS and isinstance(idx, (bytes, bytearray, np.ndarray)):
            state["faiss_index"] = faiss.deserialize_index(idx)
        self.__dict__.update(state)

    # ── query-time helpers ───────────────────────────────────────────────────
    def embed_query(self, query: str) -> np.ndarray:
        vec = self.backend.transform([query]).astype(np.float32)
        if self.metric == "cosine":
            vec = _l2_normalize(vec)
        return vec

    def dense_scores(self, query_vec: np.ndarray) -> np.ndarray:
        """Return a dense similarity score per chunk, aligned to ``self.chunks``.

        Higher is better for both metrics. FAISS is searched over the full index
        so scores can later be filtered/fused for any subset of chunks.
        """
        n = len(self.chunks)
        scores = np.full(n, -np.inf, dtype=np.float32)
        if n == 0:
            return scores
        distances, ids = self.faiss_index.search(query_vec, n)
        for dist, idx in zip(distances[0], ids[0]):
            if idx < 0:
                continue
            scores[idx] = dist if self.metric == "cosine" else 1.0 / (1.0 + float(dist))
        return scores

    def bm25_scores(self, query: str) -> Optional[np.ndarray]:
        if self.bm25 is None or not HAS_BM25:
            return None
        return np.array(self.bm25.get_scores(query.lower().split()), dtype=np.float32)


def build_vector_store(
    chunks: List[Chunk],
    backend: EmbeddingBackend,
    metric: str = "cosine",
) -> VectorStore:
    """Embed all chunks, build the FAISS dense index and the BM25 sparse index."""
    if not HAS_FAISS:
        raise RuntimeError("faiss is not installed. Run: pip install faiss-cpu")

    texts = [c.text for c in chunks]
    if not texts:
        raise RuntimeError(
            "No text chunks were produced from the document - nothing to embed. "
            "This usually means the extraction JSON layout was not recognised by "
            "rag/indexing/loader.py (expected page nodes under top-level 'children' "
            "or 'result.json.children'). Check the input file format."
        )
    print(f"Embedding {len(texts)} chunks with [{backend.backend_name}] ...")
    mat = backend.fit_transform(texts).astype(np.float32)
    dim = int(mat.shape[1]) if mat.ndim == 2 else 0

    print(f"Building FAISS index (metric={metric}, dim={dim}) ...")
    if metric == "cosine":
        mat = _l2_normalize(mat)
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)
    index.add(mat)

    bm25 = None
    if HAS_BM25:
        print("Building BM25 index ...")
        bm25 = BM25Okapi([t.lower().split() for t in texts])
    else:
        print("[info] rank_bm25 not installed - hybrid retrieval will fall back to dense.")

    return VectorStore(chunks=chunks, backend=backend, metric=metric, dim=dim,
                       faiss_index=index, bm25=bm25)


def save_store(store: VectorStore, index_dir: str) -> None:
    """Persist the store as ``index.faiss`` + ``meta.pkl`` under ``index_dir``."""
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(store.faiss_index, os.path.join(index_dir, INDEX_FILENAME))

    meta = {
        "chunks": store.chunks,
        "backend": store.backend,
        "metric": store.metric,
        "dim": store.dim,
        "bm25": store.bm25,
    }
    with open(os.path.join(index_dir, META_FILENAME), "wb") as f:
        pickle.dump(meta, f)
    print(f"Vector store saved -> {index_dir}/ (index.faiss + meta.pkl)")


def load_store(index_dir: str) -> VectorStore:
    """Load a store previously written by :func:`save_store`."""
    if not HAS_FAISS:
        raise RuntimeError("faiss is not installed. Run: pip install faiss-cpu")

    index = faiss.read_index(os.path.join(index_dir, INDEX_FILENAME))
    with open(os.path.join(index_dir, META_FILENAME), "rb") as f:
        meta = pickle.load(f)

    return VectorStore(
        chunks=meta["chunks"],
        backend=meta["backend"],
        metric=meta.get("metric", "cosine"),
        dim=meta.get("dim", index.d),
        faiss_index=index,
        bm25=meta.get("bm25"),
    )
