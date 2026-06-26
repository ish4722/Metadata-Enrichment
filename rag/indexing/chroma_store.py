"""Persistent vector store backed by ChromaDB.

    >>>>>  EMBED-ONCE VECTOR DATABASE  <<<<<

Stores OUR precomputed embeddings (we never let Chroma embed - we always pass
``embeddings=...``), so a document is embedded exactly once and every later run
serves its vectors straight from disk. Each ``(embedding_model, chunk_strategy)``
pair gets its own collection (see :attr:`RagConfig.chroma_collection`).

Idempotency: chunk ids are namespaced per document (``<doc_id>::<chunk_id>``).
:meth:`ChromaVectorStore.add_document` skips any ids already present, so
re-running a build for a doc that is already embedded is a no-op.

Retrieval scores the collection with the query embedding and returns
``(Chunk, score)`` pairs; metadata filtering (by ``doc_id`` / section / page) is
pushed down to Chroma's ``where`` clause.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMA = True
except ImportError:
    HAS_CHROMA = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

from rag.common.schema import Chunk
from rag.indexing.embeddings import EmbeddingBackend


def _minmax_normalize(arr: np.ndarray) -> np.ndarray:
    """Scale finite values to [0, 1]; non-finite become 0 (for score fusion)."""
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros_like(arr)
    mn, mx = float(finite.min()), float(finite.max())
    if mx == mn:
        return np.zeros_like(arr)
    out = (arr - mn) / (mx - mn)
    out[~np.isfinite(out)] = 0.0
    return out

# Metadata keys we persist as scalars on each Chroma record (Chroma metadata
# values must be str/int/float/bool). Lists (e.g. sibling sections) are stored
# JSON-encoded under dedicated keys and decoded back on read.
_SCALAR_META = ("page", "section_path", "section_name", "leaf_section_id", "parent_section_id")

# Chroma's local (Rust) backend rejects a single add() larger than 5461 records.
# Stay safely under that so large documents insert in multiple batches.
_MAX_ADD_BATCH = 5000


def _doc_namespaced_id(doc_id: str, chunk_id: str) -> str:
    return f"{doc_id}::{chunk_id}"


def _chunk_to_metadata(chunk: Chunk, doc_id: str) -> Dict[str, Any]:
    md: Dict[str, Any] = {
        "doc_id": doc_id,
        "source": chunk.source,
        "page": int(chunk.page),
        "section_path": chunk.section_path or "",
        "section_name": str(chunk.metadata.get("section_name", "")),
        "leaf_section_id": chunk.leaf_section_id or "",
        "parent_section_id": chunk.parent_section_id or "",
        "block_type": chunk.block_type or "",
        "order": int(chunk.order),
        "sibling_sections_json": json.dumps(chunk.metadata.get("sibling_sections", [])),
        "pages_json": json.dumps(chunk.metadata.get("pages", [chunk.page])),
    }
    return md


def _metadata_to_chunk(chunk_id: str, text: str, md: Dict[str, Any]) -> Chunk:
    try:
        siblings = json.loads(md.get("sibling_sections_json", "[]"))
    except (json.JSONDecodeError, TypeError):
        siblings = []
    try:
        pages = json.loads(md.get("pages_json", "[]"))
    except (json.JSONDecodeError, TypeError):
        pages = []
    return Chunk(
        id=chunk_id,
        source=md.get("source", ""),
        page=int(md.get("page", -1)),
        section_path=md.get("section_path", ""),
        block_type=md.get("block_type", ""),
        text=text,
        leaf_section_id=md.get("leaf_section_id", ""),
        parent_section_id=md.get("parent_section_id", ""),
        order=int(md.get("order", 0)),
        metadata={
            "section_name": md.get("section_name", ""),
            "sibling_sections": siblings,
            "pages": pages,
            "doc_id": md.get("doc_id", ""),
        },
    )


def embed_corpus(backend: EmbeddingBackend, texts: List[str]) -> np.ndarray:
    """Embed a document's chunks for storage (single corpus-level fit_transform).

    This is intentionally ONE call: backends like TF-IDF fit global state over the
    whole corpus, and API/transformer backends already batch internally. Parallelism
    across documents (for stateless API embedders) is handled by the caller.
    """
    if not texts:
        return np.zeros((0, 0), dtype=np.float32)
    return backend.fit_transform(texts).astype(np.float32)


class ChromaVectorStore:
    """Thin wrapper around a persistent Chroma collection of precomputed vectors."""

    def __init__(self, collection: Any, backend: EmbeddingBackend, metric: str = "cosine"):
        self.collection = collection
        self.backend = backend
        self.metric = metric
        # Lazily built corpus + BM25 index for hybrid retrieval, cached per scope
        # ("__all__" for the whole collection, or a doc_id when scoped to one doc).
        self._corpus_cache: Dict[str, Dict[str, Any]] = {}

    # ── build-time ───────────────────────────────────────────────────────────
    def add_document(
        self,
        doc_id: str,
        chunks: List[Chunk],
        embeddings: np.ndarray,
    ) -> int:
        """Upsert a document's chunks; skip ids already embedded. Returns #added."""
        if not chunks:
            return 0
        ids = [_doc_namespaced_id(doc_id, c.id) for c in chunks]
        existing = set()
        try:
            got = self.collection.get(ids=ids, include=[])
            existing = set(got.get("ids", []) or [])
        except Exception:
            existing = set()

        new_idx = [i for i, cid in enumerate(ids) if cid not in existing]
        if not new_idx:
            return 0

        # Chroma's local backend caps a single add() at 5461 records, so insert
        # in batches under that limit. Docs with thousands of chunks would
        # otherwise crash with "Batch size ... greater than max batch size".
        for start in range(0, len(new_idx), _MAX_ADD_BATCH):
            batch = new_idx[start:start + _MAX_ADD_BATCH]
            self.collection.add(
                ids=[ids[i] for i in batch],
                embeddings=[embeddings[i].astype(float).tolist() for i in batch],
                documents=[chunks[i].text for i in batch],
                metadatas=[_chunk_to_metadata(chunks[i], doc_id) for i in batch],
            )
        return len(new_idx)

    def has_document(self, doc_id: str) -> bool:
        try:
            got = self.collection.get(where={"doc_id": doc_id}, limit=1, include=[])
            return bool(got.get("ids"))
        except Exception:
            return False

    def fetch_chunks_for_pages(self, keys) -> List[Chunk]:
        """Return every stored chunk on the given ``(source, page)`` pages.

        Used by full-page context assembly: the matched chunks tell us which
        pages to expand, and this pulls back all chunks on those pages straight
        from the collection (no re-embedding).
        """
        out: List[Chunk] = []
        seen: set = set()
        for source, page in keys:
            try:
                got = self.collection.get(
                    where={"$and": [{"source": source}, {"page": int(page)}]},
                    include=["documents", "metadatas"],
                )
            except Exception:
                continue
            ids = got.get("ids") or []
            docs = got.get("documents") or []
            metas = got.get("metadatas") or []
            for cid, text, md in zip(ids, docs, metas):
                if cid in seen:
                    continue
                seen.add(cid)
                out.append(_metadata_to_chunk(cid, text, md or {}))
        return out

    # ── query-time ────────────────────────────────────────────────────────────
    def embed_query(self, query: str) -> np.ndarray:
        vec = self.backend.transform([query]).astype(np.float32)
        return vec

    def query(
        self,
        query: str,
        top_k: int = 25,
        doc_id: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Return the top-k ``(Chunk, score)`` for a query (higher = better)."""
        query_vec = self.embed_query(query)[0].astype(float).tolist()
        clause: Optional[Dict[str, Any]] = None
        if doc_id and where:
            clause = {"$and": [{"doc_id": doc_id}, where]}
        elif doc_id:
            clause = {"doc_id": doc_id}
        elif where:
            clause = where

        res = self.collection.query(
            query_embeddings=[query_vec],
            n_results=top_k,
            where=clause,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        out: List[Tuple[Chunk, float]] = []
        for cid, text, md, dist in zip(ids, docs, metas, dists):
            # Chroma returns a distance; convert to a higher-is-better score.
            score = 1.0 - float(dist) if self.metric == "cosine" else 1.0 / (1.0 + float(dist))
            out.append((_metadata_to_chunk(cid, text, md or {}), score))
        return out

    # ── hybrid retrieval (dense + BM25) ───────────────────────────────────────
    def _ensure_corpus(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Load the (optionally doc-scoped) corpus and build a BM25 index over it.

        The chunk text is already stored in Chroma (``documents``), so BM25 needs
        no extra persistence - it is rebuilt from those documents and cached on
        the store instance, keyed by scope. When ``doc_id`` is given only that
        document's chunks are loaded, which keeps BM25 correct (no cross-doc
        contamination) and avoids loading/scoring the whole multi-doc collection.
        """
        key = doc_id or "__all__"
        cached = self._corpus_cache.get(key)
        if cached is not None:
            return cached

        where = {"doc_id": doc_id} if doc_id else None
        got = self.collection.get(where=where, include=["documents", "metadatas"])
        ids = got.get("ids") or []
        docs = got.get("documents") or []
        metas = got.get("metadatas") or []
        chunks = [_metadata_to_chunk(cid, text, md or {}) for cid, text, md in zip(ids, docs, metas)]

        bm25 = None
        if HAS_BM25 and docs:
            bm25 = BM25Okapi([(t or "").lower().split() for t in docs])
        elif not HAS_BM25:
            print("[info] rank_bm25 not installed - hybrid retrieval falls back to dense.")

        self._corpus_cache[key] = {"ids": ids, "chunks": chunks, "bm25": bm25}
        return self._corpus_cache[key]

    def _dense_scores_full(
        self, query: str, ids: List[str], doc_id: Optional[str] = None
    ) -> np.ndarray:
        """Dense score for every chunk in ``ids`` (higher = better), aligned to ids.

        When ``doc_id`` is given the dense query is pushed down to that document,
        so ``n_results`` stays bounded by the doc's chunk count instead of the
        whole collection (which would exceed SQLite's variable limit).
        """
        n = len(ids)
        if n == 0:
            return np.zeros(0, dtype=np.float32)
        query_vec = self.embed_query(query)[0].astype(float).tolist()
        where = {"doc_id": doc_id} if doc_id else None
        res = self.collection.query(
            query_embeddings=[query_vec], n_results=n, where=where, include=["distances"]
        )
        res_ids = (res.get("ids") or [[]])[0]
        res_dists = (res.get("distances") or [[]])[0]
        score_by_id: Dict[str, float] = {}
        for cid, dist in zip(res_ids, res_dists):
            score_by_id[cid] = (
                1.0 - float(dist) if self.metric == "cosine" else 1.0 / (1.0 + float(dist))
            )
        return np.array([score_by_id.get(cid, 0.0) for cid in ids], dtype=np.float32)

    def query_hybrid(
        self,
        query: str,
        top_k: int = 25,
        hybrid_alpha: float = 0.5,
        doc_id: Optional[str] = None,
        section_filter: Optional[str] = None,
        page_range: Optional[Tuple[int, int]] = None,
    ) -> List[Tuple[Chunk, float]]:
        """Blend normalized dense + BM25 scores (``hybrid_alpha`` weights dense).

        Filtering by doc / section / page is applied in Python over the cached
        corpus, mirroring the FAISS hybrid path. Falls back to dense-only when
        BM25 is unavailable.
        """
        corpus = self._ensure_corpus(doc_id)
        ids: List[str] = corpus["ids"]
        chunks: List[Chunk] = corpus["chunks"]
        bm25 = corpus["bm25"]
        if not ids:
            return []

        dense = self._dense_scores_full(query, ids, doc_id)

        mask = np.ones(len(ids), dtype=bool)
        # corpus is already scoped to doc_id, so no doc mask is needed here.
        if section_filter:
            mask &= np.array([section_filter.lower() in (c.section_path or "").lower() for c in chunks])
        if page_range:
            lo, hi = page_range
            mask &= np.array([lo <= c.page <= hi for c in chunks])
        active = np.where(mask)[0]
        if active.size == 0:
            print("[warn] Metadata filter matched 0 chunks - ignoring filter.")
            active = np.arange(len(ids))

        dense_active = _minmax_normalize(dense[active])
        if bm25 is not None:
            bm_all = np.array(bm25.get_scores(query.lower().split()), dtype=np.float32)
            combined = hybrid_alpha * dense_active + (1 - hybrid_alpha) * _minmax_normalize(bm_all[active])
        else:
            combined = dense_active

        order = np.argsort(combined)[::-1][:top_k]
        return [(chunks[active[i]], float(combined[i])) for i in order]


def get_chroma_client(persist_dir: str) -> Any:
    """Construct a persistent Chroma client rooted at ``persist_dir``."""
    if not HAS_CHROMA:
        raise RuntimeError("chromadb is not installed. Run: pip install chromadb")
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False, allow_reset=False),
    )


def get_or_create_store(
    persist_dir: str,
    collection_name: str,
    backend: EmbeddingBackend,
    metric: str = "cosine",
) -> ChromaVectorStore:
    """Open (or create) the collection for this (model, strategy) pair."""
    client = get_chroma_client(persist_dir)
    space = "cosine" if metric == "cosine" else "l2"
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": space},
    )
    return ChromaVectorStore(collection=collection, backend=backend, metric=metric)
