"""Process-lived RAG engine that wraps the existing ``rag`` pipeline.

Loads each heavy component (embedding backend, Chroma store, cross-encoder
reranker, LLM) ONCE and caches it, then exposes streaming generators for the
two long-running operations the website drives:

* :meth:`RagEngine.build`     - embed-once a document into Chroma.
* :meth:`RagEngine.run_query` - staged retrieve -> rerank -> answer -> validate
                                -> highlight, yielding one event dict per stage.

The chunking strategy and vector backend are intentionally taken straight from
:class:`rag.config.RagConfig` (Chroma + section chunking); only the embedding
model, retrieval mode, hybrid alpha and reranker toggle are overridable from the
UI, matching the project's "change chunking in the config only" rule.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple

from rag.common.schema import Chunk
from rag.config import RagConfig
from rag.indexing.build_pipeline import chunks_from_json, make_backend
from rag.indexing.chroma_store import ChromaVectorStore, embed_corpus, get_or_create_store
from rag.indexing.embeddings import EmbeddingBackend
from rag.inference.context_builder import build_context
from rag.inference.llm import answer_with_llm, get_llm
from rag.inference.prompt import make_prompt
from rag.inference.reranker import get_reranker, rerank

import rag_evaluation

from highlights import highlights_for_matches


def _chunk_payload(chunk: Chunk, score: float) -> Dict[str, Any]:
    """Serialize a retrieved chunk for the frontend."""
    return {
        "id": chunk.id,
        "page": int(chunk.page),
        "pages": chunk.metadata.get("pages", [chunk.page]),
        "score": round(float(score), 4),
        "section_path": chunk.section_path or "",
        "section_name": chunk.metadata.get("section_name", "") or "",
        "block_type": chunk.block_type or "",
        "leaf_section_id": chunk.leaf_section_id or "",
        "text": chunk.text,
    }


class RagEngine:
    """Caches backends / stores / reranker / LLM and runs streamed operations."""

    def __init__(self) -> None:
        self._backends: Dict[str, EmbeddingBackend] = {}
        self._stores: Dict[str, ChromaVectorStore] = {}
        self._llm: Any = None
        self._llm_loaded = False
        # torch / chroma state is not safe to hit concurrently from many
        # requests; this is a single-user local tool, so serialize heavy work.
        self._lock = threading.Lock()

    # ── config + cached components ────────────────────────────────────────────
    def make_config(self, overrides: Optional[Dict[str, Any]] = None) -> RagConfig:
        cfg = RagConfig.from_env()
        overrides = overrides or {}
        if overrides.get("embedding_model"):
            cfg.embedding_model = overrides["embedding_model"]
        if overrides.get("retrieval_mode"):
            cfg.retrieval_mode = overrides["retrieval_mode"]
        if overrides.get("hybrid_alpha") is not None:
            cfg.hybrid_alpha = float(overrides["hybrid_alpha"])
        if overrides.get("use_reranker") is not None:
            cfg.use_reranker = bool(overrides["use_reranker"])
        if overrides.get("top_k") is not None:
            cfg.top_k = int(overrides["top_k"])
        return cfg

    def get_backend(self, cfg: RagConfig) -> EmbeddingBackend:
        key = cfg.embedding_slug
        if key not in self._backends:
            self._backends[key] = make_backend(cfg)
        return self._backends[key]

    def get_store(self, cfg: RagConfig) -> ChromaVectorStore:
        key = f"{cfg.chroma_dir}::{cfg.chroma_collection}"
        if key not in self._stores:
            backend = self.get_backend(cfg)
            self._stores[key] = get_or_create_store(
                cfg.chroma_dir, cfg.chroma_collection, backend, metric=cfg.faiss_metric
            )
        return self._stores[key]

    def get_llm(self, cfg: RagConfig) -> Any:
        if not self._llm_loaded:
            self._llm = get_llm(cfg)
            self._llm_loaded = True
        return self._llm

    def is_built(self, doc_id: str, cfg: RagConfig) -> bool:
        try:
            return self.get_store(cfg).has_document(doc_id)
        except Exception:
            return False

    # ── build (embed-once) ─────────────────────────────────────────────────────
    def build(
        self,
        doc_id: str,
        datalab_path: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Embed a document's chunks into Chroma, yielding progress events."""
        cfg = self.make_config(overrides)
        with self._lock:
            yield {"stage": "start", "doc_id": doc_id, "model": cfg.embedding_model,
                   "collection": cfg.chroma_collection}

            store = self.get_store(cfg)
            if store.has_document(doc_id):
                yield {"stage": "done", "already_built": True, "added": 0,
                       "message": f"{doc_id} already embedded in '{cfg.chroma_collection}'."}
                return

            yield {"stage": "chunking", "status": "start"}
            chunks = chunks_from_json(datalab_path, cfg)
            texts = [c.text for c in chunks]
            yield {"stage": "chunking", "status": "done", "num_chunks": len(texts)}

            if not texts:
                yield {"stage": "error", "message": "No chunks produced from the datalab JSON."}
                return

            backend = self.get_backend(cfg)
            yield {"stage": "embedding", "status": "start", "num_chunks": len(texts),
                   "backend": backend.backend_name}
            embeddings = embed_corpus(backend, texts)
            added = store.add_document(doc_id, chunks, embeddings)

            # Embedding cost only applies to the paid OpenAI backend; local
            # backends (tfidf / jina / minilm) are free.
            embed_tokens = 0
            embed_cost = 0.0
            if backend.backend_name == "openai":
                embed_tokens = rag_evaluation.count_tokens(texts)
                embed_cost = rag_evaluation.embedding_cost_usd(embed_tokens)
            yield {"stage": "embedding", "status": "done", "added": added,
                   "embedding_tokens": embed_tokens, "embedding_cost_usd": embed_cost}

            yield {"stage": "done", "already_built": False, "added": added,
                   "collection": cfg.chroma_collection,
                   "embedding_tokens": embed_tokens, "embedding_cost_usd": embed_cost,
                   "message": f"Embedded {added} chunks for {doc_id}."}

    # ── query (staged) ──────────────────────────────────────────────────────────
    def _retrieve(
        self, store: ChromaVectorStore, cfg: RagConfig, doc_id: str, retrieval_query: str
    ) -> List[Tuple[Chunk, float]]:
        if cfg.retrieval_mode == "hybrid":
            return store.query_hybrid(
                retrieval_query, top_k=cfg.top_k, hybrid_alpha=cfg.hybrid_alpha, doc_id=doc_id
            )
        return store.query(retrieval_query, top_k=cfg.top_k, doc_id=doc_id)

    def _corpus_for_context(
        self, store: ChromaVectorStore, cfg: RagConfig, matches: List[Tuple[Chunk, float]]
    ) -> List[Chunk]:
        if cfg.context_mode == "full_page":
            keys = {(c.source, c.page) for c, _ in matches}
            return store.fetch_chunks_for_pages(keys)
        return [c for c, _ in matches]

    def run_query(
        self,
        *,
        doc_id: str,
        datalab_path: str,
        question: str,
        retrieval_query: str,
        label: Optional[str],
        ground_truth: Optional[Dict[str, Any]],
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        cfg = self.make_config(overrides)

        with self._lock:
            yield {"stage": "config", "model": cfg.embedding_model,
                   "retrieval_mode": cfg.retrieval_mode, "hybrid_alpha": cfg.hybrid_alpha,
                   "use_reranker": cfg.use_reranker, "top_k": cfg.top_k,
                   "collection": cfg.chroma_collection}

            store = self.get_store(cfg)
            if not store.has_document(doc_id):
                yield {"stage": "error",
                       "message": f"{doc_id} is not embedded for '{cfg.embedding_model}'. Build it first."}
                return

            # 1. embed the query (shown in the flow animation)
            yield {"stage": "embed", "status": "start", "query": retrieval_query}
            qvec = store.embed_query(retrieval_query)
            dim = int(qvec.shape[-1]) if getattr(qvec, "shape", None) else 0
            embed_cost = 0.0
            if self.get_backend(cfg).backend_name == "openai":
                embed_cost = rag_evaluation.embedding_cost_usd(
                    rag_evaluation.count_tokens([retrieval_query])
                )
            yield {"stage": "embed", "status": "done", "dim": dim, "embedding_cost_usd": embed_cost}

            # 2. retrieve
            yield {"stage": "retrieve", "status": "start",
                   "mode": cfg.retrieval_mode, "hybrid_alpha": cfg.hybrid_alpha}
            matches = self._retrieve(store, cfg, doc_id, retrieval_query)
            retrieved_payload = [_chunk_payload(c, s) for c, s in matches]
            yield {"stage": "retrieve", "status": "done", "count": len(matches),
                   "chunks": retrieved_payload}

            # 3. rerank (optional)
            if cfg.use_reranker and matches:
                yield {"stage": "rerank", "status": "start", "model": cfg.reranker_model}
                reranker = get_reranker(cfg.reranker_model)
                if reranker is None:
                    yield {"stage": "rerank", "status": "unavailable",
                           "message": "Cross-encoder unavailable; keeping retrieval order."}
                    matches = matches[: cfg.rerank_top_k]
                else:
                    matches = rerank(retrieval_query, matches, cfg.reranker_model,
                                     top_k=cfg.rerank_top_k)
                reranked_payload = [_chunk_payload(c, s) for c, s in matches]
                yield {"stage": "rerank", "status": "done", "count": len(matches),
                       "chunks": reranked_payload}
            else:
                yield {"stage": "rerank", "status": "skipped"}

            # 4. context assembly
            corpus = self._corpus_for_context(store, cfg, matches)
            context_items, raw_text = build_context(matches, corpus, context_mode=cfg.context_mode)
            yield {"stage": "context", "status": "done",
                   "context_mode": cfg.context_mode, "num_items": len(context_items)}

            # 5. LLM answer
            answer = ""
            llm = self.get_llm(cfg)
            if llm is not None and context_items:
                yield {"stage": "llm", "status": "start", "model": cfg.llm_model}
                answer, usage = rag_evaluation.invoke_llm_with_usage(
                    llm, make_prompt(question, context_items)
                )
                gen_cost = rag_evaluation.token_cost_usd(
                    usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                )
                yield {"stage": "llm", "status": "done", "answer": answer,
                       "usage": usage, "cost_usd": gen_cost}
            else:
                yield {"stage": "llm", "status": "skipped",
                       "message": "LLM unavailable - showing retrieved context only."}

            # 6. validation (only when ground truth value is present)
            gt_value = (ground_truth or {}).get("value")
            gt_tokens = (ground_truth or {}).get("tokens") or []
            if label and gt_value is not None:
                retrieval_hit = rag_evaluation.answer_is_correct(raw_text, gt_tokens)
                token_match = rag_evaluation.llm_answer_is_correct(answer, gt_tokens) if answer else False
                judge_match = None
                judge_confidence = None
                is_coord = rag_evaluation.label_uses_coordinate_judge(label)
                if answer and (is_coord or llm is not None):
                    try:
                        # Coordinate labels use a deterministic comparator; others
                        # use the LLM judge.
                        judge_match, judge_confidence, _usage = rag_evaluation.judge_verdict(
                            llm, label, question, str(gt_value), answer
                        )
                    except Exception as exc:  # noqa: BLE001
                        judge_match = None
                        yield {"stage": "validate", "status": "warn",
                               "message": f"Judge failed: {exc}"}
                yield {"stage": "validate", "status": "done",
                       "ground_truth": gt_value,
                       "retrieval_hit": retrieval_hit,
                       "token_match": token_match,
                       "judge_match": judge_match,
                       "judge_confidence": judge_confidence}
            else:
                yield {"stage": "validate", "status": "skipped",
                       "message": "No ground-truth value for this query - skipping validation.",
                       "ground_truth": gt_value}

            # 7. highlights for the final chunk set
            try:
                pages = highlights_for_matches(datalab_path, matches)
                yield {"stage": "highlights", "status": "done", "pages": pages}
            except Exception as exc:  # noqa: BLE001
                yield {"stage": "highlights", "status": "error", "message": str(exc)}

            yield {"stage": "complete"}


# Module-level singleton reused across requests.
ENGINE = RagEngine()
