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

import json
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple

from rag.common.schema import Chunk
from rag.config import RagConfig
from rag.indexing.build_pipeline import chunks_from_json, make_backend
from rag.indexing.chroma_store import ChromaVectorStore, embed_corpus, get_or_create_store
from rag.indexing.embeddings import EmbeddingBackend
from rag.inference.context_builder import build_context
from rag.inference.hierarchical_answer import answer_with_hierarchical_context
from rag.inference.llm import answer_with_llm, get_llm
from rag.inference.prompt import make_prompt, parse_generator_json
from rag.inference.reranker import get_reranker, rerank
from rag.inference import verification as V

import rag_evaluation

from highlights import highlights_for_matches


# Documents that must ALWAYS be re-embedded on build (cache bypassed) so the
# embedding step can be demoed/recorded. Compared case-insensitively.
_ALWAYS_REEMBED = {"pr2610"}


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
        if overrides.get("verify") is not None:
            cfg.verify_enabled = bool(overrides["verify"])
        if overrides.get("verify_num_questions") is not None:
            cfg.verify_num_questions = int(overrides["verify_num_questions"])
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
            # Demo docs are ALWAYS re-embedded (never served from Chroma) so the
            # embedding step is visible/recordable. Existing vectors are cleared
            # first to avoid duplicates.
            force_reembed = doc_id.strip().lower() in _ALWAYS_REEMBED
            if store.has_document(doc_id):
                if force_reembed:
                    yield {"stage": "reset", "message":
                           f"Re-embedding {doc_id} (demo doc, cache bypassed)."}
                    store.delete_document(doc_id)
                else:
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

    # ── CoVe confidence verification ───────────────────────────────────────────
    @staticmethod
    def _acc_usage(total: Dict[str, int], usage: Dict[str, Any]) -> None:
        for k in ("input_tokens", "output_tokens", "total_tokens"):
            total[k] = total.get(k, 0) + int((usage or {}).get(k, 0) or 0)

    @staticmethod
    def _display_answer(raw: str) -> str:
        """Answer JSON shown in the UI: only ``answer`` + ``source_pages``.

        The generator also returns ``available`` / ``confidence`` (still used by the
        enrichment tool and as a minor CoVe input), but we hide them from the
        Explorer's answer view. Falls back to the raw string if it is not JSON.
        """
        if not raw or not raw.strip():
            return raw or ""
        parsed = parse_generator_json(raw)
        # Only re-serialize when the reply actually parsed as the JSON envelope;
        # otherwise show the original text unchanged.
        if not parsed.get("answer") and parsed.get("answer") != "":
            return raw
        display = {"answer": parsed.get("answer", ""), "source_pages": parsed.get("source_pages", [])}
        return json.dumps(display, ensure_ascii=False, indent=2)

    def _verify_stream(
        self,
        *,
        store: ChromaVectorStore,
        cfg: RagConfig,
        doc_id: str,
        question: str,
        answer_text: str,
        verbalized_conf: Optional[float],
        llm: Any,
        shared_context_items: List[Dict[str, Any]],
    ) -> Iterator[Dict[str, Any]]:
        """Run the Chain-of-Verification pass, yielding live progress events.

        Never touches the answer: it plans indirect corroboration checks, retrieves
        a related fact per check (fresh, via the same pipeline), lets the LLM judge
        each rule, and aggregates the verdicts into a confidence score. Emits:
          * ``planned``    - once, with the list of planned checks (verdicts pending)
          * ``checking``   - per check, when it starts (question + relevance)
          * ``check_done`` - per check, with the judged verdict + related fact
          * ``done``       - once, with the final confidence + aggregate breakdown
        """
        total_usage: Dict[str, int] = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

        # 1. Plan the indirect checks.
        raw_plan, usage = rag_evaluation.invoke_llm_with_usage(
            llm, V.build_plan_prompt(question, answer_text, None)
        )
        self._acc_usage(total_usage, usage)
        checks = V.parse_plan(raw_plan, cfg.verify_num_questions)
        yield {"stage": "verify", "status": "planned", "count": len(checks),
               "checks": [{"question": c["question"], "rule": c["rule"],
                           "relevance": c["relevance"]} for c in checks]}

        reranker = get_reranker(cfg.reranker_model) if cfg.use_reranker else None

        # 2. Execute each check (retrieve related fact -> LLM judges the rule).
        #    Accumulate the excerpts each check saw so a low-confidence suggestion
        #    can be grounded in exactly what the verification looked at.
        results: List[Dict[str, Any]] = []
        seen_context: List[Dict[str, Any]] = []
        seen_keys: set = set()
        for i, chk in enumerate(checks):
            yield {"stage": "verify", "status": "checking", "index": i,
                   "total": len(checks), "question": chk["question"],
                   "relevance": chk["relevance"]}
            if cfg.verify_fresh_retrieval:
                matches = self._retrieve(store, cfg, doc_id, chk["question"])
                if reranker is not None and matches:
                    matches = rerank(chk["question"], matches, cfg.reranker_model,
                                     top_k=cfg.rerank_top_k)
                elif matches:
                    matches = matches[: cfg.rerank_top_k]
                corpus = self._corpus_for_context(store, cfg, matches)
                ctx_items, _ = build_context(matches, corpus, context_mode=cfg.context_mode)
            else:
                ctx_items = shared_context_items
            for item in ctx_items:
                key = (item.get("source"), item.get("page"), (item.get("text") or "")[:80])
                if key not in seen_keys:
                    seen_keys.add(key)
                    seen_context.append(item)

            raw_judge, usage = rag_evaluation.invoke_llm_with_usage(
                llm, V.build_judge_prompt(chk, ctx_items, answer_text)
            )
            self._acc_usage(total_usage, usage)
            verdict = V.parse_judge(raw_judge)
            result = {**chk, **verdict}
            results.append(result)
            yield {"stage": "verify", "status": "check_done", "index": i,
                   "total": len(checks), "check": result}

        # 3. Aggregate into a single confidence score.
        agg = V.aggregate(results, verbalized_conf)

        # 4. When confidence is low, propose a corrected answer from the excerpts
        #    the verification saw (human accepts/rejects it in the UI).
        suggestion = None
        if agg["band"] == V.SUGGEST_BAND:
            sug_ctx = (seen_context if cfg.verify_fresh_retrieval
                       else shared_context_items)[:12]
            try:
                raw_sug, usage = rag_evaluation.invoke_llm_with_usage(
                    llm, V.build_suggestion_prompt(question, answer_text, results, sug_ctx)
                )
                self._acc_usage(total_usage, usage)
                suggestion = V.parse_suggestion(raw_sug)
            except Exception:  # noqa: BLE001
                suggestion = None

        v_cost = rag_evaluation.token_cost_usd(
            total_usage.get("input_tokens", 0), total_usage.get("output_tokens", 0)
        )
        yield {"stage": "verify", "status": "done",
               "confidence": round(agg["confidence"] * 100, 1),
               "band": agg["band"], "checks": results, "breakdown": agg["breakdown"],
               "verbalized_confidence": (round(verbalized_conf * 100, 1)
                                         if verbalized_conf is not None else None),
               "suggestion": suggestion,
               "usage": total_usage, "cost_usd": v_cost}

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
            if llm is not None and matches:
                yield {"stage": "llm", "status": "start", "model": cfg.llm_model}
                if cfg.context_mode == "hierarchical":
                    result = answer_with_hierarchical_context(
                        llm, question, matches, store, doc_id, cfg,
                        rag_evaluation.invoke_llm_with_usage,
                    )
                    answer = result.answer
                    usage = result.usage
                    context_items = result.context_items
                    gen_cost = rag_evaluation.token_cost_usd(
                        usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                    )
                    yield {"stage": "llm", "status": "done",
                           "answer": self._display_answer(answer),
                           "usage": usage, "cost_usd": gen_cost,
                           "escalation_depth": result.escalation_depth,
                           "llm_calls": result.llm_calls}
                else:
                    answer, usage = rag_evaluation.invoke_llm_with_usage(
                        llm, make_prompt(question, context_items)
                    )
                    gen_cost = rag_evaluation.token_cost_usd(
                        usage.get("input_tokens", 0), usage.get("output_tokens", 0)
                    )
                    yield {"stage": "llm", "status": "done",
                           "answer": self._display_answer(answer),
                           "usage": usage, "cost_usd": gen_cost}
            else:
                yield {"stage": "llm", "status": "skipped",
                       "message": "LLM unavailable - showing retrieved context only."}

            # 5b. CoVe confidence verification (independent of ground truth).
            #     Treats the answer as a hypothesis and corroborates it via related
            #     facts + domain-logic rules. Never modifies the answer.
            parsed = parse_generator_json(answer) if answer else {"answer": "", "confidence": None}
            answer_text = (parsed.get("answer") or answer or "").strip()
            verbalized_conf = parsed.get("confidence")
            if cfg.verify_enabled and llm is not None and answer_text:
                yield {"stage": "verify", "status": "start",
                       "num_questions": cfg.verify_num_questions}
                try:
                    yield from self._verify_stream(
                        store=store, cfg=cfg, doc_id=doc_id, question=question,
                        answer_text=answer_text, verbalized_conf=verbalized_conf,
                        llm=llm, shared_context_items=context_items,
                    )
                except Exception as exc:  # noqa: BLE001
                    yield {"stage": "verify", "status": "error", "message": str(exc)}
            else:
                yield {"stage": "verify", "status": "skipped",
                       "message": "Verification disabled or no answer to verify."}

            # 6. validation (only when ground truth value is present)
            gt_value = (ground_truth or {}).get("value")
            gt_tokens = (ground_truth or {}).get("tokens") or []
            if label and gt_value is not None:
                retrieval_hit = rag_evaluation.answer_is_correct(raw_text, gt_tokens)
                token_match = rag_evaluation.llm_answer_is_correct(answer, gt_tokens) if answer else False
                judge_match = None
                is_coord = rag_evaluation.label_uses_coordinate_judge(label)
                if answer and (is_coord or llm is not None):
                    try:
                        # Coordinate labels use a deterministic comparator; others
                        # use the LLM judge.
                        judge_match, _usage = rag_evaluation.judge_match_dispatch(
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
                       "judge_match": judge_match}
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

    # ── re-verify an (accepted) answer ─────────────────────────────────────────
    def run_verification(
        self,
        *,
        doc_id: str,
        question: str,
        answer: str,
        overrides: Optional[Dict[str, Any]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Re-run ONLY the CoVe verification against a provided answer.

        Used when a user accepts a suggested correction: the answer becomes the
        accepted value and the confidence is recomputed from scratch. Streams the
        same ``verify`` events as :meth:`run_query`, then ``complete``.
        """
        cfg = self.make_config(overrides)
        with self._lock:
            store = self.get_store(cfg)
            if not store.has_document(doc_id):
                yield {"stage": "error",
                       "message": f"{doc_id} is not embedded for '{cfg.embedding_model}'."}
                return
            llm = self.get_llm(cfg)
            answer_text = (answer or "").strip()
            if llm is None or not answer_text:
                yield {"stage": "verify", "status": "skipped",
                       "message": "No answer or LLM unavailable."}
                yield {"stage": "complete"}
                return

            # Only build a shared context when checks reuse it (fresh retrieval
            # builds its own per check).
            shared_context_items: List[Dict[str, Any]] = []
            if not cfg.verify_fresh_retrieval:
                matches = self._retrieve(store, cfg, doc_id, question)
                if cfg.use_reranker and matches:
                    reranker = get_reranker(cfg.reranker_model)
                    if reranker is not None:
                        matches = rerank(question, matches, cfg.reranker_model,
                                         top_k=cfg.rerank_top_k)
                    else:
                        matches = matches[: cfg.rerank_top_k]
                corpus = self._corpus_for_context(store, cfg, matches)
                shared_context_items, _ = build_context(
                    matches, corpus, context_mode=cfg.context_mode
                )

            yield {"stage": "verify", "status": "start",
                   "num_questions": cfg.verify_num_questions}
            try:
                yield from self._verify_stream(
                    store=store, cfg=cfg, doc_id=doc_id, question=question,
                    answer_text=answer_text, verbalized_conf=None,
                    llm=llm, shared_context_items=shared_context_items,
                )
            except Exception as exc:  # noqa: BLE001
                yield {"stage": "verify", "status": "error", "message": str(exc)}
            yield {"stage": "complete"}


# Module-level singleton reused across requests.
ENGINE = RagEngine()
