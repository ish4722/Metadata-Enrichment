"""Single-process, parallel batch runner for the RAG evaluation.

Replaces the subprocess-per-doc fan-out in ``evaluate_all.py``: it loads the
embedding backend, the cross-encoder reranker and the LLM **once**, then runs the
whole corpus in one long-lived process with bounded concurrency.

Stages and their executor (chosen by workload type):
  * build      - load + section-chunk docs on a PROCESS pool (CPU bound), then
                 embed-once into Chroma with concurrent request batches.
  * retrieval  - per (doc, query) on a THREAD pool (network/IO bound); the
                 cross-encoder is shared and guarded by a lock.
  * generation - one LLM call per (doc, query) on a THREAD pool.
  * judge      - one judge call per (doc, query) on a THREAD pool.

Determinism: every task carries its (doc_index, query_index) so results are
reassembled in order; metrics are accumulated only after the parallel sections
join (no shared mutable counters during concurrency).

NOTE: this module wires up the full run but is intended to be invoked explicitly
by the user; the actual end-to-end run is intentionally NOT triggered here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches  # noqa: E402

apply_ssl_patches()

from rag.config import RagConfig  # noqa: E402
from rag.indexing.build_pipeline import chunks_from_json, derive_doc_id, make_backend  # noqa: E402
from rag.indexing.chroma_store import (  # noqa: E402
    ChromaVectorStore,
    embed_corpus,
    get_or_create_store,
)
from rag.inference.context_builder import build_context  # noqa: E402
from rag.inference.hierarchical_answer import answer_with_hierarchical_context  # noqa: E402
from rag.inference.llm import get_llm  # noqa: E402
from rag.inference.prompt import make_prompt  # noqa: E402
from rag.inference.reranker import get_reranker, rerank  # noqa: E402

import evaluate_all  # noqa: E402  (reuse doc discovery)
import rag_evaluation  # noqa: E402


# ---------------------------------------------------------------------------
# Process-pool worker: load + chunk a single document (CPU bound)
# ---------------------------------------------------------------------------


def _load_and_chunk(args: Tuple[str, RagConfig]) -> Tuple[str, List[Any]]:
    """Top-level (picklable) worker: returns (doc_id, chunks) for one doc."""
    input_path, config = args
    chunks = chunks_from_json(input_path, config)
    return derive_doc_id(input_path), chunks


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class QueryResult:
    doc_id: str
    label: str
    question: str
    retrieval_query: str
    ground_truth: Any = None
    gt_tokens: List[str] = field(default_factory=list)
    # Retrieved (chunk, score) matches, kept transient for hierarchical escalation
    # at generation time; not serialized into the summary.
    matches: List[Tuple[Any, float]] = field(default_factory=list)
    context_items: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_text: str = ""
    retrieval_hit: bool = False
    answer: str = ""
    gen_usage: Dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    judge_usage: Dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    token_match: Optional[bool] = None
    judge_match: Optional[bool] = None
    llm_hit: Optional[bool] = None
    # Hierarchical escalation bookkeeping (0 depth / 1 call for other modes).
    escalation_depth: int = 0
    llm_calls: int = 0
    context_tokens_by_level: List[int] = field(default_factory=list)
    retrieval_time_s: float = 0.0
    gen_time_s: float = 0.0
    judge_time_s: float = 0.0


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class BatchRunner:
    def __init__(self, config: RagConfig, use_judge: bool = True, inject_name: bool = True):
        self.config = config
        self.use_judge = use_judge
        # When True, name placeholders ("{well}") in queries are filled with the
        # document's ground-truth well name; when False they collapse to "the well"
        # (the no-name phrasing) so the same queries.txt drives both experiment arms.
        self.inject_name = inject_name
        self._rerank_lock = threading.Lock()  # cross-encoder is not thread-safe
        # Embedding-cost accounting for this run (only the OpenAI backend is paid;
        # local backends embed for free).
        self.embed_tokens = 0
        self.embed_cost_usd = 0.0
        self._embed_stat_lock = threading.Lock()

        print("Loading models once (embedding backend + reranker + LLM) ...")
        self.backend = make_backend(config)
        self.store: ChromaVectorStore = get_or_create_store(
            config.chroma_dir, config.chroma_collection, self.backend, metric=config.faiss_metric
        )
        self.reranker = get_reranker(config.reranker_model) if config.use_reranker else None
        self.llm = get_llm(config)
        if self.llm is None:
            print("[warn] LLM unavailable - will measure retrieval only.")

    def llm_usage_snapshot(self) -> Dict[str, Any]:
        """OpenAI usage/metadata captured by the LLM callback for this run."""
        tracker = getattr(self.llm, "usage_tracker", None)
        return tracker.snapshot() if tracker is not None else {}

    # ── build phase ──────────────────────────────────────────────────────────
    def build_all(self, docs: List[Dict[str, str]]) -> None:
        """Ensure every doc is embedded into Chroma (embed-once).

        Load+chunk runs on a process pool (CPU bound). Embedding is one
        ``fit_transform`` per doc; for a stateless API backend (OpenAI) those
        per-doc embeds run concurrently across docs, otherwise sequentially
        (TF-IDF / local torch models share mutable state and are not safe to run
        in parallel on one backend instance).
        """
        if self.backend.backend_name == "tfidf":
            print("[warn] TF-IDF builds per-document vocabularies, which are NOT "
                  "consistent across docs in a shared collection. Use an OpenAI or "
                  "Jina embedder for the embed-once Chroma design.")

        pending = [d for d in docs if not self.store.has_document(d["id"])]
        if not pending:
            print(f"All {len(docs)} docs already embedded in '{self.config.chroma_collection}'.")
            return

        print(f"Building (load+chunk on {self.config.load_max_workers} procs) for {len(pending)} doc(s) ...")
        args = [(d["input"], self.config) for d in pending]
        chunked: List[Tuple[str, List[Any]]] = []
        if self.config.load_max_workers > 1 and len(pending) > 1:
            with ProcessPoolExecutor(max_workers=self.config.load_max_workers) as pool:
                for res in pool.map(_load_and_chunk, args):
                    chunked.append(res)
        else:
            chunked = [_load_and_chunk(a) for a in args]

        add_lock = threading.Lock()

        def _embed_and_add(item: Tuple[str, List[Any]]) -> None:
            doc_id, chunks = item
            texts = [c.text for c in chunks]
            if not texts:
                print(f"  [skip] {doc_id}: no chunks produced")
                return
            t0 = time.perf_counter()
            embeddings = embed_corpus(self.backend, texts)
            with add_lock:
                added = self.store.add_document(doc_id, chunks, embeddings)
            if self.backend.backend_name == "openai":
                tok = rag_evaluation.count_tokens(texts)
                with self._embed_stat_lock:
                    self.embed_tokens += tok
                    self.embed_cost_usd += rag_evaluation.embedding_cost_usd(tok)
            print(f"  [chroma] {doc_id}: embedded {len(texts)} chunks, added {added} "
                  f"({time.perf_counter() - t0:.1f}s)")

        stateless = self.backend.backend_name == "openai"
        if stateless and self.config.embed_max_workers > 1 and len(chunked) > 1:
            with ThreadPoolExecutor(max_workers=self.config.embed_max_workers) as pool:
                list(pool.map(_embed_and_add, chunked))
        else:
            for item in chunked:
                _embed_and_add(item)

    # ── retrieval phase ────────────────────────────────────────────────────────
    def _retrieve_one(self, qr: QueryResult) -> QueryResult:
        t0 = time.perf_counter()
        if self.config.retrieval_mode == "hybrid":
            matches = self.store.query_hybrid(
                qr.retrieval_query, top_k=self.config.top_k,
                hybrid_alpha=self.config.hybrid_alpha, doc_id=qr.doc_id,
            )
        else:
            matches = self.store.query(qr.retrieval_query, top_k=self.config.top_k, doc_id=qr.doc_id)
        if self.reranker is not None and matches:
            with self._rerank_lock:
                matches = rerank(qr.retrieval_query, matches, self.config.reranker_model,
                                 top_k=self.config.rerank_top_k)
        qr.matches = matches
        # Full-page context must see EVERY chunk on the matched pages, not just
        # the matched chunks; fetch them from the store (like the query pipeline).
        if self.config.context_mode == "full_page":
            keys = {(c.source, c.page) for c, _ in matches}
            corpus = self.store.fetch_chunks_for_pages(keys)
        else:
            corpus = [c for c, _ in matches]
        items, raw_text = build_context(matches, corpus, context_mode=self.config.context_mode)
        qr.context_items = items
        qr.retrieved_text = raw_text
        qr.retrieval_hit = rag_evaluation.answer_is_correct(raw_text, qr.gt_tokens)
        qr.retrieval_time_s = time.perf_counter() - t0
        return qr

    # ── generation phase ──────────────────────────────────────────────────────
    def _generate_one(self, qr: QueryResult) -> QueryResult:
        if self.llm is None or not qr.matches:
            return qr
        t0 = time.perf_counter()
        if self.config.context_mode == "hierarchical":
            result = answer_with_hierarchical_context(
                self.llm, qr.question, qr.matches, self.store, qr.doc_id, self.config,
                rag_evaluation.invoke_llm_with_usage,
            )
            qr.answer = result.answer
            qr.gen_usage = result.usage or qr.gen_usage
            qr.context_items = result.context_items
            qr.escalation_depth = result.escalation_depth
            qr.llm_calls = result.llm_calls
            qr.context_tokens_by_level = result.context_tokens_by_level
        else:
            prompt = make_prompt(qr.question, qr.context_items)
            answer, usage = rag_evaluation.invoke_llm_with_usage(self.llm, prompt)
            qr.answer = answer
            qr.gen_usage = usage
            qr.llm_calls = 1
        qr.gen_time_s = time.perf_counter() - t0
        qr.token_match = rag_evaluation.llm_answer_is_correct(qr.answer, qr.gt_tokens)
        return qr

    # ── judge phase ───────────────────────────────────────────────────────────
    def _judge_one(self, qr: QueryResult) -> QueryResult:
        if not self.use_judge or not qr.answer or qr.ground_truth in (None, ""):
            qr.llm_hit = qr.token_match
            return qr
        # Coordinate labels use the deterministic comparator (no LLM needed);
        # other labels need the LLM judge, so skip if it is unavailable.
        is_coord = rag_evaluation.label_uses_coordinate_judge(qr.label)
        if not is_coord and self.llm is None:
            qr.llm_hit = qr.token_match
            return qr
        t0 = time.perf_counter()
        is_match, usage = rag_evaluation.judge_match_dispatch(
            self.llm, qr.label, qr.question, str(qr.ground_truth), qr.answer
        )
        qr.judge_match = is_match
        qr.judge_usage = usage
        qr.judge_time_s = time.perf_counter() - t0
        qr.llm_hit = is_match
        return qr

    # ── orchestration ─────────────────────────────────────────────────────────
    def run(self, docs: List[Dict[str, str]], queries: List[Dict[str, str]]) -> List[QueryResult]:
        # assemble all (doc, query) tasks with ground truth
        tasks: List[QueryResult] = []
        for doc in docs:
            with open(doc["ground_truth"], "r", encoding="utf-8") as f:
                matched = json.load(f)
            # Resolve this doc's well name (ground truth) for the inject arm; the
            # no-inject arm uses the neutral "the well" so {well} -> "the well".
            well_name = rag_evaluation.resolve_well_name(matched) if self.inject_name else "the well"
            for q in queries:
                gt = rag_evaluation.extract_ground_truth(q["label"], matched)
                question = rag_evaluation.inject_name(q["question"], well_name)
                retrieval_query = rag_evaluation.inject_name(
                    q.get("retrieval_query") or q["question"], well_name
                )
                tasks.append(QueryResult(
                    doc_id=doc["id"], label=q["label"], question=question,
                    retrieval_query=retrieval_query,
                    ground_truth=gt["value"], gt_tokens=gt["tokens"],
                ))

        def _map(stage_name: str, fn, items: List[QueryResult], workers: int) -> List[QueryResult]:
            print(f"  {stage_name}: {len(items)} tasks across {workers} workers ...")
            out: List[Optional[QueryResult]] = [None] * len(items)
            if workers <= 1:
                return [fn(it) for it in items]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(fn, it): i for i, it in enumerate(items)}
                for fut in futs:
                    out[futs[fut]] = fut.result()
            return [o for o in out if o is not None]

        print("Retrieval phase ...")
        tasks = _map("retrieve", self._retrieve_one, tasks, self.config.embed_max_workers)
        if self.llm is not None:
            print("Generation phase ...")
            tasks = _map("generate", self._generate_one, tasks, self.config.llm_max_workers)
            if self.use_judge:
                print("Judge phase ...")
                tasks = _map("judge", self._judge_one, tasks, self.config.llm_max_workers)
            else:
                for qr in tasks:
                    qr.llm_hit = qr.token_match
        return tasks


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------


def _qr_to_row(qr: QueryResult) -> Dict[str, Any]:
    return {
        "doc_id": qr.doc_id,
        "label": qr.label,
        "question": qr.question,
        "ground_truth": qr.ground_truth,
        "retrieval_hit": qr.retrieval_hit,
        "token_match": qr.token_match,
        "judge_match": qr.judge_match,
        "llm_hit": qr.llm_hit,
        "answer": qr.answer,
        "input_tokens": qr.gen_usage.get("input_tokens", 0),
        "output_tokens": qr.gen_usage.get("output_tokens", 0),
        "tokens": qr.gen_usage.get("total_tokens", 0),
        "judge_input_tokens": qr.judge_usage.get("input_tokens", 0),
        "judge_output_tokens": qr.judge_usage.get("output_tokens", 0),
        "escalation_depth": qr.escalation_depth,
        "llm_calls": qr.llm_calls,
        "context_tokens_by_level": qr.context_tokens_by_level,
        "retrieval_time_s": qr.retrieval_time_s,
        "gen_time_s": qr.gen_time_s,
        "judge_time_s": qr.judge_time_s,
    }


def _has_ground_truth(row: Dict[str, Any]) -> bool:
    """A row is scorable only when it has a non-empty ground-truth value."""
    gt = row.get("ground_truth")
    if gt is None:
        return False
    if isinstance(gt, str) and not gt.strip():
        return False
    if isinstance(gt, (list, tuple, dict)) and len(gt) == 0:
        return False
    return True


def summarize(
    results: List[QueryResult],
    embed_tokens: int = 0,
    embed_cost_usd: float = 0.0,
    llm_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    rows = [_qr_to_row(r) for r in results]
    total = len(rows)

    def _is_hit(v: Any) -> bool:
        return v is True

    def _acc(correct: int, denom: int) -> Optional[float]:
        return (correct / denom) if denom else None

    # Accuracy denominators exclude rows with no ground truth: a question we
    # cannot grade (null ground truth) must not count against accuracy.
    scorable = sum(1 for r in rows if _has_ground_truth(r))
    null_ground_truth = total - scorable

    llm_correct = sum(1 for r in rows if _is_hit(r["llm_hit"]))
    token_correct = sum(1 for r in rows if _is_hit(r["token_match"]))
    retr_correct = sum(1 for r in rows if _is_hit(r["retrieval_hit"]))

    # Per-attribute (query label) breakdown, in first-seen order. Accuracy for
    # each label is measured over that label's scorable (non-null GT) rows only.
    by_label: "OrderedDict[str, Dict[str, int]]" = OrderedDict()
    for r in rows:
        b = by_label.setdefault(r["label"], {"total": 0, "scorable": 0, "llm": 0, "token": 0, "retrieval": 0})
        b["total"] += 1
        b["scorable"] += 1 if _has_ground_truth(r) else 0
        b["llm"] += 1 if _is_hit(r["llm_hit"]) else 0
        b["token"] += 1 if _is_hit(r["token_match"]) else 0
        b["retrieval"] += 1 if _is_hit(r["retrieval_hit"]) else 0
    per_label = [
        {
            "label": label,
            "total": b["total"],
            "null_ground_truth": b["total"] - b["scorable"],
            "scorable": b["scorable"],
            "llm_correct": b["llm"],
            "token_correct": b["token"],
            "retrieval_correct": b["retrieval"],
            "llm_accuracy": _acc(b["llm"], b["scorable"]),
            "token_accuracy": _acc(b["token"], b["scorable"]),
            "retrieval_accuracy": _acc(b["retrieval"], b["scorable"]),
        }
        for label, b in by_label.items()
    ]

    gen_in = sum(r["input_tokens"] for r in rows)
    gen_out = sum(r["output_tokens"] for r in rows)
    judge_in = sum(r["judge_input_tokens"] for r in rows)
    judge_out = sum(r["judge_output_tokens"] for r in rows)
    total_llm_calls = sum(r.get("llm_calls", 0) for r in rows)
    escalated = [r for r in rows if r.get("llm_calls", 0) > 0]
    avg_escalation_depth = (
        sum(r.get("escalation_depth", 0) for r in escalated) / len(escalated)
        if escalated else None
    )

    # Costs (USD). Generation + judge use the chat-model pricing; embedding uses
    # the embedding-model pricing accumulated during the build phase.
    gen_cost = rag_evaluation.token_cost_usd(gen_in, gen_out)
    judge_cost = rag_evaluation.token_cost_usd(judge_in, judge_out)
    total_cost = gen_cost + judge_cost + embed_cost_usd

    # Wall-clock time spent per phase (summed across all tasks).
    retrieval_time = sum(r["retrieval_time_s"] for r in rows)
    gen_time = sum(r["gen_time_s"] for r in rows)
    judge_time = sum(r["judge_time_s"] for r in rows)

    return {
        "total_questions": total,
        "scorable_questions": scorable,
        "null_ground_truth": null_ground_truth,
        "combined_llm_accuracy": _acc(llm_correct, scorable),
        "combined_token_accuracy": _acc(token_correct, scorable),
        "combined_retrieval_accuracy": _acc(retr_correct, scorable),
        "llm_correct": llm_correct,
        "token_correct": token_correct,
        "retrieval_correct": retr_correct,
        "per_label": per_label,
        "escalation": {
            "total_llm_calls": total_llm_calls,
            "avg_llm_calls_per_query": (total_llm_calls / total) if total else None,
            "avg_escalation_depth": avg_escalation_depth,
        },
        "tokens": {
            "embed_input": embed_tokens,
            "gen_input": gen_in, "gen_output": gen_out,
            "judge_input": judge_in, "judge_output": judge_out,
            "total": gen_in + gen_out + judge_in + judge_out,
        },
        "cost_usd": {
            "embedding": embed_cost_usd,
            "generation": gen_cost,
            "judge": judge_cost,
            "total": total_cost,
        },
        "time_s": {
            "retrieval": retrieval_time,
            "generation": gen_time,
            "judge": judge_time,
            "total": retrieval_time + gen_time + judge_time,
        },
        # OpenAI usage/cost callback metadata (successful requests, cached +
        # reasoning tokens, per-model split, finish reasons). Empty when the LLM
        # was unavailable or the callback captured nothing.
        "llm_meta": llm_meta or {},
        "per_query": rows,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    def acc(v: Optional[float]) -> str:
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    # Accuracy is measured over scorable questions (null ground truth excluded).
    denom = summary.get("scorable_questions", summary["total_questions"])
    print("\n" + "=" * 70)
    print("COMBINED ACCURACY (null ground truth excluded)")
    print("=" * 70)
    print(rag_evaluation._render_ascii_table(
        ["Metric", "Correct / Scorable", "Accuracy"],
        [
            ["LLM (judge)", f"{summary['llm_correct']} / {denom}", acc(summary["combined_llm_accuracy"])],
            ["Literal token", f"{summary['token_correct']} / {denom}", acc(summary["combined_token_accuracy"])],
            ["Retrieval hit", f"{summary['retrieval_correct']} / {denom}", acc(summary["combined_retrieval_accuracy"])],
        ],
    ))
    if summary.get("null_ground_truth"):
        print(f"({summary['null_ground_truth']} of {summary['total_questions']} questions had "
              f"null ground truth and were excluded from accuracy.)")
    per_label = summary.get("per_label", [])
    if per_label:
        print("\n" + "=" * 70)
        print("ACCURACY BY ATTRIBUTE (null ground truth excluded)")
        print("=" * 70)
        print(rag_evaluation._render_ascii_table(
            ["Attribute", "LLM (judge)", "Token", "Retrieval"],
            [
                [
                    lab["label"],
                    f"{lab['llm_correct']}/{lab.get('scorable', lab['total'])} ({acc(lab['llm_accuracy'])})",
                    acc(lab["token_accuracy"]),
                    acc(lab["retrieval_accuracy"]),
                ]
                for lab in per_label
            ],
        ))

    esc = summary.get("escalation", {})
    if esc.get("total_llm_calls"):
        def _num(v: Optional[float]) -> str:
            return f"{v:.2f}" if v is not None else "N/A"
        print("\n" + "=" * 70)
        print("HIERARCHICAL ESCALATION")
        print("=" * 70)
        print(rag_evaluation._render_ascii_table(
            ["Metric", "Value"],
            [
                ["Total LLM calls", str(esc.get("total_llm_calls", 0))],
                ["Avg LLM calls / query", _num(esc.get("avg_llm_calls_per_query"))],
                ["Avg escalation depth", _num(esc.get("avg_escalation_depth"))],
            ],
        ))

    t = summary["tokens"]
    print("\n" + "=" * 70)
    print("TOKEN USAGE (input vs output)")
    print("=" * 70)
    print(rag_evaluation._render_ascii_table(
        ["Phase", "Input tok", "Output tok"],
        [
            ["Embedding", str(t.get("embed_input", 0)), "-"],
            ["Generation", str(t["gen_input"]), str(t["gen_output"])],
            ["Judge", str(t["judge_input"]), str(t["judge_output"])],
            ["TOTAL", str(t["gen_input"] + t["judge_input"]), str(t["gen_output"] + t["judge_output"])],
        ],
    ))

    c = summary.get("cost_usd", {})
    print("\n" + "=" * 70)
    print("COST (USD)")
    print("=" * 70)
    print(rag_evaluation._render_ascii_table(
        ["Phase", "Cost (USD)"],
        [
            ["Embedding (this run)", f"${c.get('embedding', 0.0):.4f}"],
            ["Generation (LLM)", f"${c.get('generation', 0.0):.4f}"],
            ["Judge (LLM)", f"${c.get('judge', 0.0):.4f}"],
            ["TOTAL", f"${c.get('total', 0.0):.4f}"],
        ],
    ))

    tm = summary.get("time_s", {})
    print("\n" + "=" * 70)
    print("TIME (seconds, summed across tasks)")
    print("=" * 70)
    print(rag_evaluation._render_ascii_table(
        ["Phase", "Time (s)"],
        [
            ["Retrieval", f"{tm.get('retrieval', 0.0):.1f}"],
            ["Generation", f"{tm.get('generation', 0.0):.1f}"],
            ["Judge", f"{tm.get('judge', 0.0):.1f}"],
            ["TOTAL", f"{tm.get('total', 0.0):.1f}"],
        ],
    ))
    print("\nQuery-wise analysis:")
    print(rag_evaluation.render_query_table(summary["per_query"]))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-process, parallel RAG batch runner.")
    p.add_argument("--docs-dir", default=evaluate_all.DEFAULT_DOCS_DIR)
    p.add_argument("--matched-dir", default=evaluate_all.DEFAULT_MATCHED_DIR)
    p.add_argument("--queries", default=evaluate_all.DEFAULT_QUERIES)
    p.add_argument("--only", nargs="+", default=None, help="Only these doc ids (e.g. PR2610)")
    p.add_argument("--summary-output", default=os.path.join("output_table", "batch_runner_summary.json"))
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--build-only", action="store_true", help="Only embed-once into Chroma, skip inference")
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--openai-embedding-model", default=None,
                   help="Override config.openai_embedding_model (e.g. text-embedding-3-large)")
    p.add_argument("--openai-embedding-dimensions", type=int, default=None,
                   help="Truncated (Matryoshka) output dimension for OpenAI embeddings")
    p.add_argument("--use-reranker", dest="use_reranker", action="store_true", default=None,
                   help="Force the cross-encoder reranker ON")
    p.add_argument("--no-reranker", dest="use_reranker", action="store_false",
                   help="Force the cross-encoder reranker OFF")
    p.add_argument("--chunk-strategy", default=None)
    p.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default=None,
                   help="Override config.retrieval_mode")
    p.add_argument("--hybrid-alpha", type=float, default=None,
                   help="Override config.hybrid_alpha (0=BM25 .. 1=dense)")
    p.add_argument("--context-mode", choices=["chunk", "full_page", "hierarchical"], default=None,
                   help="Override config.context_mode")
    p.add_argument("--hierarchical-max-depth", type=int, default=None,
                   help="Max ancestor levels to climb in hierarchical mode")
    p.add_argument("--load-max-workers", type=int, default=None,
                   help="Process-pool workers for load+chunk (use 1 on Windows / large JSONs to avoid MemoryError)")
    p.add_argument("--embed-max-workers", type=int, default=None,
                   help="Concurrent embedding requests during build")
    p.add_argument("--inject-name", action=argparse.BooleanOptionalAction, default=True,
                   help="Fill {well} placeholders with the ground-truth name (--no-inject-name to disable)")
    return p.parse_args()


def main() -> None:
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    config = RagConfig.from_env()
    if args.embedding_model:
        config.embedding_model = args.embedding_model
    if args.openai_embedding_model:
        config.openai_embedding_model = args.openai_embedding_model
    if args.openai_embedding_dimensions is not None:
        config.openai_embedding_dimensions = args.openai_embedding_dimensions
    if args.use_reranker is not None:
        config.use_reranker = args.use_reranker
    if args.chunk_strategy:
        config.chunk_strategy = args.chunk_strategy
    if args.retrieval_mode:
        config.retrieval_mode = args.retrieval_mode
    if args.hybrid_alpha is not None:
        config.hybrid_alpha = args.hybrid_alpha
    if args.context_mode:
        config.context_mode = args.context_mode
    if args.hierarchical_max_depth is not None:
        config.hierarchical_max_depth = args.hierarchical_max_depth
    if args.load_max_workers is not None:
        config.load_max_workers = args.load_max_workers
    if args.embed_max_workers is not None:
        config.embed_max_workers = args.embed_max_workers

    matched, skipped = evaluate_all.discover_docs(args.docs_dir, args.matched_dir, only=args.only)
    print(f"Discovered {len(matched)} doc(s)" + (f", skipped {len(skipped)}" if skipped else ""))
    if not matched:
        print("[error] No docs with matching ground truth.")
        sys.exit(1)

    print(f"Config: retrieval_mode={config.retrieval_mode} hybrid_alpha={config.hybrid_alpha} "
          f"reranker={config.use_reranker} inject_name={args.inject_name}")
    queries = rag_evaluation.parse_queries_file(args.queries)
    runner = BatchRunner(config, use_judge=not args.no_judge, inject_name=args.inject_name)

    t0 = time.perf_counter()
    runner.build_all(matched)
    print(f"Build phase done in {time.perf_counter() - t0:.1f}s")
    if args.build_only:
        print("--build-only set; skipping inference.")
        return

    t1 = time.perf_counter()
    results = runner.run(matched, queries)
    print(f"Inference phase done in {time.perf_counter() - t1:.1f}s")

    summary = summarize(results, embed_tokens=runner.embed_tokens,
                        embed_cost_usd=runner.embed_cost_usd,
                        llm_meta=runner.llm_usage_snapshot())
    print_summary(summary)

    run_config = {
        "embedding_model": config.embedding_model,
        "openai_embedding_model": config.openai_embedding_model,
        "openai_embedding_dimensions": config.openai_embedding_dimensions,
        "retrieval_mode": config.retrieval_mode,
        "hybrid_alpha": config.hybrid_alpha,
        "use_reranker": config.use_reranker,
        "reranker_model": config.reranker_model,
        "chunk_strategy": config.chunk_strategy,
        "context_mode": config.context_mode,
        "hierarchical_max_depth": config.hierarchical_max_depth,
        "top_k": config.top_k,
        "rerank_top_k": config.rerank_top_k,
        "inject_name": args.inject_name,
    }
    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump({"config_collection": config.chroma_collection, "run_config": run_config,
                   "summary": summary}, f, ensure_ascii=False, indent=2)
    print(f"\nSaved summary -> {args.summary_output}")


if __name__ == "__main__":
    main()
