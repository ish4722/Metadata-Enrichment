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
    context_items: List[Dict[str, Any]] = field(default_factory=list)
    retrieved_text: str = ""
    retrieval_hit: bool = False
    answer: str = ""
    gen_usage: Dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    judge_usage: Dict[str, int] = field(default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
    token_match: Optional[bool] = None
    judge_match: Optional[bool] = None
    llm_hit: Optional[bool] = None
    retrieval_time_s: float = 0.0
    gen_time_s: float = 0.0
    judge_time_s: float = 0.0


# ---------------------------------------------------------------------------
# The runner
# ---------------------------------------------------------------------------


class BatchRunner:
    def __init__(self, config: RagConfig, use_judge: bool = True):
        self.config = config
        self.use_judge = use_judge
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
        items, raw_text = build_context(matches, [c for c, _ in matches], context_mode=self.config.context_mode)
        qr.context_items = items
        qr.retrieved_text = raw_text
        qr.retrieval_hit = rag_evaluation.answer_is_correct(raw_text, qr.gt_tokens)
        qr.retrieval_time_s = time.perf_counter() - t0
        return qr

    # ── generation phase ──────────────────────────────────────────────────────
    def _generate_one(self, qr: QueryResult) -> QueryResult:
        if self.llm is None or not qr.context_items:
            return qr
        t0 = time.perf_counter()
        prompt = make_prompt(qr.question, qr.context_items)
        answer, usage = rag_evaluation.invoke_llm_with_usage(self.llm, prompt)
        qr.answer = answer
        qr.gen_usage = usage
        qr.gen_time_s = time.perf_counter() - t0
        qr.token_match = rag_evaluation.answer_is_correct(answer, qr.gt_tokens)
        return qr

    # ── judge phase ───────────────────────────────────────────────────────────
    def _judge_one(self, qr: QueryResult) -> QueryResult:
        if self.llm is None or not self.use_judge or not qr.answer or not qr.ground_truth:
            qr.llm_hit = qr.token_match
            return qr
        t0 = time.perf_counter()
        is_match, usage = rag_evaluation.llm_judge_match(
            self.llm, qr.question, str(qr.ground_truth), qr.answer
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
            for q in queries:
                gt = rag_evaluation.extract_ground_truth(q["label"], matched)
                tasks.append(QueryResult(
                    doc_id=doc["id"], label=q["label"], question=q["question"],
                    retrieval_query=q["retrieval_query"],
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
        "retrieval_time_s": qr.retrieval_time_s,
        "gen_time_s": qr.gen_time_s,
        "judge_time_s": qr.judge_time_s,
    }


def summarize(
    results: List[QueryResult],
    embed_tokens: int = 0,
    embed_cost_usd: float = 0.0,
) -> Dict[str, Any]:
    rows = [_qr_to_row(r) for r in results]
    total = len(rows)

    def _is_hit(v: Any) -> bool:
        return v is True

    llm_correct = sum(1 for r in rows if _is_hit(r["llm_hit"]))
    token_correct = sum(1 for r in rows if _is_hit(r["token_match"]))
    retr_correct = sum(1 for r in rows if _is_hit(r["retrieval_hit"]))

    gen_in = sum(r["input_tokens"] for r in rows)
    gen_out = sum(r["output_tokens"] for r in rows)
    judge_in = sum(r["judge_input_tokens"] for r in rows)
    judge_out = sum(r["judge_output_tokens"] for r in rows)

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
        "combined_llm_accuracy": (llm_correct / total) if total else None,
        "combined_token_accuracy": (token_correct / total) if total else None,
        "combined_retrieval_accuracy": (retr_correct / total) if total else None,
        "llm_correct": llm_correct,
        "token_correct": token_correct,
        "retrieval_correct": retr_correct,
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
        "per_query": rows,
    }


def print_summary(summary: Dict[str, Any]) -> None:
    def acc(v: Optional[float]) -> str:
        return f"{v * 100:.1f}%" if v is not None else "N/A"

    print("\n" + "=" * 70)
    print("COMBINED ACCURACY")
    print("=" * 70)
    print(rag_evaluation._render_ascii_table(
        ["Metric", "Correct / Total", "Accuracy"],
        [
            ["LLM (judge)", f"{summary['llm_correct']} / {summary['total_questions']}", acc(summary["combined_llm_accuracy"])],
            ["Literal token", f"{summary['token_correct']} / {summary['total_questions']}", acc(summary["combined_token_accuracy"])],
            ["Retrieval hit", f"{summary['retrieval_correct']} / {summary['total_questions']}", acc(summary["combined_retrieval_accuracy"])],
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
    p.add_argument("--chunk-strategy", default=None)
    return p.parse_args()


def main() -> None:
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    config = RagConfig.from_env()
    if args.embedding_model:
        config.embedding_model = args.embedding_model
    if args.chunk_strategy:
        config.chunk_strategy = args.chunk_strategy

    matched, skipped = evaluate_all.discover_docs(args.docs_dir, args.matched_dir, only=args.only)
    print(f"Discovered {len(matched)} doc(s)" + (f", skipped {len(skipped)}" if skipped else ""))
    if not matched:
        print("[error] No docs with matching ground truth.")
        sys.exit(1)

    queries = rag_evaluation.parse_queries_file(args.queries)
    runner = BatchRunner(config, use_judge=not args.no_judge)

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
                        embed_cost_usd=runner.embed_cost_usd)
    print_summary(summary)

    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump({"config_collection": config.chroma_collection, "summary": summary}, f,
                  ensure_ascii=False, indent=2)
    print(f"\nSaved summary -> {args.summary_output}")


if __name__ == "__main__":
    main()
