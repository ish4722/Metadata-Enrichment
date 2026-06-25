"""Validate the LIVE query pipeline against ground truth.

Unlike ``python -m rag.main evaluate`` (which rebuilds a fresh FAISS index via
the rag_evaluation driver), this script exercises the EXACT path used by
``python -m rag.main query`` - the persistent Chroma store, hybrid retrieval and
the cross-encoder reranker - then scores each answer against the matched-JSON
ground truth using the shared ``rag_evaluation`` helpers.

For every query in the queries file it reports:
  * retrieval_hit - did the retrieved context contain the ground-truth token(s)?
  * token_match   - does the LLM answer literally contain the ground-truth value?
  * judge_match   - does an LLM judge consider the answer semantically correct?

Usage (run from the repo root):

  python validate_query.py --ground-truth matched_json/PR2610.json
  python validate_query.py --ground-truth matched_json/PR2610.json --retrieval-mode hybrid --hybrid-alpha 0.5
  python validate_query.py --ground-truth matched_json/PR2610.json --only "lat+long" "facility name"
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches

apply_ssl_patches()  # must run before transformers / huggingface_hub import

import rag_evaluation  # noqa: E402
from rag.config import RagConfig  # noqa: E402
from rag.inference.context_builder import build_context  # noqa: E402
from rag.inference.llm import answer_with_llm, get_llm  # noqa: E402
from rag.inference.prompt import make_prompt  # noqa: E402
from rag.inference.query_pipeline import load_vector_store, retrieve_and_build_context  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate the live RAG query pipeline against ground truth.")
    p.add_argument("--ground-truth", required=True, help="Path to the matched JSON (e.g. matched_json/PR2610.json)")
    p.add_argument("--queries", default="queries.txt", help="Path to the queries file")
    p.add_argument("--only", nargs="+", default=None, help="Only run these query labels (e.g. 'lat+long')")
    p.add_argument("--no-judge", action="store_true", help="Disable the LLM judge (literal token match only)")
    p.add_argument("--show-chunks", action="store_true", help="Print the retrieved chunks for each query")
    p.add_argument("--metrics-output", default=None, help="Optional path to write the results JSON")
    # config overrides (mirror rag.main)
    p.add_argument("--embedding-model", default=None)
    p.add_argument("--retrieval-mode", choices=["dense", "hybrid"], default=None)
    p.add_argument("--hybrid-alpha", type=float, default=None)
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--no-reranker", dest="use_reranker", action="store_false", default=None)
    return p.parse_args()


def build_config(args: argparse.Namespace) -> RagConfig:
    cfg = RagConfig.from_env()
    if args.embedding_model:
        cfg.embedding_model = args.embedding_model
    if args.retrieval_mode:
        cfg.retrieval_mode = args.retrieval_mode
    if args.hybrid_alpha is not None:
        cfg.hybrid_alpha = args.hybrid_alpha
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.use_reranker is False:
        cfg.use_reranker = False
    return cfg


def main() -> None:
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()
    config = build_config(args)

    if not os.path.isfile(args.ground_truth):
        print(f"[error] ground-truth not found: {args.ground_truth}")
        sys.exit(1)
    with open(args.ground_truth, "r", encoding="utf-8") as f:
        matched = json.load(f)

    queries = rag_evaluation.parse_queries_file(args.queries)
    if args.only:
        wanted = {q.strip().lower() for q in args.only}
        queries = [q for q in queries if q["label"].strip().lower() in wanted]
    if not queries:
        print("[error] no queries to run (check --queries / --only).")
        sys.exit(1)

    store = load_vector_store(config)
    llm = get_llm(config)
    use_judge = (llm is not None) and not args.no_judge
    if llm is None:
        print("[warn] LLM unavailable - scoring retrieval only (no generated answer / judge).")

    rows = []
    for q in queries:
        label = q["label"]
        question = q["question"]
        retrieval_query = q.get("retrieval_query") or question
        gt = rag_evaluation.extract_ground_truth(label, matched)

        # retrieve with the keyword-augmented query (matches batch_runner behaviour)
        matches, context_items, raw_text = retrieve_and_build_context(retrieval_query, store, config)
        retrieval_hit = rag_evaluation.answer_is_correct(raw_text, gt["tokens"])

        if args.show_chunks:
            print(f"\n### [{label}] retrieved {len(matches)} chunks")
            for i, (c, s) in enumerate(matches, 1):
                print(f"  [{i}] score={s:.4f} page={c.page} section={c.section_path or c.block_type}")

        # generate the answer with the natural-language question
        answer = ""
        if llm is not None and context_items:
            prompt = make_prompt(question, context_items)
            answer = answer_with_llm(llm, prompt)

        token_match = rag_evaluation.answer_is_correct(answer, gt["tokens"]) if answer else False
        judge_match = None
        if use_judge and gt["value"] is not None and answer:
            judge_match, _usage = rag_evaluation.llm_judge_match(llm, question, str(gt["value"]), answer)

        rows.append({
            "label": label,
            "ground_truth": gt["value"],
            "retrieval_hit": retrieval_hit,
            "token_match": token_match,
            "judge_match": judge_match,
            "answer": answer,
        })

    _report(rows, use_judge)

    if args.metrics_output:
        os.makedirs(os.path.dirname(args.metrics_output) or ".", exist_ok=True)
        with open(args.metrics_output, "w", encoding="utf-8") as f:
            json.dump({"ground_truth": args.ground_truth, "config_collection": config.chroma_collection,
                       "rows": rows}, f, ensure_ascii=False, indent=2)
        print(f"\nSaved results -> {args.metrics_output}")


def _report(rows, use_judge: bool) -> None:
    def mark(v) -> str:
        if v is True:
            return "PASS"
        if v is False:
            return "FAIL"
        return "N/A"

    headers = ["Query label", "Ground truth", "Retrieval", "Token", "Judge"]
    table_rows = [
        [
            r["label"],
            (str(r["ground_truth"])[:40] if r["ground_truth"] is not None else "(none)"),
            mark(r["retrieval_hit"]),
            mark(r["token_match"]),
            mark(r["judge_match"]),
        ]
        for r in rows
    ]
    print("\n" + rag_evaluation._render_ascii_table(headers, table_rows))

    total = len(rows)
    retr = sum(1 for r in rows if r["retrieval_hit"] is True)
    tok = sum(1 for r in rows if r["token_match"] is True)
    print(f"\nRetrieval accuracy: {retr}/{total} = {retr / total * 100:.1f}%")
    print(f"Token accuracy:     {tok}/{total} = {tok / total * 100:.1f}%")
    if use_judge:
        jud = sum(1 for r in rows if r["judge_match"] is True)
        print(f"Judge accuracy:     {jud}/{total} = {jud / total * 100:.1f}%")


if __name__ == "__main__":
    main()
