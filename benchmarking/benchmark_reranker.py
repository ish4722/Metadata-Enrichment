"""Task 3 - Reranker impact on the best embedding model.

Takes the winning embedding model from Task 1 (auto-selected by end-to-end LLM
accuracy from ``output_table/benchmarking/embeddings/comparison.json``, or forced
with ``--model``) and quantifies what the cross-encoder reranker adds, using the
same metrics as the other tasks:

  * Retrieval ranking metrics PRE vs POST rerank (MRR, MAP, Hit@k, Recall@k,
    nDCG@k, Distractor@k) - both stages come from one ``score_matryoshka_retrieval``
    pass, reusing the existing collection (no re-embedding).
  * (default) End-to-end accuracy with the reranker OFF vs ON (two BatchRunner
    runs) so the retrieval gain can be tied to final LLM-judge accuracy.

Run from the repo root:

    python -m benchmarking.benchmark_reranker
    python -m benchmarking.benchmark_reranker --model openai_large
    python -m benchmarking.benchmark_reranker --no-end-to-end   # retrieval metrics only
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, Optional, Tuple

from benchmarking import common
from benchmarking.benchmark_embeddings import CONFIGS as EMB_CONFIGS

OUT_DIR = os.path.join(common.OUTPUT_ROOT, "reranker")
EMB_COMPARISON = os.path.join(common.OUTPUT_ROOT, "embeddings", "comparison.json")

# tag -> (build_config kwargs, embed-spec key). Reuse Task 1's definitions.
EMB_MAP: Dict[str, Tuple[Dict[str, Any], str]] = {
    tag: (dict(kwargs), embed_spec) for tag, kwargs, embed_spec in EMB_CONFIGS
}

RETRIEVAL_KEYS = ["mrr", "map", "hit@1", "hit@3", "hit@5", "hit@10",
                  "recall@5", "recall@10", "ndcg@5", "ndcg@10", "distractor@5"]


def resolve_best_tag(explicit: Optional[str]) -> str:
    """Return the embedding config tag to use for the reranker study."""
    if explicit:
        if explicit not in EMB_MAP:
            raise SystemExit(f"[error] unknown --model '{explicit}'. "
                             f"Choose from: {', '.join(EMB_MAP)}")
        return explicit
    if not os.path.isfile(EMB_COMPARISON):
        raise SystemExit(
            f"[error] {EMB_COMPARISON} not found. Run Task 1 first "
            "(python -m benchmarking.benchmark_embeddings) or pass --model.")
    data = json.load(open(EMB_COMPARISON, encoding="utf-8"))
    best_tag, best_acc = None, -1.0
    for r in data.get("results", []):
        if r.get("status") != "ok":
            continue
        acc = r.get("summary", {}).get("combined_llm_accuracy")
        if isinstance(acc, (int, float)) and acc > best_acc and r["tag"] in EMB_MAP:
            best_tag, best_acc = r["tag"], acc
    if best_tag is None:
        raise SystemExit("[error] no successful Task-1 config found; pass --model.")
    print(f"Auto-selected best embedding model: {best_tag} (LLM acc={common.pct(best_acc)})")
    return best_tag


def _kwargs_without_reranker(tag: str) -> Tuple[Dict[str, Any], str]:
    kwargs, embed_spec = EMB_MAP[tag]
    kwargs = dict(kwargs)
    kwargs.pop("use_reranker", None)
    return kwargs, embed_spec


def build_report(payload: Dict[str, Any]) -> str:
    tag = payload["model_tag"]
    parts = [f"# Task 3 - Reranker Impact ({tag})\n",
             f"Reranker model: `{payload['reranker_model']}`. "
             "Retrieval metrics from the graded gold set; pre = embedding-only "
             "hybrid ranking, post = after the cross-encoder.\n"]

    pre = payload["retrieval"]["pre_rerank"]["macro"]
    post = payload["retrieval"]["post_rerank"]["macro"]
    rows = []
    for k in RETRIEVAL_KEYS:
        pv, qv = pre.get(k), post.get(k)
        delta = (qv - pv) if isinstance(pv, (int, float)) and isinstance(qv, (int, float)) else None
        rows.append([k, common.f3(pv), common.f3(qv),
                     f"{delta:+.3f}" if delta is not None else "N/A"])
    parts.append("## Retrieval metrics: pre vs post rerank\n")
    parts.append(common.md_table(["Metric", "Pre-rerank", "Post-rerank", "Delta"], rows))
    parts.append("")

    e2e = payload.get("end_to_end")
    if e2e:
        off, on = e2e["reranker_off"], e2e["reranker_on"]

        def _delta(a, b):
            return (f"{(b - a) * 100:+.1f} pp"
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "N/A")

        rows = [
            ["LLM (judge) accuracy",
             common.pct(off.get("combined_llm_accuracy")),
             common.pct(on.get("combined_llm_accuracy")),
             _delta(off.get("combined_llm_accuracy"), on.get("combined_llm_accuracy"))],
            ["Token-match accuracy",
             common.pct(off.get("combined_token_accuracy")),
             common.pct(on.get("combined_token_accuracy")),
             _delta(off.get("combined_token_accuracy"), on.get("combined_token_accuracy"))],
            ["Retrieval accuracy",
             common.pct(off.get("combined_retrieval_accuracy")),
             common.pct(on.get("combined_retrieval_accuracy")),
             _delta(off.get("combined_retrieval_accuracy"), on.get("combined_retrieval_accuracy"))],
        ]
        parts.append("## End-to-end accuracy: reranker OFF vs ON\n")
        parts.append(common.md_table(["Metric", "Reranker OFF", "Reranker ON", "Delta"], rows))
        parts.append("")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3: reranker impact on the best model.")
    p.add_argument("--model", default=None,
                   help="Embedding config tag to use (default: auto from Task 1). "
                        f"Choices: {', '.join(EMB_MAP)}")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--goldset", default=common.DEFAULT_GOLDSET)
    p.add_argument("--queries", default="queries.txt")
    p.add_argument("--end-to-end", action=argparse.BooleanOptionalAction, default=True,
                   help="Also run end-to-end reranker OFF vs ON (default: on)")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--inject-name", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    import rag_evaluation
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    tag = resolve_best_tag(args.model)
    kwargs, embed_spec = _kwargs_without_reranker(tag)

    # Retrieval pre/post rerank in one pass (reranker ON so post uses it).
    cfg_rr = common.build_config(use_reranker=True, **kwargs)
    print(f"\nScoring retrieval metrics (pre/post rerank) for {tag} "
          f"(collection={cfg_rr.chroma_collection}) ...")
    retrieval = common.run_retrieval_metrics(cfg_rr, args.goldset, args.queries)

    payload: Dict[str, Any] = {
        "task": "reranker_impact",
        "model_tag": tag,
        "collection": cfg_rr.chroma_collection,
        "reranker_model": cfg_rr.reranker_model,
        "retrieval": retrieval,
    }

    if args.end_to_end:
        matched = common.discover(only=args.only, limit=args.limit)
        queries = common.load_queries(args.queries)
        if matched:
            print("\n=== End-to-end with reranker OFF ===")
            cfg_off = common.build_config(use_reranker=False, **kwargs)
            t0 = time.perf_counter()
            off = common.run_end_to_end(cfg_off, matched, queries,
                                        use_judge=not args.no_judge, inject_name=args.inject_name)
            print(f"  reranker OFF done in {time.perf_counter() - t0:.1f}s")

            print("\n=== End-to-end with reranker ON ===")
            t0 = time.perf_counter()
            on = common.run_end_to_end(cfg_rr, matched, queries,
                                       use_judge=not args.no_judge, inject_name=args.inject_name)
            print(f"  reranker ON done in {time.perf_counter() - t0:.1f}s")
            payload["end_to_end"] = {"reranker_off": off, "reranker_on": on}

    common.write_json(os.path.join(OUT_DIR, "metrics.json"), payload)
    common.save_report(os.path.join(OUT_DIR, "report.md"), build_report(payload))
    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
