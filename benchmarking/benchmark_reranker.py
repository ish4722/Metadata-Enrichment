"""Task 3 - Reranker impact across ALL embedding models.

By default this quantifies what the cross-encoder reranker adds for EVERY
successful Task-1 embedding config (from
``output_table/benchmarking/embeddings/comparison.json``), not just the best one.
Restrict with ``--models`` (several tags) or ``--model`` (a single tag). For each
model it reports the same metrics:

  * Retrieval ranking metrics PRE vs POST rerank (MRR, MAP, Hit@k, Recall@k,
    nDCG@k, Distractor@k) - both stages come from one ``score_matryoshka_retrieval``
    pass, reusing the existing collection (no re-embedding).
  * (default) End-to-end accuracy with the reranker OFF vs ON (two BatchRunner
    runs) so the retrieval gain can be tied to final LLM-judge accuracy.

Run from the repo root:

    python -m benchmarking.benchmark_reranker                     # all models
    python -m benchmarking.benchmark_reranker --models openai_large openai_small
    python -m benchmarking.benchmark_reranker --model openai_large   # single model
    python -m benchmarking.benchmark_reranker --no-end-to-end        # retrieval metrics only
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


def resolve_tags(explicit_single: Optional[str], explicit_multi: Optional[list]) -> list:
    """Return the list of embedding config tags to run the reranker study for.

    Priority: ``--models`` (list) > ``--model`` (single) > all successful Task-1
    configs from the comparison file (in CONFIG order).
    """
    if explicit_multi:
        for t in explicit_multi:
            if t not in EMB_MAP:
                raise SystemExit(f"[error] unknown --models entry '{t}'. "
                                 f"Choose from: {', '.join(EMB_MAP)}")
        return list(explicit_multi)
    if explicit_single:
        return [resolve_best_tag(explicit_single)]
    if not os.path.isfile(EMB_COMPARISON):
        raise SystemExit(
            f"[error] {EMB_COMPARISON} not found. Run Task 1 first "
            "(python -m benchmarking.benchmark_embeddings) or pass --model/--models.")
    data = json.load(open(EMB_COMPARISON, encoding="utf-8"))
    ok_tags = {r["tag"] for r in data.get("results", [])
               if r.get("status") == "ok" and r.get("tag") in EMB_MAP}
    tags = [t for t in EMB_MAP if t in ok_tags]  # preserve CONFIG order
    if not tags:
        raise SystemExit("[error] no successful Task-1 config found; pass --model/--models.")
    print(f"Running reranker impact for {len(tags)} model(s): {', '.join(tags)}")
    return tags


def _kwargs_without_reranker(tag: str) -> Tuple[Dict[str, Any], str]:
    kwargs, embed_spec = EMB_MAP[tag]
    kwargs = dict(kwargs)
    kwargs.pop("use_reranker", None)
    return kwargs, embed_spec


def _pp_delta(a, b) -> str:
    return (f"{(b - a) * 100:+.1f} pp"
            if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "N/A")


def build_report(models: list, reranker_model: str) -> str:
    """Cross-model summary followed by a per-model detailed section."""
    parts = ["# Task 3 - Reranker Impact (all models)\n",
             f"Reranker model: `{reranker_model}`. Retrieval metrics from the graded "
             "gold set; pre = embedding-only hybrid ranking, post = after the "
             "cross-encoder. Models compared: "
             + ", ".join(f"`{m['model_tag']}`" for m in models) + ".\n"]

    # Cross-model summary: the headline pre/post deltas per model.
    srows = []
    for m in models:
        pre = m["retrieval"]["pre_rerank"]["macro"]
        post = m["retrieval"]["post_rerank"]["macro"]
        d_ndcg = post.get("ndcg@10", 0) - pre.get("ndcg@10", 0)
        d_mrr = post.get("mrr", 0) - pre.get("mrr", 0)
        row = [m["model_tag"],
               common.f3(pre.get("ndcg@10")), common.f3(post.get("ndcg@10")), f"{d_ndcg:+.3f}",
               common.f3(pre.get("mrr")), common.f3(post.get("mrr")), f"{d_mrr:+.3f}"]
        e2e = m.get("end_to_end")
        if e2e:
            off = e2e["reranker_off"].get("combined_llm_accuracy")
            on = e2e["reranker_on"].get("combined_llm_accuracy")
            row += [common.pct(off), common.pct(on), _pp_delta(off, on)]
        else:
            row += ["N/A", "N/A", "N/A"]
        srows.append(row)
    parts.append("## Summary across models\n")
    parts.append(common.md_table(
        ["Model", "nDCG@10 pre", "nDCG@10 post", "Δ nDCG@10",
         "MRR pre", "MRR post", "Δ MRR",
         "LLM acc OFF", "LLM acc ON", "Δ LLM acc"], srows))
    parts.append("")

    for m in models:
        parts.append(build_model_section(m))
    return "\n".join(parts)


def build_model_section(payload: Dict[str, Any]) -> str:
    tag = payload["model_tag"]
    parts = [f"## {tag}\n"]

    pre = payload["retrieval"]["pre_rerank"]["macro"]
    post = payload["retrieval"]["post_rerank"]["macro"]
    rows = []
    for k in RETRIEVAL_KEYS:
        pv, qv = pre.get(k), post.get(k)
        delta = (qv - pv) if isinstance(pv, (int, float)) and isinstance(qv, (int, float)) else None
        rows.append([k, common.f3(pv), common.f3(qv),
                     f"{delta:+.3f}" if delta is not None else "N/A"])
    parts.append("### Retrieval metrics: pre vs post rerank\n")
    parts.append(common.md_table(["Metric", "Pre-rerank", "Post-rerank", "Delta"], rows))
    parts.append("")

    e2e = payload.get("end_to_end")
    if e2e:
        off, on = e2e["reranker_off"], e2e["reranker_on"]
        _delta = _pp_delta
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
        parts.append("### End-to-end accuracy: reranker OFF vs ON\n")
        parts.append(common.md_table(["Metric", "Reranker OFF", "Reranker ON", "Delta"], rows))
        parts.append("")

        # OpenAI usage/cost callback metadata for both arms.
        llm_model = payload.get("llm_model")
        parts.append("### LLM call metadata (OpenAI callback)\n")
        parts.append(common.llm_meta_table(
            [("Reranker OFF", off, llm_model), ("Reranker ON", on, llm_model)],
            first_col="Arm"))
        parts.append("")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3: reranker impact across all embedding models.")
    p.add_argument("--models", nargs="+", default=None,
                   help="Embedding config tags to run (default: all successful Task-1 "
                        f"configs). Choices: {', '.join(EMB_MAP)}")
    p.add_argument("--model", default=None,
                   help="Single embedding config tag (overrides the default all-models run). "
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


def run_model(tag: str, args: argparse.Namespace,
              matched: Optional[list], queries: Optional[list]) -> Dict[str, Any]:
    """Compute pre/post-rerank retrieval (and optional end-to-end) for one model."""
    kwargs, _embed_spec = _kwargs_without_reranker(tag)

    # Retrieval pre/post rerank in one pass (reranker ON so post uses it).
    cfg_rr = common.build_config(use_reranker=True, **kwargs)
    print("\n" + "#" * 78)
    print(f"# RERANKER IMPACT for {tag}  (collection={cfg_rr.chroma_collection})")
    print("#" * 78)
    print(f"Scoring retrieval metrics (pre/post rerank) for {tag} ...")
    retrieval = common.run_retrieval_metrics(cfg_rr, args.goldset, args.queries)

    payload: Dict[str, Any] = {
        "model_tag": tag,
        "collection": cfg_rr.chroma_collection,
        "reranker_model": cfg_rr.reranker_model,
        "llm_model": cfg_rr.llm_model,
        "retrieval": retrieval,
    }

    if args.end_to_end and matched:
        print(f"\n=== [{tag}] End-to-end with reranker OFF ===")
        cfg_off = common.build_config(use_reranker=False, **kwargs)
        t0 = time.perf_counter()
        off = common.run_end_to_end(cfg_off, matched, queries,
                                    use_judge=not args.no_judge, inject_name=args.inject_name)
        print(f"  [{tag}] reranker OFF done in {time.perf_counter() - t0:.1f}s")

        print(f"\n=== [{tag}] End-to-end with reranker ON ===")
        t0 = time.perf_counter()
        on = common.run_end_to_end(cfg_rr, matched, queries,
                                   use_judge=not args.no_judge, inject_name=args.inject_name)
        print(f"  [{tag}] reranker ON done in {time.perf_counter() - t0:.1f}s")
        payload["end_to_end"] = {"reranker_off": off, "reranker_on": on}

    return payload


def main() -> None:
    import rag_evaluation
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    tags = resolve_tags(args.model, args.models)

    matched, queries = None, None
    if args.end_to_end:
        matched = common.discover(only=args.only, limit=args.limit)
        queries = common.load_queries(args.queries)
        if not matched:
            print("[warn] no docs with ground truth; skipping end-to-end (retrieval metrics only).")

    models = []
    for tag in tags:
        try:
            models.append(run_model(tag, args, matched, queries))
            # checkpoint after each model so a long run is resumable-by-hand
            common.write_json(os.path.join(OUT_DIR, "metrics.json"),
                              {"task": "reranker_impact", "models": models})
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] FAILED: {exc}")
            import traceback
            traceback.print_exc()

    if not models:
        raise SystemExit("[error] no models produced results.")

    reranker_model = models[0].get("reranker_model")
    common.write_json(os.path.join(OUT_DIR, "metrics.json"),
                      {"task": "reranker_impact", "models": models})
    common.save_report(os.path.join(OUT_DIR, "report.md"),
                       build_report(models, reranker_model))
    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
