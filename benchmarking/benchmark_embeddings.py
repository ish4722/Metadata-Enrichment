"""Task 1 - Benchmark embedding models (reranker OFF).

Compares four embedding configurations on the same corpus, queries and gold set:

    a) jina-api WITH late chunking      (collection: jina_api_latechunk)
    b) jina-api WITHOUT late chunking   (collection: jina_api)
    c) text-embedding-3-large (full)    (collection: openai_text-embedding-3-large)
    d) text-embedding-3-small (full)    (collection: openai_text-embedding-3-small)

The cross-encoder reranker is OFF for every config so we isolate the embedding.
Each config is compared on:
  * final (LLM-judge) accuracy, plus token-match and retrieval accuracy,
  * retrieval ranking metrics (MRR, Hit@k, Recall@k, nDCG@k, MAP) - embedding-only,
  * cost (per-model pricing; jina is self-hosted -> infra-only), and
  * context window.

Existing Chroma collections are reused (embed-once). Only ``jina_api`` (late
chunking off) has no vectors yet; it is built once unless ``--skip-missing``.

Run from the repo root:

    python -m benchmarking.benchmark_embeddings
    python -m benchmarking.benchmark_embeddings --only PR2610 PR2611
    python -m benchmarking.benchmark_embeddings --limit 10 --skip-missing
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

from benchmarking import common, model_specs

OUT_DIR = os.path.join(common.OUTPUT_ROOT, "embeddings")

# (tag, kwargs for build_config, embed-spec key for pricing/context window).
CONFIGS = [
    ("jina_latechunk",
     dict(embedding_model="jina-api", jina_late_chunking=True, use_reranker=False),
     "jina-api"),
    ("jina_nolatechunk",
     dict(embedding_model="jina-api", jina_late_chunking=False, use_reranker=False),
     "jina-api"),
    ("openai_large",
     dict(embedding_model="openai", openai_model="text-embedding-3-large",
          dimensions=None, use_reranker=False),
     "text-embedding-3-large"),
    ("openai_small",
     dict(embedding_model="openai", openai_model="text-embedding-3-small",
          dimensions=None, use_reranker=False),
     "text-embedding-3-small"),
]


def run_one(
    tag: str,
    cfg_kwargs: Dict[str, Any],
    embed_spec: str,
    matched: List[Dict[str, str]],
    queries: List[Dict[str, str]],
    goldset: str,
    use_judge: bool,
    inject_name: bool,
    skip_missing: bool,
) -> Dict[str, Any]:
    cfg = common.build_config(**cfg_kwargs)
    print("\n" + "#" * 78)
    print(f"# EMBEDDING CONFIG {tag}  (collection={cfg.chroma_collection}, reranker OFF)")
    print("#" * 78)

    n_vecs = common.collection_vector_count(cfg)
    if n_vecs == 0 and skip_missing:
        print(f"[skip] {tag}: collection '{cfg.chroma_collection}' is empty and "
              "--skip-missing is set.")
        return {"tag": tag, "status": "skipped_missing_collection",
                "collection": cfg.chroma_collection}
    if n_vecs == 0:
        print(f"[build] {tag}: collection '{cfg.chroma_collection}' is empty; "
              "embedding the corpus once (this is the only missing config).")

    t0 = time.perf_counter()
    summary = common.run_end_to_end(
        cfg, matched, queries, use_judge=use_judge, inject_name=inject_name)
    retrieval = common.run_retrieval_metrics(cfg, goldset)
    cost = common.recompute_costs(summary, llm_model=cfg.llm_model, embed_spec_model=embed_spec)

    result = {
        "tag": tag,
        "status": "ok",
        "embedding_model": cfg.embedding_model,
        "openai_embedding_model": cfg.openai_embedding_model if cfg.embedding_model == "openai" else None,
        "jina_late_chunking": cfg.jina_api_late_chunking if cfg.embedding_model == "jina-api" else None,
        "use_reranker": cfg.use_reranker,
        "collection": cfg.chroma_collection,
        "context_window": model_specs.embedding_context_window(embed_spec),
        "embed_price_per_1m": model_specs.EMBEDDING_SPECS.get(embed_spec, {}).get("price_per_1m"),
        "summary": summary,
        "retrieval_metrics": retrieval,
        "cost_recomputed": cost,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    common.write_json(os.path.join(OUT_DIR, f"summary_{tag}.json"), result)

    s = summary
    pre = retrieval.get("pre_rerank", {}).get("macro", {})
    print(f"[{tag}] LLM acc={common.pct(s.get('combined_llm_accuracy'))}  "
          f"retrieval acc={common.pct(s.get('combined_retrieval_accuracy'))}  "
          f"MRR={common.f3(pre.get('mrr'))}  nDCG@10={common.f3(pre.get('ndcg@10'))}  "
          f"total cost={common.usd(cost.get('total'))}")
    return result


def build_report(results: List[Dict[str, Any]]) -> str:
    ok = [r for r in results if r.get("status") == "ok"]
    parts = ["# Task 1 - Embedding Model Benchmark (reranker OFF)\n",
             f"Configs run: {len(ok)} of {len(results)}. "
             "Retrieval metrics are embedding-only (pre-rerank).\n"]

    # Accuracy + cost + context window
    rows = []
    for r in ok:
        s = r["summary"]
        c = r["cost_recomputed"]
        pre = r["retrieval_metrics"].get("pre_rerank", {}).get("macro", {})
        rows.append([
            r["tag"],
            common.pct(s.get("combined_llm_accuracy")),
            common.pct(s.get("combined_token_accuracy")),
            common.pct(s.get("combined_retrieval_accuracy")),
            common.f3(pre.get("mrr")),
            common.f3(pre.get("hit@1")),
            common.f3(pre.get("hit@5")),
            common.f3(pre.get("recall@5")),
            common.f3(pre.get("ndcg@10")),
            common.f3(pre.get("map")),
            common.usd(c.get("total")),
            common.num(r.get("context_window")),
        ])
    parts.append("## Comparison\n")
    parts.append(common.md_table(
        ["Config", "LLM acc", "Token acc", "Retr acc", "MRR", "Hit@1", "Hit@5",
         "Recall@5", "nDCG@10", "MAP", "Total cost", "Ctx window"], rows))
    parts.append("")

    # Cost breakdown
    crows = []
    for r in ok:
        c = r["cost_recomputed"]
        t = r["summary"].get("tokens", {})
        crows.append([
            r["tag"],
            common.usd(c.get("embedding")) if c.get("embedding") is not None else "N/A (self-hosted)",
            common.usd(c.get("generation")),
            common.usd(c.get("judge")),
            common.usd(c.get("total")),
            common.num(t.get("embed_input")),
            common.num((t.get("gen_input", 0) + t.get("gen_output", 0)
                        + t.get("judge_input", 0) + t.get("judge_output", 0))),
        ])
    parts.append("## Cost breakdown\n")
    parts.append(common.md_table(
        ["Config", "Embed $", "Generation $", "Judge $", "Total $",
         "Embed tokens", "LLM tokens"], crows))
    parts.append("")

    skipped = [r for r in results if r.get("status") != "ok"]
    if skipped:
        parts.append("## Skipped\n")
        for r in skipped:
            parts.append(f"- `{r['tag']}` ({r.get('collection')}): {r.get('status')}")
        parts.append("")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 1: benchmark embedding models (reranker off).")
    p.add_argument("--only", nargs="+", default=None, help="Only these doc ids")
    p.add_argument("--limit", type=int, default=None, help="Limit number of docs")
    p.add_argument("--goldset", default=common.DEFAULT_GOLDSET)
    p.add_argument("--queries", default="queries.txt")
    p.add_argument("--configs", nargs="+", default=None,
                   help="Restrict to these config tags "
                        "(jina_latechunk jina_nolatechunk openai_large openai_small)")
    p.add_argument("--skip-missing", action="store_true",
                   help="Skip configs whose Chroma collection is empty instead of building it")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--inject-name", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    import rag_evaluation
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    matched = common.discover(only=args.only, limit=args.limit)
    if not matched:
        print("[error] No docs with matching ground truth.")
        return
    queries = common.load_queries(args.queries)

    configs = CONFIGS
    if args.configs:
        wanted = set(args.configs)
        configs = [c for c in CONFIGS if c[0] in wanted]
    print(f"Running {len(configs)} embedding config(s) x {len(matched)} docs x {len(queries)} queries")

    results = []
    for tag, cfg_kwargs, embed_spec in configs:
        try:
            results.append(run_one(
                tag, cfg_kwargs, embed_spec, matched, queries, args.goldset,
                use_judge=not args.no_judge, inject_name=args.inject_name,
                skip_missing=args.skip_missing))
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] FAILED: {exc}")
            import traceback
            traceback.print_exc()
            results.append({"tag": tag, "status": f"failed: {exc}"})

    common.write_json(os.path.join(OUT_DIR, "comparison.json"),
                      {"task": "embedding_benchmark", "reranker": False, "results": results})
    common.save_report(os.path.join(OUT_DIR, "report.md"), build_report(results))
    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
