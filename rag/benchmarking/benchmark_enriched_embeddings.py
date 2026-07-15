"""Task 1 (enriched gold set) - Benchmark embedding models (reranker OFF).

Same four embedding configurations as :mod:`benchmarking.benchmark_embeddings`,
but scored against the human-verified ``enriched_ground_truth/`` files instead of
the OSDU ``matched_json/`` records. Only the attributes each enriched file
populated are queried (per document), and MTEB retrieval ranking metrics are NOT
computed - accuracy is purely end-to-end (LLM-judge / token / retrieval hit).

Run from the repo root:

    python -m benchmarking.benchmark_enriched_embeddings
    python -m benchmarking.benchmark_enriched_embeddings --only PR3403 PR3693
    python -m benchmarking.benchmark_enriched_embeddings --configs openai_large
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

from rag.benchmarking import common, enriched_dataset, model_specs
from rag.benchmarking.benchmark_embeddings import CONFIGS

OUT_DIR = os.path.join(enriched_dataset.OUTPUT_ROOT, "embeddings")


def run_one(
    tag: str,
    cfg_kwargs: Dict[str, Any],
    embed_spec: str,
    docs: List[Dict[str, Any]],
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
              "embedding the corpus once.")

    t0 = time.perf_counter()
    summary = common.run_end_to_end(
        cfg, docs, [], use_judge=use_judge, inject_name=inject_name,
        gt_extractor=enriched_dataset.extract_enriched_ground_truth)
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
        "cost_recomputed": cost,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    common.write_json(os.path.join(OUT_DIR, f"summary_{tag}.json"), result)

    s = summary
    print(f"[{tag}] LLM acc={common.pct(s.get('combined_llm_accuracy'))}  "
          f"token acc={common.pct(s.get('combined_token_accuracy'))}  "
          f"retrieval acc={common.pct(s.get('combined_retrieval_accuracy'))}  "
          f"total cost={common.usd(cost.get('total'))}")
    return result


def build_report(results: List[Dict[str, Any]]) -> str:
    ok = [r for r in results if r.get("status") == "ok"]
    parts = ["# Task 1 (enriched) - Embedding Model Benchmark (reranker OFF)\n",
             f"Configs run: {len(ok)} of {len(results)}. Scored against the "
             "human-verified enriched gold set; accuracy is end-to-end only "
             "(no MTEB retrieval ranking metrics).\n"]

    rows = []
    for r in ok:
        s = r["summary"]
        c = r["cost_recomputed"]
        rows.append([
            r["tag"],
            common.pct(s.get("combined_llm_accuracy")),
            common.pct(s.get("combined_token_accuracy")),
            common.pct(s.get("combined_retrieval_accuracy")),
            common.usd(c.get("total")),
            common.num(r.get("context_window")),
        ])
    parts.append("## Accuracy and cost\n")
    parts.append(common.md_table(
        ["Config", "LLM acc", "Token acc", "Retrieval acc", "Total cost", "Ctx window"], rows))
    parts.append("")

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

    parts.append("## LLM call metadata (OpenAI callback)\n")
    parts.append("Captured live from every generation/judge call. Cached and "
                 "reasoning tokens come straight from the OpenAI usage payload.\n")
    parts.append(common.llm_meta_table(
        [(r["tag"], r["summary"], r["cost_recomputed"].get("llm_model")) for r in ok]))
    parts.append("")

    skipped = [r for r in results if r.get("status") != "ok"]
    if skipped:
        parts.append("## Skipped\n")
        for r in skipped:
            parts.append(f"- `{r['tag']}` ({r.get('collection')}): {r.get('status')}")
        parts.append("")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 1 (enriched): benchmark embedding models (reranker off).")
    p.add_argument("--only", nargs="+", default=None, help="Only these doc ids")
    p.add_argument("--limit", type=int, default=None, help="Limit number of docs")
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
    from rag.evaluation import core as rag_evaluation
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    queries = common.load_queries(args.queries)
    docs = enriched_dataset.discover_enriched(only=args.only, limit=args.limit, queries=queries)
    if not docs:
        print("[error] No enriched docs with matching datalab output.")
        return

    common.write_json(os.path.join(enriched_dataset.OUTPUT_ROOT, "manifest.json"),
                      enriched_dataset.build_manifest(docs))

    configs = CONFIGS
    if args.configs:
        wanted = set(args.configs)
        configs = [c for c in CONFIGS if c[0] in wanted]
    total_tasks = sum(len(d["queries"]) for d in docs)
    print(f"Running {len(configs)} embedding config(s) over {len(docs)} docs "
          f"({total_tasks} query tasks each)")

    results = []
    for tag, cfg_kwargs, embed_spec in configs:
        try:
            results.append(run_one(
                tag, cfg_kwargs, embed_spec, docs,
                use_judge=not args.no_judge, inject_name=args.inject_name,
                skip_missing=args.skip_missing))
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] FAILED: {exc}")
            import traceback
            traceback.print_exc()
            results.append({"tag": tag, "status": f"failed: {exc}"})

    common.write_json(os.path.join(OUT_DIR, "comparison.json"),
                      {"task": "embedding_benchmark_enriched", "reranker": False, "results": results})
    common.save_report(os.path.join(OUT_DIR, "report.md"), build_report(results))
    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
