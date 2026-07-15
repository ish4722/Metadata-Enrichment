"""Task 4 (enriched gold set) - Benchmark generator LLMs on the best embedding.

Holds the pipeline fixed (best enriched Task-1 embedding + reranker ON +
hierarchical context) and swaps only the generation/judge LLM across the standard
sweep, scored against the human-verified enriched gold set. Models the SLB
endpoint does not serve are detected by a cheap pre-flight call and skipped. No
MTEB retrieval ranking metrics are computed.

Run from the repo root:

    python -m benchmarking.benchmark_enriched_llms
    python -m benchmarking.benchmark_enriched_llms --models gpt-5 gpt-5-mini gpt-4o-mini
    python -m benchmarking.benchmark_enriched_llms --model openai_large   # force embedding
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List

from rag.benchmarking import common, enriched_dataset, model_specs
from rag.benchmarking.benchmark_llms import _llm_available
from rag.benchmarking.benchmark_reranker import EMB_MAP, _kwargs_without_reranker

OUT_DIR = os.path.join(enriched_dataset.OUTPUT_ROOT, "llms")
EMB_COMPARISON = os.path.join(enriched_dataset.OUTPUT_ROOT, "embeddings", "comparison.json")

GT_EXTRACTOR = enriched_dataset.extract_enriched_ground_truth


def resolve_best_tag(explicit: str = None) -> str:
    """Best enriched Task-1 embedding config (by LLM accuracy), or an explicit tag."""
    if explicit:
        if explicit not in EMB_MAP:
            raise SystemExit(f"[error] unknown --model '{explicit}'. "
                             f"Choose from: {', '.join(EMB_MAP)}")
        return explicit
    if not os.path.isfile(EMB_COMPARISON):
        raise SystemExit(
            f"[error] {EMB_COMPARISON} not found. Run the enriched embedding "
            "benchmark first (python -m benchmarking.benchmark_enriched_embeddings) "
            "or pass --model.")
    data = json.load(open(EMB_COMPARISON, encoding="utf-8"))
    best_tag, best_acc = None, -1.0
    for r in data.get("results", []):
        if r.get("status") != "ok":
            continue
        acc = r.get("summary", {}).get("combined_llm_accuracy")
        if isinstance(acc, (int, float)) and acc > best_acc and r["tag"] in EMB_MAP:
            best_tag, best_acc = r["tag"], acc
    if best_tag is None:
        raise SystemExit("[error] no successful enriched Task-1 config found; pass --model.")
    print(f"Auto-selected best embedding model: {best_tag} (LLM acc={common.pct(best_acc)})")
    return best_tag


def run_one(
    llm_model: str,
    base_kwargs: Dict[str, Any],
    embed_spec: str,
    docs: List[Dict[str, Any]],
    use_judge: bool,
    inject_name: bool,
    preflight: bool,
) -> Dict[str, Any]:
    cfg = common.build_config(use_reranker=True, llm_model=llm_model, **base_kwargs)
    print("\n" + "#" * 78)
    print(f"# LLM {llm_model}  (embedding collection={cfg.chroma_collection})")
    print("#" * 78)

    if preflight and not _llm_available(cfg):
        return {"llm_model": llm_model, "status": "unavailable"}

    t0 = time.perf_counter()
    try:
        summary = common.run_end_to_end(
            cfg, docs, [], use_judge=use_judge, inject_name=inject_name,
            gt_extractor=GT_EXTRACTOR)
    except Exception as exc:  # noqa: BLE001
        print(f"[{llm_model}] FAILED during run: {exc}")
        import traceback
        traceback.print_exc()
        return {"llm_model": llm_model, "status": f"failed: {exc}"}

    cost = common.recompute_costs(summary, llm_model=llm_model, embed_spec_model=embed_spec)
    spec = model_specs.LLM_SPECS.get(llm_model, {})
    result = {
        "llm_model": llm_model,
        "status": "ok",
        "context_window": spec.get("context_window"),
        "input_price_per_1m": spec.get("input_per_1m"),
        "output_price_per_1m": spec.get("output_per_1m"),
        "summary": summary,
        "cost_recomputed": cost,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    common.write_json(os.path.join(OUT_DIR, f"summary_{llm_model}.json"), result)

    print(f"[{llm_model}] LLM acc={common.pct(summary.get('combined_llm_accuracy'))}  "
          f"token acc={common.pct(summary.get('combined_token_accuracy'))}  "
          f"cost={common.usd(cost.get('total'))}  ctx={common.num(spec.get('context_window'))}")
    return result


def build_report(model_tag: str, results: List[Dict[str, Any]]) -> str:
    ok = [r for r in results if r.get("status") == "ok"]
    parts = [f"# Task 4 (enriched) - Generator LLM Benchmark (embedding: {model_tag}, reranker ON)\n",
             f"Models run: {len(ok)} of {len(results)}. Scored against the enriched "
             "gold set. Cost uses each model's list price (USD/1M tokens); accuracy "
             "is measured over scorable questions.\n"]

    rows = []
    for r in ok:
        s = r["summary"]
        c = r["cost_recomputed"]
        acc = s.get("combined_llm_accuracy")
        total = c.get("total")
        acc_per_dollar = (acc / total) if isinstance(acc, (int, float)) and total else None
        rows.append([
            r["llm_model"],
            common.pct(acc),
            common.pct(s.get("combined_token_accuracy")),
            common.pct(s.get("combined_retrieval_accuracy")),
            common.usd(c.get("generation")),
            common.usd(c.get("judge")),
            common.usd(total),
            common.f3(acc_per_dollar),
            common.num(r.get("context_window")),
        ])
    parts.append("## Comparison\n")
    parts.append(common.md_table(
        ["LLM", "LLM acc", "Token acc", "Retr acc", "Gen $", "Judge $",
         "Total $", "LLM acc / $", "Ctx window"], rows))
    parts.append("")

    trows = []
    for r in ok:
        t = r["summary"].get("tokens", {})
        trows.append([
            r["llm_model"],
            common.num(t.get("gen_input")),
            common.num(t.get("gen_output")),
            common.num(t.get("judge_input")),
            common.num(t.get("judge_output")),
            common.usd(r.get("input_price_per_1m"), 2) + " / " + common.usd(r.get("output_price_per_1m"), 2),
        ])
    parts.append("## Token usage and unit price (per 1M in / out)\n")
    parts.append(common.md_table(
        ["LLM", "Gen in", "Gen out", "Judge in", "Judge out", "Price in/out"], trows))
    parts.append("")

    parts.append("## LLM call metadata (OpenAI callback)\n")
    parts.append(common.llm_meta_table(
        [(r["llm_model"], r["summary"], r["llm_model"]) for r in ok], first_col="LLM"))
    parts.append("")

    skipped = [r for r in results if r.get("status") != "ok"]
    if skipped:
        parts.append("## Unavailable / failed\n")
        for r in skipped:
            parts.append(f"- `{r['llm_model']}`: {r['status']}")
        parts.append("")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 4 (enriched): benchmark generator LLMs on the best model.")
    p.add_argument("--model", default=None,
                   help="Embedding config tag (default: auto from enriched Task 1)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Generator LLMs to benchmark (default: the standard sweep)")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--queries", default="queries.txt")
    p.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True,
                   help="Pre-flight each model with a tiny call to skip unavailable ones")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--inject-name", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    from rag.evaluation import core as rag_evaluation
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    model_tag = resolve_best_tag(args.model)
    base_kwargs, embed_spec = _kwargs_without_reranker(model_tag)

    queries = common.load_queries(args.queries)
    docs = enriched_dataset.discover_enriched(only=args.only, limit=args.limit, queries=queries)
    if not docs:
        print("[error] No enriched docs with matching datalab output.")
        return

    llm_models = args.models or model_specs.DEFAULT_LLM_SWEEP
    print(f"Benchmarking {len(llm_models)} LLM(s) on embedding '{model_tag}' "
          f"over {len(docs)} docs")

    results = []
    for llm_model in llm_models:
        results.append(run_one(
            llm_model, base_kwargs, embed_spec, docs,
            use_judge=not args.no_judge, inject_name=args.inject_name,
            preflight=args.preflight))

    common.write_json(os.path.join(OUT_DIR, "comparison.json"),
                      {"task": "llm_benchmark_enriched", "embedding_model": model_tag,
                       "reranker": True, "results": results})
    common.save_report(os.path.join(OUT_DIR, "report.md"), build_report(model_tag, results))
    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
