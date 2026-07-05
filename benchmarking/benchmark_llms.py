"""Task 4 - Benchmark generator LLMs on the best embedding model.

Holds the pipeline fixed (best embedding model from Task 1 + reranker ON +
hierarchical context) and swaps only the generation/judge LLM across:

    gpt-5.2, gpt-5.1, gpt-5-nano, gpt-5-mini, gpt-5, gpt-4o-mini, gpt-4o

Each model is compared on final (LLM-judge) accuracy, token/retrieval accuracy,
cost (per-model pricing), context window, token usage and time. Models that the
SLB endpoint does not serve are detected by a cheap pre-flight call and skipped;
the report lists which ran and which were unavailable.

Run from the repo root:

    python -m benchmarking.benchmark_llms
    python -m benchmarking.benchmark_llms --models gpt-5 gpt-5-mini gpt-4o-mini
    python -m benchmarking.benchmark_llms --model openai_large   # force embedding model
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List

from benchmarking import common, model_specs
from benchmarking.benchmark_reranker import _kwargs_without_reranker, resolve_best_tag

OUT_DIR = os.path.join(common.OUTPUT_ROOT, "llms")


def _llm_available(cfg) -> bool:
    """Cheap pre-flight: can the endpoint actually serve ``cfg.llm_model``?"""
    from rag.inference.llm import get_llm

    try:
        llm = get_llm(cfg)
        if llm is None:
            return False
        resp = llm.invoke("ping")
        return resp is not None
    except Exception as exc:  # noqa: BLE001
        print(f"  [unavailable] {cfg.llm_model}: {exc}")
        return False


def run_one(
    llm_model: str,
    base_kwargs: Dict[str, Any],
    embed_spec: str,
    matched: List[Dict[str, str]],
    queries: List[Dict[str, str]],
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
            cfg, matched, queries, use_judge=use_judge, inject_name=inject_name)
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
    parts = [f"# Task 4 - Generator LLM Benchmark (embedding: {model_tag}, reranker ON)\n",
             f"Models run: {len(ok)} of {len(results)}. Cost uses each model's list "
             "price (USD/1M tokens); accuracy is measured over scorable questions.\n"]

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

    # Token usage + price reference
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

    skipped = [r for r in results if r.get("status") != "ok"]
    if skipped:
        parts.append("## Unavailable / failed\n")
        for r in skipped:
            parts.append(f"- `{r['llm_model']}`: {r['status']}")
        parts.append("")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 4: benchmark generator LLMs on the best model.")
    p.add_argument("--model", default=None,
                   help="Embedding config tag (default: auto from Task 1)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Generator LLMs to benchmark (default: the 7-model sweep)")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--goldset", default=common.DEFAULT_GOLDSET)
    p.add_argument("--queries", default="queries.txt")
    p.add_argument("--preflight", action=argparse.BooleanOptionalAction, default=True,
                   help="Pre-flight each model with a tiny call to skip unavailable ones")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--inject-name", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    import rag_evaluation
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    model_tag = resolve_best_tag(args.model)
    base_kwargs, embed_spec = _kwargs_without_reranker(model_tag)

    matched = common.discover(only=args.only, limit=args.limit)
    if not matched:
        print("[error] No docs with matching ground truth.")
        return
    queries = common.load_queries(args.queries)

    llm_models = args.models or model_specs.DEFAULT_LLM_SWEEP
    print(f"Benchmarking {len(llm_models)} LLM(s) on embedding '{model_tag}' "
          f"x {len(matched)} docs x {len(queries)} queries")

    results = []
    for llm_model in llm_models:
        results.append(run_one(
            llm_model, base_kwargs, embed_spec, matched, queries,
            use_judge=not args.no_judge, inject_name=args.inject_name,
            preflight=args.preflight))

    common.write_json(os.path.join(OUT_DIR, "comparison.json"),
                      {"task": "llm_benchmark", "embedding_model_tag": model_tag,
                       "results": results})
    common.save_report(os.path.join(OUT_DIR, "report.md"), build_report(model_tag, results))
    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
