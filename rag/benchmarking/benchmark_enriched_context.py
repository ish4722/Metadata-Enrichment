"""Task 5 (enriched gold set) - Context-expansion strategy comparison.

Holds the pipeline fixed (best enriched Task-1 embedding, one reranker setting,
one generator LLM) and swaps only how retrieved matches are expanded into the
context sent to the LLM:

    1. ``chunk``          - send just the retrieved chunks.
    2. ``full_page``      - expand each match to every chunk on its page(s).
    3. ``hierarchical``   - climb the section hierarchy on demand, re-prompting
                            until the LLM can answer (depth capped, default 3).

Scored against the human-verified enriched gold set. Reports, side by side, the
accuracy (LLM-judge / token / retrieval), token usage, wall-clock time and the
number of LLM calls each strategy costs (hierarchical can call the LLM more than
once per query via escalation).

Run from the repo root:

    python -m benchmarking.benchmark_enriched_context
    python -m benchmarking.benchmark_enriched_context --model openai_large --reranker
    python -m benchmarking.benchmark_enriched_context --modes chunk hierarchical --depth 3
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Dict, List, Optional

from rag.benchmarking import common, enriched_dataset, model_specs
from rag.benchmarking.benchmark_enriched_llms import resolve_best_tag
from rag.benchmarking.benchmark_reranker import _kwargs_without_reranker

OUT_DIR = os.path.join(enriched_dataset.OUTPUT_ROOT, "context")

GT_EXTRACTOR = enriched_dataset.extract_enriched_ground_truth

# The context-expansion strategies under test. ``hierarchical`` also carries the
# depth cap; the other two ignore it.
ALL_MODES = ["chunk", "full_page", "hierarchical"]


def _strategy_label(mode: str, depth: int) -> str:
    return f"hierarchical_d{depth}" if mode == "hierarchical" else mode


def run_one(
    mode: str,
    depth: int,
    base_kwargs: Dict[str, Any],
    embed_spec: str,
    llm_model: str,
    use_reranker: bool,
    docs: List[Dict[str, Any]],
    use_judge: bool,
    inject_name: bool,
) -> Dict[str, Any]:
    label = _strategy_label(mode, depth)
    cfg = common.build_config(
        use_reranker=use_reranker,
        context_mode=mode,
        hierarchical_max_depth=depth,
        **base_kwargs,
    )
    print("\n" + "#" * 78)
    print(f"# CONTEXT STRATEGY {label}  (mode={mode}, reranker={use_reranker}, "
          f"collection={cfg.chroma_collection})")
    print("#" * 78)

    t0 = time.perf_counter()
    summary = common.run_end_to_end(cfg, docs, [], use_judge=use_judge,
                                    inject_name=inject_name, gt_extractor=GT_EXTRACTOR)
    cost = common.recompute_costs(summary, llm_model=cfg.llm_model, embed_spec_model=embed_spec)

    out = {
        "task": "context_strategy_enriched",
        "strategy": label,
        "context_mode": mode,
        "hierarchical_max_depth": depth if mode == "hierarchical" else None,
        "status": "ok",
        "use_reranker": use_reranker,
        "llm_model": cfg.llm_model,
        "collection": cfg.chroma_collection,
        "summary": summary,
        "cost_recomputed": cost,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    common.write_json(os.path.join(OUT_DIR, f"summary_{label}.json"), out)

    esc = summary.get("escalation", {})
    print(f"[{label}] LLM acc={common.pct(summary.get('combined_llm_accuracy'))}  "
          f"token acc={common.pct(summary.get('combined_token_accuracy'))}  "
          f"retr acc={common.pct(summary.get('combined_retrieval_accuracy'))}  "
          f"LLM calls={esc.get('total_llm_calls')}  cost={common.usd(cost.get('total'))}")
    return out


def _secs(v: Optional[float]) -> str:
    return f"{v:.1f}" if isinstance(v, (int, float)) else "N/A"


def _f2(v: Optional[float]) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"


def build_report(model_tag: str, llm_model: str, use_reranker: bool,
                 results: List[Dict[str, Any]]) -> str:
    ok = [r for r in results if r.get("status") == "ok"]
    parts = [
        "# Task 5 (enriched) - Context-Expansion Strategy Comparison\n",
        f"Embedding: `{model_tag}`  |  Generator LLM: `{llm_model}`  |  "
        f"Reranker: {'ON' if use_reranker else 'OFF'}. Scored against the "
        "human-verified enriched gold set. Only the context-expansion strategy "
        "varies; everything else is locked to the baseline pipeline.\n",
    ]

    # ---- Headline comparison (accuracy + token + time + calls) -------------
    rows = []
    for r in ok:
        s = r["summary"]
        tok = s.get("tokens", {})
        tm = s.get("time_s", {})
        esc = s.get("escalation", {})
        rows.append([
            r["strategy"],
            common.pct(s.get("combined_llm_accuracy")),
            common.pct(s.get("combined_token_accuracy")),
            common.pct(s.get("combined_retrieval_accuracy")),
            common.num(tok.get("total")),
            _secs(tm.get("total")),
            common.num(esc.get("total_llm_calls")),
            _f2(esc.get("avg_llm_calls_per_query")),
        ])
    parts.append("## Comparison (accuracy / tokens / time / LLM calls)\n")
    parts.append(common.md_table(
        ["Strategy", "LLM acc", "Token acc", "Retr acc", "Total tokens",
         "Total time (s)", "LLM calls", "Avg calls/query"], rows))
    parts.append("")

    # ---- Token breakdown --------------------------------------------------
    trows = []
    for r in ok:
        t = r["summary"].get("tokens", {})
        trows.append([
            r["strategy"],
            common.num(t.get("gen_input")),
            common.num(t.get("gen_output")),
            common.num(t.get("judge_input")),
            common.num(t.get("judge_output")),
            common.num(t.get("total")),
            common.usd(r["cost_recomputed"].get("total")),
        ])
    parts.append("## Token usage and cost\n")
    parts.append(common.md_table(
        ["Strategy", "Gen in", "Gen out", "Judge in", "Judge out",
         "Total tokens", "Total cost"], trows))
    parts.append("")

    # ---- Time breakdown ---------------------------------------------------
    wrows = []
    for r in ok:
        tm = r["summary"].get("time_s", {})
        wrows.append([
            r["strategy"],
            _secs(tm.get("retrieval")),
            _secs(tm.get("generation")),
            _secs(tm.get("judge")),
            _secs(tm.get("total")),
        ])
    parts.append("## Wall-clock time (seconds, summed across tasks)\n")
    parts.append(common.md_table(
        ["Strategy", "Retrieval", "Generation", "Judge", "Total"], wrows))
    parts.append("")

    # ---- Escalation / call count ------------------------------------------
    erows = []
    for r in ok:
        esc = r["summary"].get("escalation", {})
        erows.append([
            r["strategy"],
            common.num(esc.get("total_llm_calls")),
            _f2(esc.get("avg_llm_calls_per_query")),
            _f2(esc.get("avg_escalation_depth")),
        ])
    parts.append("## LLM call count (hierarchical escalation)\n")
    parts.append(common.md_table(
        ["Strategy", "Total LLM calls", "Avg calls/query", "Avg escalation depth"], erows))
    parts.append("")

    # ---- OpenAI callback metadata -----------------------------------------
    parts.append("## LLM call metadata (OpenAI callback)\n")
    parts.append(common.llm_meta_table(
        [(r["strategy"], r["summary"], r.get("llm_model")) for r in ok],
        first_col="Strategy"))
    parts.append("")

    skipped = [r for r in results if r.get("status") != "ok"]
    if skipped:
        parts.append("## Skipped / failed\n")
        for r in skipped:
            parts.append(f"- `{r.get('strategy')}`: {r.get('status')}")
        parts.append("")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Task 5 (enriched): compare context-expansion strategies, end-to-end only.")
    p.add_argument("--model", default=None,
                   help="Embedding config tag (default: auto from enriched Task 1)")
    p.add_argument("--llm-model", default=None,
                   help="Generator/judge LLM (default: pipeline default from RagConfig)")
    p.add_argument("--modes", nargs="+", default=ALL_MODES, choices=ALL_MODES,
                   help="Context-expansion strategies to compare (default: all three)")
    p.add_argument("--depth", type=int, default=3,
                   help="hierarchical_max_depth for the hierarchical strategy (default: 3)")
    p.add_argument("--reranker", dest="use_reranker", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Cross-encoder reranker ON/OFF for all strategies (default: OFF)")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--queries", default="queries.txt")
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
        raise SystemExit("[error] No enriched docs with matching datalab output.")

    # LLM model: explicit override, else whatever the baseline RagConfig uses.
    llm_model = args.llm_model or common.build_config(**base_kwargs).llm_model

    print(f"Comparing {len(args.modes)} context strategy(ies) {args.modes} on embedding "
          f"'{model_tag}' (LLM={llm_model}, reranker={'ON' if args.use_reranker else 'OFF'}) "
          f"over {len(docs)} docs")

    results: List[Dict[str, Any]] = []
    for mode in args.modes:
        try:
            results.append(run_one(
                mode, args.depth, base_kwargs, embed_spec, llm_model, args.use_reranker,
                docs, use_judge=not args.no_judge, inject_name=args.inject_name))
        except Exception as exc:  # noqa: BLE001
            label = _strategy_label(mode, args.depth)
            print(f"[{label}] FAILED: {exc}")
            import traceback
            traceback.print_exc()
            results.append({"strategy": label, "context_mode": mode, "status": f"failed: {exc}"})
        common.write_json(os.path.join(OUT_DIR, "comparison.json"), {
            "task": "context_strategy_enriched",
            "embedding_model": model_tag,
            "llm_model": llm_model,
            "use_reranker": args.use_reranker,
            "results": results,
        })

    common.save_report(os.path.join(OUT_DIR, "report.md"),
                       build_report(model_tag, llm_model, args.use_reranker, results))
    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
