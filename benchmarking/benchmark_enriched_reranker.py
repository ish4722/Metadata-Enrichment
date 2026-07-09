"""Task 3 (enriched gold set) - Reranker impact (end-to-end only).

Quantifies what the cross-encoder reranker adds for each successful enriched
Task-1 embedding config, scored against the human-verified enriched gold set. For
every model it runs two end-to-end passes - reranker OFF then ON - and reports the
delta in LLM-judge / token / retrieval-hit accuracy. MTEB retrieval ranking
metrics (pre/post-rerank MRR, nDCG, ...) are NOT computed.

Run from the repo root:

    python -m benchmarking.benchmark_enriched_reranker                 # all models
    python -m benchmarking.benchmark_enriched_reranker --models openai_large openai_small
    python -m benchmarking.benchmark_enriched_reranker --model openai_large
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

from benchmarking import common, enriched_dataset
from benchmarking.benchmark_reranker import EMB_MAP, _kwargs_without_reranker, _pp_delta

OUT_DIR = os.path.join(enriched_dataset.OUTPUT_ROOT, "reranker")
EMB_COMPARISON = os.path.join(enriched_dataset.OUTPUT_ROOT, "embeddings", "comparison.json")

GT_EXTRACTOR = enriched_dataset.extract_enriched_ground_truth


def resolve_tags(explicit_single: Optional[str], explicit_multi: Optional[list]) -> list:
    """Embedding config tags to run: --models > --model > all successful enriched Task-1."""
    if explicit_multi:
        for t in explicit_multi:
            if t not in EMB_MAP:
                raise SystemExit(f"[error] unknown --models entry '{t}'. "
                                 f"Choose from: {', '.join(EMB_MAP)}")
        return list(explicit_multi)
    if explicit_single:
        if explicit_single not in EMB_MAP:
            raise SystemExit(f"[error] unknown --model '{explicit_single}'. "
                             f"Choose from: {', '.join(EMB_MAP)}")
        return [explicit_single]
    if not os.path.isfile(EMB_COMPARISON):
        raise SystemExit(
            f"[error] {EMB_COMPARISON} not found. Run the enriched embedding "
            "benchmark first (python -m benchmarking.benchmark_enriched_embeddings) "
            "or pass --model/--models.")
    data = json.load(open(EMB_COMPARISON, encoding="utf-8"))
    ok_tags = {r["tag"] for r in data.get("results", [])
               if r.get("status") == "ok" and r.get("tag") in EMB_MAP}
    tags = [t for t in EMB_MAP if t in ok_tags]  # preserve CONFIG order
    if not tags:
        raise SystemExit("[error] no successful enriched Task-1 config found; pass --model/--models.")
    print(f"Running reranker impact for {len(tags)} model(s): {', '.join(tags)}")
    return tags


def run_model(tag: str, args: argparse.Namespace, docs: list) -> Dict[str, Any]:
    """End-to-end accuracy with the reranker OFF vs ON for one embedding config."""
    kwargs, _embed_spec = _kwargs_without_reranker(tag)

    print("\n" + "#" * 78)
    print(f"# RERANKER IMPACT for {tag}")
    print("#" * 78)

    cfg_off = common.build_config(use_reranker=False, **kwargs)
    print(f"\n=== [{tag}] End-to-end with reranker OFF ===")
    t0 = time.perf_counter()
    off = common.run_end_to_end(cfg_off, docs, [], use_judge=not args.no_judge,
                                inject_name=args.inject_name, gt_extractor=GT_EXTRACTOR)
    print(f"  [{tag}] reranker OFF done in {time.perf_counter() - t0:.1f}s")

    cfg_on = common.build_config(use_reranker=True, **kwargs)
    print(f"\n=== [{tag}] End-to-end with reranker ON ===")
    t0 = time.perf_counter()
    on = common.run_end_to_end(cfg_on, docs, [], use_judge=not args.no_judge,
                               inject_name=args.inject_name, gt_extractor=GT_EXTRACTOR)
    print(f"  [{tag}] reranker ON done in {time.perf_counter() - t0:.1f}s")

    return {
        "model_tag": tag,
        "collection": cfg_on.chroma_collection,
        "reranker_model": cfg_on.reranker_model,
        "llm_model": cfg_on.llm_model,
        "end_to_end": {"reranker_off": off, "reranker_on": on},
    }


def build_report(models: list, reranker_model: str) -> str:
    parts = ["# Task 3 (enriched) - Reranker Impact (end-to-end only)\n",
             f"Reranker model: `{reranker_model}`. Scored against the enriched gold "
             "set; accuracy is end-to-end (no MTEB retrieval ranking metrics). "
             "Models compared: "
             + ", ".join(f"`{m['model_tag']}`" for m in models) + ".\n"]

    srows = []
    for m in models:
        off = m["end_to_end"]["reranker_off"]
        on = m["end_to_end"]["reranker_on"]
        off_acc = off.get("combined_llm_accuracy")
        on_acc = on.get("combined_llm_accuracy")
        srows.append([
            m["model_tag"],
            common.pct(off_acc), common.pct(on_acc), _pp_delta(off_acc, on_acc),
            common.pct(off.get("combined_token_accuracy")),
            common.pct(on.get("combined_token_accuracy")),
            common.pct(off.get("combined_retrieval_accuracy")),
            common.pct(on.get("combined_retrieval_accuracy")),
        ])
    parts.append("## Summary across models\n")
    parts.append(common.md_table(
        ["Model", "LLM acc OFF", "LLM acc ON", "Δ LLM acc",
         "Token OFF", "Token ON", "Retr OFF", "Retr ON"], srows))
    parts.append("")

    for m in models:
        tag = m["model_tag"]
        off, on = m["end_to_end"]["reranker_off"], m["end_to_end"]["reranker_on"]
        parts.append(f"## {tag}\n")
        rows = [
            ["LLM (judge) accuracy",
             common.pct(off.get("combined_llm_accuracy")),
             common.pct(on.get("combined_llm_accuracy")),
             _pp_delta(off.get("combined_llm_accuracy"), on.get("combined_llm_accuracy"))],
            ["Token-match accuracy",
             common.pct(off.get("combined_token_accuracy")),
             common.pct(on.get("combined_token_accuracy")),
             _pp_delta(off.get("combined_token_accuracy"), on.get("combined_token_accuracy"))],
            ["Retrieval accuracy",
             common.pct(off.get("combined_retrieval_accuracy")),
             common.pct(on.get("combined_retrieval_accuracy")),
             _pp_delta(off.get("combined_retrieval_accuracy"), on.get("combined_retrieval_accuracy"))],
        ]
        parts.append("### End-to-end accuracy: reranker OFF vs ON\n")
        parts.append(common.md_table(["Metric", "Reranker OFF", "Reranker ON", "Delta"], rows))
        parts.append("")
        parts.append("### LLM call metadata (OpenAI callback)\n")
        parts.append(common.llm_meta_table(
            [("Reranker OFF", off, m.get("llm_model")),
             ("Reranker ON", on, m.get("llm_model"))],
            first_col="Arm"))
        parts.append("")
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 3 (enriched): reranker impact, end-to-end only.")
    p.add_argument("--models", nargs="+", default=None,
                   help=f"Embedding config tags to run. Choices: {', '.join(EMB_MAP)}")
    p.add_argument("--model", default=None,
                   help=f"Single embedding config tag. Choices: {', '.join(EMB_MAP)}")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--queries", default="queries.txt")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--inject-name", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    import rag_evaluation
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    tags = resolve_tags(args.model, args.models)

    queries = common.load_queries(args.queries)
    docs = enriched_dataset.discover_enriched(only=args.only, limit=args.limit, queries=queries)
    if not docs:
        raise SystemExit("[error] No enriched docs with matching datalab output.")

    models: List[Dict[str, Any]] = []
    for tag in tags:
        try:
            models.append(run_model(tag, args, docs))
            common.write_json(os.path.join(OUT_DIR, "metrics.json"),
                              {"task": "reranker_impact_enriched", "models": models})
        except Exception as exc:  # noqa: BLE001
            print(f"[{tag}] FAILED: {exc}")
            import traceback
            traceback.print_exc()

    if not models:
        raise SystemExit("[error] no models produced results.")

    reranker_model = models[0].get("reranker_model")
    common.write_json(os.path.join(OUT_DIR, "metrics.json"),
                      {"task": "reranker_impact_enriched", "models": models})
    common.save_report(os.path.join(OUT_DIR, "report.md"),
                       build_report(models, reranker_model))
    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
