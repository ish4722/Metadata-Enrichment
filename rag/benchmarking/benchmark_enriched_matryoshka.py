"""Task 2 (enriched gold set) - Matryoshka embedding-dimension study.

Runs the RAG pipeline end-to-end for text-embedding-3-small / large at several
truncated (Matryoshka) dimensions, scored against the human-verified enriched
gold set. Unlike :mod:`run_matryoshka_study`, this does NOT compute MTEB
retrieval ranking metrics - only end-to-end accuracy (LLM-judge / token /
retrieval hit), cost and LLM-call metadata, plotted against dimension.

Every non-embedding pipeline setting is locked to the baseline batch run so the
only variable is the (model, dimension) pair. Existing Chroma collections are
reused (embed-once); missing ones are built.

Run from the repo root:

    python -m benchmarking.benchmark_enriched_matryoshka
    python -m benchmarking.benchmark_enriched_matryoshka --models text-embedding-3-large --dims 256 512
    python -m benchmarking.benchmark_enriched_matryoshka --only PR3403
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

from rag.benchmarking import common, enriched_dataset
from rag.scripts.run_matryoshka_study import STUDY_MATRIX, _run_tag, build_config

OUT_DIR = os.path.join(enriched_dataset.OUTPUT_ROOT, "matryoshka")

GT_EXTRACTOR = enriched_dataset.extract_enriched_ground_truth


def run_one(
    model: str,
    dim: int,
    docs: List[Dict[str, Any]],
    use_judge: bool,
    inject_name: bool,
) -> Dict[str, Any]:
    cfg = build_config(model, dim)
    tag = _run_tag(model, dim)
    print("\n" + "#" * 78)
    print(f"# CONFIG {tag}: {model} @ {dim} dims  (collection={cfg.chroma_collection})")
    print("#" * 78)

    t0 = time.perf_counter()
    summary = common.run_end_to_end(cfg, docs, [], use_judge=use_judge,
                                    inject_name=inject_name, gt_extractor=GT_EXTRACTOR)
    embed_spec = model  # spec key in model_specs.EMBEDDING_SPECS
    cost = common.recompute_costs(summary, llm_model=cfg.llm_model, embed_spec_model=embed_spec)

    out = {
        "study": "matryoshka_enriched",
        "tag": tag,
        "status": "ok",
        "embedding_model": cfg.embedding_model,
        "openai_embedding_model": model,
        "openai_embedding_dimensions": dim,
        "use_reranker": cfg.use_reranker,
        "collection": cfg.chroma_collection,
        "summary": summary,
        "cost_recomputed": cost,
        "elapsed_s": round(time.perf_counter() - t0, 1),
    }
    common.write_json(os.path.join(OUT_DIR, f"summary_{tag}.json"), out)

    print(f"[{tag}] LLM acc={common.pct(summary.get('combined_llm_accuracy'))}  "
          f"token acc={common.pct(summary.get('combined_token_accuracy'))}  "
          f"retrieval acc={common.pct(summary.get('combined_retrieval_accuracy'))}  "
          f"cost={common.usd(cost.get('total'))}")
    return out


def _by_model(results: List[Dict[str, Any]]) -> "dict[str, List[Dict[str, Any]]]":
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        if r.get("status") == "ok":
            grouped[r["openai_embedding_model"]].append(r)
    for model in grouped:
        grouped[model].sort(key=lambda r: r["openai_embedding_dimensions"])
    return grouped


def build_aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    runs = []
    for r in results:
        if r.get("status") != "ok":
            continue
        s = r["summary"]
        runs.append({
            "tag": r["tag"],
            "model": r["openai_embedding_model"],
            "dim": r["openai_embedding_dimensions"],
            "llm_accuracy": s.get("combined_llm_accuracy"),
            "token_accuracy": s.get("combined_token_accuracy"),
            "retrieval_accuracy": s.get("combined_retrieval_accuracy"),
            "total_cost_usd": r["cost_recomputed"].get("total"),
        })
    return {"study": "matryoshka_enriched", "runs": runs}


def _retention(v: Optional[float], full: Optional[float]) -> str:
    if isinstance(v, (int, float)) and isinstance(full, (int, float)) and full:
        return f"{v / full * 100:.0f}%"
    return "N/A"


def build_report(results: List[Dict[str, Any]]) -> str:
    grouped = _by_model(results)
    parts = ["# Task 2 (enriched) - Matryoshka Dimension Study (end-to-end only)\n",
             "Scored against the human-verified enriched gold set. Accuracy is "
             "end-to-end (no MTEB retrieval ranking metrics). Retention is relative "
             "to each model's full dimension.\n"]

    parts.append("## Quality vs dimension\n")
    for model, runs in grouped.items():
        full = runs[-1]
        full_acc = full["summary"].get("combined_llm_accuracy")
        parts.append(f"### {model}\n")
        rows = []
        for r in runs:
            s = r["summary"]
            rows.append([
                str(r["openai_embedding_dimensions"]),
                common.pct(s.get("combined_llm_accuracy")),
                _retention(s.get("combined_llm_accuracy"), full_acc),
                common.pct(s.get("combined_token_accuracy")),
                common.pct(s.get("combined_retrieval_accuracy")),
                common.usd(r["cost_recomputed"].get("total")),
            ])
        parts.append(common.md_table(
            ["Dim", "LLM acc", "LLM retention", "Token acc", "Retrieval acc", "Total cost"], rows))
        parts.append("")

    parts.append("## LLM call metadata (OpenAI callback)\n")
    for model, runs in grouped.items():
        parts.append(f"### {model}\n")
        parts.append(common.llm_meta_table(
            [(str(r["openai_embedding_dimensions"]), r["summary"],
              r["cost_recomputed"].get("llm_model")) for r in runs],
            first_col="Dim"))
        parts.append("")

    skipped = [r for r in results if r.get("status") != "ok"]
    if skipped:
        parts.append("## Skipped / failed\n")
        for r in skipped:
            parts.append(f"- `{r.get('tag')}`: {r.get('status')}")
        parts.append("")
    return "\n".join(parts)


def _study_matrix(models: Optional[List[str]], dims: Optional[List[int]]):
    matrix = STUDY_MATRIX
    if models:
        matrix = [(m, d) for (m, d) in matrix if m in set(models)]
    if dims:
        matrix = [(m, d) for (m, d) in matrix if d in set(dims)]
    return matrix


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 2 (enriched): Matryoshka dimension study, end-to-end only.")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--queries", default="queries.txt")
    p.add_argument("--models", nargs="+", default=None,
                   help="Restrict OpenAI embedding models (default: both)")
    p.add_argument("--dims", nargs="+", type=int, default=None,
                   help="Restrict dimensions (default: full grid per model)")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--inject-name", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def main() -> None:
    from rag.evaluation import core as rag_evaluation
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    matrix = _study_matrix(args.models, args.dims)
    if not matrix:
        raise SystemExit("[error] no (model, dim) configs selected.")

    queries = common.load_queries(args.queries)
    docs = enriched_dataset.discover_enriched(only=args.only, limit=args.limit, queries=queries)
    if not docs:
        raise SystemExit("[error] No enriched docs with matching datalab output.")

    print(f"Running {len(matrix)} (model, dim) config(s) over {len(docs)} docs")

    results: List[Dict[str, Any]] = []
    for model, dim in matrix:
        try:
            results.append(run_one(model, dim, docs,
                                    use_judge=not args.no_judge, inject_name=args.inject_name))
        except Exception as exc:  # noqa: BLE001
            print(f"[{_run_tag(model, dim)}] FAILED: {exc}")
            import traceback
            traceback.print_exc()
            results.append({"tag": _run_tag(model, dim), "status": f"failed: {exc}"})

    common.write_json(os.path.join(OUT_DIR, "aggregate.json"), build_aggregate(results))
    common.save_report(os.path.join(OUT_DIR, "analysis_report.md"), build_report(results))
    print("\nDone. Outputs in", OUT_DIR)


if __name__ == "__main__":
    main()
