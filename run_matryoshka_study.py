"""Matryoshka embedding-dimension study orchestrator.

Runs a full end-to-end RAG evaluation (build -> retrieve -> rerank -> hierarchical
LLM -> judge) for text-embedding-3-small and text-embedding-3-large at several
truncated (Matryoshka) dimensions, then scores MTEB-style retrieval metrics for
each run against the graded gold set. Every configuration writes its own summary
JSON, and the run is resumable (finished configs are skipped unless --force).

All non-embedding pipeline settings are LOCKED to match the baseline batch run so
the only variable is the embedding (model, dimension) pair.

Run from the repo root:

    python run_matryoshka_study.py                 # full 10-config study
    python run_matryoshka_study.py --only PR2610   # subset of docs (smoke test)
    python run_matryoshka_study.py --force         # re-run configs already done
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches  # noqa: E402

apply_ssl_patches()

import evaluate_all  # noqa: E402
import rag_evaluation  # noqa: E402
from rag.config import RagConfig  # noqa: E402
from rag.evaluation.retrieval_metrics import score_matryoshka_retrieval  # noqa: E402
from rag.inference.batch_runner import BatchRunner, summarize  # noqa: E402

# (model, dimensions) grid. Full size = 1536 (small) / 3072 (large).
STUDY_MATRIX: List[Tuple[str, int]] = [
    ("text-embedding-3-small", 256),
    ("text-embedding-3-small", 512),
    ("text-embedding-3-small", 1024),
    ("text-embedding-3-small", 1536),
    ("text-embedding-3-large", 256),
    ("text-embedding-3-large", 512),
    ("text-embedding-3-large", 1024),
    ("text-embedding-3-large", 1536),
    ("text-embedding-3-large", 2048),
    ("text-embedding-3-large", 3072),
]

DEFAULT_OUTPUT_DIR = os.path.join("output_table", "matryoshka")
DEFAULT_GOLDSET = os.path.join("output_table", "retrieval_goldset.json")


def _short_model(model: str) -> str:
    if model.endswith("large"):
        return "large"
    if model.endswith("small"):
        return "small"
    return model.replace("text-embedding-", "").replace("/", "_")


def _run_tag(model: str, dim: int) -> str:
    return f"{_short_model(model)}_d{dim}"


def build_config(model: str, dim: int) -> RagConfig:
    """A RagConfig with every non-embedding setting locked to the baseline run."""
    cfg = RagConfig.from_env()
    cfg.embedding_model = "openai"
    cfg.openai_embedding_model = model
    cfg.openai_embedding_dimensions = dim
    cfg.chunk_strategy = "section"
    cfg.retrieval_mode = "hybrid"
    cfg.hybrid_alpha = 0.85
    cfg.use_reranker = True
    cfg.context_mode = "hierarchical"
    cfg.hierarchical_max_depth = 5
    return cfg


def run_one(
    model: str,
    dim: int,
    matched: List[dict],
    queries: List[dict],
    goldset_path: str,
    output_dir: str,
    use_judge: bool,
    inject_name: bool,
) -> dict:
    """Build + evaluate + MTEB-score one (model, dimension) configuration."""
    cfg = build_config(model, dim)
    tag = _run_tag(model, dim)
    print("\n" + "#" * 78)
    print(f"# CONFIG {tag}: {model} @ {dim} dims  (collection={cfg.chroma_collection})")
    print("#" * 78)

    runner = BatchRunner(cfg, use_judge=use_judge, inject_name=inject_name)

    t_build = time.perf_counter()
    runner.build_all(matched)
    print(f"[{tag}] build phase done in {time.perf_counter() - t_build:.1f}s")

    t_run = time.perf_counter()
    results = runner.run(matched, queries)
    print(f"[{tag}] inference phase done in {time.perf_counter() - t_run:.1f}s")

    summary = summarize(results, embed_tokens=runner.embed_tokens,
                        embed_cost_usd=runner.embed_cost_usd,
                        llm_meta=runner.llm_usage_snapshot())

    print(f"[{tag}] scoring MTEB retrieval metrics against gold set ...")
    mteb = score_matryoshka_retrieval(cfg, goldset_path, queries_path="queries.txt")

    run_config = {
        "embedding_model": cfg.embedding_model,
        "openai_embedding_model": cfg.openai_embedding_model,
        "openai_embedding_dimensions": cfg.openai_embedding_dimensions,
        "retrieval_mode": cfg.retrieval_mode,
        "hybrid_alpha": cfg.hybrid_alpha,
        "use_reranker": cfg.use_reranker,
        "reranker_model": cfg.reranker_model,
        "chunk_strategy": cfg.chunk_strategy,
        "context_mode": cfg.context_mode,
        "hierarchical_max_depth": cfg.hierarchical_max_depth,
        "top_k": cfg.top_k,
        "rerank_top_k": cfg.rerank_top_k,
        "inject_name": inject_name,
    }

    out = {
        "study": "matryoshka",
        "tag": tag,
        "config_collection": cfg.chroma_collection,
        "run_config": run_config,
        "summary": summary,
        "mteb_metrics": mteb,
    }
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"summary_{tag}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[{tag}] saved -> {out_path}")

    s = summary
    print(f"[{tag}] LLM acc={_pct(s['combined_llm_accuracy'])}  "
          f"retrieval acc={_pct(s['combined_retrieval_accuracy'])}  "
          f"cost=${s['cost_usd']['total']:.2f}  "
          f"pre_rerank nDCG@10={mteb['pre_rerank']['macro'].get('ndcg@10', 0):.3f}")
    return out


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.1f}%" if v is not None else "N/A"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Matryoshka embedding-dimension study runner.")
    p.add_argument("--docs-dir", default=evaluate_all.DEFAULT_DOCS_DIR)
    p.add_argument("--matched-dir", default=evaluate_all.DEFAULT_MATCHED_DIR)
    p.add_argument("--queries", default=evaluate_all.DEFAULT_QUERIES)
    p.add_argument("--only", nargs="+", default=None, help="Only these doc ids (e.g. PR2610)")
    p.add_argument("--goldset", default=DEFAULT_GOLDSET)
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--models", nargs="+", default=None,
                   help="Restrict to these OpenAI models (default: both)")
    p.add_argument("--dims", nargs="+", type=int, default=None,
                   help="Restrict to these dimensions (default: full grid per model)")
    p.add_argument("--force", action="store_true", help="Re-run configs whose summary already exists")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--inject-name", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-analysis", action="store_true", help="Do not run the analyzer at the end")
    return p.parse_args()


def main() -> None:
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    if not os.path.isfile(args.goldset):
        print(f"[error] Gold set not found at {args.goldset}.")
        print("        MTEB metrics need it. Build it first, e.g.:")
        print("        python build_retrieval_goldset.py --num-docs 40 --restart "
              "--concurrency 12 --batch-size 8")
        sys.exit(1)

    matrix = [
        (m, d) for (m, d) in STUDY_MATRIX
        if (args.models is None or m in args.models)
        and (args.dims is None or d in args.dims)
    ]
    if not matrix:
        print("[error] No (model, dimension) configs selected.")
        sys.exit(1)

    matched, skipped = evaluate_all.discover_docs(args.docs_dir, args.matched_dir, only=args.only)
    print(f"Discovered {len(matched)} doc(s)" + (f", skipped {len(skipped)}" if skipped else ""))
    if not matched:
        print("[error] No docs with matching ground truth.")
        sys.exit(1)
    queries = rag_evaluation.parse_queries_file(args.queries)

    print(f"Study matrix: {len(matrix)} config(s) x {len(matched)} doc(s) x {len(queries)} queries")

    t0 = time.perf_counter()
    done, skipped_existing, failed = 0, 0, 0
    for i, (model, dim) in enumerate(matrix, 1):
        tag = _run_tag(model, dim)
        out_path = os.path.join(args.output_dir, f"summary_{tag}.json")
        if os.path.isfile(out_path) and not args.force:
            print(f"[{i}/{len(matrix)}] {tag}: summary exists, skipping (use --force to rerun).")
            skipped_existing += 1
            continue
        print(f"\n[{i}/{len(matrix)}] starting {tag} "
              f"(elapsed {time.perf_counter() - t0:.0f}s so far) ...")
        try:
            run_one(model, dim, matched, queries, args.goldset, args.output_dir,
                    use_judge=not args.no_judge, inject_name=args.inject_name)
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[{i}/{len(matrix)}] {tag} FAILED: {exc}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 78)
    print(f"Study finished in {time.perf_counter() - t0:.0f}s: "
          f"{done} run, {skipped_existing} skipped (existing), {failed} failed.")
    print("=" * 78)

    if not args.skip_analysis:
        try:
            import analyze_matryoshka_study
            analyze_matryoshka_study.run(args.output_dir)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] analysis step failed: {exc}")


if __name__ == "__main__":
    main()
