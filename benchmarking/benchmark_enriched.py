"""Orchestrator for the enriched-gold-set benchmark suite.

Runs all four enriched benchmark tasks in dependency order:

    embeddings  ->  matryoshka + reranker  ->  llms

The embedding benchmark must run first because reranker and LLM tasks auto-pick
their embedding config from ``benchmarking_enriched/embeddings/comparison.json``.
Every task is scored against the human-verified ``enriched_ground_truth/`` files
and omits MTEB retrieval ranking metrics.

Run from the repo root:

    python -m benchmarking.benchmark_enriched
    python -m benchmarking.benchmark_enriched --only PR3403 PR3693
    python -m benchmarking.benchmark_enriched --embeddings-only
    python -m benchmarking.benchmark_enriched --matryoshka-only --dims 512 1024
"""

from __future__ import annotations

import argparse
import sys

from benchmarking import (
    benchmark_enriched_context as context,
    benchmark_enriched_embeddings as embeddings,
    benchmark_enriched_llms as llms,
    benchmark_enriched_matryoshka as matryoshka,
    benchmark_enriched_reranker as reranker,
)


def _forward(args: argparse.Namespace, extra: "list[str]") -> "list[str]":
    """Rebuild a sys.argv-style list of shared flags for a sub-benchmark."""
    argv = ["prog"]
    if args.only:
        argv += ["--only", *args.only]
    if args.limit is not None:
        argv += ["--limit", str(args.limit)]
    if args.queries:
        argv += ["--queries", args.queries]
    if args.no_judge:
        argv += ["--no-judge"]
    if not args.inject_name:
        argv += ["--no-inject-name"]
    argv += extra
    return argv


def _run(module, args: argparse.Namespace, extra: "list[str]") -> None:
    saved = sys.argv
    sys.argv = _forward(args, extra)
    try:
        module.main()
    finally:
        sys.argv = saved


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the full enriched benchmark suite.")
    p.add_argument("--only", nargs="+", default=None, help="Only these doc ids")
    p.add_argument("--limit", type=int, default=None, help="Limit number of docs")
    p.add_argument("--queries", default="queries.txt")
    p.add_argument("--no-judge", action="store_true")
    p.add_argument("--inject-name", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--configs", nargs="+", default=None,
                   help="Embedding config tags for Task 1 (default: all four)")
    p.add_argument("--models", nargs="+", default=None,
                   help="Reranker embedding tags / LLM sweep, forwarded to those tasks")
    p.add_argument("--dims", nargs="+", type=int, default=None,
                   help="Matryoshka dimensions to restrict to")
    p.add_argument("--modes", nargs="+", default=None,
                   help="Context-expansion strategies for the context task "
                        "(chunk full_page hierarchical)")
    p.add_argument("--depth", type=int, default=None,
                   help="hierarchical_max_depth for the context task (default: 3)")

    p.add_argument("--embeddings-only", action="store_true")
    p.add_argument("--matryoshka-only", action="store_true")
    p.add_argument("--reranker-only", action="store_true")
    p.add_argument("--llms-only", action="store_true")
    p.add_argument("--context-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    only_flags = [args.embeddings_only, args.matryoshka_only, args.reranker_only,
                  args.llms_only, args.context_only]
    run_all = not any(only_flags)

    if run_all or args.embeddings_only:
        print("\n" + "=" * 78 + "\n= ENRICHED TASK 1: EMBEDDINGS\n" + "=" * 78)
        extra = ["--configs", *args.configs] if args.configs else []
        _run(embeddings, args, extra)

    if run_all or args.matryoshka_only:
        print("\n" + "=" * 78 + "\n= ENRICHED TASK 2: MATRYOSHKA\n" + "=" * 78)
        extra = ["--dims", *[str(d) for d in args.dims]] if args.dims else []
        _run(matryoshka, args, extra)

    if run_all or args.reranker_only:
        print("\n" + "=" * 78 + "\n= ENRICHED TASK 3: RERANKER\n" + "=" * 78)
        extra = ["--models", *args.models] if args.models else []
        _run(reranker, args, extra)

    if run_all or args.llms_only:
        print("\n" + "=" * 78 + "\n= ENRICHED TASK 4: LLMS\n" + "=" * 78)
        extra = ["--models", *args.models] if args.models else []
        _run(llms, args, extra)

    if run_all or args.context_only:
        print("\n" + "=" * 78 + "\n= ENRICHED TASK 5: CONTEXT EXPANSION\n" + "=" * 78)
        extra: "list[str]" = []
        if args.modes:
            extra += ["--modes", *args.modes]
        if args.depth is not None:
            extra += ["--depth", str(args.depth)]
        _run(context, args, extra)

    print("\nAll requested enriched benchmarks complete.")


if __name__ == "__main__":
    main()
