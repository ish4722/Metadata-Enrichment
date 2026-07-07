"""CLI: build a per-document queries file from an uploaded schema/record.

Run from the repo root::

    # From an OSDU record (e.g. a matched_json well) - one query per attribute
    python -m query_generation.from_schema matched_json/PR2610.json

    # Just list the attributes that would be turned into queries (no LLM calls)
    python -m query_generation.from_schema matched_json/PR2610.json --list

    # Cap cost / focus on a subset
    python -m query_generation.from_schema well.json --limit 15
    python -m query_generation.from_schema well.json --include "operator" "basin"
    python -m query_generation.from_schema well.json --exclude "id" --no-null

    # Custom name / destination
    python -m query_generation.from_schema well.json --name PR2610 --out-dir generated_queries

The resulting file plugs straight into the RAG eval::

    python eval_all_queries.py --queries generated_queries/PR2610.txt --skip-build --llm-workers 8
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches

apply_ssl_patches()  # before the internal LLM client is built

from rag.config import RagConfig  # noqa: E402
from rag.inference.llm import get_llm  # noqa: E402

from query_generation.batch import DEFAULT_OUTPUT_DIR, build_query_file_from_schema  # noqa: E402
from query_generation.schema_ingest import extract_attributes  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract attributes from an OSDU schema/record and generate a "
        "per-document queries file for the RAG pipeline."
    )
    p.add_argument("source", help="Path to a JSON schema, an OSDU record, or a list of attribute names")
    p.add_argument("--name", default=None, help="Document name (defaults to the source file stem)")
    p.add_argument("--out-dir", default=DEFAULT_OUTPUT_DIR, help="Output folder for <name>.txt")
    p.add_argument("--output", default=None, help="Full output path (overrides --out-dir/--name)")
    p.add_argument("--no-null", action="store_true", help="Skip attributes whose value is null/empty")
    p.add_argument("--include", nargs="+", default=None, help="Only labels containing any of these substrings")
    p.add_argument("--exclude", nargs="+", default=None, help="Drop labels containing any of these substrings")
    p.add_argument("--limit", type=int, default=None, help="Cap the number of attributes (after filtering)")
    p.add_argument("--list", action="store_true", help="Only list extracted attributes (no LLM calls)")
    return p.parse_args()


def main() -> None:
    import rag_evaluation

    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    source = args.source
    if not os.path.isabs(source) and os.path.exists(os.path.join(_REPO_ROOT, source)):
        source = os.path.join(_REPO_ROOT, source)

    # Dry run: just show what would be generated (cheap, no LLM).
    if args.list:
        attrs = extract_attributes(source, include_null=not args.no_null)
        print(f"{len(attrs)} attribute(s) from {args.source}:")
        for a in attrs:
            marker = "" if a.is_null is None else ("  (null)" if a.is_null else "  (filled)")
            alias = "" if a.label == a.raw_name else f"  -> {a.label}"
            print(f"  - {a.raw_name}{alias}{marker}")
        return

    config = RagConfig.from_env()
    llm = get_llm(config)
    if llm is None:
        raise SystemExit(
            "[error] No LLM available. Ensure langchain-openai + sauth_token are "
            "installed and the internal endpoint is reachable."
        )

    def _progress(i: int, total: int, label: str, error: Exception | None) -> None:
        status = "ok" if error is None else f"fail ({error})"
        print(f"  [{i}/{total}] {label} - {status}")

    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(_REPO_ROOT, args.out_dir)

    print(f"Extracting attributes from {args.source} ...")
    result = build_query_file_from_schema(
        source,
        llm=llm,
        config=config,
        name=args.name,
        output_dir=out_dir,
        output_path=args.output,
        include_null=not args.no_null,
        include=args.include,
        exclude=args.exclude,
        limit=args.limit,
        on_progress=_progress,
    )

    print("=" * 72)
    print(f"Document : {result.document_name}")
    print(f"Attributes attempted : {len(result.attributes)}")
    print(f"Queries generated    : {result.num_generated}")
    if result.failures:
        print(f"Failures ({len(result.failures)}):")
        for f in result.failures:
            print(f"  - {f}")
    print(f"Saved -> {result.output_path}")
    print("\nRun it through the RAG pipeline with:")
    rel = os.path.relpath(result.output_path, _REPO_ROOT)
    print(f"  python eval_all_queries.py --queries \"{rel}\" --skip-build --llm-workers 8")


if __name__ == "__main__":
    main()
