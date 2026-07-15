"""CLI for generating OSDU-attribute retrieval queries.

Run from the repo root::

    python -m query_generation "marker names"
    python -m query_generation "marker names" --preview           # don't write
    python -m query_generation "basin name" --overwrite           # replace block
    python -m query_generation "field name" "completion date"     # several at once
    python -m query_generation "marker names" --queries queries.txt
"""

from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches

apply_ssl_patches()  # must run before the internal LLM client is built

from rag.config import RagConfig  # noqa: E402
from rag.inference.llm import get_llm  # noqa: E402

from query_generation.generator import generate_query_for_attribute  # noqa: E402
from query_generation.queries_file import has_label, write_generated_query  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate a domain-aware retrieval query for an OSDU attribute "
        "and add it to queries.txt (marked with a trailing '*')."
    )
    p.add_argument("attributes", nargs="+", help="Attribute name(s), e.g. \"marker names\"")
    p.add_argument("--queries", default="queries.txt", help="Path to the queries file")
    p.add_argument("--preview", action="store_true", help="Generate but do NOT write the file")
    p.add_argument("--overwrite", action="store_true", help="Replace an existing generated block")
    return p.parse_args()


def main() -> None:
    from rag.evaluation import core as rag_evaluation

    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    config = RagConfig.from_env()
    llm = get_llm(config)
    if llm is None:
        raise SystemExit(
            "[error] No LLM available. Ensure langchain-openai + sauth_token are "
            "installed and the internal endpoint is reachable."
        )

    queries_path = args.queries
    if not os.path.isabs(queries_path):
        queries_path = os.path.join(_REPO_ROOT, queries_path)

    failures = 0
    added = 0
    skipped = 0
    for attribute in args.attributes:
        print("=" * 72)
        print(f"Attribute: {attribute}")
        try:
            generated = generate_query_for_attribute(attribute, llm=llm, config=config)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  [fail] {exc}")
            continue

        print(f"  OSDU:     {generated.osdu_interpretation or '(n/a)'}")
        if generated.synonyms:
            print(f"  Synonyms: {', '.join(generated.synonyms)}")
        print(f"  Label:    {generated.label}")
        print(f"  Query:    {generated.query}")
        if generated.keywords:
            print(f"  Keywords: {generated.keywords}")

        if args.preview:
            print("  [preview] not written")
            continue

        # Add only when this attribute's generated ('*') query is not already
        # present. Re-running is safe: existing entries are left untouched
        # unless --overwrite is given.
        if has_label(generated.label, queries_path) and not args.overwrite:
            skipped += 1
            print(f"  [skip] {generated.label} already in {os.path.basename(queries_path)} - not re-added")
            continue

        write_generated_query(generated, queries_path, overwrite=args.overwrite)
        added += 1
        print(f"  [ok] added to {queries_path}")

    print("=" * 72)
    print(f"Done. added={added}, skipped(existing)={skipped}, failed={failures}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
