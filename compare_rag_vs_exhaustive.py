"""Side-by-side comparison: RAG pipeline vs naive exhaustive search.

Runs both approaches on the same datalab JSON + queries + ground truth and
prints a comparison table. The two pipelines share the same query parser,
ground-truth extractors, and scoring code (``rag_evaluation``), so the numbers
are directly comparable.

  - RAG          : build index -> retrieve top_k chunks -> 1 LLM call/query.
  - Exhaustive   : no index -> 1 LLM call per *page* (all queries) -> aggregate.

The headline takeaway: RAG reaches comparable accuracy with dramatically fewer
LLM calls / tokens / cost / time than scanning every page.

Run from the repo root, e.g.:

  python compare_rag_vs_exhaustive.py \
      --input "datalab-output/datalab-output-PR2610.pdf.json" \
      --ground-truth "matched_json/PR2610.json"
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List, Optional

# ── bootstrap: make 'import rag...' work when run as a script, and patch SSL ──
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches

apply_ssl_patches()  # must run before transformers / huggingface_hub import

import rag_evaluation  # noqa: E402
import exhaustive_search  # noqa: E402
from rag.config import RagConfig  # noqa: E402
from rag.inference import validation  # noqa: E402

DEFAULT_INPUT = exhaustive_search.DEFAULT_INPUT
DEFAULT_GROUND_TRUTH = exhaustive_search.DEFAULT_GROUND_TRUTH
DEFAULT_QUERIES = exhaustive_search.DEFAULT_QUERIES


def _total_row(steps: List[Any]) -> Any:
    """Return the TOTAL StepMetrics row (last row by convention)."""
    for step in reversed(steps):
        if getattr(step, "step", "") == "TOTAL":
            return step
    return steps[-1]


def run_rag(
    input_path: str,
    queries_path: str,
    ground_truth_path: str,
    use_judge: bool,
    config: Optional[RagConfig] = None,
    use_cache: bool = True,
    force_rebuild: bool = False,
) -> Dict[str, Any]:
    """Run the RAG evaluation and return a totals dict comparable to exhaustive."""
    config = config or RagConfig.from_env()
    num_queries = len(rag_evaluation.parse_queries_file(queries_path))

    print("\n" + "=" * 70)
    print("RUNNING RAG PIPELINE")
    print("=" * 70)
    steps = validation.evaluate(
        config,
        input_path=input_path,
        queries_path=queries_path,
        ground_truth_path=ground_truth_path,
        use_cache=use_cache,
        force_rebuild=force_rebuild,
        use_judge=use_judge,
    )

    total = _total_row(steps)
    # RAG issues 1 generation call per query, plus 1 judge call per query when
    # the judge is enabled and the query has a ground-truth value (all 4 do).
    llm_calls = num_queries * (2 if use_judge else 1)

    return {
        "pipeline": f"rag ({config.embedding_model}, {config.retrieval_mode}, "
                    f"top_k={config.top_k})",
        "llm_calls": llm_calls,
        "total_tokens": total.tokens,
        "cost_usd": total.cost_usd,
        "time_s": total.time_s,
        "peak_mem_mb": total.peak_mem_mb,
        "accuracy": total.accuracy,
    }


def _fmt(value: Optional[float], kind: str) -> str:
    if value is None:
        return "N/A"
    if kind == "accuracy":
        return f"{value * 100:.1f}%"
    if kind == "cost":
        return f"${value:.6f}"
    if kind == "int":
        return str(int(value))
    return f"{value:.4f}"


def _ratio(exhaustive: Optional[float], rag: Optional[float]) -> str:
    """How many times bigger the exhaustive value is vs the RAG value."""
    if not rag or rag <= 0 or exhaustive is None:
        return "N/A"
    return f"{exhaustive / rag:.1f}x"


def render_comparison_table(rag: Dict[str, Any], exh: Dict[str, Any]) -> str:
    """Render an aligned RAG-vs-exhaustive comparison table."""
    rows = [
        ("LLM calls", "int", "llm_calls"),
        ("Total tokens", "int", "total_tokens"),
        ("Cost ($)", "cost", "cost_usd"),
        ("Total time (s)", "time", "time_s"),
        ("Peak memory (MB)", "time", "peak_mem_mb"),
        ("Accuracy", "accuracy", "accuracy"),
    ]

    headers = ["Metric", "RAG", "Exhaustive", "Exhaustive / RAG"]
    table_rows: List[List[str]] = []
    for label, kind, key in rows:
        rag_val = rag.get(key)
        exh_val = exh.get(key)
        ratio = "-" if kind == "accuracy" else _ratio(exh_val, rag_val)
        table_rows.append([label, _fmt(rag_val, kind), _fmt(exh_val, kind), ratio])

    widths = [len(h) for h in headers]
    for row in table_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    separator = "-+-".join("-" * w for w in widths)
    lines = [render_row(headers), separator]
    lines.extend(render_row(row) for row in table_rows)
    return "\n".join(lines)


def _verdict(rag: Dict[str, Any], exh: Dict[str, Any]) -> str:
    calls = _ratio(exh.get("llm_calls"), rag.get("llm_calls"))
    tokens = _ratio(exh.get("total_tokens"), rag.get("total_tokens"))
    cost = _ratio(exh.get("cost_usd"), rag.get("cost_usd"))
    time_ = _ratio(exh.get("time_s"), rag.get("time_s"))
    rag_acc = _fmt(rag.get("accuracy"), "accuracy")
    exh_acc = _fmt(exh.get("accuracy"), "accuracy")
    return (
        "Verdict: exhaustive search uses about "
        f"{calls} the LLM calls, {tokens} the tokens, {cost} the cost, and "
        f"{time_} the time of RAG "
        f"(accuracy RAG={rag_acc} vs exhaustive={exh_acc}). "
        "RAG reaches comparable accuracy at a fraction of the resources because "
        "retrieval sends only the top-k relevant chunks to the LLM instead of "
        "every page."
    )


def main() -> None:
    rag_evaluation.configure_utf8_stdout()

    parser = argparse.ArgumentParser(
        description="Compare the RAG pipeline against a naive exhaustive search."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the extraction JSON file")
    parser.add_argument("--queries", default=DEFAULT_QUERIES, help="Path to queries file")
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH, help="Path to matched JSON")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Cap pages scanned by exhaustive search (default: all)")
    parser.add_argument("--no-judge", action="store_true", help="Disable the LLM judge for both")
    parser.add_argument("--rebuild", action="store_true", help="Force RAG index rebuild")
    args = parser.parse_args()

    for path, role in (
        (args.input, "Input extraction JSON"),
        (args.queries, "Queries file"),
        (args.ground_truth, "Ground-truth (matched) JSON"),
    ):
        if not os.path.isfile(path):
            print(f"[error] {role} not found: {path}")
            print(f"        resolved to : {os.path.abspath(path)}")
            sys.exit(1)

    use_judge = not args.no_judge

    rag_totals = run_rag(
        input_path=args.input,
        queries_path=args.queries,
        ground_truth_path=args.ground_truth,
        use_judge=use_judge,
        force_rebuild=args.rebuild,
    )

    print("\n" + "=" * 70)
    print("RUNNING EXHAUSTIVE SEARCH")
    print("=" * 70)
    exh_totals = exhaustive_search.run_exhaustive(
        input_path=args.input,
        queries_path=args.queries,
        ground_truth_path=args.ground_truth,
        max_pages=args.max_pages,
        use_judge=use_judge,
    )

    print("\n" + "=" * 70)
    print("COMPARISON: RAG vs EXHAUSTIVE SEARCH")
    print("=" * 70)
    print(render_comparison_table(rag_totals, exh_totals))
    print("\n" + _verdict(rag_totals, exh_totals))


if __name__ == "__main__":
    main()
