"""Hyperparameter tuning for the RAG chunking knobs: ``chunk_size`` & ``overlap``.

These are the two parameters in ``rag/config.py`` (lines 38-39):

    chunk_size: int = 800
    overlap: int = 200

This script sweeps every (chunk_size, overlap) combination by invoking the SAME
evaluation command the user runs by hand, only adding ``--chunk-size`` /
``--overlap`` / a per-combo ``--metrics-output``:

    python -m rag.main evaluate \
        --input "<datalab json>" \
        --queries queries.txt \
        --ground-truth "matched_json/PR2610.json" \
        --metrics-output output_table/<combo>.json \
        --chunk-size <cs> --overlap <ov>

After all runs it parses each metrics JSON, compiles one results table, and
reports the best combination (highest LLM accuracy, ties broken by lower total
time, then lower cost).

Note: combinations where ``overlap >= chunk_size`` are skipped (the chunking
sliding window step is ``chunk_size - overlap`` and would not advance).

Run from the repo root:

    python tune_chunking.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INPUT = r"datalab-output\datalab-output-PR2610.pdf.json"
DEFAULT_QUERIES = "queries.txt"
DEFAULT_GROUND_TRUTH = r"matched_json\PR2610.json"
DEFAULT_OUTPUT_DIR = "output_table"

DEFAULT_CHUNK_SIZES = [200, 400, 800, 1200]
DEFAULT_OVERLAPS = [0, 100, 200, 400]


# ---------------------------------------------------------------------------
# Running a single combination
# ---------------------------------------------------------------------------


def build_command(
    chunk_size: int,
    overlap: int,
    input_path: str,
    queries_path: str,
    ground_truth_path: str,
    metrics_output: str,
    no_judge: bool,
) -> List[str]:
    """Build the exact ``rag.main evaluate`` command for one combination."""
    cmd = [
        sys.executable, "-m", "rag.main", "evaluate",
        "--input", input_path,
        "--queries", queries_path,
        "--ground-truth", ground_truth_path,
        "--metrics-output", metrics_output,
        "--chunk-size", str(chunk_size),
        "--overlap", str(overlap),
    ]
    if no_judge:
        cmd.append("--no-judge")
    return cmd


def parse_metrics(path: str) -> Dict[str, Any]:
    """Pull the headline numbers out of a run's metrics JSON."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    steps = payload.get("steps", [])
    total = next((s for s in reversed(steps) if s.get("step") == "TOTAL"), None)
    retrieval = next((s for s in steps if "Retrieval" in s.get("step", "")), None)

    return {
        "retrieval_accuracy": retrieval.get("accuracy") if retrieval else None,
        "llm_accuracy": total.get("accuracy") if total else None,
        "total_tokens": total.get("tokens") if total else None,
        "cost_usd": total.get("cost_usd") if total else None,
        "time_s": total.get("time_s") if total else None,
        "peak_mem_mb": total.get("peak_mem_mb") if total else None,
    }


def run_combo(
    chunk_size: int,
    overlap: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run one (chunk_size, overlap) evaluation and return its parsed results."""
    metrics_output = os.path.join(
        args.output_dir, f"PR2610_cs{chunk_size}_ov{overlap}.json"
    )
    cmd = build_command(
        chunk_size, overlap, args.input, args.queries, args.ground_truth,
        metrics_output, args.no_judge,
    )

    print("\n" + "=" * 70)
    print(f"COMBO chunk_size={chunk_size}, overlap={overlap}")
    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("=" * 70)

    result: Dict[str, Any] = {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "metrics_output": metrics_output,
        "status": "ok",
    }

    proc = subprocess.run(cmd, cwd=_REPO_ROOT)
    if proc.returncode != 0:
        print(f"[warn] combo cs={chunk_size}, ov={overlap} exited {proc.returncode}")
        result["status"] = f"exit {proc.returncode}"
        return result

    if not os.path.isfile(metrics_output):
        print(f"[warn] metrics file not produced: {metrics_output}")
        result["status"] = "no metrics file"
        return result

    result.update(parse_metrics(metrics_output))
    print(f"  -> accuracy={_fmt(result.get('llm_accuracy'), 'accuracy')}, "
          f"time={_fmt(result.get('time_s'), 'time')}s, "
          f"cost={_fmt(result.get('cost_usd'), 'cost')}")
    return result


# ---------------------------------------------------------------------------
# Table rendering + ranking
# ---------------------------------------------------------------------------


def _fmt(value: Optional[float], kind: str) -> str:
    if value is None:
        return "N/A"
    if kind == "accuracy":
        return f"{value * 100:.1f}%"
    if kind == "cost":
        return f"{value:.6f}"
    if kind == "int":
        return str(int(value))
    return f"{value:.4f}"


def _rank_key(row: Dict[str, Any]) -> Tuple[float, float, float]:
    """Sort key: highest accuracy, then lowest time, then lowest cost."""
    acc = row.get("llm_accuracy")
    acc = acc if acc is not None else -1.0
    time_s = row.get("time_s")
    time_s = time_s if time_s is not None else float("inf")
    cost = row.get("cost_usd")
    cost = cost if cost is not None else float("inf")
    return (-acc, time_s, cost)


def render_results_table(rows: List[Dict[str, Any]], best: Optional[Dict[str, Any]]) -> str:
    headers = [
        "chunk_size", "overlap", "Retr. acc", "LLM acc",
        "Tokens", "Cost ($)", "Time (s)", "Mem (MB)", "Best",
    ]

    table_rows: List[List[str]] = []
    for row in rows:
        is_best = best is not None and row is best
        table_rows.append([
            str(row["chunk_size"]),
            str(row["overlap"]),
            _fmt(row.get("retrieval_accuracy"), "accuracy"),
            _fmt(row.get("llm_accuracy"), "accuracy"),
            _fmt(row.get("total_tokens"), "int"),
            _fmt(row.get("cost_usd"), "cost"),
            _fmt(row.get("time_s"), "time"),
            _fmt(row.get("peak_mem_mb"), "time"),
            "  <==" if is_best else ("" if row.get("status") == "ok" else row.get("status", "")),
        ])

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep chunk_size x overlap with the rag.main evaluate command."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=DEFAULT_CHUNK_SIZES)
    parser.add_argument("--overlaps", type=int, nargs="+", default=DEFAULT_OVERLAPS)
    parser.add_argument("--no-judge", action="store_true", help="Disable the LLM judge")
    parser.add_argument("--summary-output", default=None,
                        help="Path for the summary JSON (default: <output-dir>/chunking_tuning_summary.json)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    summary_output = args.summary_output or os.path.join(
        args.output_dir, "chunking_tuning_summary.json"
    )

    combos: List[Tuple[int, int]] = []
    skipped: List[Tuple[int, int]] = []
    for cs in args.chunk_sizes:
        for ov in args.overlaps:
            if ov >= cs:
                skipped.append((cs, ov))
            else:
                combos.append((cs, ov))

    print(f"Planned {len(combos)} combinations "
          f"(skipped {len(skipped)} where overlap >= chunk_size).")
    if skipped:
        print("  skipped:", ", ".join(f"(cs={c},ov={o})" for c, o in skipped))

    rows: List[Dict[str, Any]] = []
    for cs, ov in combos:
        rows.append(run_combo(cs, ov, args))

    ok_rows = [r for r in rows if r.get("status") == "ok" and r.get("llm_accuracy") is not None]
    rows_sorted = sorted(rows, key=_rank_key)
    best = min(ok_rows, key=_rank_key) if ok_rows else None

    print("\n" + "=" * 70)
    print("CHUNKING HYPERPARAMETER TUNING RESULTS (ranked best -> worst)")
    print("=" * 70)
    print(render_results_table(rows_sorted, best))

    if best is not None:
        print(f"\nBest combination: chunk_size={best['chunk_size']}, "
              f"overlap={best['overlap']} "
              f"(LLM accuracy={_fmt(best.get('llm_accuracy'), 'accuracy')}, "
              f"time={_fmt(best.get('time_s'), 'time')}s, "
              f"cost={_fmt(best.get('cost_usd'), 'cost')}, "
              f"tokens={_fmt(best.get('total_tokens'), 'int')}).")
        print("Ranking rule: highest LLM accuracy, ties broken by lower total time, then cost.")
    else:
        print("\nNo successful runs to rank.")

    payload = {
        "input": args.input,
        "queries": args.queries,
        "ground_truth": args.ground_truth,
        "chunk_sizes": args.chunk_sizes,
        "overlaps": args.overlaps,
        "skipped_combos": [{"chunk_size": c, "overlap": o} for c, o in skipped],
        "results": rows_sorted,
        "best": best,
    }
    with open(summary_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved tuning summary to {summary_output}")


if __name__ == "__main__":
    main()
