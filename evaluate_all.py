"""Batch-evaluate every datalab doc and report ONE combined accuracy.

For each extraction JSON in ``datalab-output/`` this runs the exact same
command the user runs by hand:

    python -m rag.main evaluate \
        --input "<datalab json>" \
        --queries queries.txt \
        --ground-truth "matched_json/<PRid>.json" \
        --metrics-output output_table/batch/<PRid>.json

The ground-truth file is matched automatically by the doc's PR id (the datalab
filename ``datalab-output-PR2610.pdf.json`` maps to ``matched_json/PR2610.json``).

Afterwards every doc's per-query results are pooled into a single combined
accuracy: with N docs and 4 queries that is ``correct / (N * 4)`` questions
(e.g. 50 docs -> 200 questions). Per-label and per-doc breakdowns are printed
too, and the whole summary is saved to ``output_table/batch_summary.json``.

Run from the repo root:

    python evaluate_all.py
    python evaluate_all.py --only PR2610 PR2611      # subset
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_DOCS_DIR = "datalab-output"
DEFAULT_MATCHED_DIR = "matched_json"
DEFAULT_QUERIES = "queries.txt"
DEFAULT_OUTPUT_DIR = os.path.join("output_table", "batch")
DEFAULT_SUMMARY = os.path.join("output_table", "batch_summary.json")


# ---------------------------------------------------------------------------
# Step 1 - discovery + ground-truth matching
# ---------------------------------------------------------------------------


def derive_doc_id(filename: str) -> str:
    """Turn a datalab filename into its ground-truth id (e.g. 'PR2610').

    Handles the ``datalab-output-`` prefix, the ``.pdf``/``.json`` suffixes, and
    a trailing ``" (n)"`` copy marker, with a ``PR\\d+`` regex fallback.
    """
    stem = os.path.basename(filename)
    if stem.lower().endswith(".json"):
        stem = stem[:-5]
    stem = re.sub(r"\s*\(\d+\)\s*$", "", stem)          # trailing " (1)"
    if stem.lower().endswith(".pdf"):
        stem = stem[:-4]
    stem = re.sub(r"^datalab-output-", "", stem, flags=re.IGNORECASE)
    stem = stem.strip()

    match = re.search(r"PR\d+", stem)
    return match.group(0) if match else stem


def discover_docs(
    docs_dir: str,
    matched_dir: str,
    only: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    """Return (matched, skipped) doc descriptors.

    Each descriptor has ``id``, ``input`` and (for matched) ``ground_truth``.
    """
    matched: List[Dict[str, str]] = []
    skipped: List[Dict[str, str]] = []

    for path in sorted(glob.glob(os.path.join(docs_dir, "*.json"))):
        doc_id = derive_doc_id(path)
        if only and doc_id not in only:
            continue
        gt_path = os.path.join(matched_dir, f"{doc_id}.json")
        if os.path.isfile(gt_path):
            matched.append({"id": doc_id, "input": path, "ground_truth": gt_path})
        else:
            skipped.append({"id": doc_id, "input": path, "reason": f"no ground truth at {gt_path}"})

    return matched, skipped


# ---------------------------------------------------------------------------
# Step 2 - run the evaluate command per doc
# ---------------------------------------------------------------------------


def build_command(doc: Dict[str, str], args: argparse.Namespace, metrics_output: str) -> List[str]:
    """Assemble the exact ``rag.main evaluate`` command for one doc."""
    cmd = [
        sys.executable, "-m", "rag.main", "evaluate",
        "--input", doc["input"],
        "--queries", args.queries,
        "--ground-truth", doc["ground_truth"],
        "--metrics-output", metrics_output,
    ]
    if args.no_judge:
        cmd.append("--no-judge")
    if args.rebuild:
        cmd.append("--rebuild")
    if args.chunk_size is not None:
        cmd += ["--chunk-size", str(args.chunk_size)]
    if args.overlap is not None:
        cmd += ["--overlap", str(args.overlap)]
    if args.embedding_model is not None:
        cmd += ["--embedding-model", args.embedding_model]
    return cmd


def extract_doc_metrics(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pull whole-doc resource factors out of a run's per-step metrics.

    The single-doc evaluate writes a step list ending in a ``TOTAL`` row; the
    LLM-phase steps (generation + judge) are summed separately as the pure
    "inference" cost so we can report it distinct from build/retrieval.
    """
    steps = payload.get("steps", [])
    total = next((s for s in reversed(steps) if s.get("step") == "TOTAL"), None)
    llm_steps = [s for s in steps if "LLM" in s.get("step", "")]

    return {
        "total_tokens": (total or {}).get("tokens"),
        "total_time_s": (total or {}).get("time_s"),
        "peak_mem_mb": (total or {}).get("peak_mem_mb"),
        "cost_usd": (total or {}).get("cost_usd"),
        "inference_time_s": sum(s.get("time_s", 0.0) for s in llm_steps),
        "inference_tokens": sum(s.get("tokens", 0) for s in llm_steps),
    }


def run_doc(doc: Dict[str, str], args: argparse.Namespace) -> Dict[str, Any]:
    """Run one doc's evaluation and return its per-query rows + whole-doc metrics."""
    metrics_output = os.path.join(args.output_dir, f"{doc['id']}.json")
    cmd = build_command(doc, args, metrics_output)

    print("\n" + "=" * 70)
    print(f"DOC {doc['id']}")

    result: Dict[str, Any] = {
        "id": doc["id"],
        "input": doc["input"],
        "ground_truth": doc["ground_truth"],
        "metrics_output": metrics_output,
        "status": "ok",
        "per_query": [],
        "metrics": {},
    }

    # Reuse a previously produced metrics file instead of re-running the LLM.
    if args.reuse_existing and os.path.isfile(metrics_output):
        print(f"  [reuse] {metrics_output}")
        with open(metrics_output, "r", encoding="utf-8") as f:
            payload = json.load(f)
        result["per_query"] = payload.get("per_query", [])
        result["metrics"] = extract_doc_metrics(payload)
        return result

    print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    print("=" * 70)

    proc = subprocess.run(cmd, cwd=_REPO_ROOT)
    if proc.returncode != 0:
        print(f"[warn] {doc['id']} exited with code {proc.returncode}")
        result["status"] = f"exit {proc.returncode}"
        return result

    if not os.path.isfile(metrics_output):
        print(f"[warn] {doc['id']}: metrics file not produced ({metrics_output})")
        result["status"] = "no metrics file"
        return result

    with open(metrics_output, "r", encoding="utf-8") as f:
        payload = json.load(f)
    result["per_query"] = payload.get("per_query", [])
    result["metrics"] = extract_doc_metrics(payload)
    return result


# ---------------------------------------------------------------------------
# Step 3 - aggregation
# ---------------------------------------------------------------------------


def _is_hit(value: Any) -> bool:
    """A hit is strictly True; None / False / missing count as a miss."""
    return value is True


def _accuracy(correct: int, total: int) -> Optional[float]:
    return (correct / total) if total else None


def _fmt_acc(acc: Optional[float]) -> str:
    return f"{acc * 100:.1f}%" if acc is not None else "N/A"


def _fmt_int(value: Optional[float]) -> str:
    return f"{int(value):,}" if value is not None else "N/A"


def _fmt_f(value: Optional[float], decimals: int = 2) -> str:
    return f"{value:.{decimals}f}" if value is not None else "N/A"


def render_table(headers: List[str], rows: List[List[str]]) -> str:
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def render_row(cells: List[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    separator = "-+-".join("-" * w for w in widths)
    return "\n".join([render_row(headers), separator] + [render_row(r) for r in rows])


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pool every per-query row across docs into combined metrics."""
    pooled: List[Dict[str, Any]] = []
    per_doc: List[Dict[str, Any]] = []

    for res in results:
        rows = res.get("per_query", [])
        doc_total = len(rows)
        doc_llm = sum(1 for r in rows if _is_hit(r.get("llm_hit")))
        doc_token = sum(1 for r in rows if _is_hit(r.get("token_match")))
        doc_retr = sum(1 for r in rows if _is_hit(r.get("retrieval_hit")))
        m = res.get("metrics", {}) or {}
        per_doc.append({
            "id": res["id"],
            "status": res["status"],
            "questions": doc_total,
            "llm_correct": doc_llm,
            "llm_accuracy": _accuracy(doc_llm, doc_total),
            "token_accuracy": _accuracy(doc_token, doc_total),
            "retrieval_accuracy": _accuracy(doc_retr, doc_total),
            "total_tokens": m.get("total_tokens"),
            "total_time_s": m.get("total_time_s"),
            "inference_time_s": m.get("inference_time_s"),
            "peak_mem_mb": m.get("peak_mem_mb"),
            "cost_usd": m.get("cost_usd"),
        })
        for r in rows:
            pooled.append({**r, "doc_id": res["id"]})

    total = len(pooled)
    llm_correct = sum(1 for r in pooled if _is_hit(r.get("llm_hit")))
    token_correct = sum(1 for r in pooled if _is_hit(r.get("token_match")))
    retr_correct = sum(1 for r in pooled if _is_hit(r.get("retrieval_hit")))
    null_gt = sum(1 for r in pooled if r.get("ground_truth") in (None, "", []))

    # per-label breakdown
    by_label: Dict[str, Dict[str, int]] = {}
    for r in pooled:
        label = r.get("label", "?")
        bucket = by_label.setdefault(label, {"total": 0, "llm": 0, "token": 0, "retrieval": 0})
        bucket["total"] += 1
        bucket["llm"] += 1 if _is_hit(r.get("llm_hit")) else 0
        bucket["token"] += 1 if _is_hit(r.get("token_match")) else 0
        bucket["retrieval"] += 1 if _is_hit(r.get("retrieval_hit")) else 0

    per_label = [
        {
            "label": label,
            "total": b["total"],
            "llm_correct": b["llm"],
            "llm_accuracy": _accuracy(b["llm"], b["total"]),
            "token_accuracy": _accuracy(b["token"], b["total"]),
            "retrieval_accuracy": _accuracy(b["retrieval"], b["total"]),
        }
        for label, b in by_label.items()
    ]

    # Combined resource factors across docs: tokens / time / cost are summed,
    # peak memory is a max (it's a high-water mark, not additive).
    def _sum(key: str) -> float:
        return sum(d.get(key) or 0 for d in per_doc)

    peak_mems = [d["peak_mem_mb"] for d in per_doc if d.get("peak_mem_mb") is not None]

    resources = {
        "total_tokens": _sum("total_tokens"),
        "total_time_s": _sum("total_time_s"),
        "inference_time_s": _sum("inference_time_s"),
        "total_cost_usd": _sum("cost_usd"),
        "peak_mem_mb_max": max(peak_mems) if peak_mems else None,
        "avg_time_s_per_doc": (_sum("total_time_s") / len(per_doc)) if per_doc else None,
        "avg_tokens_per_doc": (_sum("total_tokens") / len(per_doc)) if per_doc else None,
    }

    return {
        "num_docs": len(results),
        "total_questions": total,
        "null_ground_truth": null_gt,
        "combined_llm_accuracy": _accuracy(llm_correct, total),
        "combined_token_accuracy": _accuracy(token_correct, total),
        "combined_retrieval_accuracy": _accuracy(retr_correct, total),
        "llm_correct": llm_correct,
        "token_correct": token_correct,
        "retrieval_correct": retr_correct,
        "resources": resources,
        "per_doc": per_doc,
        "per_label": per_label,
    }


def print_summary(agg: Dict[str, Any], skipped: List[Dict[str, str]]) -> None:
    print("\n" + "=" * 70)
    print("PER-DOC RESULTS")
    print("=" * 70)
    doc_rows = [
        [
            d["id"],
            str(d["questions"]),
            _fmt_acc(d["llm_accuracy"]),
            _fmt_acc(d["token_accuracy"]),
            _fmt_acc(d["retrieval_accuracy"]),
            _fmt_int(d.get("total_tokens")),
            _fmt_f(d.get("total_time_s")),
            _fmt_f(d.get("inference_time_s")),
            _fmt_f(d.get("peak_mem_mb")),
            _fmt_f(d.get("cost_usd"), 6),
        ]
        for d in agg["per_doc"]
    ]
    print(render_table(
        ["Doc", "Qs", "LLM acc", "Token acc", "Retr acc",
         "Tokens", "Time(s)", "Infer(s)", "Mem(MB)", "Cost($)"],
        doc_rows,
    ))

    print("\n" + "=" * 70)
    print("PER-QUERY-LABEL RESULTS (pooled across docs)")
    print("=" * 70)
    label_rows = [
        [
            lab["label"],
            f"{lab['llm_correct']}/{lab['total']}",
            _fmt_acc(lab["llm_accuracy"]),
            _fmt_acc(lab["token_accuracy"]),
            _fmt_acc(lab["retrieval_accuracy"]),
        ]
        for lab in agg["per_label"]
    ]
    print(render_table(
        ["Query label", "LLM hits", "LLM acc", "Token acc", "Retr acc"],
        label_rows,
    ))

    print("\n" + "=" * 70)
    print("COMBINED ACCURACY (all docs x all queries pooled)")
    print("=" * 70)
    print(render_table(
        ["Metric", "Correct / Total", "Accuracy"],
        [
            ["LLM (judge)", f"{agg['llm_correct']} / {agg['total_questions']}",
             _fmt_acc(agg["combined_llm_accuracy"])],
            ["Literal token match", f"{agg['token_correct']} / {agg['total_questions']}",
             _fmt_acc(agg["combined_token_accuracy"])],
            ["Retrieval hit", f"{agg['retrieval_correct']} / {agg['total_questions']}",
             _fmt_acc(agg["combined_retrieval_accuracy"])],
        ],
    ))
    res = agg.get("resources", {})
    print("\n" + "=" * 70)
    print("COMBINED RESOURCE FACTORS (whole-doc totals across all docs)")
    print("=" * 70)
    print(render_table(
        ["Factor", "Combined", "Avg / doc"],
        [
            ["Tokens (total)", _fmt_int(res.get("total_tokens")),
             _fmt_int(res.get("avg_tokens_per_doc"))],
            ["Total time (s)", _fmt_f(res.get("total_time_s")),
             _fmt_f(res.get("avg_time_s_per_doc"))],
            ["Inference time (s) [LLM+judge]", _fmt_f(res.get("inference_time_s")), "-"],
            ["Peak memory (MB) [max]", _fmt_f(res.get("peak_mem_mb_max")), "-"],
            ["Cost ($)", _fmt_f(res.get("total_cost_usd"), 6), "-"],
        ],
    ))

    print(f"\nDocs evaluated: {agg['num_docs']} | "
          f"Total questions: {agg['total_questions']} | "
          f"Questions with null ground truth (still counted): {agg['null_ground_truth']}")
    if skipped:
        print(f"Skipped {len(skipped)} doc(s) with no ground truth: "
              + ", ".join(s["id"] for s in skipped))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-evaluate all datalab docs and report one combined accuracy."
    )
    parser.add_argument("--docs-dir", default=DEFAULT_DOCS_DIR)
    parser.add_argument("--matched-dir", default=DEFAULT_MATCHED_DIR)
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Where per-doc metrics JSONs are written")
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY)
    parser.add_argument("--only", nargs="+", default=None,
                        help="Only evaluate these doc ids (e.g. PR2610 PR2611)")
    parser.add_argument("--reuse-existing", action="store_true",
                        help="Aggregate from existing per-doc metrics JSONs without re-running the LLM")
    # pass-through evaluate flags
    parser.add_argument("--no-judge", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    parser.add_argument("--embedding-model", default=None)
    return parser.parse_args()


def main() -> None:
    import rag_evaluation
    rag_evaluation.configure_utf8_stdout()

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    matched, skipped = discover_docs(args.docs_dir, args.matched_dir, only=args.only)
    print(f"Discovered {len(matched)} doc(s) with ground truth"
          + (f", skipped {len(skipped)} without" if skipped else "") + ".")
    if skipped:
        for s in skipped:
            print(f"  [skip] {s['id']}: {s['reason']}")
    if not matched:
        print("[error] No docs with matching ground truth were found.")
        sys.exit(1)

    results = [run_doc(doc, args) for doc in matched]
    agg = aggregate(results)
    print_summary(agg, skipped)

    payload = {
        "docs_dir": args.docs_dir,
        "matched_dir": args.matched_dir,
        "queries": args.queries,
        "skipped": skipped,
        "summary": agg,
        "per_doc_metrics": {r["id"]: r["metrics_output"] for r in results},
    }
    os.makedirs(os.path.dirname(args.summary_output) or ".", exist_ok=True)
    with open(args.summary_output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nSaved combined summary to {args.summary_output}")


if __name__ == "__main__":
    main()
