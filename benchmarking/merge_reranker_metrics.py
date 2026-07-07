"""Merge multiple reranker metrics.json checkpoints into one file.

Use when separate ``--models`` runs overwrote earlier results::

    python -m benchmarking.merge_reranker_metrics \\
        output_table/benchmarking/reranker/metrics_jina.json \\
        output_table/benchmarking/reranker/metrics_openai_large_only.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from benchmarking import common  # noqa: E402
from benchmarking.benchmark_embeddings import CONFIGS as EMB_CONFIGS  # noqa: E402
from benchmarking.benchmark_reranker import build_report  # noqa: E402

OUT_DIR = os.path.join(common.OUTPUT_ROOT, "reranker")
DEFAULT_OUT = os.path.join(OUT_DIR, "metrics.json")
TAG_ORDER = [tag for tag, _, _ in EMB_CONFIGS]


def merge_payloads(paths: List[str]) -> Dict[str, Any]:
    by_tag: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        data = json.load(open(path, encoding="utf-8"))
        for model in data.get("models", []):
            tag = model.get("model_tag")
            if tag:
                by_tag[tag] = model
    tags = [t for t in TAG_ORDER if t in by_tag]
    tags += sorted(t for t in by_tag if t not in tags)
    models = [by_tag[t] for t in tags]
    if not models:
        raise SystemExit("[error] no models found in inputs.")
    return {"task": "reranker_impact", "models": models}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Merge reranker metrics.json checkpoints.")
    p.add_argument("inputs", nargs="+", help="metrics.json files to merge (later files win on tag clash)")
    p.add_argument("-o", "--output", default=DEFAULT_OUT, help=f"Output path (default: {DEFAULT_OUT})")
    p.add_argument("--no-report", action="store_true", help="Skip regenerating report.md")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    payload = merge_payloads(args.inputs)
    common.write_json(args.output, payload)
    print(f"Merged {len(payload['models'])} model(s): {', '.join(m['model_tag'] for m in payload['models'])}")
    print(f"  -> {args.output}")
    if not args.no_report:
        reranker_model = payload["models"][0].get("reranker_model")
        report_path = os.path.join(OUT_DIR, "report.md")
        common.save_report(report_path, build_report(payload["models"], reranker_model))
        print(f"  -> {report_path}")


if __name__ == "__main__":
    main()
