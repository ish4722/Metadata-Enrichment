from __future__ import annotations

import json
import os
from glob import glob
from typing import Any, Dict, List, Optional


def _fmt(value: Optional[float], kind: str) -> str:
    if value is None:
        return "N/A"
    if kind == "acc":
        return f"{value * 100:.1f}%"
    if kind == "cost":
        return f"{value:.6f}"
    if kind == "time":
        return f"{value:.2f}"
    if kind == "int":
        return str(int(value))
    return str(value)


def main() -> None:
    summaries = sorted(glob(os.path.join("output_table", "tune_PR*", "chunking_tuning_summary.json")))
    if not summaries:
        raise SystemExit("No summaries found under output_table/tune_PR*/chunking_tuning_summary.json")

    rows: List[Dict[str, Any]] = []
    for path in summaries:
        pr = os.path.basename(os.path.dirname(path)).replace("tune_", "")
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        best = payload.get("best") or {}
        rows.append(
            {
                "PR": pr,
                "best_chunk_size": best.get("chunk_size"),
                "best_overlap": best.get("overlap"),
                "llm_accuracy": best.get("llm_accuracy"),
                "retrieval_accuracy": best.get("retrieval_accuracy"),
                "time_s": best.get("time_s"),
                "cost_usd": best.get("cost_usd"),
                "tokens": best.get("total_tokens"),
                "mem_mb": best.get("peak_mem_mb"),
                "summary_path": path,
            }
        )

    print("PR | best(cs,ov) | LLM acc | Retr acc | time(s) | cost($) | tokens | mem(MB)")
    print("-" * 90)
    for r in rows:
        print(
            f"{r['PR']} | ({r['best_chunk_size']},{r['best_overlap']}) | "
            f"{_fmt(r['llm_accuracy'], 'acc')} | {_fmt(r['retrieval_accuracy'], 'acc')} | "
            f"{_fmt(r['time_s'], 'time')} | {_fmt(r['cost_usd'], 'cost')} | "
            f"{_fmt(r['tokens'], 'int')} | {_fmt(r['mem_mb'], 'time')}"
        )

    out_path = os.path.join("output_table", "tuning_best_10files.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()

