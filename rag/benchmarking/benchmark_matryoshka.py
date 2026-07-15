"""Task 2 - Matryoshka embedding-dimension study (reuse-first).

The truncated-dimension Chroma collections already exist (openai_text-embedding-3-
small/large at d256..d3072) and per-config summaries live in
``output_table/matryoshka/``. So by default this does NOT re-embed anything: it
re-scores the MTEB retrieval metrics (cheap query embeds only, reusing the
collections) and regenerates the analysis report.

  Default:   rescore_matryoshka_mteb.py  ->  analyze_matryoshka_study.py
  --rerun:   run_matryoshka_study.py (builds/evaluates any missing config, then
             analyzes). Forwards --only / --models / --dims / --force.

Run from the repo root:

    python -m benchmarking.benchmark_matryoshka
    python -m benchmarking.benchmark_matryoshka --rerun --models text-embedding-3-large --dims 256 512
    python -m benchmarking.benchmark_matryoshka --rerun --force
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OUTPUT_DIR = os.path.join("output_table", "matryoshka")
DEFAULT_GOLDSET = os.path.join("output_table", "retrieval_goldset.json")


def _run(cmd: list) -> int:
    printable = " ".join(f'"{c}"' if " " in c else c for c in cmd)
    print("\n$ " + printable)
    return subprocess.run(cmd, cwd=_REPO_ROOT).returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 2: Matryoshka dimension study (reuse-first).")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--goldset", default=DEFAULT_GOLDSET)
    p.add_argument("--rerun", action="store_true",
                   help="Re-run the full study (build/evaluate missing configs) instead of reuse-only")
    # Forwarded to run_matryoshka_study.py when --rerun is set.
    p.add_argument("--only", nargs="+", default=None, help="Only these doc ids (with --rerun)")
    p.add_argument("--models", nargs="+", default=None, help="Restrict OpenAI models (with --rerun)")
    p.add_argument("--dims", nargs="+", type=int, default=None, help="Restrict dimensions (with --rerun)")
    p.add_argument("--force", action="store_true", help="Re-run configs with existing summaries (with --rerun)")
    p.add_argument("--no-judge", action="store_true", help="(with --rerun) skip the LLM judge")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not os.path.isfile(os.path.join(_REPO_ROOT, args.goldset)):
        print(f"[error] gold set not found: {args.goldset}")
        sys.exit(1)

    if args.rerun:
        cmd = [sys.executable, "run_matryoshka_study.py",
               "--goldset", args.goldset, "--output-dir", args.output_dir]
        if args.only:
            cmd += ["--only", *args.only]
        if args.models:
            cmd += ["--models", *args.models]
        if args.dims:
            cmd += ["--dims", *[str(d) for d in args.dims]]
        if args.force:
            cmd.append("--force")
        if args.no_judge:
            cmd.append("--no-judge")
        # run_matryoshka_study runs the analyzer itself at the end.
        rc = _run(cmd)
        sys.exit(rc)

    # Reuse-only: rescore MTEB metrics against existing collections, then analyze.
    rc = _run([sys.executable, "rescore_matryoshka_mteb.py",
               "--goldset", args.goldset, "--output-dir", args.output_dir])
    if rc != 0:
        print("[warn] rescore step returned non-zero; continuing to analysis with existing metrics.")
    rc = _run([sys.executable, "analyze_matryoshka_study.py", "--output-dir", args.output_dir])
    print(f"\nDone. Reports in {args.output_dir} (aggregate.json, analysis_report.md).")
    sys.exit(rc)


if __name__ == "__main__":
    main()
