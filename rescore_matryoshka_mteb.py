"""Re-score MTEB retrieval metrics for existing Matryoshka summaries.

The original study ran with the cross-encoder reranker silently disabled (its
sentence-transformers loader failed on a network/SSL call), so ``post_rerank``
equalled ``pre_rerank``. The reranker now has an offline fallback, so this
re-scores retrieval against the gold set for every existing summary and patches
its ``mteb_metrics`` in place. It re-uses the already-built Chroma collections
(no re-embedding of the corpus; only cheap query embeds), so it is fast and does
not touch the end-to-end LLM results.

Run:
    python rescore_matryoshka_mteb.py
    python rescore_matryoshka_mteb.py --output-dir output_table/matryoshka
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches  # noqa: E402

apply_ssl_patches()

import rag_evaluation  # noqa: E402
from rag.config import RagConfig  # noqa: E402
from rag.evaluation.retrieval_metrics import score_matryoshka_retrieval  # noqa: E402

DEFAULT_OUTPUT_DIR = os.path.join("output_table", "matryoshka")
DEFAULT_GOLDSET = os.path.join("output_table", "retrieval_goldset.json")


def _config_from_run(rc: dict) -> RagConfig:
    cfg = RagConfig.from_env()
    cfg.embedding_model = rc.get("embedding_model", "openai")
    cfg.openai_embedding_model = rc.get("openai_embedding_model", cfg.openai_embedding_model)
    cfg.openai_embedding_dimensions = rc.get("openai_embedding_dimensions")
    cfg.chunk_strategy = rc.get("chunk_strategy", "section")
    cfg.retrieval_mode = rc.get("retrieval_mode", "hybrid")
    cfg.hybrid_alpha = rc.get("hybrid_alpha", cfg.hybrid_alpha)
    cfg.use_reranker = rc.get("use_reranker", True)
    cfg.reranker_model = rc.get("reranker_model", cfg.reranker_model)
    cfg.top_k = rc.get("top_k", cfg.top_k)
    cfg.rerank_top_k = rc.get("rerank_top_k", cfg.rerank_top_k)
    return cfg


def main() -> None:
    rag_evaluation.configure_utf8_stdout()
    p = argparse.ArgumentParser(description="Re-score MTEB metrics for existing Matryoshka summaries.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--goldset", default=DEFAULT_GOLDSET)
    p.add_argument("--only", nargs="+", default=None, help="Only summaries whose tag matches these")
    args = p.parse_args()

    if not os.path.isfile(args.goldset):
        print(f"[error] gold set not found: {args.goldset}")
        sys.exit(1)

    paths = sorted(glob.glob(os.path.join(args.output_dir, "summary_*.json")))
    if not paths:
        print(f"[error] no summary_*.json in {args.output_dir}")
        sys.exit(1)

    for i, path in enumerate(paths, 1):
        data = json.load(open(path, encoding="utf-8"))
        tag = data.get("tag", os.path.basename(path))
        if args.only and tag not in args.only:
            continue
        rc = data.get("run_config", {})
        cfg = _config_from_run(rc)
        print(f"\n[{i}/{len(paths)}] rescoring {tag} "
              f"({cfg.openai_embedding_model} @ {cfg.openai_embedding_dimensions}, "
              f"collection={cfg.chroma_collection}) ...")
        try:
            mteb = score_matryoshka_retrieval(cfg, args.goldset, queries_path="queries.txt")
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}")
            import traceback
            traceback.print_exc()
            continue
        data["mteb_metrics"] = mteb
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        pre = mteb["pre_rerank"]["macro"].get("ndcg@10", 0)
        post = mteb["post_rerank"]["macro"].get("ndcg@10", 0)
        print(f"    reranker={mteb.get('reranker')}  pre nDCG@10={pre:.3f}  post nDCG@10={post:.3f}  "
              f"gain={post - pre:+.3f}")

    print("\nDone. Now regenerate the report:  python analyze_matryoshka_study.py")


if __name__ == "__main__":
    main()
