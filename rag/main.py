"""Single entry point for the modular RAG system.

Commands:
  build        Extraction JSON  ->  chunks  ->  embeddings  ->  FAISS store
  query        One-shot question against a built store
  interactive  REPL with persistent chat history
  evaluate     Per-step metrics (time / memory / tokens / cost / accuracy)

All hyperparameters default to :class:`rag.config.RagConfig` and can be
overridden by the flags below. Run from the repo root, e.g.:

  python -m rag.main build --input "datalab-output-PR3216.pdf.json"
  python -m rag.main query --query "What is the operator name?"
  python -m rag.main evaluate --input doc.json --ground-truth matched.json
"""

from __future__ import annotations

import argparse
import os
import sys

# ── bootstrap: make 'import rag...' work when run as a script, and patch SSL ──
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches

apply_ssl_patches()  # must run before transformers / huggingface_hub import

from rag.config import RagConfig  # noqa: E402


def add_config_args(parser: argparse.ArgumentParser) -> None:
    """Attach hyperparameter override flags shared by every command."""
    g = parser.add_argument_group("hyperparameters (override rag/config.py)")
    g.add_argument("--embedding-model", help="'jina' | 'minilm' | 'tfidf' | any HF id")
    g.add_argument("--chunk-size", type=int)
    g.add_argument("--overlap", type=int)
    g.add_argument("--no-cross-page-merge", action="store_true", help="Disable cross-page section merge")
    g.add_argument("--naive", action="store_true", help="Naive mode: chunk raw JSON blocks without any section merge")
    g.add_argument("--index-dir", help="Directory for the FAISS store (index.faiss + meta.pkl)")
    g.add_argument("--faiss-metric", choices=["cosine", "l2"])
    g.add_argument("--retrieval-mode", choices=["dense", "hybrid"])
    g.add_argument("--hybrid-alpha", type=float)
    g.add_argument("--top-k", type=int)
    g.add_argument("--use-reranker", dest="use_reranker", action="store_true")
    g.add_argument("--no-reranker", dest="use_reranker", action="store_false")
    g.set_defaults(use_reranker=None)
    g.add_argument("--reranker-model")
    g.add_argument("--rerank-top-k", type=int)
    g.add_argument("--context-mode", choices=["chunk", "full_page", "hierarchical"])
    g.add_argument("--hierarchical-max-depth", type=int)
    g.add_argument("--save-pages", action="store_true", help="Save the context sent to the LLM")
    g.add_argument("--model-path", help="Local model folder for dense embeddings (e.g. ./jina-v3)")
    g.add_argument("--llm-model")


def build_config(args: argparse.Namespace) -> RagConfig:
    """Start from defaults, then apply any provided CLI overrides."""
    cfg = RagConfig.from_env()

    overrides = {
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "overlap": args.overlap,
        "index_dir": args.index_dir,
        "faiss_metric": args.faiss_metric,
        "retrieval_mode": args.retrieval_mode,
        "hybrid_alpha": args.hybrid_alpha,
        "top_k": args.top_k,
        "reranker_model": args.reranker_model,
        "rerank_top_k": args.rerank_top_k,
        "context_mode": args.context_mode,
        "hierarchical_max_depth": args.hierarchical_max_depth,
        "model_path": args.model_path,
        "llm_model": args.llm_model,
    }
    for key, value in overrides.items():
        if value is not None:
            setattr(cfg, key, value)

    if getattr(args, "use_reranker", None) is not None:
        cfg.use_reranker = args.use_reranker
    if getattr(args, "no_cross_page_merge", False):
        cfg.merge_across_pages = False
    if getattr(args, "naive", False):
        cfg.merge_sections = False
    if getattr(args, "save_pages", False):
        cfg.save_pages_sent_to_llm = True

    return cfg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Modular RAG pipeline (indexing + inference).")
    sub = parser.add_subparsers(dest="command", required=True)

    bp = sub.add_parser("build", help="Build the FAISS store from extraction JSON")
    bp.add_argument("--input", required=True, help="Path to the extraction JSON file")
    bp.add_argument("--chunks-output", default=None, help="Optional JSONL dump of the chunks")
    add_config_args(bp)

    qp = sub.add_parser("query", help="Ask one question against a built store")
    qp.add_argument("--query", required=True, help="Question text")
    qp.add_argument("--section-filter", default=None, help="Filter by section name substring")
    qp.add_argument("--page-range", nargs=2, type=int, metavar=("LO", "HI"), default=None)
    add_config_args(qp)

    ip = sub.add_parser("interactive", help="Interactive query REPL")
    add_config_args(ip)

    ev = sub.add_parser("evaluate", help="Run the pipeline and report per-step metrics")
    ev.add_argument("--input", required=True, help="Path to extraction JSON file")
    ev.add_argument("--queries", default="queries.txt", help="Path to queries file")
    ev.add_argument("--ground-truth", required=True, help="Path to matched JSON with ground-truth")
    ev.add_argument("--metrics-output", default=None, help="Optional path to save metrics JSON")
    ev.add_argument("--cache-root", default=None, help="Root folder for cached indexes")
    ev.add_argument("--no-cache", action="store_true", help="Disable reading/writing the index cache")
    ev.add_argument("--rebuild", action="store_true", help="Ignore any cached index and rebuild")
    ev.add_argument("--no-judge", action="store_true", help="Disable the LLM judge")
    add_config_args(ev)

    return parser.parse_args()


def _require_file(path: str, role: str) -> None:
    """Fail early with a clear message when an input file is missing."""
    if not os.path.isfile(path):
        print(f"[error] {role} not found: {path}")
        print(f"        resolved to : {os.path.abspath(path)}")
        print(f"        current dir : {os.getcwd()}")
        print("        Pass the correct path with the matching flag (use quotes if it contains spaces).")
        sys.exit(1)


def main() -> None:
    import rag_evaluation
    rag_evaluation.configure_utf8_stdout()

    args = parse_args()
    config = build_config(args)

    if args.command == "build":
        _require_file(args.input, "Input extraction JSON")
        from rag.indexing.build_pipeline import build
        build(args.input, config, chunks_output=args.chunks_output)

    elif args.command == "query":
        from rag.inference.query_pipeline import run_query
        page_range = tuple(args.page_range) if args.page_range else None
        run_query(config, args.query, section_filter=args.section_filter, page_range=page_range)

    elif args.command == "interactive":
        from rag.inference.query_pipeline import interactive_loop
        interactive_loop(config)

    elif args.command == "evaluate":
        _require_file(args.input, "Input extraction JSON")
        _require_file(args.queries, "Queries file")
        _require_file(args.ground_truth, "Ground-truth (matched) JSON")
        from rag.inference.validation import evaluate
        evaluate(
            config,
            input_path=args.input,
            queries_path=args.queries,
            ground_truth_path=args.ground_truth,
            metrics_output=args.metrics_output,
            cache_root=args.cache_root or rag_evaluation.DEFAULT_CACHE_ROOT,
            use_cache=not args.no_cache,
            force_rebuild=args.rebuild,
            use_judge=not args.no_judge,
        )


if __name__ == "__main__":
    main()
