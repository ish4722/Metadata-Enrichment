# Benchmarking suite

Four self-contained benchmarks for the RAG pipeline. Each has its own script,
reuses the existing engines (`rag.inference.batch_runner` for end-to-end
accuracy/cost/time and `rag.evaluation.retrieval_metrics` for MTEB-style
retrieval metrics), and writes results under `output_table/benchmarking/`.

Run everything from the **repository root**. Existing Chroma collections are
reused (embed-once); nothing is re-embedded when a collection already exists.

## Prerequisites

- The retrieval gold set at `output_table/retrieval_goldset.json` (needed for
  retrieval metrics). Build it with:
  ```bash
  python -m rag.scripts.build_retrieval_goldset --num-docs 40 --restart --concurrency 12 --batch-size 8
  ```
- Embeddings already built in `chroma_store/` (they are, except jina-api without
  late chunking — see Task 1).

## Shared modules

| File | Purpose |
|---|---|
| `model_specs.py` | Per-model pricing (USD/1M tokens) and context windows for embeddings + LLMs; cost helpers. |
| `common.py` | Config building, `run_end_to_end` (BatchRunner+summarize), `run_retrieval_metrics`, per-model cost recompute, JSON/markdown writers. |

Cost is recomputed per model from stored token counts, because the pipeline's
own accounting bakes in a single fixed price.

## Task 1 — Embedding models (reranker OFF)

Compares jina-api (late chunking on/off), `text-embedding-3-large`, and
`text-embedding-3-small` on final LLM accuracy, retrieval metrics (MRR, Hit@k,
Recall@k, nDCG@k, MAP), cost, and context window.

```bash
python -m rag.benchmarking.benchmark_embeddings
python -m rag.benchmarking.benchmark_embeddings --only PR2610 PR2611
python -m rag.benchmarking.benchmark_embeddings --skip-missing        # don't build jina (late-off)
python -m rag.benchmarking.benchmark_embeddings --configs openai_large openai_small
```

Only `jina_api` (late chunking OFF) has no vectors yet; it is built once unless
`--skip-missing`. Output: `output_table/benchmarking/embeddings/`
(`summary_<tag>.json`, `comparison.json`, `report.md`).

## Task 2 — Matryoshka dimension study (reuse-first)

The truncated-dimension collections and summaries already exist, so by default
this only re-scores retrieval metrics and regenerates the analysis (no
re-embedding). Uses the fixed OpenAI dimension wiring in `rag/config.py`.

```bash
python -m rag.benchmarking.benchmark_matryoshka                       # rescore + analyze
python -m rag.benchmarking.benchmark_matryoshka --rerun --force       # full re-run
python -m rag.benchmarking.benchmark_matryoshka --rerun --models text-embedding-3-large --dims 256 512
```

Output: `output_table/matryoshka/` (`aggregate.json`, `analysis_report.md`).

## Task 3 — Reranker impact (best model)

Auto-selects the Task-1 winner (by LLM accuracy) or `--model <tag>`, then reports
retrieval metrics pre vs post rerank plus (by default) end-to-end accuracy with
the reranker OFF vs ON.

```bash
python -m rag.benchmarking.benchmark_reranker
python -m rag.benchmarking.benchmark_reranker --model openai_large
python -m rag.benchmarking.benchmark_reranker --no-end-to-end         # retrieval metrics only
```

Output: `output_table/benchmarking/reranker/` (`metrics.json`, `report.md`).

## Task 4 — Generator LLMs (best model, reranker ON)

Swaps only the generation/judge LLM across
`gpt-5.2, gpt-5.1, gpt-5-nano, gpt-5-mini, gpt-5, gpt-4o-mini, gpt-4o`
and compares accuracy, cost, context window, tokens and time. Unavailable models
are detected by a pre-flight call and skipped.

```bash
python -m rag.benchmarking.benchmark_llms
python -m rag.benchmarking.benchmark_llms --models gpt-5 gpt-5-mini gpt-4o-mini
python -m rag.benchmarking.benchmark_llms --model openai_large        # force embedding model
python -m rag.benchmarking.benchmark_llms --no-preflight
```

Output: `output_table/benchmarking/llms/` (`summary_<model>.json`,
`comparison.json`, `report.md`).

## Enriched gold-set benchmarks (no MTEB retrieval metrics)

The same four tasks can be scored against the human-verified
`enriched_ground_truth/` files instead of the OSDU `matched_json/` records. Each
enriched file holds only the attributes a reviewer verified for that document, so
these runs query just those attributes (per document) and read ground truth
straight from the flat enriched record. MTEB retrieval ranking metrics (MRR,
nDCG@k, Hit@k, ...) are **not** computed here — accuracy is purely end-to-end
(LLM-judge / token / retrieval hit), so no `retrieval_goldset.json` is required.

Shared utilities live in `enriched_dataset.py` (discovery, per-doc query
filtering, the enriched ground-truth extractor and a run manifest). The
`BatchRunner` accepts a `gt_extractor` hook so the same engine drives both gold
standards.

```bash
# Full suite: embeddings -> matryoshka + reranker -> llms
python -m rag.benchmarking.benchmark_enriched

# Individual tasks
python -m rag.benchmarking.benchmark_enriched_embeddings --only PR3403 PR3693
python -m rag.benchmarking.benchmark_enriched_matryoshka --dims 512 1024
python -m rag.benchmarking.benchmark_enriched_reranker --models openai_large
python -m rag.benchmarking.benchmark_enriched_llms --model openai_large

# Suite subsets
python -m rag.benchmarking.benchmark_enriched --embeddings-only
python -m rag.benchmarking.benchmark_enriched --reranker-only --models openai_large
```

The reranker and LLM tasks auto-pick their embedding config from the enriched
Task-1 `comparison.json`, so run the embedding benchmark first (the orchestrator
does this automatically). Output: `output_table/benchmarking_enriched/`
(`manifest.json`, plus `embeddings/`, `matryoshka/`, `reranker/`, `llms/`
subfolders mirroring the standard reports).

## Turning reports into Word

```bash
python -m rag.reports.md_to_docx output_table/benchmarking/embeddings/report.md
```

## Common flags

| Flag | Meaning |
|---|---|
| `--only PR2610 PR2611` | Restrict to specific doc ids |
| `--limit N` | Use only the first N docs |
| `--no-judge` | Skip the LLM judge (token-match accuracy only) |
| `--no-inject-name` | Don't fill `{well}` placeholders with the ground-truth name |
| `--goldset PATH` | Custom gold-set path |
