# Benchmarking & evaluation

There are three complementary ways to measure quality, from cheapest to most
complete.

```mermaid
flowchart LR
    A[Retrieval metrics - no LLM] --> B[End-to-end eval - LLM judge]
    B --> C[Component benchmarks - sweep models]
```

## 1. Retrieval metrics (no generation cost)

Scores how well retrieval ranks the relevant chunks — no LLM answering involved.

```bash
# Build the graded gold set once (correct=2 / distractor=1 / omitted); resumable
python -m rag.scripts.build_retrieval_goldset --num-docs 40 --restart --concurrency 12 --batch-size 8

# Score dense + hybrid (α=0.5/0.7/0.85) × inject/no-inject
python -m rag.scripts.evaluate_retrieval
python -m rag.scripts.evaluate_retrieval --rerank      # add cross-encoder reranking
```

Metrics: MRR, Hit@k, Recall@k, nDCG@k, MAP.
Outputs: `output_table/retrieval_goldset.json`, `output_table/retrieval_metrics.json`.

## 2. End-to-end evaluation (LLM judge)

Runs the full pipeline and judges each answer against ground truth, recording
accuracy, cost, tokens, time, and peak memory.

```bash
# Whole corpus, single process (recommended)
python rag/inference/batch_runner.py --context-mode hierarchical --load-max-workers 1

# One document, per-step detail
python -m rag.main evaluate \
  --input "datalab-output/datalab-output-PR2610.pdf.json" \
  --ground-truth "matched_json/PR2610.json"
```

Output: `output_table/batch_runner_summary.json`.

The judge (in `rag/evaluation/core.py`) applies domain-aware tolerances: dates
±1 day, coordinate-system conversion (DMS/NZMG/NZTM → WGS84), and
formatting/synonym normalisation.

### Two ground-truth standards

| Standard | Folder | Notes |
|---|---|---|
| OSDU matched | `matched_json/` | Full structured records; supports MTEB retrieval metrics |
| Enriched | `enriched_ground_truth/` | Only reviewer-verified attributes; end-to-end accuracy only |

## 3. Component benchmarks

Four benchmarks sweep one component at a time, reusing existing Chroma
collections (nothing re-embeds when vectors already exist). Full flags in
[`rag/benchmarking/README.md`](../rag/benchmarking/README.md).

| Benchmark | Command | What it compares |
|---|---|---|
| Embeddings | `python -m rag.benchmarking.benchmark_embeddings` | jina-api (late on/off), `text-embedding-3-large/small`: accuracy, retrieval metrics, cost, context window |
| Matryoshka | `python -m rag.benchmarking.benchmark_matryoshka` | Truncated embedding dimensions (256–3072) |
| Reranker | `python -m rag.benchmarking.benchmark_reranker` | Retrieval + accuracy, reranker OFF vs ON |
| Generator LLMs | `python -m rag.benchmarking.benchmark_llms` | Swaps only the answer/judge LLM |

Enriched-ground-truth variants (end-to-end accuracy only, no gold set required):

```bash
python -m rag.benchmarking.benchmark_enriched                 # full suite
python -m rag.benchmarking.benchmark_enriched_embeddings
python -m rag.benchmarking.benchmark_enriched_matryoshka
python -m rag.benchmarking.benchmark_enriched_reranker
python -m rag.benchmarking.benchmark_enriched_llms
```

Outputs: `output_table/benchmarking/` and `output_table/benchmarking_enriched/`.

## Other studies

```bash
# Matryoshka embedding-dimension study (standalone driver)
python -m rag.scripts.run_matryoshka_study
python -m rag.scripts.analyze_matryoshka_study

# Chunk size / overlap sweep
python -m rag.scripts.tune_chunking --chunk-sizes 400 800 1200 --overlaps 0 100 200
python -m rag.scripts.aggregate_tuning_results

# RAG vs naive exhaustive page scan
python -m rag.scripts.compare_rag_vs_exhaustive --input <json> --ground-truth <json>
```

## Turning results into Word

```bash
python -m rag.reports.md_to_docx output_table/benchmarking/embeddings/report.md
python -m rag.reports.generate_benchmarking_results_docx
python -m rag.reports.generate_embedding_report_docx
```

## Common benchmarking flags

| Flag | Meaning |
|---|---|
| `--only PR2610 PR2611` | Restrict to specific doc ids |
| `--limit N` | Use only the first N docs |
| `--no-judge` | Skip the LLM judge (token-match accuracy only) |
| `--no-inject-name` | Don't fill `{well}` placeholders with the ground-truth name |
| `--goldset PATH` | Custom gold-set path |
