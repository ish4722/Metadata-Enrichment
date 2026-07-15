# CLI reference

Every command runs from the **repository root**. All commands accept `--help`.
Hyperparameter flags override [`rag/config.py`](../rag/config.py); anything you
don't pass keeps its config default.

## `rag.main` — core pipeline

```bash
python -m rag.main <build|query|interactive|evaluate> [flags]
```

### Shared hyperparameter flags (all subcommands)

| Flag | Meaning |
|---|---|
| `--embedding-model` | `jina`\|`minilm`\|`tfidf`\|`openai`\|`jina-api`\| HF id |
| `--chunk-size N` | Chunk size (chars) |
| `--overlap N` | Chunk overlap (chars) |
| `--no-cross-page-merge` | Disable cross-page section merge |
| `--naive` | Chunk raw JSON blocks without section merge |
| `--index-dir DIR` | FAISS store directory |
| `--faiss-metric cosine\|l2` | FAISS distance metric |
| `--retrieval-mode dense\|hybrid` | Retrieval mode |
| `--hybrid-alpha F` | Dense/BM25 blend (0..1) |
| `--top-k N` | Candidates retrieved |
| `--use-reranker` / `--no-reranker` | Toggle reranking |
| `--reranker-model M` | Cross-encoder model |
| `--rerank-top-k N` | Candidates kept after rerank |
| `--context-mode chunk\|full_page\|hierarchical` | Context assembly |
| `--hierarchical-max-depth N` | Ancestor levels to climb |
| `--save-pages` | Save context sent to the LLM |
| `--model-path DIR` | Local dense-embedding model folder |
| `--llm-model M` | Generation/judge LLM |

### `build`

```bash
python -m rag.main build --input "datalab-output/datalab-output-PR2610.pdf.json" \
  [--chunks-output useful_chunks.jsonl]
```

### `query`

```bash
python -m rag.main query --query "What is the well name?" \
  [--section-filter "location"] [--page-range 1 20]
```

### `interactive`

```bash
python -m rag.main interactive
```

### `evaluate`

```bash
python -m rag.main evaluate \
  --input "datalab-output/datalab-output-PR2610.pdf.json" \
  --ground-truth "matched_json/PR2610.json" \
  --queries queries.txt
```

| Extra flag | Meaning |
|---|---|
| `--queries PATH` | Queries file (default `queries.txt`) |
| `--metrics-output PATH` | Save metrics JSON |
| `--cache-root DIR` | Root folder for cached indexes |
| `--no-cache` | Disable index cache read/write |
| `--rebuild` | Ignore cached index and rebuild |
| `--no-judge` | Disable the LLM judge (token match only) |

## `rag/inference/batch_runner.py` — main corpus batch eval

Loads the embedder, reranker, and LLM once, then runs the corpus with bounded
parallelism. Output: `output_table/batch_runner_summary.json`.

```bash
python rag/inference/batch_runner.py [flags]
```

| Flag | Meaning |
|---|---|
| `--docs-dir DIR` | Extraction JSON folder (default `datalab-output`) |
| `--matched-dir DIR` | Ground-truth folder (default `matched_json`) |
| `--queries PATH` | Queries file (default `queries.txt`) |
| `--only PR2610 …` | Restrict to specific doc ids |
| `--summary-output PATH` | Output summary JSON path |
| `--no-judge` | Skip the LLM judge |
| `--build-only` | Only embed-once into Chroma, skip inference |
| `--embedding-model M` | Embedding backend |
| `--openai-embedding-model M` | OpenAI embedding model |
| `--openai-embedding-dimensions N` | Matryoshka truncated dimension |
| `--use-reranker` / `--no-reranker` | Toggle reranking |
| `--chunk-strategy S` | `section` or `legacy` |
| `--retrieval-mode dense\|hybrid` | Retrieval mode |
| `--hybrid-alpha F` | Dense/BM25 blend |
| `--context-mode …` | Context assembly |
| `--hierarchical-max-depth N` | Ancestor depth |
| `--load-max-workers N` | Load+chunk process pool (use `1` on Windows if unstable) |
| `--embed-max-workers N` | Concurrent embedding requests |
| `--inject-name` / `--no-inject-name` | Fill `{well}` placeholders with the GT name |

## `rag.evaluation.batch` — legacy subprocess batch eval

Spawns `python -m rag.main evaluate` once per document.

```bash
python -m rag.evaluation.batch [--only PR2610 PR2611] [--reuse-existing]
                               [--rebuild] [--no-judge]
                               [--chunk-size N] [--overlap N] [--embedding-model M]
```

Output: per-doc JSON in `output_table/batch/`, combined `output_table/batch_summary.json`.

## `rag.scripts.build_retrieval_goldset`

Builds a graded retrieval gold set (correct=2 / distractor=1 / omitted).
Checkpointed and resumable. Output: `output_table/retrieval_goldset.json`.

```bash
python -m rag.scripts.build_retrieval_goldset --num-docs 40 --restart \
  --concurrency 12 --batch-size 8
```

| Flag | Default | Meaning |
|---|---|---|
| `--num-docs N` | all | Use the N fewest-chunk docs |
| `--max-chunks N` | `1000` | Cap chunks per doc |
| `--concurrency N` | `12` | Concurrent LLM calls |
| `--batch-size N` | `8` | Chunks per LLM call |
| `--only PR… ` | — | Restrict to doc ids |
| `--limit N` | — | First N docs |
| `--matched-dir` / `--docs-dir` / `--queries` | defaults | Input locations |
| `--output PATH` | `output_table/retrieval_goldset.json` | Output |
| `--restart` | off | Ignore existing output and rebuild all |

## `rag.scripts.evaluate_retrieval`

Scores 8 configs (dense + hybrid α=0.5/0.7/0.85 × inject/no-inject) vs the gold set.

```bash
python -m rag.scripts.evaluate_retrieval [--rerank] [--output PATH]
```

Output: `output_table/retrieval_metrics.json` (or `_rerank.json` with `--rerank`).

## Matryoshka study

```bash
python -m rag.scripts.run_matryoshka_study                       # full study (resumable)
python -m rag.scripts.run_matryoshka_study --only PR2610         # smoke test
python -m rag.scripts.run_matryoshka_study --force               # re-run finished configs
python -m rag.scripts.run_matryoshka_study --models text-embedding-3-large --dims 256 512 1024
python -m rag.scripts.run_matryoshka_study --skip-analysis
python -m rag.scripts.analyze_matryoshka_study [--output-dir output_table/matryoshka]
python -m rag.scripts.rescore_matryoshka_mteb
```

## Chunking tuning

```bash
python -m rag.scripts.tune_chunking \
  [--input <json>] [--ground-truth <json>] \
  --chunk-sizes 400 800 1200 --overlaps 0 100 200
python -m rag.scripts.aggregate_tuning_results
```

## Baselines & live validation

```bash
python -m rag.scripts.compare_rag_vs_exhaustive \
  --input <json> --ground-truth <json> [--max-pages 20] [--rebuild] [--no-judge]

python -m rag.scripts.exhaustive_search \
  --input <json> --ground-truth <json> --queries queries.txt

python -m rag.scripts.validate_query --ground-truth matched_json/PR2610.json [flags]
```

`validate_query` flags: `--queries`, `--only "lat+long" …`, `--no-judge`,
`--show-chunks`, `--metrics-output`, `--embedding-model`, `--retrieval-mode`,
`--hybrid-alpha`, `--top-k`, `--no-reranker`.

## Benchmarking

```bash
python -m rag.benchmarking.benchmark_embeddings [--only …] [--skip-missing] [--configs …]
python -m rag.benchmarking.benchmark_matryoshka [--rerun] [--force] [--models …] [--dims …]
python -m rag.benchmarking.benchmark_reranker  [--model TAG] [--no-end-to-end]
python -m rag.benchmarking.benchmark_llms      [--models …] [--model TAG] [--no-preflight]
python -m rag.benchmarking.benchmark_enriched  [--embeddings-only|--reranker-only|…]
```

Full details: [benchmarking.md](benchmarking.md) and
[`rag/benchmarking/README.md`](../rag/benchmarking/README.md).

## Reports & data prep

```bash
python -m rag.reports.md_to_docx <file.md>
python -m rag.reports.generate_benchmarking_results_docx
python -m rag.reports.generate_embedding_report_docx
python -m rag.data_prep.download
python -m rag.data_prep.linking          # edit the CONFIGURATION block first
```

## Web app

```bash
uvicorn webapp.backend.main:app --reload --port 8000
cd webapp/frontend && npm run dev
```
