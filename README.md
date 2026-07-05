# Metadata Enrichment — RAG Evaluation

End-to-end Retrieval-Augmented Generation (RAG) pipeline for oil & gas well completion reports. Documents are extracted to JSON (Datalab), chunked and embedded into Chroma, retrieved with hybrid dense+BM25 search, reranked, and answered by an LLM. The repo includes batch evaluation, retrieval gold-set building, Matryoshka embedding-dimension studies, and a web UI for interactive exploration.

Run all commands from the **repository root** unless noted otherwise.

---

## Prerequisites

```bash
pip install -r rag/requirements.txt
```

Optional: download a local Jina v3 snapshot (avoids HuggingFace at runtime):

```bash
python download.py
```

For the web app, also install backend deps and frontend packages:

```bash
pip install -r webapp/backend/requirements.txt
cd webapp/frontend && npm install
```

### Environment variables

| Variable | Purpose |
|---|---|
| `LLM_MODEL` | Override default LLM model (`gpt-5.2`) |
| `LLM_BASE_URL` | Internal model-ops API base URL |
| `SLB_PARTITION_ID` | Deployment partition ID |

Hyperparameters (embedding model, retrieval mode, reranker, context mode, etc.) live in `rag/config.py` and can be overridden via CLI flags on most scripts.

---

## Data layout

| Path | Description |
|---|---|
| `datalab-output/` | Datalab extraction JSON files (`datalab-output-PR2610.pdf.json`, …) |
| `matched_json/` | Ground-truth JSON per document (`PR2610.json`, …) |
| `queries.txt` | Evaluation queries (facility name, lat+long, openv ID, spud date, country, operator, basin, field, markers, completion date, vertical measurements). Labels ending in `2` are `[facility name]` phrasing variants. |
| `chroma_store/` | Persistent Chroma vector store (embed-once per model/strategy) |
| `output_table/` | Evaluation summaries, gold sets, charts, and study reports |

Doc IDs are derived from filenames (e.g. `datalab-output-PR2610.pdf.json` → `PR2610`).

---

## Project structure

```
Metadata-Enrichment/
├── rag/                    # Modular RAG pipeline (see rag/README.md)
│   ├── main.py             # build / query / interactive / evaluate
│   ├── config.py           # central hyperparameters
│   ├── indexing/           # chunking, embeddings, Chroma/FAISS
│   └── inference/          # retrieval, rerank, context, LLM, batch_runner
├── webapp/                 # FastAPI + React RAG Explorer UI
├── evaluate_all.py         # subprocess batch eval (legacy driver)
├── build_retrieval_goldset.py
├── evaluate_retrieval.py
├── run_matryoshka_study.py
├── analyze_matryoshka_study.py
├── tune_chunking.py
├── compare_rag_vs_exhaustive.py
├── exhaustive_search.py
└── validate_query.py
```

---

## Quick start — single document

```bash
# Build index from one extraction JSON
python -m rag.main build --input "datalab-output/datalab-output-PR2610.pdf.json" --chunks-output useful_chunks.jsonl

# Ask one question
python -m rag.main query --query "What is the well name?"

# Interactive REPL
python -m rag.main interactive

# Full per-step evaluation (retrieval + generation + judge)
python -m rag.main evaluate \
  --input "datalab-output/datalab-output-PR2610.pdf.json" \
  --ground-truth "matched_json/PR2610.json" \
  --queries queries.txt
```

Common overrides (apply to `build`, `query`, `evaluate`, `interactive`):

```bash
python -m rag.main query --query "..." \
  --embedding-model openai \
  --retrieval-mode hybrid --hybrid-alpha 0.85 \
  --use-reranker \
  --context-mode hierarchical --hierarchical-max-depth 5
```

---

## Batch evaluation (full corpus)

### Recommended: single-process batch runner

Loads embedding backend, reranker, and LLM **once**, then runs the whole corpus with bounded parallelism. Faster and more reliable than spawning one subprocess per document.

```bash
# Baseline end-to-end run (default config from rag/config.py)
python rag/inference/batch_runner.py

# Hierarchical context + limit process pool (recommended on Windows / large JSONs)
python rag/inference/batch_runner.py \
  --context-mode hierarchical \
  --load-max-workers 1

# Smoke test on one document
python rag/inference/batch_runner.py --only PR2610

# Build Chroma index only (skip inference)
python rag/inference/batch_runner.py --build-only

# Override retrieval / embedding settings
python rag/inference/batch_runner.py \
  --retrieval-mode hybrid --hybrid-alpha 0.85 \
  --openai-embedding-model text-embedding-3-large \
  --openai-embedding-dimensions 512 \
  --chunk-strategy section \
  --embed-max-workers 8

# Skip LLM judge (token match only)
python rag/inference/batch_runner.py --no-judge

# Disable well-name injection in queries ({well} → plain phrasing)
python rag/inference/batch_runner.py --no-inject-name
```

Output: `output_table/batch_runner_summary.json`

| Flag | Description |
|---|---|
| `--docs-dir` | Datalab JSON folder (default: `datalab-output`) |
| `--matched-dir` | Ground-truth folder (default: `matched_json`) |
| `--queries` | Queries file (default: `queries.txt`) |
| `--only PR2610 PR2611` | Restrict to specific doc IDs |
| `--summary-output` | Custom output JSON path |
| `--load-max-workers` | Process pool for load+chunk |
| `--embed-max-workers` | Concurrent embedding requests during build |

### Legacy: subprocess-per-doc driver

Spawns `python -m rag.main evaluate` once per document.

```bash
python evaluate_all.py
python evaluate_all.py --only PR2610 PR2611
python evaluate_all.py --reuse-existing          # skip docs with cached metrics
python evaluate_all.py --rebuild --no-judge
python evaluate_all.py --chunk-size 800 --overlap 200 --embedding-model openai
```

Output: per-doc JSON in `output_table/batch/`, combined `output_table/batch_summary.json`

---

## Matryoshka embedding dimension study

Compares `text-embedding-3-small` and `text-embedding-3-large` at truncated dimensions (256–3072). All non-embedding settings are locked to the baseline batch run.

### Prerequisites

Build the retrieval gold set first (needed for MTEB-style retrieval metrics):

```bash
python build_retrieval_goldset.py --num-docs 40 --restart --concurrency 12 --batch-size 8
```

### Run the study

```bash
# Full 10-config study (resumable — skips configs with existing summaries)
python run_matryoshka_study.py

# Smoke test on one document
python run_matryoshka_study.py --only PR2610

# Re-run configs that already finished
python run_matryoshka_study.py --force

# Restrict models or dimensions
python run_matryoshka_study.py --models text-embedding-3-large --dims 256 512 1024

# Skip the automatic analysis step at the end
python run_matryoshka_study.py --skip-analysis
```

Output per config: `output_table/matryoshka/summary_{large\|small}_d{dim}.json`

### Analyze results

```bash
# Standalone analysis (after summaries exist)
python analyze_matryoshka_study.py
python analyze_matryoshka_study.py --output-dir output_table/matryoshka

# Convert markdown report to Word
python md_to_docx.py output_table/matryoshka/analysis_report.md
```

Analysis outputs:

| File | Description |
|---|---|
| `output_table/matryoshka/aggregate.json` | Machine-readable comparison across runs |
| `output_table/matryoshka/analysis_report.md` | Human-readable report (quality vs dim, crossover, cost, MTEB curves) |
| `output_table/matryoshka/analysis_report.docx` | Word version (via `md_to_docx.py`) |

---

## Retrieval evaluation

### Build graded gold set

Labels each (doc, query, chunk) as **correct** (grade 2), **wrong/distractor** (grade 1), or omitted. Checkpointed and resumable.

```bash
python build_retrieval_goldset.py --num-docs 40 --restart --concurrency 12 --batch-size 8

# Other useful flags
python build_retrieval_goldset.py --only PR2610 PR2611
python build_retrieval_goldset.py --max-chunks 1000 --limit 10
python build_retrieval_goldset.py --output output_table/retrieval_goldset.json
```

Output: `output_table/retrieval_goldset.json`

### Score retrieval configurations

Scores 8 configs (dense + hybrid α=0.5/0.7/0.85 × inject/no-inject) against the gold set.

```bash
python evaluate_retrieval.py
python evaluate_retrieval.py --rerank          # add cross-encoder reranking
python evaluate_retrieval.py --output output_table/retrieval_metrics.json
```

Output: `output_table/retrieval_metrics.json` (or `retrieval_metrics_rerank.json` with `--rerank`)

---

## Chunking hyperparameter tuning

Sweeps `chunk_size` × `overlap` on a single document via `rag.main evaluate`.

```bash
python tune_chunking.py
python tune_chunking.py \
  --input "datalab-output/datalab-output-PR2610.pdf.json" \
  --ground-truth "matched_json/PR2610.json" \
  --chunk-sizes 400 800 1200 \
  --overlaps 0 100 200
```

Aggregate results across multiple PR runs:

```bash
python aggregate_tuning_results.py
```

Output: `output_table/tuning_best_10files.json`

---

## Baselines and live validation

### RAG vs exhaustive (naive page scan)

```bash
python compare_rag_vs_exhaustive.py \
  --input "datalab-output/datalab-output-PR2610.pdf.json" \
  --ground-truth "matched_json/PR2610.json"

python compare_rag_vs_exhaustive.py --max-pages 20 --rebuild --no-judge
```

### Exhaustive search only

```bash
python exhaustive_search.py \
  --input "datalab-output/datalab-output-PR2610.pdf.json" \
  --ground-truth "matched_json/PR2610.json" \
  --queries queries.txt
```

### Validate live query path (Chroma + hybrid + reranker)

Exercises the same code path as `rag.main query` (not a fresh rebuild).

```bash
python validate_query.py --ground-truth matched_json/PR2610.json
python validate_query.py --ground-truth matched_json/PR2610.json \
  --retrieval-mode hybrid --hybrid-alpha 0.5
python validate_query.py --ground-truth matched_json/PR2610.json \
  --only "lat+long" "facility name" --show-chunks
```

---

## Web app (RAG Explorer)

```bash
# Terminal 1 — API
uvicorn webapp.backend.main:app --reload --port 8000

# Terminal 2 — frontend
cd webapp/frontend && npm run dev
```

The UI supports document discovery, embed-once build into Chroma, staged query streaming (retrieve → rerank → LLM → validate → PDF highlight), and gold-set review.

---

## Data preparation

### Link PDFs to ground-truth JSON

`linking.py` matches PDFs (via Excel well names) to extraction JSON facility names and copies matched JSON into `matched_json/`. Edit the `CONFIGURATION` block at the top of the file, then:

```bash
pip install pandas openpyxl
python linking.py
```

### Download local embedding model

```bash
python download.py
```

---

## Typical analysis workflows

### 1. Baseline end-to-end accuracy (full corpus)

```bash
python rag/inference/batch_runner.py --context-mode hierarchical --load-max-workers 1
```

Review: `output_table/batch_runner_summary.json`

### 2. Retrieval-only analysis (no generation cost)

```bash
python build_retrieval_goldset.py --num-docs 40 --restart --concurrency 12 --batch-size 8
python evaluate_retrieval.py
python evaluate_retrieval.py --rerank
```

### 3. Embedding dimension trade-off study

```bash
python build_retrieval_goldset.py --num-docs 40 --restart --concurrency 12 --batch-size 8
python run_matryoshka_study.py
python analyze_matryoshka_study.py
python md_to_docx.py output_table/matryoshka/analysis_report.md
```

### 4. Single-document deep dive

```bash
python -m rag.main evaluate \
  --input "datalab-output/datalab-output-PR2610.pdf.json" \
  --ground-truth "matched_json/PR2610.json"

python validate_query.py --ground-truth matched_json/PR2610.json --show-chunks

python compare_rag_vs_exhaustive.py \
  --input "datalab-output/datalab-output-PR2610.pdf.json" \
  --ground-truth "matched_json/PR2610.json"
```

### 5. Chunking A/B

```bash
python tune_chunking.py --chunk-sizes 400 800 1200 --overlaps 0 100 200
python aggregate_tuning_results.py
```

---

## Default pipeline configuration

Current defaults in `rag/config.py` (override via CLI or edit the file):

| Setting | Default |
|---|---|
| Embedding | OpenAI `text-embedding-3-small` |
| Chunk strategy | `section` (hierarchy-aware) |
| Vector store | Chroma (`chroma_store/`) |
| Retrieval | Hybrid, α = 0.85 |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Context mode | `hierarchical` (escalate to ancestor sections) |
| LLM | `gpt-5.2` |

See [`rag/README.md`](rag/README.md) for component swap points (embedding backend, reranker, prompt, LLM) and how to reproduce the original `datalab_rag_pipeline*.py` scripts.

---

## Command reference (all scripts)

| Script | Purpose |
|---|---|
| `python -m rag.main build` | Index one JSON → Chroma/FAISS |
| `python -m rag.main query` | Single question |
| `python -m rag.main interactive` | Interactive REPL |
| `python -m rag.main evaluate` | Per-step metrics on one doc |
| `python rag/inference/batch_runner.py` | **Main** corpus batch eval |
| `python evaluate_all.py` | Legacy subprocess batch eval |
| `python build_retrieval_goldset.py` | Build graded retrieval gold set |
| `python evaluate_retrieval.py` | Score retrieval configs vs gold set |
| `python run_matryoshka_study.py` | Matryoshka dimension sweep |
| `python analyze_matryoshka_study.py` | Analyze Matryoshka summaries |
| `python md_to_docx.py` | Markdown → Word report |
| `python tune_chunking.py` | Sweep chunk_size / overlap |
| `python aggregate_tuning_results.py` | Aggregate chunking tuning runs |
| `python compare_rag_vs_exhaustive.py` | RAG vs naive page scan |
| `python exhaustive_search.py` | Naive exhaustive baseline |
| `python validate_query.py` | Validate live Chroma query path |
| `python download.py` | Download Jina v3 locally |
| `python linking.py` | Match PDFs to JSON ground truth |
| `uvicorn webapp.backend.main:app` | Start web API |
