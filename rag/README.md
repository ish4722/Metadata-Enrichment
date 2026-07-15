# The `rag` package

The RAG pipeline plus all of its tooling, organised so that changing one
component (embedding model, reranker, LLM, vector DB, prompt) means editing one
file only. All hyperparameters live in `config.py`.

Run everything from the **repository root** as a module (e.g.
`python -m rag.main ...`, `python -m rag.scripts.<name>`).

## Layout

```
rag/
├── main.py            # entry point: build / query / interactive / evaluate
├── config.py          # ALL hyperparameters + component toggles
│
├── common/            # shared helpers
│   ├── schema.py      # Chunk dataclass
│   ├── io_utils.py    # chunk JSONL save/load
│   ├── token_utils.py # token counting / auth helpers
│   ├── ssl_patch.py   # corporate SSL / HuggingFace patch
│   └── sauth_token.py # internal auth token helper (git-ignored)
│
├── indexing/          # JSON -> chunks -> embeddings -> vector store
│   ├── loader.py      # load + flatten extraction JSON
│   ├── merging.py     # section + cross-page merge
│   ├── chunking.py / section_chunking.py   # sliding-window / hierarchy-aware
│   ├── embeddings.py  # EmbeddingBackend        <- swap embedding model here
│   ├── vector_store.py# FAISS + BM25            <- swap vector DB here
│   ├── chroma_store.py# persistent Chroma store (embed-once)
│   └── build_pipeline.py
│
├── inference/         # query -> retrieve -> rerank -> context -> LLM -> verify
│   ├── retrieval.py        # dense / hybrid + metadata filter
│   ├── reranker.py         # cross-encoder      <- swap reranker here
│   ├── context_builder.py  # chunk / full-page / hierarchical context
│   ├── hierarchical_answer.py  # ancestor-escalation answering
│   ├── prompt.py           # prompt template    <- edit prompt here
│   ├── llm.py              # LLM client         <- swap LLM here
│   ├── verification.py     # Chain-of-Verification confidence pass
│   ├── validation.py       # wraps rag.evaluation.core for metrics
│   ├── query_pipeline.py
│   └── batch_runner.py     # single-process corpus batch eval
│
├── evaluation/        # metrics + batch drivers
│   ├── core.py             # metrics machinery + run_evaluation (was rag_evaluation.py)
│   ├── batch.py            # subprocess-per-doc batch driver (was evaluate_all.py)
│   └── retrieval_metrics.py# MTEB-style retrieval scoring
│
├── query_generation/  # schema-driven query synthesis
├── scripts/           # CLI drivers   →  python -m rag.scripts.<name>
├── benchmarking/      # embedding / reranker / LLM / Matryoshka benchmarks
├── reports/           # docx / markdown report generators
└── data_prep/         # download (model), linking (PDF↔JSON), starter
```

`inference/validation.py` and other modules import the metrics driver as
`from rag.evaluation import core as rag_evaluation`.

## Install

```bash
pip install -r rag/requirements.txt
```

`faiss-cpu` is required. The dense embedding / reranker deps (`torch`,
`transformers`, `sentence-transformers`) are optional — without them the system
falls back to TF-IDF dense retrieval.

## Usage

Run from the repository root.

```bash
# 1) Build the vector store from an extraction JSON
python -m rag.main build --input "datalab-output/datalab-output-PR3216.pdf.json" --chunks-output useful_chunks.jsonl

# 2) Ask a question
python -m rag.main query --query "What is the operator name?"

# 3) Interactive mode
python -m rag.main interactive

# 4) Evaluate against ground truth
python -m rag.main evaluate \
  --input "datalab-output/datalab-output-PR3216.pdf.json" \
  --ground-truth "matched_json/PR3216.json" --queries queries.txt
```

## Preset configurations

The pipeline started life as three separate scripts; they are now a single
config-driven pipeline. These preset combinations (toggle via `config.py` or CLI
flags) reproduce the original behaviours:

| Preset | embedding_model | retrieval_mode | use_reranker | context_mode |
|---|---|---|---|---|
| v1 | `tfidf` | `dense`  | off | `chunk`     |
| v2 | `jina`  | `hybrid` | on  | `full_page` |
| v3 | `tfidf` | `dense`  | off | `full_page` |

Example (reproduce v2):

```bash
python -m rag.main query --query "..." \
  --embedding-model jina --retrieval-mode hybrid --use-reranker --context-mode full_page
```

## Swap a component (the whole point)

- **Embedding model** → edit `indexing/embeddings.py` (or set `--embedding-model`).
- **Vector DB** → edit `indexing/vector_store.py`.
- **Reranker** → edit `inference/reranker.py` (or `--reranker-model`).
- **Prompt** → edit `inference/prompt.py`.
- **LLM** → edit `inference/llm.py` (or `--llm-model`).
