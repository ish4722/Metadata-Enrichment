# Modular RAG pipeline

A modular re-organisation of `datalab_rag_pipeline*.py` into two pipelines plus a
single entry point. Each pipeline stage lives in its own file, so changing one
component (embedding model, reranker, LLM, vector DB, prompt) means editing one
file only.

## Layout

```
rag/
├── main.py            # single entry point: build / query / interactive / evaluate
├── config.py          # ALL hyperparameters + component toggles
│
├── common/            # shared
│   ├── schema.py      # Chunk dataclass
│   ├── io_utils.py    # chunk JSONL save/load
│   └── ssl_patch.py   # corporate SSL / HuggingFace patch
│
├── indexing/          # FOLDER 1: JSON -> chunks -> embeddings -> FAISS store
│   ├── loader.py      # load + flatten extraction JSON
│   ├── merging.py     # section + cross-page merge
│   ├── chunking.py    # sliding-window chunking
│   ├── embeddings.py  # EmbeddingBackend   <- swap embedding model here
│   ├── vector_store.py# FAISS + BM25       <- swap vector DB here
│   └── build_pipeline.py
│
└── inference/         # FOLDER 2: query -> retrieval -> rerank -> context -> LLM -> answer
    ├── retrieval.py       # dense / hybrid + metadata filter
    ├── reranker.py        # cross-encoder      <- swap reranker here
    ├── context_builder.py # chunk / full-page context
    ├── prompt.py          # prompt template    <- edit prompt here
    ├── llm.py             # LLM client         <- swap LLM here
    ├── validation.py      # wraps rag_evaluation for metrics
    └── query_pipeline.py
```

The existing root scripts (`datalab_rag_pipeline*.py`, `rag_evaluation.py`) are
left untouched. `validation.py` imports `rag_evaluation` from the repo root.

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
python -m rag.main build --input "datalab-output-PR3216.pdf.json" --chunks-output useful_chunks.jsonl

# 2) Ask a question
python -m rag.main query --query "What is the operator name?"

# 3) Interactive mode
python -m rag.main interactive

# 4) Evaluate against ground truth
python -m rag.main evaluate --input doc.json --ground-truth matched.json --queries queries.txt
```

## Reproducing the three original scripts

All three are now one config-driven pipeline. Toggle via `config.py` or CLI flags:

| Original | embedding_model | retrieval_mode | use_reranker | context_mode |
|---|---|---|---|---|
| `datalab_rag_pipeline.py` (v1)  | `tfidf` | `dense`  | off | `chunk`     |
| `datalab_rag_pipeline2.py` (v2) | `jina`  | `hybrid` | on  | `full_page` |
| `datalab_rag_pipeline3.py` (v3) | `tfidf` | `dense`  | off | `full_page` |

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
