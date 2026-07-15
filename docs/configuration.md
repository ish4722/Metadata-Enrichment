# Configuration reference

All behaviour is controlled by the `RagConfig` dataclass in
[`rag/config.py`](../rag/config.py). Three ways to set it (highest precedence last):

1. **Defaults** in `RagConfig`.
2. **Environment variables** (a subset — see [below](#environment-variables)),
   applied by `RagConfig.from_env()`.
3. **CLI flags** on `rag.main` and most scripts (see [cli-reference.md](cli-reference.md)).

`DEFAULT_CONFIG = RagConfig.from_env()` is the singleton other modules import.

## Chunking

| Field | Default | Meaning |
|---|---|---|
| `chunk_size` | `800` | Sliding-window chunk size (chars), legacy strategy |
| `overlap` | `200` | Overlap between windows (chars) |
| `merge_across_pages` | `True` | Merge a section split across pages |
| `merge_sections` | `True` | `False` = naive: chunk raw JSON blocks (A/B baseline) |
| `chunk_strategy` | `"section"` | `"section"` (hierarchy-aware) or `"legacy"` |

## Embedding

| Field | Default | Meaning |
|---|---|---|
| `embedding_model` | `"openai"` | `jina-api`\|`jina`\|`minilm`\|`openai`\|`tfidf`\| HF id |
| `openai_embedding_model` | `text-embedding-3-small` | OpenAI-compatible model |
| `openai_embedding_base_url` | SLB deployment URL | Embeddings API base URL |
| `openai_embedding_partition_id` | `sandbox-...` | Deployment partition |
| `openai_embedding_dimensions` | `None` | Matryoshka truncated width; `None` = native. Each dim gets its own collection (`_d{dim}`) |

### Hosted Jina v3 (`embedding_model="jina-api"`)

| Field | Default | Meaning |
|---|---|---|
| `jina_api_url` | Cloud Run URL | Remote `POST /embed` server |
| `jina_api_late_chunking` | `True` | Encode each window once and mean-pool per chunk (whole-window context) |
| `jina_api_window_tokens` | `3072` | Token budget per late-chunking pass |

## Vector store

| Field | Default | Meaning |
|---|---|---|
| `vector_backend` | `"chroma"` | `"chroma"` (persistent, embed-once) or `"faiss"` (legacy in-memory) |
| `faiss_metric` | `"cosine"` | `"cosine"` or `"l2"` (faiss only) |
| `index_dir` | `"rag/faiss_store"` | FAISS `index.faiss` + `meta.pkl` location |
| `chroma_dir` | `"chroma_store"` | Chroma `PersistentClient` path |

## Parallelism (bounded, to respect API rate limits)

| Field | Default | Meaning |
|---|---|---|
| `embed_batch_size` | `64` | Texts per embedding request |
| `embed_max_workers` | `8` | Concurrent embedding requests |
| `llm_max_workers` | `8` | Concurrent LLM / judge calls |
| `doc_max_workers` | `4` | Documents processed concurrently |
| `load_max_workers` | `4` | Process-pool workers for load+chunk (set `1` on Windows if unstable) |

## Retrieval

| Field | Default | Meaning |
|---|---|---|
| `retrieval_mode` | `"hybrid"` | `"dense"` or `"hybrid"` |
| `hybrid_alpha` | `0.85` | `0` = pure BM25 … `1` = pure dense |
| `top_k` | `25` | Candidates retrieved before reranking |

## Reranker

| Field | Default | Meaning |
|---|---|---|
| `use_reranker` | `True` | Toggle cross-encoder reranking |
| `reranker_model` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `rerank_top_k` | `10` | Candidates kept after reranking |

## Context assembly

| Field | Default | Meaning |
|---|---|---|
| `context_mode` | `"hierarchical"` | `"chunk"`\|`"full_page"`\|`"hierarchical"` |
| `hierarchical_max_depth` | `3` | Ancestor levels to climb in hierarchical mode |
| `save_pages_sent_to_llm` | `False` | Persist the exact context sent to the LLM |

## LLM

| Field | Default | Meaning |
|---|---|---|
| `llm_model` | `"gpt-5-mini"` | Generation + judge model (env: `LLM_MODEL`) |
| `llm_temperature` | `0.0` | `0.0` = deterministic; `None` = omit (reasoning models) |
| `llm_base_url` | SLB deployment URL | Model-ops API base URL (env: `LLM_BASE_URL`) |
| `partition_id` | `sandbox-...` | Deployment partition (env: `SLB_PARTITION_ID`) |
| `model_path` | `None` | Local model folder for Jina (auto-detects `./jina-v3`) |

## Chain-of-Verification (RAG Explorer)

| Field | Default | Meaning |
|---|---|---|
| `verify_enabled` | `True` | Run the CoVe confidence pass after answering |
| `verify_num_questions` | `5` | Corroboration checks per query |
| `verify_fresh_retrieval` | `True` | Re-retrieve context per check (higher signal) |
| `max_verify_rounds` | `3` | Enrichment only: max low-confidence correction rounds |

## Derived helpers (read-only)

`RagConfig` computes several values you may see in paths and logs:

- `resolved_embedding_model` — alias → concrete model id (or `tfidf`).
- `embedding_slug` — filesystem-safe per-model identifier.
- `chroma_collection` — `"<embedding_slug>__<chunk_strategy>"`.
- `resolved_index_dir` — `"<index_dir>/<embedding_slug>"`.
- `backend_tag()` — cache namespace per (model, context, merge).

## Environment variables

`RagConfig.from_env()` reads these:

| Variable | Overrides |
|---|---|
| `LLM_MODEL` | `llm_model` |
| `LLM_BASE_URL` | `llm_base_url` |
| `SLB_PARTITION_ID` | `partition_id` |
| `JINA_API_URL` | `jina_api_url` |

Token pricing for cost accounting can be overridden via
`LLM_INPUT_PRICE_PER_1K` and `LLM_OUTPUT_PRICE_PER_1K` (read by
`rag/evaluation/core.py`).
