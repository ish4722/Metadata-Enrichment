# Troubleshooting

## Imports & environment

**`ModuleNotFoundError: No module named 'rag'` (or `webapp`)**
- Run from the **repository root**, not from inside `rag/` or `webapp/`.
- Activate your virtual environment (`.\.venv\Scripts\Activate.ps1` on Windows,
  `source .venv/bin/activate` elsewhere).
- Use the module form: `python -m rag.main ...`, not `python rag/main.py` from a
  subfolder.

**`ImportError` for `faiss`, `torch`, `transformers`, `chromadb`**
- `pip install -r rag/requirements.txt`.
- `torch` / `transformers` / `sentence-transformers` are optional; without them
  the system falls back to TF-IDF dense retrieval (`--embedding-model tfidf`).

## SSL / model download

**`SSL: CERTIFICATE_VERIFY_FAILED` when loading a model / calling the API**
- This is handled by `rag/common/ssl_patch.py`, which must import **before**
  `transformers` / `huggingface_hub`. Entry points call `apply_ssl_patches()`
  early; if you write a new script, do the same.
- For embeddings, prefer a local snapshot: `python -m rag.data_prep.download`
  then `--embedding-model jina` (auto-detects `./jina-v3`).
- The message `Returning existing local_dir jina-v3 as remote repo cannot be
  accessed` is benign — it means the local snapshot is being used.

## Authentication / LLM

**`401`/`403` or auth errors from the LLM or embeddings API**
- Check `LLM_BASE_URL`, `SLB_PARTITION_ID`, and the embeddings base URL/partition
  in `rag/config.py`.
- Auth tokens come from `rag/common/sauth_token.py` (a local, git-ignored helper);
  make sure it exists and returns a valid token.

**Model not available in the LLM benchmark**
- `benchmark_llms` pre-flights each model and skips unavailable ones. Use
  `--no-preflight` to force, or `--models` to restrict to known-good models.

## Performance & stability

**Build hangs or crashes on Windows during load+chunk**
- Add `--load-max-workers 1` to `rag/inference/batch_runner.py` (limits the
  process pool).

**Runs are slow / hit rate limits**
- Lower `embed_max_workers`, `llm_max_workers`, `doc_max_workers` in
  `rag/config.py` (or the matching CLI flags).

**"It re-embedded everything again"**
- Embed-once keys on (embedding model × chunk strategy). Changing
  `--embedding-model`, `--chunk-strategy`, or `--openai-embedding-dimensions`
  creates a **new** Chroma collection by design. Reuse the exact same settings to
  hit the cache.

## Evaluation

**Missing gold set**
- Retrieval metrics and the Matryoshka study need
  `output_table/retrieval_goldset.json`. Build it first:
  `python -m rag.scripts.build_retrieval_goldset --num-docs 40 --restart --concurrency 12 --batch-size 8`.

**Accuracy looks wrong for dates/coordinates**
- The judge already tolerates dates ±1 day and converts coordinate systems to
  WGS84. If a value still mismatches, inspect the saved context with
  `--save-pages` (or `validate_query --show-chunks`).

## Web app

**Frontend can't reach the backend**
- Start the backend first: `uvicorn webapp.backend.main:app --reload --port 8000`.
- Check the API base URL in `webapp/frontend/src/api.js` and CORS settings.

**PDF doesn't render / highlight**
- Ensure the source PDF exists in `pdfs/` for that doc id; the backend serves it
  via `GET /api/pdf/{doc_id}`.

## Still stuck?

- Re-run the failing command with `--help` to confirm flags.
- Reproduce on a single document with `--only PR2610` (or the `evaluate` command)
  to shrink the problem.
- Check the relevant guide: [pipeline.md](pipeline.md),
  [configuration.md](configuration.md), [cli-reference.md](cli-reference.md).
