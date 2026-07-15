# Web app — RAG Explorer

An interactive UI over the `rag` pipeline: pick a document, build its index,
ask questions, watch the pipeline stream stage by stage, and see the answer
highlighted back in the source PDF. It also supports metadata enrichment and
gold-set review.

## Stack

| Layer | Tech | Location |
|---|---|---|
| Backend | FastAPI + uvicorn | [`webapp/backend/`](../webapp/backend/) |
| Frontend | React 18 + Vite | [`webapp/frontend/`](../webapp/frontend/) |
| PDF rendering | `pdfjs-dist` | frontend |

The backend imports the `rag` package directly, so the UI and CLI share the exact
same pipeline behaviour.

## Run it

Two terminals, both from the repo root:

```bash
# Terminal 1 — API on http://localhost:8000
uvicorn webapp.backend.main:app --reload --port 8000

# Terminal 2 — UI on http://localhost:5173
cd webapp/frontend
npm install          # first time only
npm run dev
```

Build a static frontend bundle with `npm run build` (output in `dist/`), preview
it with `npm run preview`.

## What you can do

| Feature | Description |
|---|---|
| Document discovery | Browse the docs available in `datalab-output/` |
| Build | Embed-once a document's chunks into Chroma for a chosen embedding model |
| Query stream | Ask a question and watch retrieve → rerank → LLM → verify stream live (NDJSON) |
| Answer validation | See the answer, confidence score, and supporting chunks |
| PDF highlight | Jump to and highlight the source passage in the original PDF |
| Enrichment | Run the enrichment flow that fills verified metadata attributes |
| Ground truth | Generate and save ground-truth records |
| Gold-set review | Inspect and edit the retrieval gold set |

## Backend API (selected endpoints)

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Liveness check |
| `GET /api/embedding-models` | List available embedding backends |
| `GET /api/documents` | List documents |
| `GET /api/documents/{doc_id}` | Document detail |
| `GET /api/queries` | Evaluation query labels |
| `GET /api/matched/{doc_id}` | Ground-truth record |
| `POST /api/upload` | Upload a PDF/JSON |
| `POST /api/build` | Build (embed-once) a document's index (streamed) |
| `POST /api/query` | Run a staged query (NDJSON stream) |
| `POST /api/reverify` | Re-run verification on an answer |
| `GET /api/enrichment/documents` · `GET /api/enrichment/source/{doc_id}` · `POST /api/enrichment/run` | Enrichment flow |
| `GET /api/ground-truth/documents` · `POST /api/ground-truth/run` · `POST /api/ground-truth/save` | Ground-truth flow |
| `GET /api/goldset/docs` · `GET /api/goldset/doc/{doc_id}` · `POST /api/goldset/edit` | Gold-set review |
| `GET /api/pdf/{doc_id}` | Serve the source PDF (range-enabled) |
| `GET /review` | Static gold-set review page |

Long-running build/query endpoints stream **newline-delimited JSON (NDJSON)**; the
frontend reads the response body incrementally to drive the flow animation.

## Backend modules (`webapp/backend/`)

| File | Responsibility |
|---|---|
| `main.py` | FastAPI app, routes, NDJSON streaming |
| `engine.py` | `ENGINE` singleton wrapping the rag build/query pipeline |
| `enrichment.py` | Metadata enrichment flow (uses verification) |
| `ground_truth.py` | Ground-truth generation/save |
| `goldset_review.py` | Gold-set inspection & editing |
| `docs.py` | Document discovery helpers |
| `highlights.py` | Map answers back to PDF spans for highlighting |

## Frontend components (`webapp/frontend/src/components/`)

`Home`, `DocumentPicker`, `BuildPanel`, `QueryPanel`, `PipelineFlow`,
`ChunksView`, `AnswerValidation`, `CostPanel`, `PdfViewer`, `Enrichment`,
`GroundTruth`. The API client lives in `src/api.js`.

## Configuration

The backend uses the same `RagConfig` defaults as the CLI (see
[configuration.md](configuration.md)); set `LLM_MODEL`, `LLM_BASE_URL`,
`SLB_PARTITION_ID`, etc. in the environment before launching uvicorn. If the
frontend cannot reach the API, check the base URL in `src/api.js` and that the
backend is running on port 8000.
