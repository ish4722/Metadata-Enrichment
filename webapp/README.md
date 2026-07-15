# RAG Explorer — Web App

An interactive UI over the `rag` pipeline: pick a document, build its index, ask
questions, watch the pipeline stream stage by stage, and see the answer
highlighted back in the source PDF. Also supports metadata enrichment, ground-truth
generation, and gold-set review.

> This is the quick reference for working inside `webapp/`. For the full
> architecture, endpoint list, and component breakdown, see
> [`../docs/webapp.md`](../docs/webapp.md).

## Stack

| Layer | Tech | Folder |
|---|---|---|
| Backend | FastAPI + uvicorn | [`backend/`](backend/) |
| Frontend | React 18 + Vite | [`frontend/`](frontend/) |
| PDF rendering | `pdfjs-dist` | frontend |

The backend imports the `rag` package directly, so the UI and CLI always share
the exact same pipeline behaviour.

## Prerequisites

Install from the **repository root** (one level up):

```bash
# Pipeline deps (required by the backend, which wraps the rag package)
pip install -r rag/requirements.txt

# Backend web deps
pip install -r webapp/backend/requirements.txt

# Frontend deps
cd webapp/frontend && npm install && cd ../..
```

Node.js 18+ is required for the frontend.

## Run it (two terminals, from the repo root)

```bash
# Terminal 1 — API on http://localhost:8000
uvicorn webapp.backend.main:app --reload --port 8000

# Terminal 2 — UI on http://localhost:5173
cd webapp/frontend
npm run dev
```

Open http://localhost:5173 in your browser.

### Build a production frontend bundle

```bash
cd webapp/frontend
npm run build      # outputs to dist/
npm run preview    # serve the built bundle locally
```

## Configuration

The backend uses the same `RagConfig` defaults as the CLI. Set these environment
variables **before** launching uvicorn (see
[`../docs/configuration.md`](../docs/configuration.md)):

| Variable | Purpose |
|---|---|
| `LLM_MODEL` | Generation + judge LLM |
| `LLM_BASE_URL` | Internal model-ops API base URL |
| `SLB_PARTITION_ID` | Deployment partition ID |
| `JINA_API_URL` | Hosted Jina v3 embedding server |

The frontend's API base URL lives in [`frontend/src/api.js`](frontend/src/api.js).

## Folder layout

```
webapp/
├── backend/
│   ├── main.py            # FastAPI app, routes, NDJSON streaming
│   ├── engine.py          # ENGINE singleton wrapping the rag build/query pipeline
│   ├── enrichment.py      # metadata enrichment flow (uses verification)
│   ├── ground_truth.py    # ground-truth generation / save
│   ├── goldset_review.py  # gold-set inspection & editing
│   ├── docs.py            # document discovery helpers
│   ├── highlights.py      # map answers back to PDF spans for highlighting
│   ├── static/            # standalone review / analyzer HTML pages
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── vite.config.js
    ├── package.json
    └── src/
        ├── main.jsx  App.jsx  api.js  styles.css
        └── components/   # Home, DocumentPicker, BuildPanel, QueryPanel,
                          # PipelineFlow, ChunksView, AnswerValidation,
                          # CostPanel, PdfViewer, Enrichment, GroundTruth
```

## Key API endpoints

| Method & path | Purpose |
|---|---|
| `GET /api/health` | Liveness check |
| `GET /api/documents` | List documents |
| `GET /api/embedding-models` | Available embedding backends |
| `POST /api/build` | Build (embed-once) a document's index — NDJSON stream |
| `POST /api/query` | Run a staged query (retrieve → rerank → LLM → verify) — NDJSON stream |
| `POST /api/reverify` | Re-run verification on an answer |
| `POST /api/enrichment/run` | Run the metadata enrichment flow |
| `GET /api/pdf/{doc_id}` | Serve the source PDF (range-enabled) |

Full endpoint list in [`../docs/webapp.md`](../docs/webapp.md). Long-running
build/query endpoints stream **newline-delimited JSON (NDJSON)**; the frontend
reads the response body incrementally to drive the flow animation.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Frontend can't reach the API | Start the backend on port 8000 first; check the base URL in `frontend/src/api.js` |
| `ModuleNotFoundError: rag...` | Launch uvicorn from the **repo root**, with your venv active |
| Auth / 401 from the LLM | Check `LLM_BASE_URL`, `SLB_PARTITION_ID`, and `rag/common/sauth_token.py` |
| PDF won't render/highlight | Ensure the source PDF exists in `pdfs/` for that doc id |

More in [`../docs/troubleshooting.md`](../docs/troubleshooting.md).
