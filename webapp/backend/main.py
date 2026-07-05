"""FastAPI app for the RAG Explorer website.

Wraps the existing ``rag`` package: document discovery, embedding-model build
(embed-once into Chroma), a staged query stream (embed -> retrieve -> rerank ->
LLM -> validate -> highlight) and range-enabled PDF serving.

Run from the repo root:

    uvicorn webapp.backend.main:app --reload --port 8000

Long-running build/query endpoints stream newline-delimited JSON (NDJSON); the
frontend reads the response body incrementally to drive the flow animation.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterator, Optional

# ── make the repo root importable, then patch SSL before heavy imports ────────
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
# The backend package dir itself (so ``import engine`` / ``import highlights`` work).
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from rag.common.ssl_patch import apply_ssl_patches  # noqa: E402

apply_ssl_patches()

from fastapi import FastAPI, HTTPException, Request, UploadFile  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse  # noqa: E402

import rag_evaluation  # noqa: E402

import docs  # noqa: E402
import goldset_review  # noqa: E402
from engine import ENGINE  # noqa: E402

app = FastAPI(title="RAG Explorer")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

QUERIES_PATH = os.path.join(_REPO_ROOT, "queries.txt")
STATIC_DIR = os.path.join(_BACKEND_DIR, "static")


# ── helpers ───────────────────────────────────────────────────────────────────
def _ndjson(events: Iterator[Dict[str, Any]]) -> StreamingResponse:
    """Stream an iterator of event dicts as newline-delimited JSON."""

    def gen() -> Iterator[str]:
        try:
            for event in events:
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as exc:  # noqa: BLE001
            yield json.dumps({"stage": "error", "message": str(exc)}) + "\n"

    return StreamingResponse(gen(), media_type="application/x-ndjson")


def _load_matched(doc_id: str) -> Optional[Dict[str, Any]]:
    path = docs.matched_path(doc_id)
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ── metadata endpoints ──────────────────────────────────────────────────────
@app.get("/api/health")
def health() -> Dict[str, Any]:
    return {"ok": True}


@app.get("/api/embedding-models")
def embedding_models() -> Dict[str, Any]:
    return {"models": docs.EMBEDDING_MODELS, "default": "jina-api"}


@app.get("/api/documents")
def list_documents() -> Dict[str, Any]:
    return {"documents": docs.list_documents()}


@app.get("/api/documents/{doc_id}")
def get_document(doc_id: str) -> Dict[str, Any]:
    info = docs.describe_document(doc_id)
    if not info["has_datalab"] and not info["has_pdf"]:
        raise HTTPException(status_code=404, detail=f"No document for id '{doc_id}'.")
    return info


@app.get("/api/queries")
def list_queries() -> Dict[str, Any]:
    """Predefined, validatable queries parsed from queries.txt."""
    if not os.path.isfile(QUERIES_PATH):
        return {"queries": []}
    parsed = rag_evaluation.parse_queries_file(QUERIES_PATH)
    out = []
    for q in parsed:
        label = q.get("label", "")
        validatable = label.strip().lower() in rag_evaluation.GROUND_TRUTH_EXTRACTORS
        out.append({
            "label": label,
            "question": q.get("question", ""),
            "keywords": q.get("keywords", ""),
            "retrieval_query": q.get("retrieval_query", q.get("question", "")),
            "validatable": validatable,
        })
    return {"queries": out}


@app.post("/api/upload")
async def upload_pdf(file: UploadFile) -> JSONResponse:
    content = await file.read()
    info = docs.save_uploaded_pdf(file.filename or "upload.pdf", content)
    return JSONResponse(info)


# ── goldset review ───────────────────────────────────────────────────────────
@app.get("/review")
def review_page() -> Response:
    path = os.path.join(STATIC_DIR, "review.html")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="review.html not found.")
    return FileResponse(path, media_type="text/html")


@app.get("/api/goldset/docs")
def goldset_docs() -> Dict[str, Any]:
    return {"documents": goldset_review.list_docs()}


@app.get("/api/goldset/doc/{doc_id}")
def goldset_doc(doc_id: str) -> Dict[str, Any]:
    try:
        return goldset_review.doc_detail(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"No gold-set entry for '{doc_id}'.")


@app.post("/api/goldset/edit")
async def goldset_edit(request: Request) -> Dict[str, Any]:
    body = await request.json()
    doc_id = body.get("doc_id")
    label = body.get("label")
    chunk_id = body.get("chunk_id")
    action = body.get("action")
    if not all([doc_id, label, chunk_id, action]):
        raise HTTPException(status_code=400,
                            detail="doc_id, label, chunk_id and action are required.")
    try:
        return goldset_review.apply_edit(doc_id, label, chunk_id, action)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


# ── build ──────────────────────────────────────────────────────────────────
@app.post("/api/build")
async def build(request: Request) -> StreamingResponse:
    body = await request.json()
    doc_id = body.get("doc_id")
    model = body.get("embedding_model")
    if not doc_id or not model:
        raise HTTPException(status_code=400, detail="doc_id and embedding_model are required.")
    path = docs.datalab_path(doc_id)
    if not path:
        raise HTTPException(status_code=404, detail=f"No datalab output for '{doc_id}'.")
    return _ndjson(ENGINE.build(doc_id, path, overrides={"embedding_model": model}))


# ── query ────────────────────────────────────────────────────────────────────
@app.post("/api/query")
async def query(request: Request) -> StreamingResponse:
    body = await request.json()
    doc_id = body.get("doc_id")
    if not doc_id:
        raise HTTPException(status_code=400, detail="doc_id is required.")
    datalab = docs.datalab_path(doc_id)
    if not datalab:
        raise HTTPException(status_code=404, detail=f"No datalab output for '{doc_id}'.")

    label = (body.get("label") or "").strip() or None
    question = (body.get("question") or "").strip()
    retrieval_query = (body.get("retrieval_query") or "").strip() or question

    # Resolve ground truth for predefined (validatable) queries, and fill in any
    # name placeholders (e.g. "{well}") using the document's ground-truth name.
    matched = _load_matched(doc_id)
    ground_truth: Optional[Dict[str, Any]] = None
    if label and matched is not None:
        ground_truth = rag_evaluation.extract_ground_truth(label, matched)
    if matched is not None:
        well_name = rag_evaluation.resolve_well_name(matched)
        question = rag_evaluation.inject_name(question, well_name)
        retrieval_query = rag_evaluation.inject_name(retrieval_query, well_name)

    if not question:
        raise HTTPException(status_code=400, detail="A question (or label) is required.")

    overrides = {
        "embedding_model": body.get("embedding_model"),
        "retrieval_mode": body.get("retrieval_mode"),
        "hybrid_alpha": body.get("hybrid_alpha"),
        "use_reranker": body.get("use_reranker"),
        "top_k": body.get("top_k"),
    }

    return _ndjson(ENGINE.run_query(
        doc_id=doc_id,
        datalab_path=datalab,
        question=question,
        retrieval_query=retrieval_query,
        label=label,
        ground_truth=ground_truth,
        overrides=overrides,
    ))


# ── PDF serving (Range-enabled for large files) ──────────────────────────────
@app.get("/api/pdf/{doc_id}")
def get_pdf(doc_id: str, request: Request) -> Response:
    path = docs.pdf_path(doc_id)
    if not path or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail=f"No PDF for '{doc_id}'.")

    file_size = os.path.getsize(path)
    range_header = request.headers.get("range") or request.headers.get("Range")
    if not range_header:
        return FileResponse(path, media_type="application/pdf",
                            headers={"Accept-Ranges": "bytes"})

    # Parse "bytes=start-end"
    try:
        units, _, rng = range_header.partition("=")
        start_s, _, end_s = rng.partition("-")
        start = int(start_s) if start_s else 0
        end = int(end_s) if end_s else file_size - 1
    except ValueError:
        raise HTTPException(status_code=416, detail="Invalid Range header.")
    start = max(0, start)
    end = min(end, file_size - 1)
    if start > end:
        raise HTTPException(status_code=416, detail="Range not satisfiable.")

    chunk_size = end - start + 1

    def iter_file() -> Iterator[bytes]:
        with open(path, "rb") as f:
            f.seek(start)
            remaining = chunk_size
            block = 1024 * 1024
            while remaining > 0:
                data = f.read(min(block, remaining))
                if not data:
                    break
                remaining -= len(data)
                yield data

    headers = {
        "Content-Range": f"bytes {start}-{end}/{file_size}",
        "Accept-Ranges": "bytes",
        "Content-Length": str(chunk_size),
        "Content-Type": "application/pdf",
    }
    return StreamingResponse(iter_file(), status_code=206, headers=headers,
                            media_type="application/pdf")
