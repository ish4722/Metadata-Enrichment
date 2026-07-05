"""Document discovery + PDF upload, keyed by the project's ``PR####`` doc id.

A document is the trio that share a doc id:

    pdfs/<id>.pdf          - the source PDF (rendered in the browser)
    datalab-output/<id>.json - the extraction JSON (chunks + bboxes)
    matched_json/<id>.json - the structured ground truth (for validation)

The doc id is derived with the same ``derive_doc_id`` used by the indexing
pipeline, so filenames like ``datalab-output-PR2610 (1).json`` resolve to
``PR2610`` and line up across folders.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from rag.config import EMBEDDING_ALIASES, RagConfig
from rag.indexing.build_pipeline import derive_doc_id
from rag.indexing.chroma_store import get_chroma_client

# Resolved relative to the repo root (set on import by main.py via REPO_ROOT).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PDF_DIR = os.path.join(REPO_ROOT, "pdfs")
DATALAB_DIR = os.path.join(REPO_ROOT, "datalab-output")
MATCHED_DIR = os.path.join(REPO_ROOT, "matched_json")

# Embedding models exposed in the UI (friendly aliases from the config).
# "jina-api" = the remotely hosted jina-embeddings-v3 server (late chunking on).
EMBEDDING_MODELS = ["jina-api", "openai", "jina", "minilm", "tfidf"]


def _scan_ids(directory: str, ext: str) -> Dict[str, str]:
    """Map doc id -> absolute file path for every ``ext`` file in a directory.

    ``ext`` is matched case-insensitively (e.g. ".pdf"). This matters because the
    ``pdfs/`` folder also contains stray .json copies that share a doc id; only
    actual PDFs should resolve as the PDF path.
    """
    out: Dict[str, str] = {}
    if not os.path.isdir(directory):
        return out
    ext = ext.lower()
    for name in os.listdir(directory):
        if not name.lower().endswith(ext):
            continue
        doc_id = derive_doc_id(name)
        if doc_id and doc_id not in out:
            out[doc_id] = os.path.join(directory, name)
    return out


def datalab_path(doc_id: str) -> Optional[str]:
    return _scan_ids(DATALAB_DIR, ".json").get(doc_id)


def pdf_path(doc_id: str) -> Optional[str]:
    return _scan_ids(PDF_DIR, ".pdf").get(doc_id)


def matched_path(doc_id: str) -> Optional[str]:
    return _scan_ids(MATCHED_DIR, ".json").get(doc_id)


# ── built-model detection (cheap: no embedding model is loaded) ───────────────
_client_cache: Dict[str, Any] = {}


def _client(chroma_dir: str) -> Any:
    abs_dir = chroma_dir if os.path.isabs(chroma_dir) else os.path.join(REPO_ROOT, chroma_dir)
    if abs_dir not in _client_cache:
        _client_cache[abs_dir] = get_chroma_client(abs_dir)
    return _client_cache[abs_dir]


def built_models(doc_id: str) -> List[str]:
    """Which embedding models already have this doc embedded in Chroma.

    Resolves each model's collection name via :class:`RagConfig` (a pure
    property) and asks Chroma whether the doc id is present - no heavyweight
    embedding backend is constructed.
    """
    built: List[str] = []
    for model in EMBEDDING_MODELS:
        cfg = RagConfig.from_env()
        cfg.embedding_model = model
        try:
            client = _client(cfg.chroma_dir)
            collection = client.get_collection(name=cfg.chroma_collection)
            got = collection.get(where={"doc_id": doc_id}, limit=1, include=[])
            if got.get("ids"):
                built.append(model)
        except Exception:
            continue
    return built


def describe_document(doc_id: str, *, with_built: bool = True) -> Dict[str, Any]:
    return {
        "id": doc_id,
        "has_pdf": pdf_path(doc_id) is not None,
        "has_datalab": datalab_path(doc_id) is not None,
        "has_matched": matched_path(doc_id) is not None,
        "built_models": built_models(doc_id) if with_built else [],
    }


def list_documents() -> List[Dict[str, Any]]:
    """All docs that have a datalab extraction (the minimum needed to build)."""
    ids = sorted(set(_scan_ids(DATALAB_DIR, ".json").keys()) | set(_scan_ids(PDF_DIR, ".pdf").keys()))
    return [describe_document(doc_id) for doc_id in ids]


def save_uploaded_pdf(filename: str, content: bytes) -> Dict[str, Any]:
    """Persist an uploaded PDF into ``pdfs/`` and report what it matches."""
    os.makedirs(PDF_DIR, exist_ok=True)
    safe_name = os.path.basename(filename) or "upload.pdf"
    if not safe_name.lower().endswith(".pdf"):
        safe_name += ".pdf"
    dest = os.path.join(PDF_DIR, safe_name)
    with open(dest, "wb") as f:
        f.write(content)

    doc_id = derive_doc_id(safe_name)
    info = describe_document(doc_id)
    info["uploaded_as"] = safe_name
    return info
