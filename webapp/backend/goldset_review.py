"""Backend for the standalone Goldset Review page.

Loads the graded retrieval gold set (``output_table/retrieval_goldset.json``),
joins each relevant chunk back to its text (from Chroma) and its bounding boxes
on the rendered PDF (reusing :mod:`highlights`), and supports manual edits
(flip a chunk's verdict between correct/wrong, or drop it) that are persisted
back to the gold set file with the same recomputed status/summary the builder
produces.

Grading scheme (matches build_retrieval_goldset.py):
    grade 2 = correct (tick)   grade 1 = wrong (cross)   dropped = removed
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional

from rag.config import RagConfig
from rag.indexing.chroma_store import _metadata_to_chunk
from rag.indexing.chroma_store import get_chroma_client

import docs as docs_mod
from highlights import highlights_for_matches

# Labels (order shown in the UI) and grading constants, mirrored from the builder.
ALL_LABELS = ["facility name", "openv ID", "facilityevent spud date", "lat+long"]
VERDICT_GRADE = {"correct": 2, "wrong": 1}
GRADE_VERDICT = {2: "correct", 1: "wrong"}

GOLDSET_PATH = os.path.join(docs_mod.REPO_ROOT, "output_table", "retrieval_goldset.json")

_LOCK = threading.RLock()
# Cache of doc_id -> {chunk_id: Chunk} reconstructed from Chroma.
_chunk_cache: Dict[str, Dict[str, Any]] = {}
_collection_cache: Dict[str, Any] = {}


# ── gold set IO ──────────────────────────────────────────────────────────────
def _load() -> Dict[str, Any]:
    with open(GOLDSET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_atomic(data: Dict[str, Any]) -> None:
    tmp = GOLDSET_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, GOLDSET_PATH)


# ── Chroma chunk text/metadata ───────────────────────────────────────────────
def _collection() -> Any:
    cfg = RagConfig.from_env()
    key = f"{cfg.chroma_dir}::{cfg.chroma_collection}"
    if key not in _collection_cache:
        client = get_chroma_client(cfg.chroma_dir)
        _collection_cache[key] = client.get_collection(name=cfg.chroma_collection)
    return _collection_cache[key]


def _doc_chunks(doc_id: str) -> Dict[str, Any]:
    """Return {chunk_id -> Chunk} for a doc, reconstructed from Chroma (cached)."""
    if doc_id in _chunk_cache:
        return _chunk_cache[doc_id]
    out: Dict[str, Any] = {}
    try:
        got = _collection().get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
        ids = got.get("ids") or []
        texts = got.get("documents") or []
        metas = got.get("metadatas") or []
        for cid, text, md in zip(ids, texts, metas):
            out[cid] = _metadata_to_chunk(cid, text or "", md or {})
    except Exception:
        out = {}
    _chunk_cache[doc_id] = out
    return out


# ── status / summary recomputation (mirrors the builder) ─────────────────────
def _status_for(value: Any, relevant: List[Dict[str, Any]]) -> str:
    if value is None:
        return "no_ground_truth"
    if any(r["grade"] == 2 for r in relevant):
        return "answerable"
    if relevant:
        return "only_wrong"
    return "not_grounded"


def _recompute_label(entry: Dict[str, Any]) -> None:
    relevant = sorted(entry.get("relevant", []), key=lambda r: -r["grade"])
    entry["relevant"] = relevant
    entry["status"] = _status_for(entry.get("value"), relevant)
    entry["correct_chunk_ids"] = [r["chunk_id"] for r in relevant if r["grade"] == 2]
    entry["relevant_pages"] = sorted({r["page"] for r in relevant if r["grade"] == 2})


def _recompute_summary(data: Dict[str, Any]) -> None:
    goldset = data.get("docs", {})
    per_label: Dict[str, Any] = {}
    grade_counts: Dict[str, Any] = {}
    for lbl in ALL_LABELS:
        c: Dict[str, int] = defaultdict(int)
        g = {"correct": 0, "wrong": 0}
        for entry in goldset.values():
            if lbl in entry:
                c[entry[lbl]["status"]] += 1
                g["correct"] += sum(1 for r in entry[lbl]["relevant"] if r["grade"] == 2)
                g["wrong"] += sum(1 for r in entry[lbl]["relevant"] if r["grade"] == 1)
        per_label[lbl] = dict(c)
        grade_counts[lbl] = g
    not_grounded = [
        {"doc_id": doc_id, "label": lbl, "value": entry[lbl]["value"]}
        for doc_id, entry in goldset.items() for lbl in entry
        if entry[lbl]["status"] == "not_grounded"
    ]
    data["not_grounded"] = not_grounded
    summary = data.setdefault("summary", {})
    summary["per_label"] = per_label
    summary["grade_counts"] = grade_counts


# ── public API ───────────────────────────────────────────────────────────────
def list_docs() -> List[Dict[str, Any]]:
    """Doc ids present in the gold set, with correct/wrong chunk counts."""
    data = _load()
    goldset = data.get("docs", {})
    out: List[Dict[str, Any]] = []
    for doc_id in sorted(goldset.keys()):
        entry = goldset[doc_id]
        correct = wrong = 0
        for lbl in ALL_LABELS:
            for r in (entry.get(lbl, {}) or {}).get("relevant", []):
                if r["grade"] == 2:
                    correct += 1
                elif r["grade"] == 1:
                    wrong += 1
        out.append({
            "doc_id": doc_id,
            "correct": correct,
            "wrong": wrong,
            "has_pdf": docs_mod.pdf_path(doc_id) is not None,
        })
    return out


def _rects_for_chunk(doc_id: str, chunk: Any) -> List[Dict[str, Any]]:
    datalab = docs_mod.datalab_path(doc_id)
    if not datalab or chunk is None:
        return []
    try:
        pages = highlights_for_matches(datalab, [(chunk, 1.0)])
    except Exception:
        return []
    rects: List[Dict[str, Any]] = []
    for page in pages:
        for r in page.get("rects", []):
            rects.append({"page": page["page"], "x": r["x"], "y": r["y"],
                          "w": r["w"], "h": r["h"]})
    return rects


def doc_detail(doc_id: str) -> Dict[str, Any]:
    data = _load()
    entry = (data.get("docs", {}) or {}).get(doc_id)
    if entry is None:
        raise KeyError(doc_id)
    chunks_by_id = _doc_chunks(doc_id)

    fields: List[Dict[str, Any]] = []
    for lbl in ALL_LABELS:
        label_entry = entry.get(lbl)
        if not label_entry:
            continue
        chunk_rows: List[Dict[str, Any]] = []
        for r in label_entry.get("relevant", []):
            chunk = chunks_by_id.get(r["chunk_id"])
            chunk_rows.append({
                "chunk_id": r["chunk_id"],
                "page": r["page"],
                "grade": r["grade"],
                "verdict": r.get("verdict", GRADE_VERDICT.get(r["grade"], "")),
                "matched_by": r.get("matched_by", ""),
                "text": getattr(chunk, "text", "") if chunk else "",
                "section_path": getattr(chunk, "section_path", "") if chunk else "",
                "rects": _rects_for_chunk(doc_id, chunk),
            })
        fields.append({
            "label": lbl,
            "value": label_entry.get("value"),
            "status": label_entry.get("status"),
            "chunks": chunk_rows,
        })
    return {
        "doc_id": doc_id,
        "has_pdf": docs_mod.pdf_path(doc_id) is not None,
        "fields": fields,
    }


def apply_edit(doc_id: str, label: str, chunk_id: str, action: str) -> Dict[str, Any]:
    """action in {'correct','wrong','drop'}; persists and returns the new field entry."""
    if action not in ("correct", "wrong", "drop"):
        raise ValueError(f"unknown action '{action}'")
    with _LOCK:
        data = _load()
        entry = (data.get("docs", {}) or {}).get(doc_id)
        if entry is None or label not in entry:
            raise KeyError(f"{doc_id} / {label}")
        label_entry = entry[label]
        relevant = label_entry.get("relevant", [])

        if action == "drop":
            label_entry["relevant"] = [r for r in relevant if r["chunk_id"] != chunk_id]
        else:
            found = False
            for r in relevant:
                if r["chunk_id"] == chunk_id:
                    r["grade"] = VERDICT_GRADE[action]
                    r["verdict"] = action
                    r["matched_by"] = "manual"
                    found = True
                    break
            if not found:
                raise KeyError(f"chunk {chunk_id} not in {doc_id}/{label}")

        _recompute_label(label_entry)
        _recompute_summary(data)
        _save_atomic(data)

    # Return a refreshed detail entry for this label.
    chunks_by_id = _doc_chunks(doc_id)
    chunk_rows: List[Dict[str, Any]] = []
    for r in label_entry.get("relevant", []):
        chunk = chunks_by_id.get(r["chunk_id"])
        chunk_rows.append({
            "chunk_id": r["chunk_id"],
            "page": r["page"],
            "grade": r["grade"],
            "verdict": r.get("verdict", GRADE_VERDICT.get(r["grade"], "")),
            "matched_by": r.get("matched_by", ""),
            "text": getattr(chunk, "text", "") if chunk else "",
            "section_path": getattr(chunk, "section_path", "") if chunk else "",
            "rects": _rects_for_chunk(doc_id, chunk),
        })
    return {
        "label": label,
        "value": label_entry.get("value"),
        "status": label_entry.get("status"),
        "chunks": chunk_rows,
    }
