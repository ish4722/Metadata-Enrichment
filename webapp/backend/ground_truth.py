"""Persist a human-verified, corrected ground-truth dataset for a document.

The Ground Truth Enrichment tool lets a reviewer accept / correct the values the
RAG pipeline proposes for each attribute, then saves the result as an enhanced
ground-truth record. Files land in ``enriched_ground_truth/<doc_id>.json`` in the
OSDU ``{"data": {...}}`` shape (real field keys), so the existing evaluation
tooling can read them straight away.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

# Resolved relative to the repo root (webapp/backend/ -> repo root is two up).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ENRICHED_DIR = os.path.join(_REPO_ROOT, "enriched_ground_truth")


def save_ground_truth(doc_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
    """Write ``{"data": data}`` to ``enriched_ground_truth/<doc_id>.json``.

    ``doc_id`` is reduced to a basename so it can never escape the target folder.
    Returns ``{"ok", "path", "count"}``.
    """
    if not isinstance(data, dict):
        raise ValueError("data must be an object of attribute -> value.")

    safe_id = os.path.basename(str(doc_id or "").strip()) or "document"
    if not safe_id.lower().endswith(".json"):
        safe_id = f"{safe_id}.json"

    os.makedirs(ENRICHED_DIR, exist_ok=True)
    dest = os.path.join(ENRICHED_DIR, safe_id)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump({"data": data}, f, indent=2, ensure_ascii=False)

    return {"ok": True, "path": dest, "count": len(data)}
