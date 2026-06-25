"""Map retrieved chunks back to bounding boxes on the rendered PDF.

Retrieved chunks are section-merged, so a chunk has no single bbox. But every
datalab content block carries a ``bbox`` and a resolvable ``leaf_section_id``
(via ``section_hierarchy``), and each chunk records the ``leaf_section_id`` of
the section it was built from. So we rebuild, per document, a map

    leaf_section_id -> [ {page, bbox} ]

and per-page block lists (for blocks that resolve to no section), reusing the
exact loader helpers so the section ids line up with what indexing produced.

Bounding boxes are normalized to ``[0, 1]`` against each page's own bbox (e.g.
``[0, 0, 1596, 2240]``) so the frontend can scale them onto a PDF.js viewport of
any size. The parsed index is cached per datalab path (these files are large).
"""

from __future__ import annotations

import json
import threading
from typing import Any, Dict, List, Optional, Tuple

from rag.common.schema import Chunk
from rag.indexing.loader import (
    KEEP_BLOCK_TYPES,
    _extraction_root,
    _iter_nodes,
    build_header_index,
    resolve_section_chain,
)

# datalab_path -> parsed index
_INDEX_CACHE: Dict[str, "DocIndex"] = {}
_CACHE_LOCK = threading.Lock()


class DocIndex:
    """Per-document geometry index built once from the datalab JSON."""

    def __init__(self) -> None:
        # page (int) -> [width, height] in datalab pixel space
        self.page_dims: Dict[int, Tuple[float, float]] = {}
        # leaf_section_id -> list of {"page": int, "bbox": [x0,y0,x1,y1]}
        self.section_blocks: Dict[str, List[Dict[str, Any]]] = {}
        # page (int) -> list of {"bbox": [...], "block_type": str}
        self.page_blocks: Dict[int, List[Dict[str, Any]]] = {}


def _build_index(datalab_path: str) -> DocIndex:
    with open(datalab_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    root = _extraction_root(data)
    header_text = build_header_index(root)
    index = DocIndex()

    # Page nodes carry no "page" attribute (only blocks do); their position in
    # document order is the 0-indexed page number that block ``page`` values use.
    for page_index_num, page_node in enumerate(root.get("children", [])):
        page_default = page_index_num
        page_bbox = page_node.get("bbox") or []
        if len(page_bbox) == 4:
            width = float(page_bbox[2]) - float(page_bbox[0])
            height = float(page_bbox[3]) - float(page_bbox[1])
            if width > 0 and height > 0:
                index.page_dims[page_index_num] = (width, height)

        for node in _iter_nodes(page_node):
            block_type = node.get("block_type", "Unknown")
            if block_type not in KEEP_BLOCK_TYPES:
                continue
            bbox = node.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            node_page = node.get("page", page_default)
            page_num = int(node_page) if isinstance(node_page, int) else -1

            ids, _texts = resolve_section_chain(node.get("section_hierarchy"), header_text)
            leaf = ids[-1] if ids else ""

            entry = {"page": page_num, "bbox": [float(v) for v in bbox], "block_type": block_type}
            if leaf:
                index.section_blocks.setdefault(leaf, []).append(entry)
            index.page_blocks.setdefault(page_num, []).append(entry)

    return index


def get_index(datalab_path: str) -> DocIndex:
    with _CACHE_LOCK:
        cached = _INDEX_CACHE.get(datalab_path)
        if cached is not None:
            return cached
    # Build outside the lock (parsing a large JSON can take seconds).
    index = _build_index(datalab_path)
    with _CACHE_LOCK:
        _INDEX_CACHE[datalab_path] = index
    return index


def _normalize(bbox: List[float], dims: Tuple[float, float]) -> Optional[Dict[str, float]]:
    width, height = dims
    if width <= 0 or height <= 0:
        return None
    x0, y0, x1, y1 = bbox
    nx0, nx1 = sorted((x0 / width, x1 / width))
    ny0, ny1 = sorted((y0 / height, y1 / height))
    return {"x": nx0, "y": ny0, "w": max(0.0, nx1 - nx0), "h": max(0.0, ny1 - ny0)}


def _chunk_pages(chunk: Chunk) -> List[int]:
    pages = chunk.metadata.get("pages") if isinstance(chunk.metadata, dict) else None
    if pages:
        try:
            return [int(p) for p in pages]
        except (TypeError, ValueError):
            pass
    return [int(chunk.page)]


def highlights_for_matches(
    datalab_path: str,
    matches: List[Tuple[Chunk, float]],
) -> List[Dict[str, Any]]:
    """Return per-page highlight rectangles for the matched chunks.

    Output shape (normalized 0..1 rectangles, ready to scale onto a viewport):

        [ { "page": int, "rects": [ {"x","y","w","h","chunk_id","score","rank"} ] } ]
    """
    index = get_index(datalab_path)
    # page -> list of rects
    by_page: Dict[int, List[Dict[str, Any]]] = {}

    for rank, (chunk, score) in enumerate(matches, start=1):
        pages = set(_chunk_pages(chunk))
        leaf = chunk.leaf_section_id or ""
        blocks: List[Dict[str, Any]] = []
        if leaf and leaf in index.section_blocks:
            blocks = [b for b in index.section_blocks[leaf] if b["page"] in pages]
        if not blocks:
            # Fallback: highlight all content blocks on the chunk's page(s).
            for page in pages:
                blocks.extend(index.page_blocks.get(page, []))

        for block in blocks:
            page = block["page"]
            dims = index.page_dims.get(page)
            if not dims:
                continue
            rect = _normalize(block["bbox"], dims)
            if rect is None:
                continue
            rect.update({"chunk_id": chunk.id, "score": round(float(score), 4), "rank": rank})
            by_page.setdefault(page, []).append(rect)

    return [{"page": page, "rects": rects} for page, rects in sorted(by_page.items())]
