"""Step 1 - Load extraction JSON and flatten it into text blocks.

Input  : the JSON produced by the PDF extraction tool (datalab output).
Output : a flat list of :class:`Chunk` (one per useful content block).

Section structure
-----------------
In the datalab layout every page node holds a *flat* list of blocks
(``Text`` / ``Table`` / ``Form`` / ``SectionHeader`` / ...). ``SectionHeader``
nodes never have children, so the document tree does NOT encode which section a
block belongs to. The HTML heading level (``<h1>``..``<h6>``) is assigned by font
size and is unreliable for nesting.

The authoritative signal is each block's ``section_hierarchy`` dict, shaped like
``{"1": "/page/0/SectionHeader/4", "2": "/page/0/SectionHeader/5"}`` - a map from
nesting level to the *global* id of the SectionHeader that owns the block at that
level. Ids are global (``/page/N/SectionHeader/M``) so they resolve to headers on
earlier pages, giving cross-page section continuity.

This loader therefore:
  1. indexes every ``SectionHeader`` id -> heading text (pass 1), then
  2. resolves each content block's ``section_hierarchy`` into a parent breadcrumb
     (level1 > level2 > ... > leaf) plus its leaf/parent section ids (pass 2).
The breadcrumb is stored as ``section_path`` and the ids as ``leaf_section_id`` /
``parent_section_id`` so the chunker can group blocks by section and prepend
parent context.
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rag.common.schema import Chunk

# Block types we keep as retrievable content. ``Form`` carries key/value
# metadata (e.g. "Well Name  Goldie-2  Date  6/03/2006"), ``ListGroup`` /
# ``TableOfContents`` / ``Caption`` carry useful text, so they are kept too.
# ``SectionHeader`` is intentionally NOT here: its text is folded into the
# breadcrumb prefix of every block beneath it, so emitting it as its own chunk
# would only add tiny duplicate fragments. Page headers/footers are noise.
KEEP_BLOCK_TYPES = {"Text", "Table", "Form", "ListGroup", "TableOfContents", "Caption"}


def html_to_text(raw_html: str) -> str:
    """Strip HTML tags / scripts / styles and collapse whitespace."""
    raw_html = html.unescape(raw_html or "")
    raw_html = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    raw_html = re.sub(r"<style.*?>.*?</style>", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
    raw_html = re.sub(r"<script.*?>.*?</script>", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
    raw_html = re.sub(r"<.*?>", "", raw_html)
    raw_html = re.sub(r"\s+", " ", raw_html)
    return raw_html.strip()


def _iter_nodes(node: Dict[str, Any]):
    """Yield ``node`` and all of its descendants (depth-first)."""
    yield node
    for child in node.get("children") or []:
        yield from _iter_nodes(child)


def build_header_index(root: Dict[str, Any]) -> Dict[str, str]:
    """Map every ``SectionHeader`` global id -> its heading text."""
    header_text: Dict[str, str] = {}
    for page_node in root.get("children", []):
        for node in _iter_nodes(page_node):
            if node.get("block_type") == "SectionHeader":
                text = html_to_text(node.get("html", ""))
                node_id = node.get("id", "")
                if text and node_id:
                    header_text[node_id] = text
    return header_text


def resolve_section_chain(
    section_hierarchy: Optional[Dict[str, str]],
    header_text: Dict[str, str],
) -> Tuple[List[str], List[str]]:
    """Resolve a ``section_hierarchy`` dict into (header_ids, header_texts).

    Ordered shallow -> deep by integer level key. Unknown ids (header text not
    found) are dropped from the text list but kept in the id list so the leaf/
    parent ids stay accurate.
    """
    if not section_hierarchy:
        return [], []
    try:
        ordered = sorted(section_hierarchy.items(), key=lambda kv: int(kv[0]))
    except (TypeError, ValueError):
        ordered = sorted(section_hierarchy.items())
    ids = [hid for _level, hid in ordered if hid]
    texts = [header_text.get(hid, "") for hid in ids]
    return ids, texts


def normalize_section_path(section_titles: List[str]) -> str:
    """Join a stack of section headings into a single ' > ' separated path."""
    return " > ".join(t.strip() for t in section_titles if t and t.strip())


def _extraction_root(data: Any) -> Dict[str, Any]:
    """Return the node whose ``children`` are the document pages.

    Supports two datalab layouts:
      * legacy:  the page tree is at the top level (``data["children"]``).
      * current: the tree is nested under an API wrapper
        (``data["result"]["json"]["children"]``).
    """
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("children"), list):
        return data

    result = data.get("result")
    if isinstance(result, dict):
        nested = result.get("json")
        if isinstance(nested, dict) and isinstance(nested.get("children"), list):
            return nested
        if isinstance(result.get("children"), list):
            return result

    nested = data.get("json")
    if isinstance(nested, dict) and isinstance(nested.get("children"), list):
        return nested

    return data


def load_extracted_json(path: str) -> List[Chunk]:
    """Load the extraction JSON file and return a flat list of Chunks.

    Each kept content block carries its resolved section breadcrumb
    (``section_path``) and section ids (``leaf_section_id`` / ``parent_section_id``)
    plus a document-order ``order`` index used to keep multi-page sections in
    reading order during chunking.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    root = _extraction_root(data)
    source = Path(path).name

    header_text = build_header_index(root)

    chunks: List[Chunk] = []
    order = 0
    for page_node in root.get("children", []):
        page_default = page_node.get("page", -1)
        for node in _iter_nodes(page_node):
            block_type = node.get("block_type", "Unknown")
            if block_type not in KEEP_BLOCK_TYPES:
                continue
            node_html = node.get("html", "")
            if not node_html:
                continue
            text = html_to_text(node_html)
            if not text:
                continue

            node_page = node.get("page", page_default)
            page_num = int(node_page) if isinstance(node_page, int) else -1

            ids, texts = resolve_section_chain(node.get("section_hierarchy"), header_text)
            section_path = normalize_section_path(texts)
            leaf_section_id = ids[-1] if ids else ""
            parent_section_id = ids[-2] if len(ids) >= 2 else ""

            chunks.append(
                Chunk(
                    id=node.get("id", "") or f"{source}::block::{order}",
                    source=source,
                    page=page_num,
                    section_path=section_path,
                    block_type=block_type,
                    text=text,
                    leaf_section_id=leaf_section_id,
                    parent_section_id=parent_section_id,
                    order=order,
                    metadata={
                        "block_id": node.get("id", ""),
                        "block_type": block_type,
                        "page": page_num,
                        "section_path": section_path,
                        "section_hierarchy_ids": ids,
                        "leaf_section_id": leaf_section_id,
                        "parent_section_id": parent_section_id,
                    },
                )
            )
            order += 1

    return chunks
