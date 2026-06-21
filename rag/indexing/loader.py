"""Step 1 - Load extraction JSON and flatten it into text blocks.

Input  : the JSON produced by the PDF extraction tool (datalab output).
Output : a flat list of :class:`Chunk` (one per useful HTML block).
"""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from rag.common.schema import Chunk

# Block types we keep as retrievable text.
KEEP_BLOCK_TYPES = {"Text", "SectionHeader", "Table", "PageHeader"}


def html_to_text(raw_html: str) -> str:
    """Strip HTML tags / scripts / styles and collapse whitespace."""
    raw_html = html.unescape(raw_html or "")
    raw_html = re.sub(r"<br\s*/?>", "\n", raw_html, flags=re.IGNORECASE)
    raw_html = re.sub(r"<style.*?>.*?</style>", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
    raw_html = re.sub(r"<script.*?>.*?</script>", "", raw_html, flags=re.IGNORECASE | re.DOTALL)
    raw_html = re.sub(r"<.*?>", "", raw_html)
    raw_html = re.sub(r"\s+", " ", raw_html)
    return raw_html.strip()


def normalize_section_path(section_titles: List[str]) -> str:
    """Join a stack of section headings into a single ' > ' separated path."""
    return " > ".join(t.strip() for t in section_titles if t and t.strip())


def flatten_blocks(
    node: Dict[str, Any],
    page: int = -1,
    section_titles: Optional[List[str]] = None,
) -> List[Chunk]:
    """Recursively walk the extraction tree, emitting one Chunk per kept block."""
    if section_titles is None:
        section_titles = []

    blocks: List[Chunk] = []
    node_page = node.get("page", page)
    node_block_type = node.get("block_type", "Unknown")
    node_id = node.get("id", "")
    node_html = node.get("html", "")

    if node_block_type == "SectionHeader" and node_html:
        heading_text = html_to_text(node_html)
        if heading_text:
            section_titles = section_titles + [heading_text]

    if node_block_type in KEEP_BLOCK_TYPES and node_html:
        text = html_to_text(node_html)
        if text:
            page_num = int(node_page) if isinstance(node_page, int) else -1
            blocks.append(
                Chunk(
                    id=node_id,
                    source="",
                    page=page_num,
                    section_path=normalize_section_path(section_titles),
                    block_type=node_block_type,
                    text=text,
                    metadata={
                        "block_id": node_id,
                        "block_type": node_block_type,
                        "page": page_num,
                        "bbox": node.get("bbox"),
                        "polygon": node.get("polygon"),
                        "section_hierarchy": node.get("section_hierarchy"),
                        "raw_html": node_html,
                    },
                )
            )

    for child in node.get("children", []):
        blocks.extend(flatten_blocks(child, page=node_page, section_titles=section_titles))

    return blocks


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
    """Load the extraction JSON file and return a flat list of Chunks."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    root = _extraction_root(data)
    source = Path(path).name
    chunks: List[Chunk] = []
    for page_node in root.get("children", []):
        page_index = page_node.get("page", -1)
        for chunk in flatten_blocks(page_node, page=page_index, section_titles=[]):
            chunk.source = source
            chunks.append(chunk)
    return chunks
