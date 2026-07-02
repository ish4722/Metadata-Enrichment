"""Step 2/3 - Section-aware chunking.

Groups the flat content blocks emitted by :mod:`rag.indexing.loader` into
section-level units using the leaf section id that the loader resolved from
datalab's ``section_hierarchy``. Blocks that belong to the same leaf section are
joined in document order (across pages), and each emitted chunk is prefixed with
its parent breadcrumb so both the embedding and the LLM get parent-section
context. Oversize sections are split with a sliding word window; every split
keeps the breadcrumb prefix.

This replaces the old two-pass ``merge_blocks_by_section`` + ``merge_across_pages``
+ blind ``build_useful_chunks`` flow with a single hierarchy-driven pass. The
legacy functions are kept in :mod:`rag.indexing.merging` / :mod:`rag.indexing.chunking`
for A/B comparison.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Dict, List

from rag.common.schema import Chunk
from rag.indexing.chunking import split_text_into_chunks


def _section_name(section_path: str) -> str:
    """Last component of a ' > ' breadcrumb (the section's own title)."""
    if not section_path:
        return ""
    return section_path.split(" > ")[-1].strip()


def _group_key(block: Chunk) -> str:
    """Stable grouping key: the leaf section id, or a per-page fallback.

    Blocks with no resolved section (e.g. title/cover pages) are grouped per
    page so they stay together instead of each becoming a lone chunk.
    """
    if block.leaf_section_id:
        return block.leaf_section_id
    return f"__nosection__::{block.source}::p{block.page}"


def build_section_chunks(
    blocks: List[Chunk],
    chunk_size: int = 800,
    overlap: int = 200,
) -> List[Chunk]:
    """Group blocks into section units, then split oversize ones.

    Parameters mirror the sliding-window chunker: ``chunk_size`` / ``overlap`` are
    measured in words and only apply when a section is longer than ``chunk_size``.
    """
    if not blocks:
        return []

    # ── group blocks by leaf section, preserving first-seen (document) order ──
    groups: "OrderedDict[str, List[Chunk]]" = OrderedDict()
    for block in blocks:
        groups.setdefault(_group_key(block), []).append(block)

    # ── sibling map: parent section id -> ordered unique child section names ──
    parent_to_children: "defaultdict[str, List[str]]" = defaultdict(list)
    key_to_parent: Dict[str, str] = {}
    key_to_name: Dict[str, str] = {}
    for key, group in groups.items():
        rep = max(group, key=lambda b: len(b.section_path or ""))
        parent_id = rep.parent_section_id or ""
        name = _section_name(rep.section_path) or rep.block_type
        key_to_parent[key] = parent_id
        key_to_name[key] = name
        if name and name not in parent_to_children[parent_id]:
            parent_to_children[parent_id].append(name)

    useful: List[Chunk] = []
    for key, group in groups.items():
        ordered = sorted(group, key=lambda b: b.order)
        rep = max(ordered, key=lambda b: len(b.section_path or ""))
        breadcrumb = rep.section_path or ""
        body = "\n\n".join(b.text for b in ordered if b.text.strip()).strip()
        if not body:
            continue

        pages = sorted({b.page for b in ordered})
        parent_id = key_to_parent.get(key, "")
        siblings = [n for n in parent_to_children.get(parent_id, []) if n != key_to_name.get(key)]
        hierarchy_ids = list(rep.metadata.get("section_hierarchy_ids", []))
        prefix = f"{breadcrumb}\n\n" if breadcrumb else ""

        for part in split_text_into_chunks(body, chunk_size, overlap):
            part = part.strip()
            if not part:
                continue
            cid = f"chunk_{len(useful)}"
            text = f"{prefix}{part}" if prefix else part
            useful.append(
                Chunk(
                    id=cid,
                    source=rep.source,
                    page=ordered[0].page,
                    section_path=breadcrumb,
                    block_type=rep.block_type,
                    text=text,
                    leaf_section_id=rep.leaf_section_id,
                    parent_section_id=parent_id,
                    order=ordered[0].order,
                    section_hierarchy_ids=hierarchy_ids,
                    metadata={
                        "chunk_id": cid,
                        "source": rep.source,
                        "page": ordered[0].page,
                        "pages": pages,
                        "section_path": breadcrumb,
                        "section_name": key_to_name.get(key, ""),
                        "leaf_section_id": rep.leaf_section_id,
                        "parent_section_id": parent_id,
                        "section_hierarchy_ids": hierarchy_ids,
                        "sibling_sections": siblings,
                        "text_length": len(text),
                    },
                )
            )
    return useful
