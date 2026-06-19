"""Step 2 - Merge blocks into section-level units.

Two passes:
  * :func:`merge_blocks_by_section` - merge blocks sharing (source, page, section)
    within a single page.
  * :func:`merge_across_pages` - join consecutive same-section blocks across
    adjacent pages (multi-page tables / flowing paragraphs).
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from rag.common.schema import Chunk


def merge_blocks_by_section(chunks: List[Chunk]) -> List[Chunk]:
    """Merge blocks that share the same (source, page, section_path)."""
    grouped: Dict[Tuple[str, int, str], List[Chunk]] = {}
    for chunk in chunks:
        key = (chunk.source, chunk.page, chunk.section_path or chunk.block_type)
        grouped.setdefault(key, []).append(chunk)

    merged: List[Chunk] = []
    for idx, (_key, group) in enumerate(grouped.items()):
        if len(group) == 1:
            merged.append(group[0])
            continue

        merged_text = "\n\n".join(c.text for c in group if c.text.strip()).strip()
        merged.append(
            Chunk(
                id=f"{group[0].source}::{group[0].page}::{idx}",
                source=group[0].source,
                page=group[0].page,
                section_path=group[0].section_path,
                block_type=group[0].block_type,
                text=merged_text,
                metadata={
                    "source": group[0].source,
                    "page": group[0].page,
                    "section_path": group[0].section_path,
                    "block_type": group[0].block_type,
                    "block_ids": [c.id for c in group],
                    "raw_html_blocks": [c.metadata.get("raw_html") for c in group],
                },
            )
        )
    return merged


def merge_across_pages(chunks: List[Chunk]) -> List[Chunk]:
    """Merge consecutive chunks sharing a section_path across adjacent pages."""
    if not chunks:
        return chunks

    result: List[Chunk] = []
    sorted_chunks = sorted(chunks, key=lambda c: (c.source, c.section_path, c.page))

    prev = sorted_chunks[0]
    for curr in sorted_chunks[1:]:
        same_source = curr.source == prev.source
        same_section = curr.section_path == prev.section_path and curr.section_path != ""
        adjacent = curr.page == prev.page + 1

        if same_source and same_section and adjacent:
            prev = Chunk(
                id=prev.id,
                source=prev.source,
                page=prev.page,
                section_path=prev.section_path,
                block_type=prev.block_type,
                text=prev.text + "\n\n" + curr.text,
                metadata={
                    **prev.metadata,
                    "merged_pages": prev.metadata.get("merged_pages", [prev.page]) + [curr.page],
                },
            )
        else:
            result.append(prev)
            prev = curr

    result.append(prev)
    return result
