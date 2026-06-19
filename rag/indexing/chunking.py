"""Step 3 - Split merged blocks into fixed-size retrieval chunks.

A sliding word window (``chunk_size`` words, ``overlap`` words) turns long
section blocks into uniform retrieval units.
"""

from __future__ import annotations

from typing import List

from rag.common.schema import Chunk


def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    result: List[str] = []
    start = 0
    while start < len(words):
        end = min(len(words), start + chunk_size)
        result.append(" ".join(words[start:end]).strip())
        if end == len(words):
            break
        start += chunk_size - overlap
    return result


def build_useful_chunks(
    chunks: List[Chunk],
    chunk_size: int = 800,
    overlap: int = 200,
) -> List[Chunk]:
    """Expand each section-level chunk into one or more retrieval chunks."""
    useful: List[Chunk] = []
    for ctr, chunk in enumerate(chunks):
        for part in split_text_into_chunks(chunk.text, chunk_size, overlap):
            if not part.strip():
                continue
            cid = f"chunk_{len(useful)}"
            useful.append(
                Chunk(
                    id=cid,
                    source=chunk.source,
                    page=chunk.page,
                    section_path=chunk.section_path,
                    block_type=chunk.block_type,
                    text=part,
                    metadata={
                        **chunk.metadata,
                        "chunk_id": cid,
                        "source": chunk.source,
                        "page": chunk.page,
                        "section_path": chunk.section_path,
                        "block_type": chunk.block_type,
                        "text_length": len(part),
                    },
                )
            )
    return useful
