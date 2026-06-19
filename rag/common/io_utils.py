"""Serialisation helpers for chunks (JSONL)."""

from __future__ import annotations

import json
from typing import List

from rag.common.schema import Chunk


def save_chunks_jsonl(chunks: List[Chunk], path: str) -> None:
    """Write chunks to a JSONL file (one JSON record per line)."""
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            record = {
                "id": c.id,
                "source": c.source,
                "page": c.page,
                "section_path": c.section_path,
                "block_type": c.block_type,
                "text": c.text,
                "metadata": c.metadata,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_chunks_jsonl(path: str) -> List[Chunk]:
    """Read chunks back from a JSONL file."""
    chunks: List[Chunk] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            chunks.append(
                Chunk(
                    id=d["id"],
                    source=d["source"],
                    page=d["page"],
                    section_path=d.get("section_path", ""),
                    block_type=d.get("block_type", ""),
                    text=d["text"],
                    metadata=d.get("metadata", {}),
                )
            )
    return chunks
