"""Shared data model for the RAG pipeline.

The ``Chunk`` dataclass is the single unit of text that flows through both the
indexing pipeline (folder 1) and the inference pipeline (folder 2). Keeping it in
one place means every module agrees on the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Chunk:
    id: str
    source: str
    page: int
    section_path: str
    block_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
