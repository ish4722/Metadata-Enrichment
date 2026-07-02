"""Shared data model for the RAG pipeline.

The ``Chunk`` dataclass is the single unit of text that flows through both the
indexing pipeline (folder 1) and the inference pipeline (folder 2). Keeping it in
one place means every module agrees on the same shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Chunk:
    id: str
    source: str
    page: int
    section_path: str
    block_type: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Section identity derived from datalab's ``section_hierarchy`` (see
    # rag.indexing.loader). ``leaf_section_id`` is the deepest SectionHeader the
    # block lives under; ``parent_section_id`` is its immediate parent header.
    # Both are global ids like "/page/0/SectionHeader/4". ``order`` is the
    # document-order position used to keep multi-page sections in reading order.
    leaf_section_id: str = ""
    parent_section_id: str = ""
    order: int = 0
    # Full ancestor chain of SectionHeader ids, shallow -> deep, ending in
    # ``leaf_section_id``. Used by hierarchical context escalation to walk up
    # parent sections when the matched chunk lacks enough context.
    section_hierarchy_ids: List[str] = field(default_factory=list)
