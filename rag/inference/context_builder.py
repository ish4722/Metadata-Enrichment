"""Step 3 - Context assembly.

Turns retrieved/reranked matches into the context blocks sent to the LLM.

Modes (from config):
  * ``context_mode="chunk"``     - use the matched chunk texts directly.
  * ``context_mode="full_page"`` - expand each matched chunk to its full page
    (all chunks on that page), deduplicated and ordered.
  * ``context_mode="hierarchical"`` - start from the matched chunks and, when the
    LLM cannot answer, progressively add ancestor section texts. This module only
    assembles the context for a given escalation ``depth``; the escalation loop
    itself lives in :mod:`rag.inference.hierarchical_answer`.

Returns a uniform list of context items (dicts with source/page/text) plus the
concatenated raw text used for retrieval-accuracy scoring.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from rag.common.schema import Chunk


def build_page_index(chunks: List[Chunk]) -> Dict[Tuple[str, int], List[Chunk]]:
    """Map (source, page) -> all chunks on that page."""
    page_index: Dict[Tuple[str, int], List[Chunk]] = {}
    for chunk in chunks:
        page_index.setdefault((chunk.source, chunk.page), []).append(chunk)
    return page_index


def _chunk_sort_key(c: Chunk) -> int:
    if c.id.startswith("chunk_"):
        try:
            return int(c.id.split("_")[-1])
        except ValueError:
            return 0
    return 0


def get_full_pages_for_matches(
    matches: List[Tuple[Chunk, float]],
    page_index: Dict[Tuple[str, int], List[Chunk]],
) -> List[Dict[str, Any]]:
    """Expand matched chunks into deduplicated full-page context objects."""
    page_best_score: Dict[Tuple[str, int], float] = {}
    page_matched_ids: Dict[Tuple[str, int], List[str]] = {}

    for chunk, score in matches:
        key = (chunk.source, chunk.page)
        if key not in page_best_score or score > page_best_score[key]:
            page_best_score[key] = score
        page_matched_ids.setdefault(key, []).append(chunk.id)

    full_pages: List[Dict[str, Any]] = []
    for key in sorted(page_best_score.keys()):
        source, page = key
        page_chunks = sorted(page_index.get(key, []), key=_chunk_sort_key)
        full_text = "\n\n".join(c.text for c in page_chunks if c.text.strip())
        full_pages.append(
            {
                "source": source,
                "page": page,
                "text": full_text,
                "matched_score": page_best_score[key],
                "matched_chunk_ids": page_matched_ids[key],
                "num_chunks_on_page": len(page_chunks),
            }
        )
    return full_pages


def build_context(
    matches: List[Tuple[Chunk, float]],
    all_chunks: List[Chunk],
    context_mode: str = "chunk",
) -> Tuple[List[Dict[str, Any]], str]:
    """Build context items + concatenated raw text for the given mode."""
    if context_mode == "full_page":
        page_index = build_page_index(all_chunks)
        items = get_full_pages_for_matches(matches, page_index)
        raw_text = "\n".join(item["text"] for item in items)
        return items, raw_text

    items = [
        {
            "source": chunk.source,
            "page": chunk.page,
            "section": chunk.section_path or chunk.block_type,
            "text": chunk.text,
            "matched_score": score,
        }
        for chunk, score in matches
    ]
    raw_text = "\n".join(chunk.text for chunk, _ in matches)
    return items, raw_text


# ── hierarchical (parent-escalation) context ──────────────────────────────────


def ancestor_section_ids(chunk: Chunk) -> List[str]:
    """Return the chunk's ancestor SectionHeader ids, shallow -> deep.

    This is the full ``section_hierarchy_ids`` chain with the chunk's own leaf
    section removed, so the first element is the outermost (top-level) parent and
    the last is the immediate parent.
    """
    ids = list(chunk.section_hierarchy_ids or chunk.metadata.get("section_hierarchy_ids", []))
    leaf = chunk.leaf_section_id
    if leaf and ids and ids[-1] == leaf:
        return ids[:-1]
    if leaf:
        return [i for i in ids if i != leaf]
    return ids


def max_ancestor_depth(matches: List[Tuple[Chunk, float]]) -> int:
    """Deepest ancestor chain across all matches (0 when none have a hierarchy)."""
    return max((len(ancestor_section_ids(c)) for c, _ in matches), default=0)


def _ancestors_to_add(
    matches: List[Tuple[Chunk, float]], depth: int
) -> List[Tuple[str, int]]:
    """Ordered ``(section_id, level)`` ancestors to include up to ``depth``.

    ``level`` is the distance from the matched leaf (1 = immediate parent). For
    each match the immediate parent is preferred first, then higher ancestors.
    Sections already present as a matched leaf are skipped (their body is already
    in the level-0 context), and each ancestor id is added at most once.
    """
    matched_leaves = {c.leaf_section_id for c, _ in matches if c.leaf_section_id}
    seen: set = set()
    result: List[Tuple[str, int]] = []
    for chunk, _ in matches:
        anc = ancestor_section_ids(chunk)  # shallow -> deep
        n = len(anc)
        for level in range(1, depth + 1):
            idx = n - level  # deepest (immediate parent) first
            if idx < 0:
                break
            sid = anc[idx]
            if not sid or sid in seen or sid in matched_leaves:
                continue
            seen.add(sid)
            result.append((sid, level))
    return result


def build_hierarchical_context(
    matches: List[Tuple[Chunk, float]],
    store: Any,
    doc_id: str,
    depth: int,
) -> Tuple[List[Dict[str, Any]], str]:
    """Assemble context for one escalation ``depth``.

    ``depth == 0`` returns just the matched chunks. Each higher depth adds the
    text of one more ancestor section per match (deduplicated across matches),
    fetched from the store via ``fetch_chunks_for_section``.
    """
    items: List[Dict[str, Any]] = [
        {
            "source": chunk.source,
            "page": chunk.page,
            "section": chunk.section_path or chunk.block_type,
            "text": chunk.text,
            "matched_score": score,
            "context_level": 0,
            "section_id": chunk.leaf_section_id,
        }
        for chunk, score in matches
    ]

    if depth > 0:
        for sid, level in _ancestors_to_add(matches, depth):
            section_chunks = store.fetch_chunks_for_section(doc_id, sid)
            text = "\n\n".join(c.text for c in section_chunks if c.text.strip()).strip()
            if not text:
                continue
            head = section_chunks[0]
            items.append(
                {
                    "source": head.source,
                    "page": head.page,
                    "section": head.section_path or head.block_type,
                    "text": text,
                    "matched_score": None,
                    "context_level": level,
                    "section_id": sid,
                }
            )

    raw_text = "\n".join(item["text"] for item in items)
    return items, raw_text


def save_pages_sent_to_llm(items: List[Dict[str, Any]], query: str, output_dir: str) -> str:
    """Write a JSON file listing every context block sent to the LLM."""
    safe_query = re.sub(r"[^\w\s-]", "", query)[:50].strip().replace(" ", "_")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / f"pages_sent_to_llm_{safe_query}.json"

    payload = {"query": query, "num_pages": len(items), "pages": items}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"Context sent to LLM written to: {output_path}")
    return str(output_path)
