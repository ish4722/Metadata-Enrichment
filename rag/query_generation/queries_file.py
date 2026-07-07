"""Read/write helpers for the queries file (``queries.txt``).

Generation stays in :mod:`query_generation.generator`; this module owns the file
I/O so appending a generated query never re-embeds or re-parses anything the RAG
pipeline cares about. Blocks are the same shape the evaluator parses:

    <label>:
    <question>

    Keywords: <...>        # optional

A generated query's label carries a trailing ``*`` (see ``GENERATED_MARKER``).
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, List, Optional, Tuple

from query_generation.generator import GeneratedQuery

DEFAULT_QUERIES_FILE = "queries.txt"


def _is_label_line(stripped: str) -> bool:
    """Mirror rag_evaluation.parse_queries_file's label detection."""
    return (
        stripped.endswith(":")
        and not stripped.lower().startswith("keywords:")
        and len(stripped) > 1
    )


def _split_blocks(lines: List[str]) -> Tuple[List[str], "Dict[str, List[int]]"]:
    """Return raw lines plus a map of label -> [start_index, ...] for each block.

    The start index points at the label line. A block runs until the next label
    line (or EOF).
    """
    label_starts: Dict[str, List[int]] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        if _is_label_line(stripped):
            label = stripped[:-1].strip()
            label_starts.setdefault(label.lower(), []).append(i)
    return lines, label_starts


def _block_bounds(lines: List[str], start: int) -> Tuple[int, int]:
    """Return [start, end) line range for the block whose label is at ``start``."""
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _is_label_line(lines[j].strip()):
            end = j
            break
    return start, end


def existing_labels(path: str = DEFAULT_QUERIES_FILE) -> List[str]:
    """Return all labels currently present in the queries file (original case)."""
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.rstrip("\n") for line in f]
    labels: List[str] = []
    for line in lines:
        stripped = line.strip()
        if _is_label_line(stripped):
            labels.append(stripped[:-1].strip())
    return labels


def has_label(label: str, path: str = DEFAULT_QUERIES_FILE) -> bool:
    return label.strip().lower() in {l.lower() for l in existing_labels(path)}


def write_generated_query(
    generated: GeneratedQuery,
    path: str = DEFAULT_QUERIES_FILE,
    *,
    overwrite: bool = False,
) -> str:
    """Append (or upsert) a generated query block into the queries file.

    - If the label is absent, the block is appended at the end.
    - If the label exists and ``overwrite`` is True, its block is replaced.
    - If the label exists and ``overwrite`` is False, a ``FileExistsError`` is
      raised so the caller can decide.

    Returns the label written.
    """
    label = generated.label
    block = generated.to_block()

    if not os.path.exists(path):
        with open(path, "w", encoding="utf-8") as f:
            f.write(block + "\n")
        return label

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()

    _, label_starts = _split_blocks(lines)
    existing = label_starts.get(label.lower())

    if existing:
        if not overwrite:
            raise FileExistsError(
                f"Label {label!r} already exists in {path}. Pass overwrite=True to replace it."
            )
        # Replace the first occurrence's block in place.
        start, end = _block_bounds(lines, existing[0])
        new_lines = lines[:start] + block.split("\n") + lines[end:]
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines).rstrip("\n") + "\n")
        return label

    # Append a fresh block separated from prior content by exactly one blank line.
    # Count trailing newlines in the raw bytes so the gap is exactly "\n\n".
    if not content:
        lead = ""
    else:
        trailing_newlines = len(content) - len(content.rstrip("\n"))
        lead = "\n" * max(0, 2 - trailing_newlines)
    with open(path, "a", encoding="utf-8") as f:
        f.write(lead + block + "\n")
    return label


def write_queries_file(
    generated: Iterable[GeneratedQuery],
    path: str,
    *,
    header: Optional[str] = None,
) -> str:
    """Write a fresh queries file containing every generated block.

    Used for the per-document query files produced from an uploaded schema. The
    optional ``header`` becomes a leading comment line (it must not end with
    ``:`` so the evaluator's parser ignores it). Overwrites ``path``.
    """
    blocks = [g.to_block() for g in generated]
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if header:
            safe = header.rstrip(":").replace("\n", " ")
            f.write(f"# {safe}\n\n")
        f.write("\n\n".join(blocks))
        if blocks:
            f.write("\n")
    return path
