"""Batch query generation: schema/record -> per-document queries file.

This is the "plug into the RAG pipeline" endpoint. Given an uploaded structured
representation (OSDU schema or record), it:
  1. extracts the attributes (see :mod:`query_generation.schema_ingest`),
  2. generates one retrieval query per attribute (reusing the single-attribute
     pipeline in :mod:`query_generation.generator`),
  3. writes a queries file named after the document, ready to feed to
     ``eval_all_queries.py --queries <path>``.

Every generated label carries the trailing ``*`` marker, so a document's
generated queries are visually and programmatically distinct from the
hand-written ``queries.txt``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from query_generation.generator import GeneratedQuery, generate_query_for_attribute
from query_generation.queries_file import write_queries_file
from query_generation.schema_ingest import ExtractedAttribute, extract_attributes

DEFAULT_OUTPUT_DIR = "generated_queries"


@dataclass
class BuildResult:
    """Outcome of building a per-document queries file."""

    output_path: str
    document_name: str
    attributes: List[str] = field(default_factory=list)   # labels attempted
    generated: List[GeneratedQuery] = field(default_factory=list)
    failures: List[str] = field(default_factory=list)      # "label: reason"

    @property
    def num_generated(self) -> int:
        return len(self.generated)


def generate_queries_for_attributes(
    attributes: List[Union[str, ExtractedAttribute]],
    *,
    llm: Any,
    config: Optional[Any] = None,
    on_progress: Optional[Callable[[int, int, str, Optional[Exception]], None]] = None,
) -> "tuple[List[GeneratedQuery], List[str]]":
    """Generate a query for each attribute; return (generated, failures).

    ``on_progress(index, total, label, error)`` is called after each attribute
    (``error`` is None on success). Failures never abort the batch.
    """
    labels = [a.label if isinstance(a, ExtractedAttribute) else str(a) for a in attributes]
    generated: List[GeneratedQuery] = []
    failures: List[str] = []
    total = len(labels)
    for i, label in enumerate(labels, 1):
        try:
            gq = generate_query_for_attribute(label, llm=llm, config=config)
            generated.append(gq)
            if on_progress:
                on_progress(i, total, label, None)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"{label}: {exc}")
            if on_progress:
                on_progress(i, total, label, exc)
    return generated, failures


def _derive_name(source: Union[str, Dict[str, Any], list], name: Optional[str]) -> str:
    if name:
        return name
    if isinstance(source, str) and os.path.exists(source):
        return os.path.splitext(os.path.basename(source))[0]
    return "generated"


def build_query_file_from_schema(
    source: Union[str, Dict[str, Any], list],
    *,
    llm: Any,
    config: Optional[Any] = None,
    name: Optional[str] = None,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    output_path: Optional[str] = None,
    include_null: bool = True,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    limit: Optional[int] = None,
    on_progress: Optional[Callable[[int, int, str, Optional[Exception]], None]] = None,
) -> BuildResult:
    """Extract attributes from ``source`` and write a per-document queries file.

    - ``name`` overrides the document name (defaults to the source file stem).
    - ``output_path`` overrides the full destination (else ``<output_dir>/<name>.txt``).
    - ``include`` / ``exclude`` filter attribute labels (case-insensitive substring).
    - ``limit`` caps the number of attributes (after filtering) - useful because a
      full OSDU record can hold ~100 attributes (one LLM call each).
    """
    if llm is None:
        raise RuntimeError("No LLM client available (see rag.inference.llm.get_llm).")

    attributes = extract_attributes(source, include_null=include_null)

    if include:
        wanted = [s.lower() for s in include]
        attributes = [a for a in attributes if any(w in a.label.lower() for w in wanted)]
    if exclude:
        unwanted = [s.lower() for s in exclude]
        attributes = [a for a in attributes if not any(w in a.label.lower() for w in unwanted)]
    if limit is not None:
        attributes = attributes[:limit]

    document_name = _derive_name(source, name)
    dest = output_path or os.path.join(output_dir, f"{document_name}.txt")

    generated, failures = generate_queries_for_attributes(
        attributes, llm=llm, config=config, on_progress=on_progress
    )

    src_desc = source if isinstance(source, str) and os.path.exists(source) else "uploaded schema"
    write_queries_file(
        generated,
        dest,
        header=f"Auto-generated queries for {document_name} from {os.path.basename(str(src_desc))} "
               f"({len(generated)} attributes). Generated by query_generation.",
    )

    return BuildResult(
        output_path=os.path.abspath(dest),
        document_name=document_name,
        attributes=[a.label for a in attributes],
        generated=generated,
        failures=failures,
    )
