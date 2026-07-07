"""Automatic OSDU-attribute -> retrieval query generation.

Turn an OSDU well/wellbore attribute name into a domain-aware retrieval query
(synonyms + associated report terms folded in) and add it to ``queries.txt``,
marked with a trailing ``*`` so it is distinguishable from a hand-written query.

Typical use::

    from query_generation import add_query_for_attribute
    from rag.config import RagConfig
    from rag.inference.llm import get_llm

    config = RagConfig.from_env()
    llm = get_llm(config)
    result = add_query_for_attribute("marker names", llm=llm, config=config)
    print(result.label, "->", result.query)

Or from the CLI (see :mod:`query_generation.cli`)::

    python -m query_generation "marker names"
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from query_generation.batch import (
    BuildResult,
    build_query_file_from_schema,
    generate_queries_for_attributes,
)
from query_generation.generator import (
    GENERATED_MARKER,
    GeneratedQuery,
    generate_query_for_attribute,
    make_label,
)
from query_generation.queries_file import (
    DEFAULT_QUERIES_FILE,
    existing_labels,
    has_label,
    write_generated_query,
    write_queries_file,
)
from query_generation.schema_ingest import ExtractedAttribute, extract_attributes

__all__ = [
    # single-attribute generation
    "GENERATED_MARKER",
    "GeneratedQuery",
    "generate_query_for_attribute",
    "make_label",
    "add_query_for_attribute",
    # queries file I/O
    "DEFAULT_QUERIES_FILE",
    "existing_labels",
    "has_label",
    "write_generated_query",
    "write_queries_file",
    # schema-driven batch generation
    "ExtractedAttribute",
    "extract_attributes",
    "BuildResult",
    "generate_queries_for_attributes",
    "build_query_file_from_schema",
]


def add_query_for_attribute(
    attribute: str,
    *,
    llm: Any,
    config: Optional[Any] = None,
    llm_holder: Optional[Dict[str, Any]] = None,
    queries_path: str = DEFAULT_QUERIES_FILE,
    write: bool = True,
    overwrite: bool = False,
) -> GeneratedQuery:
    """Generate a query for ``attribute`` and (optionally) append it to the file.

    Returns the :class:`GeneratedQuery`. When ``write`` is False the query is
    generated but the file is left untouched (useful for previews).
    """
    generated = generate_query_for_attribute(
        attribute, llm=llm, config=config, llm_holder=llm_holder
    )
    if write:
        write_generated_query(generated, queries_path, overwrite=overwrite)
    return generated
