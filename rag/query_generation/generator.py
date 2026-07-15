"""Turn an OSDU attribute name into a retrieval query via a single LLM call.

Pipeline: build prompt (:mod:`query_generation.prompt`) -> invoke the shared LLM
client (:mod:`rag.inference.llm`) -> parse the strict-JSON response into a
:class:`GeneratedQuery`. Writing the result into ``queries.txt`` lives in
:mod:`query_generation.queries_file` so generation stays independent of I/O.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from query_generation.prompt import build_generation_prompt

# Suffix appended to a generated query's label so it is visually distinct from a
# hand-written query in queries.txt.
GENERATED_MARKER = "*"


@dataclass
class GeneratedQuery:
    """Structured result of generating a retrieval query for one attribute."""

    attribute: str                         # the attribute as requested
    label: str                             # queries.txt label, e.g. "marker names*"
    query: str                             # natural-language question
    keywords: str = ""                     # comma-separated retrieval enrichment terms
    synonyms: List[str] = field(default_factory=list)
    osdu_interpretation: str = ""
    usage: Dict[str, int] = field(default_factory=dict)

    def to_block(self) -> str:
        """Render this query as a queries.txt block (label + question [+ keywords])."""
        lines = [f"{self.label}:", self.query]
        if self.keywords:
            lines.append("")
            lines.append(f"Keywords: {self.keywords}")
        return "\n".join(lines)


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort parse of a JSON object from an LLM response.

    Handles bare JSON, ```json fenced blocks, and leading/trailing prose by
    grabbing the first balanced ``{...}`` span.
    """
    if not text:
        raise ValueError("empty LLM response")
    cleaned = text.strip()
    # Strip a ```json ... ``` (or ``` ... ```) fence if present.
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Fall back to the first balanced object.
    start = cleaned.find("{")
    if start != -1:
        depth = 0
        for i in range(start, len(cleaned)):
            if cleaned[i] == "{":
                depth += 1
            elif cleaned[i] == "}":
                depth -= 1
                if depth == 0:
                    return json.loads(cleaned[start:i + 1])
    raise ValueError(f"could not parse JSON from LLM response: {text[:200]!r}")


def _normalize_keywords(value: Any) -> str:
    """Accept a list or a string and return a clean comma-separated string."""
    if isinstance(value, list):
        parts = [str(v).strip() for v in value if str(v).strip()]
    else:
        parts = [p.strip() for p in re.split(r"[,;]", str(value or "")) if p.strip()]
    # De-duplicate while preserving order.
    seen: set[str] = set()
    unique: List[str] = []
    for p in parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            unique.append(p)
    return ", ".join(unique)


def make_label(attribute: str) -> str:
    """Return the queries.txt label for a generated attribute query."""
    base = (attribute or "").strip()
    base = base.rstrip(GENERATED_MARKER).strip()
    return f"{base}{GENERATED_MARKER}"


def generate_query_for_attribute(
    attribute: str,
    *,
    llm: Any,
    config: Optional[Any] = None,
    llm_holder: Optional[Dict[str, Any]] = None,
) -> GeneratedQuery:
    """Generate one retrieval query for ``attribute`` using ``llm``.

    ``config`` (optional) enables auth-retry/token-refresh through
    :func:`rag_evaluation.invoke_llm_with_usage`.
    """
    if llm is None:
        raise RuntimeError(
            "No LLM client available. Ensure langchain-openai + sauth_token are "
            "installed and reachable (see rag.inference.llm.get_llm)."
        )
    if not (attribute or "").strip():
        raise ValueError("attribute must be a non-empty string")

    from rag.evaluation import core as rag_evaluation

    prompt = build_generation_prompt(attribute)
    answer, usage = rag_evaluation.invoke_llm_with_usage(
        llm, prompt, config=config, llm_holder=llm_holder
    )
    data = _extract_json(answer)

    query = str(data.get("query") or "").strip()
    if not query:
        raise ValueError(f"LLM response had no 'query' field: {answer[:200]!r}")

    return GeneratedQuery(
        attribute=attribute.strip(),
        label=make_label(attribute),
        query=query,
        keywords=_normalize_keywords(data.get("keywords")),
        synonyms=[str(s).strip() for s in (data.get("synonyms") or []) if str(s).strip()],
        osdu_interpretation=str(data.get("osdu_interpretation") or "").strip(),
        usage=usage or {},
    )
