"""Step 4b - Hierarchical (parent-escalation) answering.

Instead of sending a whole page to the LLM up front, start with the matched
chunks alone and re-prompt with progressively larger context only when the model
reports it cannot answer (``"available": false``). Each escalation level adds one
more ancestor section per match (see
:func:`rag.inference.context_builder.build_hierarchical_context`).

The loop stops as soon as the answer is available, or once the section hierarchy
is exhausted / ``hierarchical_max_depth`` is reached. Token usage is accumulated
across every attempt so cost accounting stays accurate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from rag.common.schema import Chunk
from rag.inference.context_builder import build_hierarchical_context, max_ancestor_depth
from rag.inference.prompt import make_prompt, parse_generator_json


@dataclass
class HierarchicalAnswerResult:
    answer: str
    available: bool
    context_items: List[Dict[str, Any]]
    raw_text: str
    escalation_depth: int
    llm_calls: int
    usage: Dict[str, int] = field(default_factory=dict)
    context_tokens_by_level: List[int] = field(default_factory=list)


def _empty_usage() -> Dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def answer_with_hierarchical_context(
    llm: Any,
    query: str,
    matches: List[Tuple[Chunk, float]],
    store: Any,
    doc_id: str,
    config: Any,
    invoke_fn: Any,
) -> HierarchicalAnswerResult:
    """Escalate context up the section hierarchy until the LLM can answer.

    ``invoke_fn`` is ``(llm, prompt) -> (answer_text, usage)`` (typically
    :func:`rag_evaluation.invoke_llm_with_usage`), injected so token accounting is
    centralized and the loop stays easy to test.
    """
    max_depth = min(
        int(getattr(config, "hierarchical_max_depth", 5)),
        max_ancestor_depth(matches),
    )

    total_usage = _empty_usage()
    tokens_by_level: List[int] = []
    llm_calls = 0

    last_answer = ""
    last_items: List[Dict[str, Any]] = []
    last_raw = ""

    for depth in range(0, max_depth + 1):
        items, raw_text = build_hierarchical_context(matches, store, doc_id, depth)

        # Nothing new was added at this depth (e.g. ancestor sections had no
        # direct text): re-prompting would be identical, so stop escalating.
        if depth > 0 and raw_text == last_raw:
            break

        answer, usage = invoke_fn(llm, make_prompt(query, items))
        llm_calls += 1
        for key in total_usage:
            total_usage[key] += int(usage.get(key, 0) or 0)
        tokens_by_level.append(int(usage.get("input_tokens", 0) or 0))

        parsed = parse_generator_json(answer)
        last_answer, last_items, last_raw = answer, items, raw_text

        if parsed["available"]:
            return HierarchicalAnswerResult(
                answer=answer,
                available=True,
                context_items=items,
                raw_text=raw_text,
                escalation_depth=depth,
                llm_calls=llm_calls,
                usage=total_usage,
                context_tokens_by_level=tokens_by_level,
            )

    return HierarchicalAnswerResult(
        answer=last_answer,
        available=False,
        context_items=last_items,
        raw_text=last_raw,
        escalation_depth=max(len(tokens_by_level) - 1, 0),
        llm_calls=llm_calls,
        usage=total_usage,
        context_tokens_by_level=tokens_by_level,
    )
