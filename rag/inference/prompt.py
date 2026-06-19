"""Step 4 - Prompt construction.

    >>>>>  EDIT THE PROMPT TEMPLATE HERE  <<<<<

Builds the LLM prompt from the context items produced by
:mod:`rag.inference.context_builder`. Works for both chunk and full-page items
(both expose ``source`` / ``page`` / ``text``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

SYSTEM_PREAMBLE = (
    "You are an expert retrieval assistant for oil and gas reports. "
    "Use ONLY the provided excerpts and cite the source page for every answer. "
    "If the answer is not contained in the excerpts, say that the information is "
    "unavailable in the document.\n\n"
)


def _history_block(chat_history: Optional[List[Dict[str, str]]]) -> str:
    if not chat_history:
        return ""
    lines: List[str] = []
    for turn in chat_history:
        lines.append(f"Previous question: {turn['query']}")
        lines.append(f"Previous answer: {turn['answer']}")
    return "\n".join(lines) + "\n\n"


def make_prompt(
    query: str,
    context_items: List[Dict[str, Any]],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Assemble the full prompt string."""
    context_parts = []
    for item in context_items:
        header = f"[source={item.get('source')}, page={item.get('page')}]"
        context_parts.append(f"{header}\n{item.get('text', '')}")
    context = "\n\n---\n\n".join(context_parts)

    return (
        SYSTEM_PREAMBLE
        + _history_block(chat_history)
        + f"Question: {query}\n\n"
        + "Relevant excerpts:\n"
        + f"{context}\n\n"
        + "Answer concisely and cite the source page when possible."
    )
