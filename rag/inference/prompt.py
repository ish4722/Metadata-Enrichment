"""Step 4 - Prompt construction.

    >>>>>  EDIT THE PROMPT TEMPLATE HERE  <<<<<

Builds the LLM prompt from the context items produced by
:mod:`rag.inference.context_builder`. Works for both chunk and full-page items
(both expose ``source`` / ``page`` / ``text``).
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# Generic references to the subject of a question, replaced with the concrete
# well/facility name at generation time (see :func:`inject_facility_name`).
_GENERIC_SUBJECT = re.compile(
    r"\b(this|the)\s+(oil and gas facility|oil and gas asset|well|facility|asset)\b",
    re.IGNORECASE,
)


def inject_facility_name(question: str, name: Optional[str]) -> str:
    """Rewrite a generic question to name the specific well/facility.

    Used ONLY at generation time (never for retrieval). Generic references like
    "this well" are replaced with ``name``; if none are present, a clarifying
    clause is appended instead. A falsy name leaves the question unchanged.
    """
    if not name:
        return question
    new_q, n = _GENERIC_SUBJECT.subn(name, question)
    if n == 0:
        new_q = f"{question.rstrip()} (The well/facility is: {name}.)"
    return new_q


# OLD prompt (plain-text answer) - kept for reference:
# SYSTEM_PREAMBLE = (
#     "You are an expert retrieval assistant for oil and gas reports. "
#     "Use ONLY the provided excerpts and cite the source page for every answer. "
#     "If the answer is not contained in the excerpts, say that the information is "
#     "unavailable in the document.\n"
#     "For questions about a well's geographic location or coordinates: if the "
#     "excerpts give location coordinates in ANY datum or format (e.g. WGS84, "
#     "NZGD2000 / NZTM, easting/northing, or degrees-minutes-seconds), report the "
#     "latitude and longitude - converting to decimal degrees when possible and "
#     "naming the datum - instead of replying that WGS84 coordinates are unavailable. "
#     "These datums are effectively interchangeable for locating the well. Only say "
#     "the location is unavailable if NO coordinates of any kind are present.\n\n"
# )

SYSTEM_PREAMBLE = (
    "You are an expert retrieval assistant for oil and gas reports. "
    "Use ONLY the provided excerpts to answer the question. "
    'If the answer is not contained in the excerpts, set "available" to false and '
    "state that the information is unavailable in the document.\n"
    "For questions about a well's geographic location or coordinates: if the "
    "excerpts give location coordinates in ANY datum or format (e.g. WGS84, "
    "NZGD2000 / NZTM, easting/northing, or degrees-minutes-seconds), report the "
    "latitude and longitude - converting to decimal degrees when possible and "
    "naming the datum - instead of saying WGS84 coordinates are unavailable. "
    "These datums are effectively interchangeable for locating the well. Only set "
    '"available" to false for location if NO coordinates of any kind are present.\n\n'
    "Respond with ONLY a valid JSON object (no markdown fences, no extra text) "
    "using this schema:\n"
    "{\n"
    '  "answer": "<concise answer text>",\n'
    '  "source_pages": [<integer page numbers cited from the excerpts>],\n'
    '  "available": <true if the excerpts contain the answer, false otherwise>\n'
    "}\n\n"
)


def _history_block(chat_history: Optional[List[Dict[str, str]]]) -> str:
    if not chat_history:
        return ""
    lines: List[str] = []
    for turn in chat_history:
        lines.append(f"Previous question: {turn['query']}")
        lines.append(f"Previous answer: {turn['answer']}")
    return "\n".join(lines) + "\n\n"


def parse_generator_response(raw: str) -> str:
    """Extract the ``answer`` field from a JSON generator reply for scoring.

    Returns ``raw`` unchanged when it is not JSON or has no ``answer`` field.
    """
    if not raw or not raw.strip():
        return raw or ""
    text = raw.strip()
    for candidate in (text,):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict) and obj.get("answer") is not None:
                return str(obj["answer"])
        except json.JSONDecodeError:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict) and obj.get("answer") is not None:
                return str(obj["answer"])
        except json.JSONDecodeError:
            pass
    return raw


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
        # OLD trailing instruction (plain-text answer) - kept for reference:
        # + "Answer concisely and cite the source page when possible."
        + "Respond with ONLY the JSON object described above."
    )
