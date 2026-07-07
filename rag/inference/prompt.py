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
    '  "available": <true if the excerpts contain the answer, false otherwise>,\n'
    '  "confidence": <number between 0.0 and 1.0 for how certain you are>\n'
    "}\n"
    "For \"confidence\": rate high (close to 1.0) when the excerpts directly and "
    "unambiguously state the answer; rate low (close to 0.0) when you must guess, "
    "the excerpts are ambiguous/conflicting, or you are confused; and set a low "
    "value together with \"available\": false when nothing in the excerpts "
    "supports an answer.\n\n"
)


def _history_block(chat_history: Optional[List[Dict[str, str]]]) -> str:
    if not chat_history:
        return ""
    lines: List[str] = []
    for turn in chat_history:
        lines.append(f"Previous question: {turn['query']}")
        lines.append(f"Previous answer: {turn['answer']}")
    return "\n".join(lines) + "\n\n"


def _extract_json_object(raw: str) -> Optional[Dict[str, Any]]:
    """Return the first JSON object parsed from ``raw`` (whole string or embedded)."""
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
    return None


def parse_generator_response(raw: str) -> str:
    """Extract the ``answer`` field from a JSON generator reply for scoring.

    Returns ``raw`` unchanged when it is not JSON or has no ``answer`` field.
    """
    if not raw or not raw.strip():
        return raw or ""
    obj = _extract_json_object(raw)
    if obj is not None and obj.get("answer") is not None:
        return str(obj["answer"])
    return raw


def _parse_confidence(value: Any) -> Optional[float]:
    """Coerce a generator ``confidence`` field to a float in [0, 1] or ``None``."""
    if value is None:
        return None
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    if conf < 0.0:
        return 0.0
    if conf > 1.0:
        return 1.0
    return conf


def parse_generator_json(raw: str) -> Dict[str, Any]:
    """Parse a generator reply into ``{answer, available, source_pages, confidence}``.

    Robust to models that wrap the JSON in prose or markdown fences. When the
    ``available`` flag is missing or malformed it defaults to ``True`` whenever a
    non-empty answer is present, so escalation does not loop forever on a reply
    that simply omitted the flag. ``confidence`` is ``None`` when the model did
    not report a usable numeric score.
    """
    obj = _extract_json_object(raw)
    if obj is None:
        text = (raw or "").strip()
        return {"answer": text, "available": bool(text), "source_pages": [],
                "confidence": None}

    answer = obj.get("answer")
    answer_str = "" if answer is None else str(answer)

    available = obj.get("available")
    if isinstance(available, bool):
        avail = available
    elif isinstance(available, str):
        avail = available.strip().lower() in {"true", "yes", "1"}
    else:
        avail = bool(answer_str.strip())

    source_pages = obj.get("source_pages")
    if not isinstance(source_pages, list):
        source_pages = []

    confidence = _parse_confidence(obj.get("confidence"))

    return {"answer": answer_str, "available": avail, "source_pages": source_pages,
            "confidence": confidence}


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
