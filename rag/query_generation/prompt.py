"""Prompt construction for OSDU attribute -> retrieval query generation.

    >>>>>  EDIT THE QUERY-GENERATION PROMPT HERE  <<<<<

Given an OSDU well/wellbore attribute, we ask the LLM to:
  1. interpret what the attribute means in the OSDU data model,
  2. list the words/phrasings actually used to refer to it in well reports
     (synonyms, associated terms, alternate terminology),
  3. write a concise natural-language question that mirrors the style of the
     hand-written queries in ``queries.txt``,
  4. return retrieval keywords (the enrichment terms) that get appended to the
     question to widen recall.

The model must return STRICT JSON so :mod:`query_generation.generator` can parse
it deterministically.
"""

from __future__ import annotations

from typing import List

from query_generation.osdu_context import OsduAttribute, lookup

# Few-shot exemplars drawn from the existing hand-written queries so generated
# queries match their style (single well, alternate terms inline, no invented
# values). Keep these short - they anchor tone, not content.
_FEWSHOT = [
    {
        "attribute": "facilityevent spud date",
        "query": "What is the spud date/kick off date of the well?",
        "keywords": "spud date, spudded, kick off, commenced drilling, spud in",
    },
    {
        "attribute": "marker names",
        "query": "What are the markers or marker names of the well? Include any "
                 "formation tops or stratigraphic tops.",
        "keywords": "formation tops, stratigraphic tops, markers, formations, "
                    "members, lithostratigraphy, geological tops",
    },
    {
        "attribute": "default vertical measurement ID",
        "query": "What is the vertical measurement datum or reference elevation "
                 "for this well (e.g. KB, Kelly Bushing, RT, Rotary Table, GL, "
                 "Ground Level, MSL)?",
        "keywords": "datum, depth reference, reference elevation, kelly bushing, "
                    "rotary table, ground level, MSL",
    },
]

SYSTEM_PREAMBLE = (
    "You are a petroleum-domain metadata expert. You know the OSDU (Open "
    "Subsurface Data Universe) Well and Wellbore master-data model, and you know "
    "how New Zealand petroleum well completion reports actually phrase things. "
    "Your job is to turn an OSDU attribute name into a retrieval query that will "
    "find that attribute inside a single well report, even when the report uses "
    "different terminology than the OSDU field name."
)

_INSTRUCTIONS = """\
Given ONE OSDU attribute, produce a retrieval query for it.

Do this:
1. Interpret the attribute in the OSDU Well/Wellbore data model (one sentence).
2. List the terms, synonyms, abbreviations and alternate phrasings used to refer
   to this attribute in real well reports (drilling reports, completion reports,
   well data summaries). Think about how the value is labelled in tables and
   prose, not just the OSDU field name. Example: "spud" is also "kick off",
   "commenced drilling"; "markers" are also "formation tops" / "stratigraphic
   tops"; a datum may be "KB", "RT", "GL", "MSL".
3. Write ONE concise natural-language question about THIS well/document, in the
   same style as the examples. Fold the 1-2 most common alternate terms directly
   into the question (as the examples do). Do NOT ask for multiple unrelated
   attributes, do NOT invent values, and do NOT reference a specific well name.
4. Provide a comma-separated `keywords` string of the remaining domain terms
   (the enrichment terms) that should widen retrieval recall. Prefer terms as
   they appear in reports; include abbreviations and expanded forms.

Rules:
- The query must be answerable from a single well report.
- Keep the question under ~40 words.
- Keep it about "this well" / "the well" (no proper names).
- Output STRICT JSON only, no markdown fences, with exactly these keys:
  {"attribute": str, "osdu_interpretation": str, "synonyms": [str, ...],
   "query": str, "keywords": str}
"""


def _hint_block(attribute: str, hint: OsduAttribute | None) -> str:
    if hint is None:
        return (
            f"Attribute: {attribute}\n"
            "OSDU hint: (not in the curated registry - infer the most likely OSDU "
            "Well/Wellbore field from the attribute name.)\n"
        )
    terms = ", ".join(hint.report_terms) if hint.report_terms else "(none provided)"
    return (
        f"Attribute: {attribute}\n"
        f"OSDU field: {hint.osdu_field}\n"
        f"OSDU meaning: {hint.meaning}\n"
        f"Known report phrasings: {terms}\n"
    )


def _fewshot_block() -> str:
    lines: List[str] = ["Examples of good output (style reference):"]
    for ex in _FEWSHOT:
        lines.append(
            '{"attribute": "%s", "query": "%s", "keywords": "%s"}'
            % (ex["attribute"], ex["query"], ex["keywords"])
        )
    return "\n".join(lines)


def build_generation_prompt(attribute: str) -> str:
    """Assemble the full prompt string for one attribute."""
    hint = lookup(attribute)
    return (
        SYSTEM_PREAMBLE
        + "\n\n"
        + _INSTRUCTIONS
        + "\n"
        + _fewshot_block()
        + "\n\n"
        + "Now generate the query for this attribute:\n"
        + _hint_block(attribute, hint)
        + "\nReturn STRICT JSON only."
    )
