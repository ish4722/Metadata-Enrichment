"""Chain-of-Verification (CoVe) confidence scoring.

This module produces a *confidence score* for an answer the RAG pipeline already
generated. It never changes the answer. The answer is treated as a HYPOTHESIS and
verified INDIRECTLY:

  1. plan   - an LLM proposes N domain-specific checks. Each check asks about a
              DIFFERENT, related fact (never restates the original question) and
              carries a domain-logic ``rule`` that must hold if the answer is
              correct (e.g. "spud date <= completion date") plus a ``relevance``.
  2. execute- for each check the caller retrieves the related fact via the same
              RAG pipeline and an LLM judge extracts the fact + evaluates the rule,
              returning ``consistent | inconsistent | not_found`` with a confidence.
  3. aggregate - the per-check verdicts are combined (relevance-weighted, with an
              asymmetric verdict scale and a contradiction veto) into a 0-1 score.

Why no restatement / grounding checks: re-asking the same question and comparing
to the answer is circular (a hallucination confirms itself); proving the answer
text appears in some passage does not prove it is correct. Every check here pulls
a *different* fact and corroborates the answer through logic.

This module has no web/store dependencies (prompts + parsing + math only) so it is
unit-testable; the orchestration that retrieves per-check context lives in the
web-app engine.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

# ── scoring constants (tune here) ─────────────────────────────────────────────

# Planner relevance -> weight. A wrong high-relevance check should dominate.
RELEVANCE_WEIGHTS: Dict[str, float] = {"high": 3.0, "medium": 2.0, "low": 1.0}
_DEFAULT_RELEVANCE = "medium"

# Verdict -> value. Asymmetric on purpose: a contradiction ("inconsistent") is a
# full penalty, while merely "not_found" (could not verify) is only a mild one.
VERDICT_VALUES: Dict[str, float] = {
    "consistent": 1.0,
    "not_found": 0.5,
    "inconsistent": 0.0,
}
_VALID_VERDICTS = set(VERDICT_VALUES)

# Final blend: corroboration score dominates; the model's own verbalized
# confidence is a minor supporting term.
SCORE_WEIGHT = 0.85
VERBALIZED_WEIGHT = 0.15

# Contradiction veto: a confident, high-weight contradiction caps the score, but
# ONLY when it is not clearly outweighed by corroborating (consistent) evidence.
# A single high-relevance contradiction against several consistent checks (often
# a flawed/over-strict judge rule) must NOT tank an otherwise well-supported
# answer.
VETO_MIN_WEIGHT = 3.0        # only high-relevance checks can trigger the veto
VETO_MIN_CONFIDENCE = 0.7    # the judge must be reasonably sure of the contradiction
VETO_CAP = 0.3               # capped final confidence when the veto fires

# Confidence bands for the UI.
BAND_HIGH = 0.75
BAND_MEDIUM = 0.5


# ── JSON extraction helpers ───────────────────────────────────────────────────
def _extract_json(raw: str) -> Optional[Any]:
    """Parse the first JSON value (object or array) from ``raw``.

    Robust to models that wrap JSON in prose or markdown fences.
    """
    if not raw or not raw.strip():
        return None
    text = raw.strip()
    # Strip markdown fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Fall back to the first embedded {...} or [...] block.
    match = re.search(r"[\[{].*[\]}]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _normalize_relevance(value: Any) -> str:
    key = str(value or "").strip().lower()
    return key if key in RELEVANCE_WEIGHTS else _DEFAULT_RELEVANCE


def _clamp01(value: Any) -> Optional[float]:
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, conf))


# ── 1. planning prompt ────────────────────────────────────────────────────────
def build_plan_prompt(question: str, answer: str, well_name: Optional[str]) -> str:
    """Prompt that asks the model to plan INDIRECT corroboration checks.

    The model must NOT restate the original question and must NOT propose checks
    that merely look for the answer text; each check targets a different, related
    fact plus a domain-logic rule.
    """
    well = (well_name or "the well").strip() or "the well"
    return (
        "You are verifying whether a proposed answer about an oil & gas well is "
        "correct, by planning INDIRECT verification checks.\n\n"
        f"Well/facility: {well}\n"
        f"Original question: {question}\n"
        f"Proposed answer (the HYPOTHESIS to test): {answer}\n\n"
        "Design verification checks that corroborate the proposed answer through "
        "OTHER, RELATED facts and domain logic. Each check must:\n"
        "  - ask about a DIFFERENT fact than the original question (NEVER restate "
        "the original question, and NEVER just ask to find the answer again),\n"
        "  - come with a short domain-logic RULE that must hold if the proposed "
        "answer is correct (e.g. 'spud date must be on or before the completion "
        "date', 'total depth reached before completion', 'the well status must be "
        "consistent with having a completion date', 'the described well must be "
        f"{well}'),\n"
        "  - be answerable from the well report.\n\n"
        "Rate each check's RELEVANCE: 'high' if a violation strongly implies the "
        "answer is wrong, 'medium' if it is supporting evidence, 'low' if it is a "
        "weak sanity check.\n\n"
        "Respond with ONLY a JSON array (no markdown, no extra text) of objects:\n"
        "[\n"
        '  {"question": "<related-fact question to retrieve>", '
        '"rule": "<logic that must hold vs the proposed answer>", '
        '"relevance": "high|medium|low"}\n'
        "]\n"
    )


def parse_plan(raw: str, max_checks: int) -> List[Dict[str, str]]:
    """Parse the planner reply into a list of ``{question, rule, relevance}``.

    Drops malformed entries and clamps to ``max_checks``.
    """
    data = _extract_json(raw)
    if not isinstance(data, list):
        return []
    checks: List[Dict[str, str]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        q = str(item.get("question") or "").strip()
        rule = str(item.get("rule") or "").strip()
        if not q or not rule:
            continue
        checks.append({
            "question": q,
            "rule": rule,
            "relevance": _normalize_relevance(item.get("relevance")),
        })
        if len(checks) >= max_checks:
            break
    return checks


# ── 2. per-check judge prompt ─────────────────────────────────────────────────
def build_judge_prompt(
    check: Dict[str, str],
    context_items: List[Dict[str, Any]],
    answer: str,
) -> str:
    """Prompt the LLM to extract the related fact and evaluate the check's rule.

    Returns instructions for a strict JSON verdict. ``context_items`` are the
    excerpts freshly retrieved for the check's own question.
    """
    parts: List[str] = []
    for item in context_items:
        header = f"[source={item.get('source')}, page={item.get('page')}]"
        parts.append(f"{header}\n{item.get('text', '')}")
    context = "\n\n---\n\n".join(parts) if parts else "(no excerpts retrieved)"
    return (
        "You are verifying a proposed answer about an oil & gas well by checking a "
        "related fact against a logic rule. Use ONLY the excerpts below.\n\n"
        f"Proposed answer being verified: {answer}\n"
        f"Related-fact question: {check.get('question')}\n"
        f"Rule that must hold if the proposed answer is correct: {check.get('rule')}\n\n"
        "Steps:\n"
        "1. Extract the related fact from the excerpts (or note it is absent).\n"
        "2. Decide the verdict:\n"
        "   - 'consistent'   : the related fact is present and the rule holds.\n"
        "   - 'inconsistent' : the related fact is present and the rule is VIOLATED.\n"
        "   - 'not_found'    : the related fact is not in the excerpts.\n\n"
        "Excerpts:\n"
        f"{context}\n\n"
        "Respond with ONLY a JSON object (no markdown, no extra text):\n"
        "{\n"
        '  "related_fact": "<the fact you found, or empty>",\n'
        '  "verdict": "consistent|inconsistent|not_found",\n'
        '  "confidence": <number 0.0-1.0 for how sure you are of the verdict>\n'
        "}\n"
    )


def parse_judge(raw: str) -> Dict[str, Any]:
    """Parse a judge reply into ``{related_fact, verdict, confidence}``.

    Unrecognized/malformed replies default to ``not_found`` with no confidence.
    """
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return {"related_fact": "", "verdict": "not_found", "confidence": None}
    verdict = str(data.get("verdict") or "").strip().lower()
    if verdict not in _VALID_VERDICTS:
        verdict = "not_found"
    return {
        "related_fact": str(data.get("related_fact") or "").strip(),
        "verdict": verdict,
        "confidence": _clamp01(data.get("confidence")),
    }


# ── 4. low-confidence suggestion ──────────────────────────────────────────────
# When the aggregated band is this (or worse), attempt to propose a corrected
# answer from the excerpts the verification actually saw.
SUGGEST_BAND = "low"


def build_suggestion_prompt(
    question: str,
    answer: str,
    checks: List[Dict[str, Any]],
    context_items: List[Dict[str, Any]],
) -> str:
    """Prompt that proposes a corrected answer from the verification evidence.

    Given the low-confidence verdicts and the excerpts the checks retrieved, the
    model decides whether the answer should change and, if so, supplies the
    corrected value strictly from those excerpts.
    """
    findings: List[str] = []
    for c in checks:
        findings.append(
            f"- check: {c.get('question')}\n"
            f"  rule: {c.get('rule')}\n"
            f"  verdict: {c.get('verdict')} | related fact found: {c.get('related_fact') or '(none)'}"
        )
    findings_block = "\n".join(findings) if findings else "(no checks)"

    parts: List[str] = []
    for item in context_items:
        header = f"[source={item.get('source')}, page={item.get('page')}]"
        parts.append(f"{header}\n{item.get('text', '')}")
    context = "\n\n---\n\n".join(parts) if parts else "(no excerpts)"

    return (
        "A retrieval system produced an answer whose confidence is LOW after "
        "verification. Using ONLY the excerpts below and the verification findings, "
        "decide whether the answer should be corrected.\n\n"
        f"Question: {question}\n"
        f"Current answer: {answer}\n\n"
        "Verification findings:\n"
        f"{findings_block}\n\n"
        "Excerpts the verification retrieved:\n"
        f"{context}\n\n"
        "If the excerpts clearly support a DIFFERENT and better answer, propose it. "
        "If the current answer is actually fine or the excerpts do not support a "
        "confident correction, set needs_change to false. Never invent values not "
        "grounded in the excerpts.\n\n"
        "Respond with ONLY a JSON object (no markdown, no extra text):\n"
        "{\n"
        '  "needs_change": <true|false>,\n'
        '  "suggested_answer": "<the corrected answer, or empty if no change>",\n'
        '  "reason": "<short justification citing the evidence>"\n'
        "}\n"
    )


def parse_suggestion(raw: str) -> Optional[Dict[str, Any]]:
    """Parse a suggestion reply into ``{needs_change, suggested_answer, reason}``.

    Returns ``None`` when no usable suggestion (no change needed or unparseable).
    """
    data = _extract_json(raw)
    if not isinstance(data, dict):
        return None
    needs = data.get("needs_change")
    if isinstance(needs, str):
        needs = needs.strip().lower() in {"true", "yes", "1"}
    suggested = str(data.get("suggested_answer") or "").strip()
    if not needs or not suggested:
        return None
    return {
        "needs_change": True,
        "suggested_answer": suggested,
        "reason": str(data.get("reason") or "").strip(),
    }


# ── 3. aggregation ────────────────────────────────────────────────────────────
def _band(score: float) -> str:
    if score >= BAND_HIGH:
        return "high"
    if score >= BAND_MEDIUM:
        return "medium"
    return "low"


def aggregate(
    checks: List[Dict[str, Any]],
    verbalized_conf: Optional[float] = None,
) -> Dict[str, Any]:
    """Combine per-check verdicts into a final confidence score.

    ``checks`` items carry at least ``relevance`` and ``verdict`` (and optionally
    ``confidence``). Returns ``{confidence, band, breakdown}`` where ``confidence``
    is in [0, 1].
    """
    if not checks:
        # No corroboration possible - fall back to the verbalized confidence if any.
        base = verbalized_conf if verbalized_conf is not None else 0.5
        return {
            "confidence": round(base, 4),
            "band": _band(base),
            "breakdown": {
                "num_checks": 0,
                "score": None,
                "verbalized_conf": verbalized_conf,
                "vetoed": False,
            },
        }

    weighted_sum = 0.0
    weight_total = 0.0
    consistent_weight = 0.0   # corroborating evidence
    contra_weight = 0.0       # confident, high-relevance contradictions
    for c in checks:
        weight = RELEVANCE_WEIGHTS.get(_normalize_relevance(c.get("relevance")), RELEVANCE_WEIGHTS[_DEFAULT_RELEVANCE])
        verdict = c.get("verdict") if c.get("verdict") in _VALID_VERDICTS else "not_found"
        value = VERDICT_VALUES[verdict]
        weighted_sum += weight * value
        weight_total += weight
        conf = c.get("confidence")
        if verdict == "consistent":
            consistent_weight += weight
        if (verdict == "inconsistent" and weight >= VETO_MIN_WEIGHT
                and conf is not None and conf >= VETO_MIN_CONFIDENCE):
            contra_weight += weight

    score = (weighted_sum / weight_total) if weight_total else 0.5

    verb = verbalized_conf if verbalized_conf is not None else score
    final = SCORE_WEIGHT * score + VERBALIZED_WEIGHT * verb
    # Only cap when confident high-relevance contradictions are NOT outweighed by
    # the corroborating evidence. A lone contradiction against several consistent
    # checks (commonly a too-strict judge rule) should not veto a strong answer.
    vetoed = contra_weight > 0 and contra_weight >= consistent_weight
    if vetoed:
        final = min(final, VETO_CAP)
    final = max(0.0, min(1.0, final))

    return {
        "confidence": round(final, 4),
        "band": _band(final),
        "breakdown": {
            "num_checks": len(checks),
            "score": round(score, 4),
            "verbalized_conf": verbalized_conf,
            "vetoed": vetoed,
            "consistent_weight": consistent_weight,
            "contra_weight": contra_weight,
            "weight_total": weight_total,
        },
    }
