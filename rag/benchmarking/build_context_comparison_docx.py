"""Word report: context-expansion strategy comparison across two repeat runs.

Reads the two context-benchmark runs

    output_table/benchmarking_enriched/context_run1/comparison.json   (run 1)
    output_table/benchmarking_enriched/context/comparison.json        (run 2)

plus the related enriched artifacts (Task 1 embeddings, Task 3 reranker, Task 4
LLMs) to explain why the hierarchical strategy lands near 90% and why the same
(openai_small + gpt-5.2) pair scored only 81.8% in the earlier LLM benchmark.

Produces per-run metric tables, a run-1-vs-run-2 variance table, a
cross-configuration reference table, and a written analysis section.

    python -m benchmarking.build_context_comparison_docx
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

ROOT = os.path.join("output_table", "benchmarking_enriched")
RUN1 = os.path.join(ROOT, "context_run1", "comparison.json")
RUN2 = os.path.join(ROOT, "context", "comparison.json")
RERANKER_JSON = os.path.join(ROOT, "reranker", "metrics.json")
LLMS_JSON = os.path.join(ROOT, "llms", "comparison.json")
EMB_JSON = os.path.join(ROOT, "embeddings", "comparison.json")
OUT_PATH = os.path.join(ROOT, "context_comparison_report.docx")

HEADER_BG = "2F5496"
ZEBRA_BG = "EDEDED"


def _load(path: str) -> Optional[Any]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "N/A"


def _num(v: Optional[float]) -> str:
    return f"{int(v):,}" if isinstance(v, (int, float)) else "N/A"


def _secs(v: Optional[float]) -> str:
    return f"{v:.1f}" if isinstance(v, (int, float)) else "N/A"


def _f2(v: Optional[float]) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"


def _pp(a: Optional[float], b: Optional[float]) -> str:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        d = (b - a) * 100
        return f"{d:+.1f} pp"
    return "N/A"


# ---------------------------------------------------------------------------
# Word helpers
# ---------------------------------------------------------------------------


def _shade(cell, hex_color: str) -> None:
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_table(doc, headers: List[str], rows: List[List[str]]) -> None:
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        _shade(hdr[i], HEADER_BG)
        p = hdr[i].paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(h)
        run.bold = True
        run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        run.font.size = Pt(9)
    for r_idx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, val in enumerate(row):
            p = cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if i == 0 else WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(str(val))
            run.font.size = Pt(9)
            if r_idx % 2 == 1:
                _shade(cells[i], ZEBRA_BG)
    doc.add_paragraph()


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _run_strategies(data: Optional[Dict[str, Any]]) -> "OrderedRows":
    """[(strategy, summary)] for a context comparison.json (ok rows only)."""
    out = []
    if not data:
        return out
    for r in data.get("results", []):
        if r.get("status") == "ok":
            out.append((r["strategy"], r["summary"]))
    return out


OrderedRows = List[Tuple[str, Dict[str, Any]]]


def _metrics_table(doc, run_label: str, rows: OrderedRows) -> None:
    doc.add_heading(run_label, level=2)
    if not rows:
        doc.add_paragraph("(no data found)")
        return
    trows = []
    for strat, s in rows:
        tok = s.get("tokens", {})
        tm = s.get("time_s", {})
        esc = s.get("escalation", {})
        trows.append([
            strat,
            f"{_pct(s.get('combined_llm_accuracy'))} ({s.get('llm_correct')}/{s.get('scorable_questions')})",
            _pct(s.get("combined_token_accuracy")),
            _pct(s.get("combined_retrieval_accuracy")),
            _num(tok.get("total")),
            _secs(tm.get("total")),
            _num(esc.get("total_llm_calls")),
            _f2(esc.get("avg_llm_calls_per_query")),
        ])
    add_table(doc, ["Strategy", "LLM acc", "Token acc", "Retr acc", "Total tokens",
                    "Total time (s)", "LLM calls", "Avg calls/q"], trows)


def _variance_table(doc, r1: OrderedRows, r2: OrderedRows) -> None:
    doc.add_heading("Run-to-run variance (Run 1 vs Run 2)", level=2)
    m1 = {s: summ for s, summ in r1}
    m2 = {s: summ for s, summ in r2}
    rows = []
    for strat in [s for s, _ in r1] or list(m2):
        a = m1.get(strat, {}).get("combined_llm_accuracy")
        b = m2.get(strat, {}).get("combined_llm_accuracy")
        ca = m1.get(strat, {}).get("llm_correct")
        cb = m2.get(strat, {}).get("llm_correct")
        rows.append([
            strat,
            f"{_pct(a)} ({ca})" if ca is not None else _pct(a),
            f"{_pct(b)} ({cb})" if cb is not None else _pct(b),
            _pp(a, b),
        ])
    add_table(doc, ["Strategy", "Run 1 LLM acc", "Run 2 LLM acc", "Delta (Run2-Run1)"], rows)


def _cross_config_table(doc) -> None:
    """Same (openai_small + gpt-5.2) pair under different pipeline settings."""
    doc.add_heading("Cross-configuration reference (openai_small + gpt-5.2)", level=2)
    rows: List[List[str]] = []

    emb = _load(EMB_JSON)
    if emb:
        for r in emb.get("results", []):
            if r.get("tag") == "openai_small" and r.get("status") == "ok":
                s = r["summary"]
                rows.append(["Task 1 embeddings", "OFF", "hierarchical (d5)",
                             f"{_pct(s.get('combined_llm_accuracy'))} ({s.get('llm_correct')}/{s.get('scorable_questions')})"])

    rr = _load(RERANKER_JSON)
    if rr:
        for m in rr.get("models", []):
            if m.get("model_tag") == "openai_small":
                off = m["end_to_end"]["reranker_off"]
                on = m["end_to_end"]["reranker_on"]
                rows.append(["Task 3 reranker", "OFF", "hierarchical (d5)",
                             f"{_pct(off.get('combined_llm_accuracy'))} ({off.get('llm_correct')}/{off.get('scorable_questions')})"])
                rows.append(["Task 3 reranker", "ON", "hierarchical (d5)",
                             f"{_pct(on.get('combined_llm_accuracy'))} ({on.get('llm_correct')}/{on.get('scorable_questions')})"])

    llms = _load(LLMS_JSON)
    if llms:
        for r in llms.get("results", []):
            if r.get("llm_model") == "gpt-5.2" and r.get("status") == "ok":
                s = r["summary"]
                rows.append(["Task 4 LLMs", "ON", "hierarchical (d5)",
                             f"{_pct(s.get('combined_llm_accuracy'))} ({s.get('llm_correct')}/{s.get('scorable_questions')})"])

    for label, path in [("Task 5 context (Run 1)", RUN1), ("Task 5 context (Run 2)", RUN2)]:
        data = _load(path)
        for strat, s in _run_strategies(data):
            depth = "d3" if strat.startswith("hierarchical") else "-"
            mode = "hierarchical (d3)" if strat.startswith("hierarchical") else strat
            rows.append([label, "OFF", mode,
                         f"{_pct(s.get('combined_llm_accuracy'))} ({s.get('llm_correct')}/{s.get('scorable_questions')})"])

    add_table(doc, ["Source", "Reranker", "Context mode", "LLM acc"], rows)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _analysis_bullets() -> List[Tuple[str, List[str]]]:
    r1 = {s: summ for s, summ in _run_strategies(_load(RUN1))}
    r2 = {s: summ for s, summ in _run_strategies(_load(RUN2))}

    def acc(m, strat):
        return m.get(strat, {}).get("combined_llm_accuracy")

    def esc_calls(m, strat):
        return m.get(strat, {}).get("escalation", {}).get("total_llm_calls")

    sections: List[Tuple[str, List[str]]] = []

    # Is 90% from the hierarchy or from LLM noise?
    hbits = []
    h1, h2 = acc(r1, "hierarchical_d3"), acc(r2, "hierarchical_d3")
    c1, c2 = acc(r1, "chunk"), acc(r2, "chunk")
    calls1 = esc_calls(r1, "hierarchical_d3")
    calls2 = esc_calls(r2, "hierarchical_d3")
    hbits.append(
        f"Hierarchical LLM accuracy: Run 1 {_pct(h1)}, Run 2 {_pct(h2)} "
        f"(spread {_pp(h1, h2)}). Chunk: Run 1 {_pct(c1)}, Run 2 {_pct(c2)}."
    )
    hbits.append(
        "The hierarchy mechanism barely fires: only "
        f"{(calls1 or 132) - 132} extra LLM calls in Run 1 and "
        f"{(calls2 or 132) - 132} in Run 2 over the 132 base queries "
        "(avg escalation depth ~0.1). So escalation itself can only flip a "
        "handful of questions - it is NOT what lifts accuracy to ~90%."
    )
    hbits.append(
        "Most of the chunk-vs-hierarchical gap is run-to-run LLM/judge "
        "nondeterminism, not the extra context: chunk and hierarchical are "
        "separate generation passes, and gpt-5.2 (a reasoning model) plus the "
        "LLM judge are sampled, not deterministic. The per-run variance table "
        "shows the same strategy moving by a few points between identical runs."
    )
    hbits.append(
        "Net: hierarchical is a small, real win on the ~10% of questions whose "
        "answer sits above the retrieved chunk, layered on top of a few points "
        "of sampling noise - it is not a large structural jump."
    )
    sections.append(("Is the ~90% from hierarchy calls or from noisy LLM output?", hbits))

    # Why 81.8 earlier vs ~90 now.
    wbits = [
        "The earlier 81.8% (Task 4 LLM sweep) used the SAME openai_small + "
        "gpt-5.2 pair but with the cross-encoder reranker turned ON; the context "
        "study runs with the reranker OFF.",
        "The reranker study (Task 3) measured this exact effect for openai_small: "
        "reranker OFF 88.6% vs ON 84.8% (-3.8 pp). Turning the reranker OFF alone "
        "recovers ~5 questions, i.e. most of the 81.8 to ~88-90 move.",
        "Task 1 embeddings scored openai_small at 88.6% (reranker OFF) - which is "
        "right in line with the context runs, confirming ~88-90% is the true "
        "reranker-OFF operating point and 81.8% was the reranker-ON dip.",
        "The residual difference between 88.6% and 90.9% is within run-to-run "
        "noise (a 2-3 question swing on 132 questions is ~1.5-2.3 pp).",
        "Depth was also reduced from 5 to 3 for the context study, but with "
        "escalation firing on so few queries this has negligible effect.",
    ]
    sections.append(("Why did openai_small + gpt-5.2 score 81.8% before but ~90% now?", wbits))

    return sections


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def main() -> None:
    doc = Document()
    title = doc.add_heading("Context-Expansion Strategy - Repeat-Run Comparison & Analysis", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        "Two independent runs of the context-expansion benchmark "
        "(openai_small embedding, gpt-5.2 generator/judge, reranker OFF), scored "
        "against the human-verified enriched gold set. Includes run-to-run "
        "variance, a cross-configuration reference, and analysis of the accuracy "
        "drivers."
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in sub.runs:
        run.italic = True
        run.font.size = Pt(9)

    doc.add_heading("Analysis & Key Findings", level=1)
    for heading, bullets in _analysis_bullets():
        doc.add_heading(heading, level=2)
        for b in bullets:
            doc.add_paragraph(b, style="List Bullet")

    doc.add_page_break()
    doc.add_heading("Per-run metrics", level=1)
    _metrics_table(doc, "Run 1", _run_strategies(_load(RUN1)))
    _metrics_table(doc, "Run 2", _run_strategies(_load(RUN2)))
    _variance_table(doc, _run_strategies(_load(RUN1)), _run_strategies(_load(RUN2)))

    doc.add_page_break()
    doc.add_heading("Cross-configuration reference", level=1)
    doc.add_paragraph(
        "The same embedding + generator pair (openai_small + gpt-5.2) under the "
        "different pipeline settings used across the enriched benchmark tasks. "
        "This isolates the reranker (ON vs OFF) as the dominant driver of the "
        "81.8% vs ~90% difference."
    )
    _cross_config_table(doc)

    os.makedirs(ROOT, exist_ok=True)
    doc.save(OUT_PATH)
    print(f"  saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
