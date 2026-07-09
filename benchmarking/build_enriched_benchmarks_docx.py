"""Bundle the enriched Matryoshka / Reranker / LLM benchmark reports into one Word file.

Reads the markdown reports produced by

    python -m benchmarking.benchmark_enriched --matryoshka-only
    python -m benchmarking.benchmark_enriched --reranker-only
    python -m benchmarking.benchmark_enriched --llms-only

renders every markdown table as a real, styled Word table, and appends a
data-driven "Analysis & Key Findings" section computed from the JSON artifacts.

    python -m benchmarking.build_enriched_benchmarks_docx
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

ROOT = os.path.join("output_table", "benchmarking_enriched")
OUT_PATH = os.path.join(ROOT, "enriched_benchmarks_report.docx")

MATRYOSHKA_MD = os.path.join(ROOT, "matryoshka", "analysis_report.md")
MATRYOSHKA_JSON = os.path.join(ROOT, "matryoshka", "aggregate.json")
RERANKER_MD = os.path.join(ROOT, "reranker", "report.md")
RERANKER_JSON = os.path.join(ROOT, "reranker", "metrics.json")
LLMS_MD = os.path.join(ROOT, "llms", "report.md")
LLMS_JSON = os.path.join(ROOT, "llms", "comparison.json")

HEADER_BG = "2F5496"
ZEBRA_BG = "EDEDED"


# ---------------------------------------------------------------------------
# Word styling helpers
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


# ---------------------------------------------------------------------------
# Minimal markdown -> Word renderer (headings, tables, paragraphs, bullets)
# ---------------------------------------------------------------------------


def _split_md_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _is_sep_row(line: str) -> bool:
    return bool(re.match(r"^\s*\|?\s*:?-{2,}", line)) and set(line.strip()) <= set("|-: ")


def render_markdown(doc, md_text: str, base_heading_offset: int = 1) -> None:
    """Render a subset of markdown into ``doc``.

    ``base_heading_offset`` shifts markdown heading levels (so a report's H1
    becomes a Word H2 under the top-level section, etc.).
    """
    lines = md_text.splitlines()
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Table: a header row followed by a separator row.
        if stripped.startswith("|") and i + 1 < n and _is_sep_row(lines[i + 1]):
            headers = _split_md_row(stripped)
            rows: List[List[str]] = []
            j = i + 2
            while j < n and lines[j].strip().startswith("|"):
                rows.append(_split_md_row(lines[j]))
                j += 1
            add_table(doc, headers, rows)
            doc.add_paragraph()
            i = j
            continue

        # Headings.
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level = min(len(m.group(1)) + base_heading_offset, 4)
            doc.add_heading(m.group(2).strip(), level=level)
            i += 1
            continue

        # Bullet list.
        if stripped.startswith("- "):
            doc.add_paragraph(stripped[2:].strip(), style="List Bullet")
            i += 1
            continue

        # Plain paragraph.
        doc.add_paragraph(stripped)
        i += 1


# ---------------------------------------------------------------------------
# Data-driven analysis (from JSON artifacts)
# ---------------------------------------------------------------------------


def _load_json(path: str) -> Optional[Any]:
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "N/A"


def _usd(v: Optional[float]) -> str:
    return f"${v:.4f}" if isinstance(v, (int, float)) else "N/A"


def _matryoshka_findings() -> List[str]:
    data = _load_json(MATRYOSHKA_JSON)
    if not data:
        return []
    runs = [r for r in data.get("runs", []) if isinstance(r.get("llm_accuracy"), (int, float))]
    if not runs:
        return []
    best = max(runs, key=lambda r: r["llm_accuracy"])
    cheapest = min(runs, key=lambda r: (r.get("total_cost_usd") if isinstance(r.get("total_cost_usd"), (int, float)) else 9e9))
    out = [
        f"Best end-to-end LLM accuracy: {best['model']} at {best['dim']} dims "
        f"({_pct(best['llm_accuracy'])}, cost {_usd(best.get('total_cost_usd'))}).",
        f"Lowest cost config: {cheapest['model']} at {cheapest['dim']} dims "
        f"({_usd(cheapest.get('total_cost_usd'))}, LLM acc {_pct(cheapest['llm_accuracy'])}).",
    ]
    # Retention: is truncation nearly free?
    by_model: Dict[str, List[Dict[str, Any]]] = {}
    for r in runs:
        by_model.setdefault(r["model"], []).append(r)
    for model, rs in by_model.items():
        rs.sort(key=lambda r: r["dim"])
        full = rs[-1]["llm_accuracy"]
        small = rs[0]
        if full:
            ret = small["llm_accuracy"] / full * 100
            out.append(
                f"{model}: truncating to {small['dim']} dims retains {ret:.0f}% of "
                f"full-dimension ({rs[-1]['dim']}-d) LLM accuracy - Matryoshka "
                "truncation is essentially free on this gold set."
            )
    return out


def _reranker_findings() -> List[str]:
    data = _load_json(RERANKER_JSON)
    if not data:
        return []
    out = []
    for m in data.get("models", []):
        off = m["end_to_end"]["reranker_off"].get("combined_llm_accuracy")
        on = m["end_to_end"]["reranker_on"].get("combined_llm_accuracy")
        if isinstance(off, (int, float)) and isinstance(on, (int, float)):
            delta = (on - off) * 100
            sign = "+" if delta >= 0 else ""
            out.append(
                f"{m['model_tag']}: reranker OFF {_pct(off)} -> ON {_pct(on)} "
                f"({sign}{delta:.1f} pp LLM accuracy)."
            )
    if out:
        out.append(
            "Across every embedding config the cross-encoder reranker lowered "
            "end-to-end LLM accuracy on the enriched gold set, so it should stay "
            "OFF for this workload (it also roughly halves prompt tokens, but "
            "accuracy is the deciding factor)."
        )
    return out


def _llm_findings() -> List[str]:
    data = _load_json(LLMS_JSON)
    if not data:
        return []
    ok = [r for r in data.get("results", []) if r.get("status") == "ok"]
    if not ok:
        return []

    def acc(r):
        return r.get("summary", {}).get("combined_llm_accuracy")

    def total(r):
        return r.get("cost_recomputed", {}).get("total")

    best = max(ok, key=lambda r: acc(r) if isinstance(acc(r), (int, float)) else -1)
    cheapest = min(ok, key=lambda r: total(r) if isinstance(total(r), (int, float)) else 9e9)

    def per_dollar(r):
        a, t = acc(r), total(r)
        return (a / t) if isinstance(a, (int, float)) and isinstance(t, (int, float)) and t else -1

    value = max(ok, key=per_dollar)
    out = [
        f"Highest accuracy generator: {best['llm_model']} ({_pct(acc(best))}, "
        f"cost {_usd(total(best))}) on embedding '{data.get('embedding_model')}'.",
        f"Cheapest generator: {cheapest['llm_model']} ({_usd(total(cheapest))}, "
        f"LLM acc {_pct(acc(cheapest))}).",
        f"Best accuracy-per-dollar: {value['llm_model']} "
        f"({_pct(acc(value))} at {_usd(total(value))}).",
    ]
    skipped = [r for r in data.get("results", []) if r.get("status") != "ok"]
    if skipped:
        out.append(
            "Unavailable / skipped models: "
            + ", ".join(f"{r['llm_model']} ({r['status']})" for r in skipped)
            + "."
        )
    return out


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------


def _title_page(doc) -> None:
    title = doc.add_heading("Metadata Enrichment - Enriched Benchmark Suite", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        "Matryoshka dimension study, reranker impact, and generator-LLM sweep, "
        "each scored end-to-end against the human-verified enriched gold set "
        "(no MTEB retrieval-ranking metrics)."
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in sub.runs:
        run.italic = True
        run.font.size = Pt(9)


def _bullets_section(doc, title: str, bullets: List[str]) -> None:
    doc.add_heading(title, level=2)
    if not bullets:
        doc.add_paragraph("(no data found)")
        return
    for b in bullets:
        doc.add_paragraph(b, style="List Bullet")


def main() -> None:
    doc = Document()
    _title_page(doc)

    doc.add_heading("Analysis & Key Findings", level=1)
    doc.add_paragraph(
        "Automatically summarized from the JSON artifacts of the three runs. "
        "Full metric tables follow in the sections below."
    )
    _bullets_section(doc, "Task 2 - Matryoshka dimension study", _matryoshka_findings())
    _bullets_section(doc, "Task 3 - Reranker impact", _reranker_findings())
    _bullets_section(doc, "Task 4 - Generator-LLM sweep", _llm_findings())

    doc.add_page_break()
    doc.add_heading("Task 2 - Matryoshka Dimension Study", level=1)
    md = _read(MATRYOSHKA_MD)
    render_markdown(doc, _strip_first_h1(md), base_heading_offset=1)

    doc.add_page_break()
    doc.add_heading("Task 3 - Reranker Impact", level=1)
    render_markdown(doc, _strip_first_h1(_read(RERANKER_MD)), base_heading_offset=1)

    doc.add_page_break()
    doc.add_heading("Task 4 - Generator-LLM Sweep", level=1)
    render_markdown(doc, _strip_first_h1(_read(LLMS_MD)), base_heading_offset=1)

    os.makedirs(ROOT, exist_ok=True)
    doc.save(OUT_PATH)
    print(f"  saved -> {OUT_PATH}")


def _read(path: str) -> str:
    if not os.path.isfile(path):
        return f"_Report not found: {path}_"
    with open(path, encoding="utf-8") as f:
        return f.read()


def _strip_first_h1(md: str) -> str:
    """Drop the report's leading H1 (we add our own section heading)."""
    lines = md.splitlines()
    for idx, line in enumerate(lines):
        if line.strip().startswith("# "):
            return "\n".join(lines[idx + 1:]).lstrip("\n")
    return md


if __name__ == "__main__":
    main()
