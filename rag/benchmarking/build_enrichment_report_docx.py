"""Generate a Word report of the enriched-benchmark metrics.

Produces ``output_table/benchmarking_enriched/embeddings/enrichment_report.docx``
with the accuracy/cost table, the ground-truth diff, and the two fair accuracy
comparisons (same-96-rows and all-132-rows).

    python -m benchmarking.build_enrichment_report_docx
"""

from __future__ import annotations

import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

OUT_DIR = os.path.join("output_table", "benchmarking_enriched", "embeddings")
OUT_PATH = os.path.join(OUT_DIR, "enrichment_report.docx")

HEADER_BG = "2F5496"  # dark blue


def _shade(cell, hex_color: str) -> None:
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_color)
    tc_pr.append(shd)


def add_table(doc, headers, rows, col_align=None):
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
        run.font.size = Pt(10)
    for r_idx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, val in enumerate(row):
            p = cells[i].paragraphs[0]
            if col_align and col_align[i]:
                p.alignment = col_align[i]
            run = p.add_run(str(val))
            run.font.size = Pt(10)
            if r_idx % 2 == 1:
                _shade(cells[i], "EDEDED")
    return table


def heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    return h


def main() -> None:
    doc = Document()

    title = doc.add_heading("Metadata Enrichment - Benchmark Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub = doc.add_paragraph(
        "Enriched gold set vs. original OSDU ground truth - embedding models "
        "(reranker OFF). Accuracy is end-to-end LLM-judge accuracy over the "
        "human-verified enriched gold set."
    )
    sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in sub.runs:
        run.italic = True
        run.font.size = Pt(9)

    C = WD_ALIGN_PARAGRAPH.CENTER
    L = WD_ALIGN_PARAGRAPH.LEFT

    # ---- Accuracy and cost ------------------------------------------------
    heading(doc, "Accuracy and Cost", level=1)
    add_table(
        doc,
        ["Config", "LLM acc", "Token acc", "Retrieval acc", "Total cost", "Ctx window"],
        [
            ["jina_latechunk", "85.6%", "54.5%", "55.3%", "$2.1557", "8,192"],
            ["openai_large", "85.6%", "55.3%", "55.3%", "$2.1947", "8,191"],
            ["openai_small", "88.6%", "57.6%", "51.5%", "$2.8467", "8,191"],
        ],
        col_align=[L, C, C, C, C, C],
    )

    heading(doc, "Cost Breakdown", level=1)
    add_table(
        doc,
        ["Config", "Embed $", "Generation $", "Judge $", "Total $", "Embed tokens", "LLM tokens"],
        [
            ["jina_latechunk", "N/A (self-hosted)", "$2.0331", "$0.1226", "$2.1557", "0", "1,173,942"],
            ["openai_large", "$0.0000", "$2.0719", "$0.1228", "$2.1947", "0", "1,190,406"],
            ["openai_small", "$0.0000", "$2.7238", "$0.1229", "$2.8467", "0", "1,563,338"],
        ],
        col_align=[L, C, C, C, C, C, C],
    )

    # ---- Part 1: GT diff --------------------------------------------------
    heading(doc, "Part 1 - Ground-Truth Diff (enriched vs. original OSDU)", level=1)
    doc.add_paragraph(
        "Across all 132 scored (doc x attribute) pairs, comparing each "
        "enriched_ground_truth/ value against the original ground truth the "
        "pipeline extracts from matched_json/:"
    )
    add_table(
        doc,
        ["Category", "Count", "%", "Meaning"],
        [
            ["verified", "80", "60.6%", "original had a value and it matches the enriched value"],
            ["improved", "16", "12.1%", "original had a value but it changed in the enriched set"],
            ["enriched", "36", "27.3%", "original was null/empty; enrichment supplied a value"],
            ["removed", "0", "0.0%", "-"],
            ["still_null", "0", "0.0%", "-"],
        ],
        col_align=[L, C, C, L],
    )

    # ---- Comparison A -----------------------------------------------------
    heading(doc, "Comparison A - Fair, Same 96 Rows (only rows scorable on original GT)", level=1)
    doc.add_paragraph(
        "Same denominator, same rows, both gold sets. Enriched now scored on the "
        "same 96 rows (not 132)."
    )
    add_table(
        doc,
        ["Model", "Enriched GT (LLM acc)", "Original GT (LLM acc)", "\u0394"],
        [
            ["jina_latechunk", "85.4% (82/96)", "79.2% (76/96)", "+6.2"],
            ["openai_large", "82.3% (79/96)", "77.1% (74/96)", "+5.2"],
            ["openai_small", "89.6% (86/96)", "83.3% (80/96)", "+6.3"],
        ],
        col_align=[L, C, C, C],
    )
    doc.add_paragraph(
        "Even on identical rows, the enriched gold set scores each model ~5-6 "
        "points higher - i.e. the enrichment fixed/clarified ground-truth values "
        "(the 16 \"improved\" attributes) that the models were actually getting "
        "right but the noisy original GT marked wrong."
    )

    # ---- Comparison B -----------------------------------------------------
    heading(doc, "Comparison B - All 132 Rows (original null GT counted as not-answerable)", level=1)
    doc.add_paragraph(
        "Here the 36 attributes with no original GT stay in the denominator; since "
        "they can't be graded against the original record, they count against "
        "original accuracy."
    )
    add_table(
        doc,
        ["Model", "Enriched GT (LLM acc)", "Original GT (LLM acc)", "\u0394"],
        [
            ["jina_latechunk", "85.6% (113/132)", "57.6% (76/132)", "+28.0"],
            ["openai_large", "85.6% (113/132)", "56.1% (74/132)", "+29.5"],
            ["openai_small", "88.6% (117/132)", "60.6% (80/132)", "+28.0"],
        ],
        col_align=[L, C, C, C],
    )
    doc.add_paragraph(
        "The ~28-point gap is almost entirely coverage: 36/132 (27%) questions "
        "have no answer at all in the original OSDU record, so the model can never "
        "be credited for them there."
    )

    os.makedirs(OUT_DIR, exist_ok=True)
    doc.save(OUT_PATH)
    print(f"  saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
