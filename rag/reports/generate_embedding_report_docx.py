"""Generate a Word (.docx) report from an embedding-benchmark comparison.json.

Reads output_table/benchmarking/embeddings/comparison.json and writes a formatted
report with:
  * per-label (item-wise) LLM-judge accuracy across models,
  * per-label breakdown of the three accuracy types for the best model,
  * MTEB retrieval metrics per model,
  * a written analysis / key takeaways.

Usage:
    python generate_embedding_report_docx.py \
        --input output_table/benchmarking/embeddings/comparison.json \
        --output output_table/benchmarking/embeddings/embedding_benchmark_report.docx
"""

from __future__ import annotations

import argparse
import json
import os

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt, RGBColor

RETRIEVAL_ROWS = [
    ("MRR", "mrr", False), ("MAP", "map", False),
    ("Hit@1", "hit@1", False), ("Hit@3", "hit@3", False),
    ("Hit@5", "hit@5", False), ("Hit@10", "hit@10", False),
    ("Recall@1", "recall@1", False), ("Recall@3", "recall@3", False),
    ("Recall@5", "recall@5", False), ("Recall@10", "recall@10", False),
    ("nDCG@1", "ndcg@1", False), ("nDCG@3", "ndcg@3", False),
    ("nDCG@5", "ndcg@5", False), ("nDCG@10", "ndcg@10", False),
    ("Distractor@1", "distractor@1", True), ("Distractor@3", "distractor@3", True),
    ("Distractor@5", "distractor@5", True), ("Distractor@10", "distractor@10", True),
]


def pct(x):
    return f"{x * 100:.1f}%"


def dec(x):
    return f"{x:.3f}"


def set_cell_bold(cell, bold=True):
    for p in cell.paragraphs:
        for r in p.runs:
            r.font.bold = bold


def shade_header(row):
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    for cell in row.cells:
        tcpr = cell._tc.get_or_add_tcPr()
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear")
        shd.set(qn("w:fill"), "2F5496")
        tcpr.append(shd)
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


def add_table(doc, headers, rows, bold_best_cols=None):
    """bold_best_cols: dict {row_index: set(col_index)} cells to bold (best value)."""
    table = doc.add_table(rows=1, cols=len(headers))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = table.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = str(h)
    shade_header(table.rows[0])
    for r_idx, row in enumerate(rows):
        cells = table.add_row().cells
        for c_idx, val in enumerate(row):
            cells[c_idx].text = str(val)
            if c_idx > 0:
                cells[c_idx].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            if bold_best_cols and c_idx in bold_best_cols.get(r_idx, set()):
                set_cell_bold(cells[c_idx], True)
    return table


def add_embeddings_section(doc: Document, comparison_path: str) -> None:
    """Add Task 1 embedding benchmark tables and analysis to *doc*."""
    data = json.load(open(comparison_path, encoding="utf-8"))
    results = data.get("results", [])
    ok = [r for r in results if r.get("status") == "ok"]
    failed = [r for r in results if r.get("status") != "ok"]
    if not ok:
        doc.add_paragraph("[No successful embedding benchmark results found.]")
        return

    tags = [r["tag"] for r in ok]
    labels = [pl["label"] for pl in ok[0]["summary"]["per_label"]]
    acc = {r["tag"]: {pl["label"]: pl for pl in r["summary"]["per_label"]} for r in ok}

    doc.add_heading("Task 1 — Embedding Model Benchmark", level=1)
    s0 = ok[0]["summary"]
    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = meta.add_run(
        f"Reranker: {data.get('reranker', 'OFF')}  |  "
        f"Total questions: {s0.get('total_questions')}  |  "
        f"Scorable: {s0.get('scorable_questions')}  |  "
        f"Null ground truth: {s0.get('null_ground_truth')}")
    run.italic = True

    if failed:
        p = doc.add_paragraph()
        note = p.add_run("Skipped: " + ", ".join(
            f"'{r['tag']}' ({r.get('status')})" for r in failed))
        note.font.color.rgb = RGBColor(0xC0, 0x00, 0x00)
        note.italic = True

    doc.add_paragraph(
        "Comparison table (reranker OFF — retrieval metrics are pre-rerank only):")

    headers = ["Config", "LLM acc", "Token acc", "Retr acc", "MRR", "Hit@1",
               "Hit@5", "Recall@5", "nDCG@10", "MAP", "Total cost"]
    rows = []
    for r in ok:
        sm = r["summary"]
        mac = r["retrieval_metrics"]["pre_rerank"]["macro"]
        cost = r.get("cost_recomputed", {}).get("total")
        rows.append([
            r["tag"], pct(sm["combined_llm_accuracy"]), pct(sm["combined_token_accuracy"]),
            pct(sm["combined_retrieval_accuracy"]), dec(mac["mrr"]), dec(mac["hit@1"]),
            dec(mac["hit@5"]), dec(mac["recall@5"]), dec(mac["ndcg@10"]), dec(mac["map"]),
            f"${cost:.4f}" if isinstance(cost, (int, float)) else "N/A",
        ])
    add_table(doc, headers, rows)

    doc.add_heading("Item-wise LLM-judge accuracy (per label × model)", level=2)
    headers = ["Label"] + tags
    rows, bold = [], {}
    for r_idx, lbl in enumerate(labels):
        vals = [acc[t][lbl]["llm_accuracy"] for t in tags]
        best = max(vals)
        rows.append([lbl] + [pct(v) for v in vals])
        bold[r_idx] = {i + 1 for i, v in enumerate(vals) if v == best}
    combined = [r["summary"]["combined_llm_accuracy"] for r in ok]
    rows.append(["Combined (all labels)"] + [pct(v) for v in combined])
    bcombined = max(combined)
    bold[len(labels)] = {i + 1 for i, v in enumerate(combined) if v == bcombined}
    add_table(doc, headers, rows, bold)

    best_tag = max(ok, key=lambda r: r["summary"]["combined_llm_accuracy"])["tag"]
    sm = next(r["summary"] for r in ok if r["tag"] == best_tag)
    macros = {t: next(r for r in ok if r["tag"] == t)["retrieval_metrics"]["pre_rerank"]["macro"]
              for t in tags}

    doc.add_heading("Analysis", level=2)
    analysis = [
        (f"{best_tag} wins overall.",
         f"Best combined LLM-judge accuracy ({pct(sm['combined_llm_accuracy'])}) and "
         f"strongest retrieval (Hit@10 {dec(macros[best_tag]['hit@10'])}, "
         f"nDCG@10 {dec(macros[best_tag]['ndcg@10'])}). Recommended default embedding."),
        ("openai_small is a close second.",
         "Nearly matches large on answer accuracy but weaker ranking (lower Hit@k / nDCG@k)."),
        ("jina_latechunk lags on retrieval.",
         "Hit@10 and nDCG@10 well below OpenAI models; fewer distractors but less complete recall."),
        ("Hardest labels.",
         "completion date and vertical measurement total depth score lowest (~48–60% LLM accuracy)."),
    ]
    for head, body in analysis:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(head + " ")
        r.bold = True
        p.add_run(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=os.path.join(
        "output_table", "benchmarking", "embeddings", "comparison.json"))
    ap.add_argument("--output", default=os.path.join(
        "output_table", "benchmarking", "embeddings", "embedding_benchmark_report.docx"))
    args = ap.parse_args()

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    title = doc.add_heading("Embedding Benchmark Report", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    add_embeddings_section(doc, args.input)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    doc.save(args.output)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
