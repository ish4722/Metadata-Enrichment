"""Generate a combined Word report for all four benchmarking tasks.

Writes:
  * output_table/benchmarking/benchmarking_results.docx  (master report)
  * output_table/benchmarking/reranker/report.docx
  * output_table/benchmarking/llms/report.docx

Sections: embeddings, matryoshka, reranker, LLM generator sweep — each with
tables from the existing markdown/json outputs and a short written analysis.

Usage:
    python generate_benchmarking_results_docx.py
"""

from __future__ import annotations

import os
import sys

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from generate_embedding_report_docx import add_embeddings_section  # noqa: E402
from md_to_docx import append_markdown_to_doc, convert  # noqa: E402

BENCH_ROOT = os.path.join("output_table", "benchmarking")
MASTER_OUT = os.path.join(BENCH_ROOT, "benchmarking_results.docx")

PATHS = {
    "embeddings_json": os.path.join(BENCH_ROOT, "embeddings", "comparison.json"),
    "matryoshka_md": os.path.join("output_table", "matryoshka", "analysis_report.md"),
    "reranker_md": os.path.join(BENCH_ROOT, "reranker", "report.md"),
    "llms_md": os.path.join(BENCH_ROOT, "llms", "report.md"),
}


def _read_lines(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        return f.read().splitlines()


def _add_analysis_bullets(doc: Document, heading: str, bullets: list[tuple[str, str]]) -> None:
    doc.add_heading(heading, level=2)
    for head, body in bullets:
        p = doc.add_paragraph(style="List Bullet")
        r = p.add_run(head + " ")
        r.bold = True
        p.add_run(body)


def add_matryoshka_section(doc: Document, md_path: str) -> None:
    doc.add_heading("Task 2 — Matryoshka Embedding Dimension Study", level=1)
    if not os.path.isfile(md_path):
        doc.add_paragraph(f"[Missing: {md_path}]")
        return
    lines = _read_lines(md_path)
    # Skip the top-level title; content starts at executive summary.
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    append_markdown_to_doc(doc, lines, heading_offset=1)
    _add_analysis_bullets(doc, "Analysis", [
        ("Large model peaks at 2048 dims, not full 3072.",
         "text-embedding-3-large reaches 92.7% LLM accuracy at 2048 dimensions while "
         "full 3072-dim is 89.6% — truncation can help as well as hurt depending on dim."),
        ("Diminishing returns are real.",
         "256-dim vectors already land within ~2 pp of full-size LLM accuracy for both "
         "models; storage and search can be cut aggressively with minimal answer loss."),
        ("End-to-end numbers use 4-query subset; retrieval rescored on 12 queries.",
         "Per-label e2e tables cover four early query types; MTEB retrieval metrics in "
         "summary JSON were re-scored against all 12 queries in queries.txt."),
    ])


def add_reranker_section(doc: Document, md_path: str) -> None:
    doc.add_heading("Task 3 — Reranker Impact", level=1)
    if not os.path.isfile(md_path):
        doc.add_paragraph(f"[Missing: {md_path}]")
        return
    lines = _read_lines(md_path)
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    append_markdown_to_doc(doc, lines, heading_offset=1)
    _add_analysis_bullets(doc, "Analysis", [
        ("Reranker clearly improves retrieval ranking.",
         "All three embedding models gain nDCG@10 (+0.06 to +0.10) and MRR (+0.10 to +0.20) "
         "after cross-encoder reranking; Hit@1 jumps +0.16 to +0.22."),
        ("End-to-end accuracy does not improve (slight drop).",
         "LLM-judge accuracy falls 0.9–2.4 pp with reranker ON despite better retrieval — "
         "reranked context may omit useful chunks or change page assembly."),
        ("Jina benefits most from reranking.",
         "jina_latechunk gains the largest retrieval lift (+0.103 nDCG@10) but starts from "
         "the weakest baseline; OpenAI models already rank well pre-rerank."),
        ("Cost trade-off.",
         "Reranker ON reduces prompt tokens (smaller reranked context) but adds CPU latency "
         "during retrieval; judge cost is unchanged when --no-judge is used."),
    ])


def add_llms_section(doc: Document, md_path: str) -> None:
    doc.add_heading("Task 4 — Generator LLM Benchmark", level=1)
    if not os.path.isfile(md_path):
        doc.add_paragraph(f"[Missing: {md_path}]")
        return
    lines = _read_lines(md_path)
    if lines and lines[0].startswith("# "):
        lines = lines[1:]
    append_markdown_to_doc(doc, lines, heading_offset=1)
    _add_analysis_bullets(doc, "Analysis", [
        ("gpt-5-mini is the accuracy winner.",
         "76.8% LLM-judge accuracy at $2.17 total cost — best balance of quality and price "
         "among models tested."),
        ("gpt-5-nano is the value pick.",
         "66.6% accuracy at $0.72 total (0.93 acc per dollar) — suitable for high-volume "
         "screening where cost dominates."),
        ("gpt-5.2 is strong but expensive.",
         "75.8% accuracy at ~$9.94 — second-tier accuracy at premium pricing; use when "
         "latency or reasoning quality matters more than cost."),
        ("Retrieval accuracy is fixed across models.",
         "All LLMs share the same openai_large + reranker retrieval stack, so 54.6% "
         "retrieval accuracy is identical — this benchmark isolates generator quality only."),
    ])


def build_master_report() -> str:
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)

    title = doc.add_heading("Benchmarking Results", level=0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    intro = doc.add_paragraph()
    intro.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = intro.add_run(
        "RAG pipeline benchmarks: embedding models, Matryoshka dimensions, "
        "cross-encoder reranker, and generator LLMs. Corpus: 48 docs, 12 query types.")
    run.italic = True

    if os.path.isfile(PATHS["embeddings_json"]):
        add_embeddings_section(doc, PATHS["embeddings_json"])
    else:
        doc.add_heading("Task 1 — Embedding Model Benchmark", level=1)
        doc.add_paragraph(f"[Missing: {PATHS['embeddings_json']}]")

    add_matryoshka_section(doc, PATHS["matryoshka_md"])
    add_reranker_section(doc, PATHS["reranker_md"])
    add_llms_section(doc, PATHS["llms_md"])

    os.makedirs(BENCH_ROOT, exist_ok=True)
    doc.save(MASTER_OUT)
    return MASTER_OUT


def main() -> None:
    master = build_master_report()
    print(f"Saved -> {master}")

    reranker_docx = os.path.join(BENCH_ROOT, "reranker", "report.docx")
    llms_docx = os.path.join(BENCH_ROOT, "llms", "report.docx")

    if os.path.isfile(PATHS["reranker_md"]):
        convert(PATHS["reranker_md"], reranker_docx)
        print(f"Saved -> {reranker_docx}")

    if os.path.isfile(PATHS["llms_md"]):
        convert(PATHS["llms_md"], llms_docx)
        print(f"Saved -> {llms_docx}")

    # Keep legacy embedding-only doc in sync (same Task 1 section, standalone title).
    from generate_embedding_report_docx import main as embed_main  # noqa: WPS433
    embed_main()


if __name__ == "__main__":
    main()
