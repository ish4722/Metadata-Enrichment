"""Convert a Markdown report into a Word (.docx) document.

Handles the constructs used by the study reports: ATX headings (``#``/``##``/
``###``), paragraphs, ``-`` bullet lists (including one level of indentation),
inline ``**bold**``, and GitHub-style pipe tables (with a ``---`` separator row).

Usage:
    python md_to_docx.py output_table/matryoshka/analysis_report.md
    python md_to_docx.py <input.md> [output.docx]
"""

from __future__ import annotations

import argparse
import os
import re
from typing import List

from docx import Document
from docx.shared import Pt


_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")


def _add_runs_with_bold(paragraph, text: str) -> None:
    """Append text to a paragraph, rendering ``**bold**`` spans as bold runs."""
    pos = 0
    for m in _BOLD_RE.finditer(text):
        if m.start() > pos:
            paragraph.add_run(text[pos:m.start()])
        paragraph.add_run(m.group(1)).bold = True
        pos = m.end()
    if pos < len(text):
        paragraph.add_run(text[pos:])


def _is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|") and line.rstrip().endswith("|")


def _is_separator_row(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return bool(cells) and all(set(c) <= set("-: ") and "-" in c for c in cells)


def _split_row(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _add_table(doc: Document, rows: List[List[str]]) -> None:
    if not rows:
        return
    ncols = max(len(r) for r in rows)
    table = doc.add_table(rows=0, cols=ncols)
    try:
        table.style = "Light Grid Accent 1"
    except KeyError:
        table.style = "Table Grid"
    for ri, row in enumerate(rows):
        cells = table.add_row().cells
        for ci in range(ncols):
            text = row[ci] if ci < len(row) else ""
            para = cells[ci].paragraphs[0]
            _add_runs_with_bold(para, text)
            if ri == 0:  # header row: bold everything
                for run in para.runs:
                    run.bold = True


def convert(md_path: str, docx_path: str) -> str:
    with open(md_path, encoding="utf-8") as f:
        lines = f.read().splitlines()

    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # Tables: a run of pipe rows (skip the --- separator row).
        if _is_table_row(line):
            table_rows: List[List[str]] = []
            while i < n and _is_table_row(lines[i]):
                if not _is_separator_row(lines[i]):
                    table_rows.append(_split_row(lines[i]))
                i += 1
            _add_table(doc, table_rows)
            continue

        # Headings.
        if stripped.startswith("### "):
            doc.add_heading(stripped[4:], level=3)
        elif stripped.startswith("## "):
            doc.add_heading(stripped[3:], level=2)
        elif stripped.startswith("# "):
            doc.add_heading(stripped[2:], level=1)
        # Bullets (one nesting level supported via leading spaces).
        elif stripped.startswith("- "):
            indent = len(line) - len(line.lstrip(" "))
            style = "List Bullet 2" if indent >= 2 else "List Bullet"
            try:
                para = doc.add_paragraph(style=style)
            except KeyError:
                para = doc.add_paragraph(style="List Bullet")
            _add_runs_with_bold(para, stripped[2:])
        else:
            para = doc.add_paragraph()
            _add_runs_with_bold(para, stripped)
        i += 1

    doc.save(docx_path)
    return docx_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a Markdown file to a .docx document.")
    p.add_argument("input", help="Path to the Markdown file.")
    p.add_argument("output", nargs="?", default=None,
                   help="Output .docx path (default: same name with .docx).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output or os.path.splitext(args.input)[0] + ".docx"
    convert(args.input, out)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
