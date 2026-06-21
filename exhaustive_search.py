"""Naive *exhaustive search* baseline (no embeddings, no retrieval).

This is the A/B counterpart to the RAG pipeline. Instead of building an index
and retrieving only the most relevant chunks, it scans **every page** of the
datalab extraction JSON and asks the LLM all queries against each page. The
point of the comparison is to show how much more expensive (calls / tokens /
cost / time) the brute-force approach is for similar accuracy.

Flow
----
1. Load the extraction JSON and group blocks into one text blob per page
   (the "naive chunk" unit -- no section merge, no embeddings).
2. For every page, send ONE prompt containing the page text + all queries and
   ask the LLM to return a strict ``<label>: <answer or NOT FOUND>`` line per
   query.
3. Aggregate per query across pages: keep the first page that yields a real
   (non "NOT FOUND") answer.
4. Validate each query's answer against the matched ground-truth JSON using the
   exact same scoring used by the RAG evaluation (token match + optional LLM
   judge).
5. Print per-query detail + a per-step metrics table, and optionally save a
   metrics JSON (same shape as the RAG ``evaluate`` output).

Run from the repo root, e.g.:

  python exhaustive_search.py \
      --input "datalab-output/datalab-output-PR2610.pdf.json" \
      --ground-truth "matched_json/PR2610.json"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict
from typing import Any, Dict, List, Optional

# ── bootstrap: make 'import rag...' work when run as a script, and patch SSL ──
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches

apply_ssl_patches()  # must run before transformers / huggingface_hub import

import rag_evaluation  # noqa: E402
from rag.common.schema import Chunk  # noqa: E402
from rag.config import RagConfig  # noqa: E402
from rag.indexing.loader import load_extracted_json  # noqa: E402
from rag.inference.llm import get_llm  # noqa: E402

DEFAULT_INPUT = "datalab-output/datalab-output-PR2610.pdf.json"
DEFAULT_GROUND_TRUTH = "matched_json/PR2610.json"
DEFAULT_QUERIES = "queries.txt"

# Sentinel the LLM is asked to use when a page does not contain an answer.
NOT_FOUND = "NOT FOUND"


# ---------------------------------------------------------------------------
# Step 1 - page assembly (naive chunking: one blob of text per page)
# ---------------------------------------------------------------------------


def build_pages(input_path: str) -> List[Dict[str, Any]]:
    """Group the flattened blocks into one text blob per page, in page order."""
    chunks: List[Chunk] = load_extracted_json(input_path)
    by_page: Dict[int, List[str]] = {}
    source = ""
    for chunk in chunks:
        source = chunk.source or source
        if chunk.text.strip():
            by_page.setdefault(chunk.page, []).append(chunk.text)

    pages: List[Dict[str, Any]] = []
    for page in sorted(by_page.keys()):
        text = "\n\n".join(by_page[page]).strip()
        if text:
            pages.append({"source": source, "page": page, "text": text})
    return pages


# ---------------------------------------------------------------------------
# Step 2 - per-page extraction prompt + response parsing
# ---------------------------------------------------------------------------


def make_page_prompt(page_text: str, queries: List[Dict[str, str]]) -> str:
    """Build a single prompt that asks every query against one page of text."""
    question_lines = []
    for q in queries:
        question_lines.append(f"- {q['label']}: {q['question']}")
    questions_block = "\n".join(question_lines)

    return (
        "You are an information extraction assistant for oil and gas reports. "
        "You are given the text of a SINGLE page from a document and a list of "
        "questions. Answer each question using ONLY the page text below.\n\n"
        "Rules:\n"
        f"- If the page does not contain the answer to a question, write exactly '{NOT_FOUND}'.\n"
        "- Do not guess. Only answer from the page text.\n"
        "- Output one line per question in the exact format '<label>: <answer>'.\n"
        "- Use the labels exactly as given.\n\n"
        "Questions (label: question):\n"
        f"{questions_block}\n\n"
        "Page text:\n"
        f"{page_text}\n\n"
        "Answers (one '<label>: <answer>' per line):"
    )


def parse_page_response(response_text: str, queries: List[Dict[str, str]]) -> Dict[str, str]:
    """Map an LLM response back to ``{label: answer}`` for the known labels."""
    labels = {q["label"].strip().lower(): q["label"] for q in queries}
    result: Dict[str, str] = {}

    for raw_line in response_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        prefix, _, answer = line.partition(":")
        key = prefix.strip().lstrip("-*•").strip().lower()
        if key in labels and labels[key] not in result:
            result[labels[key]] = answer.strip()

    return result


def _is_real_answer(answer: Optional[str]) -> bool:
    """True when the answer is present and is not a 'not found' style reply."""
    if not answer:
        return False
    normalized = rag_evaluation._normalize_for_match(answer)
    if not normalized:
        return False
    miss_markers = ("not found", "unavailable", "not available", "no answer", "none", "n a")
    return not any(marker in normalized for marker in miss_markers)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_exhaustive(
    input_path: str = DEFAULT_INPUT,
    queries_path: str = DEFAULT_QUERIES,
    ground_truth_path: str = DEFAULT_GROUND_TRUTH,
    max_pages: Optional[int] = None,
    use_judge: bool = True,
    metrics_output: Optional[str] = None,
    config: Optional[RagConfig] = None,
) -> Dict[str, Any]:
    """Run the exhaustive baseline end-to-end and return a totals dict.

    The returned dict mirrors the fields the comparison driver needs:
    ``llm_calls``, ``total_tokens``, ``cost_usd``, ``time_s``, ``peak_mem_mb``,
    ``accuracy`` plus the per-step metrics and per-query rows.
    """
    config = config or RagConfig.from_env()

    print("=== Exhaustive search (naive, no retrieval) ===")
    llm = get_llm(config)
    if llm is None:
        print(
            "[error] LLM stack unavailable (langchain_openai + sauth_token). "
            "The exhaustive baseline extracts answers with an LLM and cannot run "
            "without it."
        )
        raise SystemExit(1)

    print(f"Loading queries from {queries_path}")
    queries = rag_evaluation.parse_queries_file(queries_path)
    print(f"  -> {len(queries)} queries parsed")

    print(f"Loading ground-truth JSON from {ground_truth_path}")
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        matched_json = json.load(f)

    metrics: List[rag_evaluation.StepMetrics] = []

    # --- Step 1: load + page assembly --------------------------------------
    with rag_evaluation.StepProfiler() as prof:
        pages = build_pages(input_path)
    if max_pages is not None:
        pages = pages[:max_pages]
    metrics.append(
        rag_evaluation.StepMetrics(
            "1. Load + page assembly", time_s=prof.elapsed, peak_mem_mb=prof.peak_mem_mb
        )
    )
    print(f"  {len(pages)} pages assembled "
          f"({'all' if max_pages is None else f'capped at {max_pages}'}).")

    # --- Step 2: exhaustive per-page LLM scan ------------------------------
    # best_answers[label] = {"answer": str, "page": int}
    best_answers: Dict[str, Dict[str, Any]] = {}
    scan_time = 0.0
    scan_peak = 0.0
    scan_tokens = 0
    scan_cost = 0.0
    page_calls = 0

    for idx, page in enumerate(pages, start=1):
        prompt = make_page_prompt(page["text"], queries)
        with rag_evaluation.StepProfiler() as prof:
            answer_text, usage = rag_evaluation.invoke_llm_with_usage(llm, prompt)
        scan_time += prof.elapsed
        scan_peak = max(scan_peak, prof.peak_mem_mb)
        scan_tokens += usage.get("total_tokens", 0)
        scan_cost += rag_evaluation.token_cost_usd(
            usage.get("input_tokens", 0), usage.get("output_tokens", 0)
        )
        page_calls += 1

        page_answers = parse_page_response(answer_text, queries)
        for q in queries:
            label = q["label"]
            if label in best_answers:
                continue  # first real answer wins; stop probing this query
            candidate = page_answers.get(label)
            if _is_real_answer(candidate):
                best_answers[label] = {"answer": candidate, "page": page["page"]}

        if idx % 25 == 0 or idx == len(pages):
            found = len(best_answers)
            print(f"  scanned {idx}/{len(pages)} pages "
                  f"({found}/{len(queries)} queries answered so far)")

    metrics.append(
        rag_evaluation.StepMetrics(
            f"2. Exhaustive page scan ({page_calls} LLM calls)",
            cost_usd=scan_cost,
            tokens=scan_tokens,
            time_s=scan_time,
            peak_mem_mb=scan_peak,
        )
    )

    # --- Step 3 + 4: aggregate + validate ----------------------------------
    judge_time = 0.0
    judge_peak = 0.0
    judge_tokens = 0
    judge_cost = 0.0
    judge_calls = 0

    correct = 0
    per_query_rows: List[Dict[str, Any]] = []

    for q in queries:
        label = q["label"]
        gt = rag_evaluation.extract_ground_truth(label, matched_json)
        found = best_answers.get(label)
        answer = found["answer"] if found else ""
        from_page = found["page"] if found else None

        token_hit = rag_evaluation.answer_is_correct(answer, gt["tokens"]) if answer else False
        judge_hit: Optional[bool] = None
        if use_judge and answer and gt["value"]:
            with rag_evaluation.StepProfiler() as prof:
                judge_hit, judge_usage = rag_evaluation.llm_judge_match(
                    llm, q["question"], str(gt["value"]), answer
                )
            judge_time += prof.elapsed
            judge_peak = max(judge_peak, prof.peak_mem_mb)
            judge_tokens += judge_usage.get("total_tokens", 0)
            judge_cost += rag_evaluation.token_cost_usd(
                judge_usage.get("input_tokens", 0), judge_usage.get("output_tokens", 0)
            )
            judge_calls += 1
            final_hit = judge_hit
        else:
            final_hit = token_hit

        if final_hit:
            correct += 1

        per_query_rows.append(
            {
                "label": label,
                "question": q["question"],
                "ground_truth": gt["value"],
                "answer": answer,
                "from_page": from_page,
                "token_match": token_hit,
                "judge_match": judge_hit,
                "hit": final_hit,
            }
        )

    num_queries = len(queries) or 1
    accuracy = correct / num_queries

    metrics.append(
        rag_evaluation.StepMetrics(
            "3. Aggregate answers", time_s=0.0, peak_mem_mb=0.0
        )
    )
    if judge_calls:
        metrics.append(
            rag_evaluation.StepMetrics(
                f"4. Validation / judge ({judge_calls} LLM calls)",
                cost_usd=judge_cost,
                tokens=judge_tokens,
                accuracy=accuracy,
                time_s=judge_time,
                peak_mem_mb=judge_peak,
            )
        )
    else:
        metrics.append(
            rag_evaluation.StepMetrics(
                "4. Validation (token match)", accuracy=accuracy
            )
        )

    total_calls = page_calls + judge_calls
    total = rag_evaluation.StepMetrics(
        "TOTAL",
        cost_usd=sum(m.cost_usd for m in metrics),
        tokens=sum(m.tokens for m in metrics),
        accuracy=accuracy,
        time_s=sum(m.time_s for m in metrics),
        peak_mem_mb=max((m.peak_mem_mb for m in metrics), default=0.0),
    )

    # --- Per-query detail --------------------------------------------------
    print("\nPer-query results:")
    for row in per_query_rows:
        status = "MATCH" if row["hit"] else "miss"
        page_note = f" (page {row['from_page']})" if row["from_page"] is not None else ""
        print(f"\n  [{row['label']}] {row['question']}")
        print(f"    ground truth : {row['ground_truth']}")
        print(f"    result       : {status}{page_note}")
        if row["judge_match"] is not None:
            token_status = "match" if row["token_match"] else "no-match"
            print(f"    judge        : {'MATCH' if row['judge_match'] else 'miss'} "
                  f"(literal token check: {token_status})")
        if row["answer"]:
            print(f"    answer       : {row['answer'][:200].strip()}")

    # --- Metrics table -----------------------------------------------------
    all_rows = metrics + [total]
    print("\nExhaustive search metrics by step:")
    print(rag_evaluation.render_metrics_table(all_rows))
    print(f"\nTotal LLM calls: {total_calls} "
          f"({page_calls} page scans + {judge_calls} judge calls)")

    if metrics_output:
        payload = {
            "pipeline": "exhaustive (naive, no retrieval)",
            "input": input_path,
            "queries": queries_path,
            "ground_truth": ground_truth_path,
            "num_pages": len(pages),
            "llm_calls": total_calls,
            "steps": [asdict(m) for m in all_rows],
            "per_query": per_query_rows,
        }
        with open(metrics_output, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"\nSaved metrics to {metrics_output}")

    return {
        "pipeline": "exhaustive (naive)",
        "llm_calls": total_calls,
        "total_tokens": total.tokens,
        "cost_usd": total.cost_usd,
        "time_s": total.time_s,
        "peak_mem_mb": total.peak_mem_mb,
        "accuracy": accuracy,
        "num_pages": len(pages),
        "steps": all_rows,
        "per_query": per_query_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Exhaustive (naive, no-retrieval) search baseline over a datalab JSON."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Path to the extraction JSON file")
    parser.add_argument("--queries", default=DEFAULT_QUERIES, help="Path to queries file")
    parser.add_argument("--ground-truth", default=DEFAULT_GROUND_TRUTH, help="Path to matched JSON")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Cap the number of pages scanned (default: all pages)")
    parser.add_argument("--no-judge", action="store_true", help="Disable the LLM judge")
    parser.add_argument("--metrics-output", default=None, help="Optional path to save metrics JSON")
    return parser.parse_args()


def main() -> None:
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    for path, role in (
        (args.input, "Input extraction JSON"),
        (args.queries, "Queries file"),
        (args.ground_truth, "Ground-truth (matched) JSON"),
    ):
        if not os.path.isfile(path):
            print(f"[error] {role} not found: {path}")
            print(f"        resolved to : {os.path.abspath(path)}")
            sys.exit(1)

    run_exhaustive(
        input_path=args.input,
        queries_path=args.queries,
        ground_truth_path=args.ground_truth,
        max_pages=args.max_pages,
        use_judge=not args.no_judge,
        metrics_output=args.metrics_output,
    )


if __name__ == "__main__":
    main()
