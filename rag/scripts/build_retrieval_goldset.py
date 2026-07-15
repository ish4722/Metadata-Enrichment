"""Build a GRADED retrieval-evaluation gold set: query -> relevant section chunks.

Graded relevance (per chunk, per field):
  * grade 2 = "correct": answering the field from this chunk alone yields the
    ground-truth value.
  * grade 1 = "wrong":  the chunk DOES provide a value for the field, but a
    different one (a confusable / distractor that looks answerable but isn't).
  * (omitted) = "drop": the chunk does not provide the field at all.

Per-field routing:
  * lat+long                -> deterministic coordinate comparator (LLM is poor
                               at DMS/NZMG/NZTM -> WGS84 conversion). Coords that
                               match GT -> correct(2); coords present but a
                               different location -> wrong(1); no coords -> drop.
  * all other query labels  -> two-phase LLM, answer-BLIND. Labels are read from
    (see queries.txt)          queries.txt / rag_evaluation.LLM_FIELDS:
                               1) EXTRACT: ask each chunk what value it states for
                               the field (the ground truth is NOT shown), batching
                               several chunks per call. Chunks that state nothing
                               for a field -> drop.
                               2) VALIDATE: judge each extracted value against the
                               ground truth using the SAME CORRECT/INCORRECT
                               tolerances as the RAG pipeline's judge (dates +/-1
                               day, coordinate conversion, formatting/synonyms), but
                               BATCHED (several values per call) so cost/time stays
                               roughly flat in the number of fields -> match =
                               correct(2), different value = wrong(1).

Robustness:
  * Chunks read from the embedded Chroma ``__section`` collection.
  * Per-doc CHECKPOINTING: after each doc the partial gold set is written to the
    output file, and a re-run RESUMES (skips docs already present). Combined with
    the async retry/backoff in rag.inference.llm, an interruption never loses work.

Run from the repo root (40 fewest-chunk docs):
    python build_retrieval_goldset.py --num-docs 40 --restart --concurrency 12 --batch-size 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches  # noqa: E402

apply_ssl_patches()

from rag.evaluation import core as rag_evaluation  # noqa: E402
from rag.config import RagConfig  # noqa: E402
from rag.indexing.chroma_store import get_chroma_client  # noqa: E402
from rag.inference.llm import get_llm  # noqa: E402
from rag.evaluation import batch as evaluate_all  # noqa: E402

# LLM-judged text fields: (query label, short key, human description).
# The label MUST match both a queries.txt block label (for the judge question)
# and a GROUND_TRUTH_EXTRACTORS key in rag_evaluation (for the target value).
# lat+long is handled separately by the deterministic coordinate comparator.
LLM_FIELDS = [
    ("facility name", "facility_name", "the well/facility name"),
    ("openv ID", "operating_environment", "the operating environment (e.g. Onshore/Offshore)"),
    ("facilityevent spud date", "spud_date", "the spud date / kick-off date of the well"),
    ("country name", "country_name", "the country where the well is located"),
    ("basin name", "basin_name", "the basin the well is found in"),
    ("field name", "field_name", "the field the well is found in"),
    ("default vertical measurement ID", "vertical_datum",
     "the vertical measurement datum / reference elevation (e.g. KB, RT, GL, MSL)"),
    ("current operator ID", "current_operator", "the current operator of the well"),
    ("marker names", "marker_names",
     "the formation markers / marker names (lithology or stratigraphy levels)"),
    ("vertical measurement total depth", "total_depth_md",
     "the total depth as measured depth (MD) of the well"),
    ("completion date", "completion_date",
     "the completion date (or plugged & abandoned / rig-released date) of the well"),
]
KEY_TO_LABEL = {key: lbl for lbl, key, _ in LLM_FIELDS}
LABEL_TO_FIELD = {lbl: (key, desc) for lbl, key, desc in LLM_FIELDS}
COORD_LABEL = "lat+long"
ALL_LABELS = [l for l, _, _ in LLM_FIELDS] + [COORD_LABEL]
CHAR_CAP = 2500  # per-excerpt cap inside a batch prompt
VERDICT_GRADE = {"correct": 2, "wrong": 1}


def make_extract_prompt(excerpts, fields):
    """PHASE 1 (answer-BLIND). excerpts: list[str]; fields: [(key, desc)] (no GT).

    Ask what value each excerpt states for each field, WITHOUT revealing the
    ground truth, so the model cannot be biased toward the target.
    """
    lines = [
        "You extract specific field values from document excerpts about ONE oil & gas well.",
        "",
        "Fields to extract:",
    ]
    for key, desc in fields:
        lines.append(f"- {key}: {desc}")
    lines += [
        "",
        "For EACH excerpt, report the value it states for EACH field, using ONLY the text of",
        "that excerpt. Copy the value as written; do NOT infer, normalize, translate, or guess.",
        "If an excerpt does not state a field, omit that field for that excerpt.",
        "",
        "Excerpts:",
    ]
    for i, text in enumerate(excerpts):
        snippet = (text or "").strip().replace("\n", " ")[:CHAR_CAP]
        lines.append(f"[{i}] {snippet}")
    lines += [
        "",
        "Respond with ONLY a JSON object mapping each excerpt index (as a string) to an object of",
        '{field_key: "extracted value"}, including ONLY fields the excerpt actually states and',
        'ONLY excerpts that state at least one. Example: {"0": {"facility_name": "Maui-A"},',
        '"2": {"spud_date": "12 Jan 1998", "facility_name": "Pohokura-1"}}. If none, respond {}.',
    ]
    return "\n".join(lines)


def parse_extract_response(text, n):
    """Return {index:int -> {label: extracted_value_str}} from a JSON-ish reply."""
    out = {}
    if not text:
        return out
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    for k, v in obj.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < n) or not isinstance(v, dict):
            continue
        vals = {}
        for key, val in v.items():
            lbl = KEY_TO_LABEL.get(key)
            if lbl and val is not None and str(val).strip():
                vals[lbl] = str(val).strip()
        if vals:
            out[idx] = vals
    return out


def make_batch_judge_prompt(items):
    """PHASE 2 (batched). A batched form of rag_evaluation.make_judge_prompt: judge
    many (question, expected value, system answer) triples in ONE call, using the
    SAME CORRECT/INCORRECT tolerances the RAG judge applies.
    items: [(item_id, question, ground_truth, answer)].
    """
    lines = [
        "You are a strict evaluator for a retrieval system. For EACH item below, decide whether",
        "the SYSTEM ANSWER correctly conveys the EXPECTED VALUE for that item's question.",
        "",
        "Treat an answer as CORRECT if it states the same value as the expected value, allowing for",
        "differences in formatting, units, abbreviations, synonyms, or extra surrounding text.",
        "Apply these tolerances explicitly:",
        "- DATES: accept any answer whose date is within +/- 1 calendar day of the expected date",
        "  (this absorbs off-by-one and timezone/format differences), and ignore differences in date",
        "  format (e.g. 6/03/2006 vs 2006-03-06). The spud date is the same as the start/kick-off date.",
        "- COORDINATES: first CONVERT both the expected value and the system answer to WGS84 decimal",
        "  degrees (latitude, longitude) regardless of input format (DMS, degrees-decimal-minutes,",
        "  UTM/easting-northing, or decimal degrees), then treat them as a match if they refer to the",
        "  same location within normal rounding tolerance. Sign/hemisphere (N/S, E/W) must agree.",
        "Treat it as INCORRECT if it gives a different value, contradicts the expected value, refers to",
        "a different entity, or says the information is unavailable. Judge each item independently.",
        "",
        "Items:",
    ]
    for item_id, question, ground_truth, answer in items:
        lines.append(f"[{item_id}] Question: {question}")
        lines.append(f"      Expected value: {ground_truth}")
        lines.append(f"      System answer: {answer}")
    lines += [
        "",
        'Respond with ONLY a JSON object mapping each item id (as a string) to "YES" (correct) or',
        '"NO" (incorrect). Example: {"0": "YES", "1": "NO"}.',
    ]
    return "\n".join(lines)


def parse_batch_judge_response(text):
    """Return {item_id:str -> bool is_match} from a JSON-ish YES/NO reply."""
    out = {}
    if not text:
        return out
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return out
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return out
    for k, v in obj.items():
        out[str(k)] = str(v).strip().lower().startswith("y")
    return out


def fetch_chunk_counts(col):
    got = col.get(include=[])
    counts = defaultdict(int)
    for cid in got.get("ids") or []:
        counts[cid.split("::", 1)[0]] += 1
    return counts


def fetch_doc_chunks(col, doc_id):
    got = col.get(where={"doc_id": doc_id}, include=["documents", "metadatas"])
    ids = got.get("ids") or []
    docs = got.get("documents") or []
    metas = got.get("metadatas") or []
    return [{"id": cid, "text": t or "", "page": int((md or {}).get("page", -1))}
            for cid, t, md in zip(ids, docs, metas)]


def _status_for(value, relevant):
    """answerable if any correct(2); only_wrong if only wrong(1); else not_grounded."""
    if value is None:
        return "no_ground_truth"
    if any(r["grade"] == 2 for r in relevant):
        return "answerable"
    if relevant:
        return "only_wrong"
    return "not_grounded"


def build_output(goldset, scope, args, usage, calls, elapsed):
    per_label = {}
    grade_counts = {}
    for lbl in ALL_LABELS:
        c = defaultdict(int)
        g = {"correct": 0, "wrong": 0}
        for entry in goldset.values():
            if lbl in entry:
                c[entry[lbl]["status"]] += 1
                g["correct"] += sum(1 for r in entry[lbl]["relevant"] if r["grade"] == 2)
                g["wrong"] += sum(1 for r in entry[lbl]["relevant"] if r["grade"] == 1)
        per_label[lbl] = dict(c)
        grade_counts[lbl] = g
    not_grounded = [
        {"doc_id": doc_id, "label": lbl, "value": entry[lbl]["value"]}
        for doc_id, entry in goldset.items() for lbl in entry
        if entry[lbl]["status"] == "not_grounded"
    ]
    cost = rag_evaluation.token_cost_usd(usage["input_tokens"], usage["output_tokens"])
    return {
        "scope": {"num_docs": args.num_docs, "max_chunks": args.max_chunks,
                  "docs": scope, "completed_docs": sorted(goldset.keys())},
        "grading": {"scheme": "graded", "correct": 2, "wrong": 1, "drop": 0},
        "docs": goldset,
        "not_grounded": not_grounded,
        "summary": {"per_label": per_label, "grade_counts": grade_counts,
                    "llm_calls": calls, "tokens": usage,
                    "cost_usd": round(cost, 4), "elapsed_s": round(elapsed, 1)},
    }


def save_atomic(out, path):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description="Build GRADED retrieval gold set (LLM batched + comparator).")
    p.add_argument("--num-docs", type=int, default=None,
                   help="Select the N docs with the FEWEST chunks (recommended for cost control).")
    p.add_argument("--max-chunks", type=int, default=1000,
                   help="Used only when --num-docs is not given: keep docs with <max-chunks chunks.")
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--matched-dir", default=evaluate_all.DEFAULT_MATCHED_DIR)
    p.add_argument("--docs-dir", default=evaluate_all.DEFAULT_DOCS_DIR)
    p.add_argument("--queries", default="queries.txt",
                   help="Queries file; supplies the judge question per label (phase 2).")
    p.add_argument("--only", nargs="+", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=os.path.join("output_table", "retrieval_goldset.json"))
    p.add_argument("--restart", action="store_true", help="Ignore existing output and rebuild all")
    return p.parse_args()


async def main_async():
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()
    cfg = RagConfig.from_env()

    matched, _ = evaluate_all.discover_docs(args.docs_dir, args.matched_dir, only=args.only)
    gt_path = {d["id"]: d["ground_truth"] for d in matched}

    # Judge questions per label (phase 2), read from the same queries file the RAG
    # pipeline uses. Missing labels fall back to a generic question from the desc.
    questions = {q["label"]: q["question"]
                 for q in rag_evaluation.parse_queries_file(args.queries)}

    def question_for(lbl):
        return questions.get(lbl) or f"What is {LABEL_TO_FIELD[lbl][1]}?"

    client = get_chroma_client(cfg.chroma_dir)
    col = client.get_collection(name=cfg.chroma_collection)
    counts = fetch_chunk_counts(col)

    embedded = [d for d in gt_path if counts.get(d, 0) > 0]
    if args.num_docs is not None:
        # the N docs with the FEWEST chunks (ties broken by doc id)
        ranked = sorted(embedded, key=lambda d: (counts[d], d))
        scope = sorted(ranked[: args.num_docs])
    else:
        scope = sorted(d for d in embedded if counts[d] < args.max_chunks)
    if args.limit:
        scope = scope[: args.limit]

    # ---- resume: load existing output, skip completed docs ----
    goldset = {}
    usage = {"input_tokens": 0, "output_tokens": 0}
    calls = 0
    if os.path.exists(args.output) and not args.restart:
        try:
            prev = json.load(open(args.output, encoding="utf-8"))
            goldset = prev.get("docs", {}) or {}
            s = prev.get("summary", {})
            usage = s.get("tokens", usage) or usage
            calls = s.get("llm_calls", 0) or 0
            print(f"[resume] loaded {len(goldset)} completed docs from {args.output}")
        except Exception as e:  # noqa: BLE001
            print(f"[resume] could not read existing output ({e}); starting fresh")

    to_run = [d for d in scope if d not in goldset]
    total_chunks = sum(counts[d] for d in scope)
    print(f"Collection: {cfg.chroma_collection}")
    print(f"In-scope docs: {len(scope)} ({total_chunks} chunks) | remaining: {len(to_run)} | "
          f"batch_size={args.batch_size} concurrency={args.concurrency}")
    if not to_run:
        print("Nothing to do (all in-scope docs already completed).")
        save_atomic(build_output(goldset, scope, args, usage, calls, 0.0), args.output)
        return

    llm = get_llm(cfg)
    if llm is None:
        print("[error] LLM unavailable.")
        sys.exit(1)

    sem = asyncio.Semaphore(args.concurrency)
    stat_lock = asyncio.Lock()
    t0 = time.perf_counter()

    async def _invoke(prompt):
        """Run one LLM call, accounting for tokens/calls; return response text."""
        nonlocal calls
        async with sem:
            resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        u = rag_evaluation.extract_token_usage(resp, prompt, text)
        async with stat_lock:
            usage["input_tokens"] += u["input_tokens"]
            usage["output_tokens"] += u["output_tokens"]
            calls += 1
        return text

    async def extract_batch(slice_chunks, fields):
        """PHASE 1: {local_index -> {label: extracted_value}} for a batch of chunks."""
        text = await _invoke(make_extract_prompt([c["text"] for c in slice_chunks], fields))
        return parse_extract_response(text, len(slice_chunks))

    async def judge_batch(items):
        """PHASE 2 (batched): validate many extracted values in ONE judge call.

        Uses the same CORRECT/INCORRECT tolerances as rag_evaluation's RAG judge,
        so results match how the pipeline scores answers, but batched to keep the
        call count (and cost) roughly flat in the number of fields.
        items: [(item_id, question, ground_truth, extracted)].
        """
        text = await _invoke(make_batch_judge_prompt(items))
        return parse_batch_judge_response(text)

    for i, doc_id in enumerate(to_run, 1):
        with open(gt_path[doc_id], "r", encoding="utf-8") as f:
            matched_json = json.load(f)
        gts = {lbl: rag_evaluation.extract_ground_truth(lbl, matched_json) for lbl in ALL_LABELS}
        # answer-BLIND extraction fields (key, desc); GT withheld from the prompt
        extract_fields = [(k, d) for lbl, k, d in LLM_FIELDS if gts[lbl]["value"] is not None]
        coord_value = gts[COORD_LABEL]["value"]
        chunks = fetch_doc_chunks(col, doc_id)

        rel = defaultdict(list)
        # lat+long deterministic comparator (graded)
        if coord_value is not None:
            for c in chunks:
                got = (rag_evaluation.parse_coordinates(c["text"])
                       or rag_evaluation.parse_grid_coordinates(c["text"]))
                if not got:
                    continue  # no coords in this chunk -> drop
                is_match = rag_evaluation.coordinate_verdict(str(coord_value), c["text"])[0]
                grade = 2 if is_match else 1
                rel[COORD_LABEL].append({"chunk_id": c["id"], "page": c["page"],
                                         "grade": grade, "verdict": "correct" if is_match else "wrong",
                                         "matched_by": "comparator"})
        # text fields: PHASE 1 blind extraction, then PHASE 2 correctness check
        if extract_fields:
            batches = [chunks[j:j + args.batch_size] for j in range(0, len(chunks), args.batch_size)]
            results = await asyncio.gather(*(extract_batch(b, extract_fields) for b in batches))
            # collect every chunk that stated a value for a field
            candidates = []  # [(chunk, label, extracted_value)]
            for b_idx, res in enumerate(results):
                base = b_idx * args.batch_size
                for local_idx, vals in res.items():
                    c = chunks[base + local_idx]
                    for lbl, value in vals.items():
                        candidates.append((c, lbl, value))
            # PHASE 2: validate extracted values with the RAG judge, BATCHED
            # (several candidates per call) and run concurrently across batches.
            if candidates:
                groups = [candidates[j:j + args.batch_size]
                          for j in range(0, len(candidates), args.batch_size)]
                items_per_group = [
                    [(str(k), question_for(lbl), gts[lbl]["value"], value)
                     for k, (c, lbl, value) in enumerate(group)]
                    for group in groups
                ]
                verdict_maps = await asyncio.gather(*(judge_batch(it) for it in items_per_group))
                for group, vmap in zip(groups, verdict_maps):
                    for k, (c, lbl, value) in enumerate(group):
                        # a value was extracted, so default to wrong(1) if inconclusive
                        is_match = vmap.get(str(k), False)
                        verdict = "correct" if is_match else "wrong"
                        rel[lbl].append({"chunk_id": c["id"], "page": c["page"],
                                         "grade": VERDICT_GRADE[verdict], "verdict": verdict,
                                         "matched_by": "llm-judge", "extracted": value})

        entry = {}
        for lbl in ALL_LABELS:
            value = gts[lbl]["value"]
            relevant = sorted(rel.get(lbl, []), key=lambda r: -r["grade"])
            entry[lbl] = {
                "value": value,
                "status": _status_for(value, relevant),
                "relevant": relevant,
                "correct_chunk_ids": [r["chunk_id"] for r in relevant if r["grade"] == 2],
                "relevant_pages": sorted({r["page"] for r in relevant if r["grade"] == 2}),
            }
        goldset[doc_id] = entry

        # CHECKPOINT after each doc
        save_atomic(build_output(goldset, scope, args, usage, calls,
                                 time.perf_counter() - t0), args.output)
        print(f"  [{i}/{len(to_run)}] {doc_id}: {len(chunks)} chunks -> "
              + ", ".join(
                  f"{lbl.split()[0]}={entry[lbl]['status'][:4]}"
                  f"(c{sum(1 for r in entry[lbl]['relevant'] if r['grade'] == 2)}/"
                  f"w{sum(1 for r in entry[lbl]['relevant'] if r['grade'] == 1)})"
                  for lbl in entry)
              + "  [checkpointed]")

    elapsed = time.perf_counter() - t0
    out = build_output(goldset, scope, args, usage, calls, elapsed)
    save_atomic(out, args.output)
    print("\n" + "=" * 60)
    for lbl in ALL_LABELS:
        c = out["summary"]["per_label"][lbl]
        g = out["summary"]["grade_counts"][lbl]
        print(f"  {lbl:26s} {c}  grades: correct={g['correct']} wrong={g['wrong']}")
    print(f"\nLLM calls: {calls} | tokens {usage['input_tokens']}/{usage['output_tokens']} | "
          f"cost ${out['summary']['cost_usd']} | {elapsed:.1f}s")
    print(f"not_grounded pairs: {len(out['not_grounded'])}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    asyncio.run(main_async())
