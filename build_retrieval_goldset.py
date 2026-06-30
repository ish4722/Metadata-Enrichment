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
  * facility name / openv   -> LLM, answer-aware, graded. Chunks are BATCHED
    ID / spud date             (several per call) to cut latency/cost.

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

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches  # noqa: E402

apply_ssl_patches()

import rag_evaluation  # noqa: E402
from rag.config import RagConfig  # noqa: E402
from rag.indexing.chroma_store import get_chroma_client  # noqa: E402
from rag.inference.llm import get_llm  # noqa: E402
import evaluate_all  # noqa: E402

# LLM-judged text fields: (query label, short key, human description).
LLM_FIELDS = [
    ("facility name", "facility_name", "the well/facility name"),
    ("openv ID", "operating_environment", "the operating environment (e.g. Onshore/Offshore)"),
    ("facilityevent spud date", "spud_date", "the spud date / kick-off date of the well"),
]
KEY_TO_LABEL = {key: lbl for lbl, key, _ in LLM_FIELDS}
COORD_LABEL = "lat+long"
ALL_LABELS = [l for l, _, _ in LLM_FIELDS] + [COORD_LABEL]
CHAR_CAP = 2500  # per-excerpt cap inside a batch prompt
VERDICT_GRADE = {"correct": 2, "wrong": 1}


def make_batch_prompt(excerpts, fields):
    """excerpts: list[str]; fields: [(key, desc, value)] (non-null GT only)."""
    lines = [
        "You grade whether document excerpts answer specific fields for ONE oil & gas well.",
        "",
        "Fields and their TARGET (ground-truth) values:",
    ]
    for key, desc, value in fields:
        lines.append(f'- {key}: {desc}. Target value = "{value}"')
    lines += [
        "",
        "For EACH excerpt and EACH field decide one of:",
        '  "correct" - answering this field using ONLY this excerpt yields the target value above.',
        '  "wrong"   - this excerpt DOES state a value for this field, but a DIFFERENT one',
        "              (a different well/event/date/place). It looks answerable but is not the target.",
        "  omit      - the excerpt does not state this field at all (most excerpts).",
        "Judge each field independently. Do NOT count mere mentions of a related word.",
        "For dates, treat formats equal within 1 day as the same value.",
        "",
        "Excerpts:",
    ]
    for i, text in enumerate(excerpts):
        snippet = (text or "").strip().replace("\n", " ")[:CHAR_CAP]
        lines.append(f"[{i}] {snippet}")
    lines += [
        "",
        "Respond with ONLY a JSON object mapping each excerpt index (as a string) to an object of",
        '{field_key: "correct"|"wrong"}, including ONLY fields that are correct or wrong and ONLY',
        'excerpts that have at least one. Example: {"0": {"facility_name": "correct"},',
        '"2": {"spud_date": "wrong", "facility_name": "correct"}}. If none match, respond {}.',
    ]
    return "\n".join(lines)


def parse_batch_response(text, n):
    """Return {index:int -> {label: "correct"|"wrong"}} parsed from a JSON-ish reply."""
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
        verdicts = {}
        for key, verdict in v.items():
            lbl = KEY_TO_LABEL.get(key)
            vd = str(verdict).strip().lower()
            if lbl and vd in VERDICT_GRADE:
                verdicts[lbl] = vd
        if verdicts:
            out[idx] = verdicts
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

    async def judge_batch(slice_chunks, active_fields):
        """Return {local_index -> {label: verdict}} for a batch of chunks."""
        nonlocal calls
        prompt = make_batch_prompt([c["text"] for c in slice_chunks], active_fields)
        async with sem:
            resp = await llm.ainvoke(prompt)
        text = resp.content if hasattr(resp, "content") else str(resp)
        u = rag_evaluation.extract_token_usage(resp, prompt, text)
        async with stat_lock:
            usage["input_tokens"] += u["input_tokens"]
            usage["output_tokens"] += u["output_tokens"]
            calls += 1
        return parse_batch_response(text, len(slice_chunks))

    for i, doc_id in enumerate(to_run, 1):
        with open(gt_path[doc_id], "r", encoding="utf-8") as f:
            matched_json = json.load(f)
        gts = {lbl: rag_evaluation.extract_ground_truth(lbl, matched_json) for lbl in ALL_LABELS}
        active_fields = [(k, d, gts[lbl]["value"]) for lbl, k, d in LLM_FIELDS
                         if gts[lbl]["value"] is not None]
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
        # text fields: batched LLM (graded)
        if active_fields:
            batches = [chunks[j:j + args.batch_size] for j in range(0, len(chunks), args.batch_size)]
            results = await asyncio.gather(*(judge_batch(b, active_fields) for b in batches))
            for b_idx, res in enumerate(results):
                base = b_idx * args.batch_size
                for local_idx, verdicts in res.items():
                    c = chunks[base + local_idx]
                    for lbl, verdict in verdicts.items():
                        rel[lbl].append({"chunk_id": c["id"], "page": c["page"],
                                         "grade": VERDICT_GRADE[verdict], "verdict": verdict,
                                         "matched_by": "llm"})

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
