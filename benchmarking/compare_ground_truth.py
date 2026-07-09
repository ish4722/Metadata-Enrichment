"""Compare the enriched gold set against the original OSDU ground truth.

Two deliverables:

PART 1 - Ground-truth diff (deterministic)
    For every (doc, attribute) the enriched benchmark scored, classify the
    enriched value versus the original ground truth the pipeline extracts from
    ``matched_json/`` (via ``rag_evaluation.extract_ground_truth`` for the core
    attributes, and the raw OSDU key for the generated ``*`` attributes):

      * verified - original had a value and it MATCHES the enriched value.
      * improved - original had a value but it CHANGED in the enriched set.
      * enriched - original was null/empty; the enriched set supplies a value.
      * still_null - neither side has a value (should not happen for scored rows).

PART 2 - Accuracy on the ORIGINAL ground truth
    Re-score each model's stored answers (from the enriched embedding benchmark's
    ``comparison.json``) against the ORIGINAL ground truth, using the same
    grading pipeline the RAG model uses:

      * rows whose original GT is null      -> not scorable (excluded), just like
        the enriched run excludes null GT.
      * coordinate labels (lat+long)        -> deterministic coordinate comparator.
      * non-coordinate, original == enriched -> reuse the stored judge verdict
        (identical judge input, so identical verdict; no LLM call).
      * non-coordinate, original != enriched -> needs a live LLM judge against the
        original value; only done when ``--judge`` is passed, otherwise the
        deterministic token match is used as a fallback and flagged.

Run from the repo root:

    python -m benchmarking.compare_ground_truth
    python -m benchmarking.compare_ground_truth --judge   # live judge for 'improved' rows
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import rag_evaluation
from benchmarking import enriched_dataset

COMPARISON = os.path.join(
    "output_table", "benchmarking_enriched", "embeddings", "comparison.json"
)
OUT_DIR = os.path.dirname(COMPARISON)
MATCHED_DIR = os.path.join("matched_json")

Key = Tuple[str, str]  # (doc_id, label)


# ---------------------------------------------------------------------------
# Value resolution
# ---------------------------------------------------------------------------


def _norm(value: Any) -> str:
    return rag_evaluation._normalize_for_match(value) if value not in (None, "") else ""


def original_ground_truth(label: str, matched_json: Dict[str, Any],
                          enriched_key: Optional[str]) -> Dict[str, Any]:
    """Original GT value/tokens for a label from the full OSDU record.

    Core labels go through the registered extractor (handles nested OSDU shapes:
    FacilityEvents, GeoContexts, coordinate geometry, reference-id stripping).
    Generated ``*`` labels (and any the extractor can't resolve) fall back to the
    raw OSDU key the enriched file used, with reference-id cleaning applied.
    """
    res = rag_evaluation.extract_ground_truth(label, matched_json)
    if res.get("value") not in (None, ""):
        return res

    data = rag_evaluation._data_root(matched_json)
    if enriched_key and enriched_key in data:
        raw = data.get(enriched_key)
        if isinstance(raw, str):
            cleaned = rag_evaluation._strip_reference_id(raw) if ":" in raw else raw
            cleaned = (cleaned or "").strip()
            if cleaned:
                return {"value": cleaned, "tokens": [cleaned]}
        elif raw not in (None, "", [], {}):
            return {"value": str(raw), "tokens": [str(raw)]}
    return {"value": None, "tokens": []}


def classify(orig_val: Any, enr_val: Any, is_coord: bool,
             orig_tokens: List[str], enr_tokens: List[str]) -> str:
    orig_present = orig_val not in (None, "")
    enr_present = enr_val not in (None, "")
    if not enr_present and not orig_present:
        return "still_null"
    if not orig_present and enr_present:
        return "enriched"
    if orig_present and not enr_present:
        return "removed"
    # both present -> equal?
    if is_coord:
        # Coordinate tokens may be stored in opposite order (lat,lon vs lon,lat),
        # so compare order-insensitively.
        equal = bool(enr_tokens) and sorted(orig_tokens) == sorted(enr_tokens)
    else:
        equal = _norm(orig_val) == _norm(enr_val)
    return "verified" if equal else "improved"


# ---------------------------------------------------------------------------
# Load benchmark per-query answers
# ---------------------------------------------------------------------------


def load_models(path: str) -> "OrderedDict[str, Dict[Key, Dict[str, Any]]]":
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    models: "OrderedDict[str, Dict[Key, Dict[str, Any]]]" = OrderedDict()
    for res in data.get("results", []):
        if res.get("status") != "ok":
            continue
        indexed = {(r["doc_id"], r["label"]): r
                   for r in res.get("summary", {}).get("per_query", [])}
        models[res["tag"]] = indexed
    return models


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import rag_evaluation as _re
    _re.configure_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true",
                    help="Call the live LLM judge to re-grade 'improved' rows "
                         "against the original ground truth.")
    args = ap.parse_args()

    docs = enriched_dataset.discover_enriched()
    models = load_models(COMPARISON)
    tags = list(models)

    llm = None
    if args.judge:
        from benchmarking import common
        cfg = common.build_config()
        from rag.inference.llm import get_llm
        llm = get_llm(cfg)

    # ---- PART 1: build the GT diff, indexed by (doc, label) ---------------
    diff_index: Dict[Key, Dict[str, Any]] = {}
    per_doc: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()
    cat_counts: "OrderedDict[str, int]" = OrderedDict(
        (c, 0) for c in ("verified", "improved", "enriched", "removed", "still_null"))

    for doc in docs:
        doc_id = doc["id"]
        with open(doc["ground_truth"], "r", encoding="utf-8") as f:
            enriched_json = json.load(f)
        enr_data = rag_evaluation._data_root(enriched_json)
        matched_path = os.path.join(MATCHED_DIR, f"{doc_id}.json")
        if not os.path.exists(matched_path):
            print(f"[warn] no matched_json for {doc_id}; skipping")
            continue
        with open(matched_path, "r", encoding="utf-8") as f:
            matched_json = json.load(f)

        rows: List[Dict[str, Any]] = []
        for q in doc["queries"]:
            label = q["label"]
            enriched_key = enriched_dataset.field_key_for_label(label, enr_data)
            enr = enriched_dataset.extract_enriched_ground_truth(label, enriched_json)
            orig = original_ground_truth(label, matched_json, enriched_key)
            is_coord = rag_evaluation.label_uses_coordinate_judge(label)
            category = classify(orig["value"], enr["value"], is_coord,
                                orig["tokens"], enr["tokens"])
            cat_counts[category] = cat_counts.get(category, 0) + 1
            entry = {
                "doc_id": doc_id,
                "label": label,
                "field_key": enriched_key,
                "category": category,
                "original_value": orig["value"],
                "enriched_value": enr["value"],
            }
            rows.append(entry)
            diff_index[(doc_id, label)] = {
                "category": category,
                "orig_value": orig["value"],
                "orig_tokens": orig["tokens"],
                "enr_value": enr["value"],
                "is_coord": is_coord,
            }
        per_doc[doc_id] = rows

    diff_out = {
        "source_enriched": enriched_dataset.ENRICHED_DIR,
        "source_original": MATCHED_DIR,
        "legend": {
            "verified": "original had a value and it matches the enriched value",
            "improved": "original had a value but it changed in the enriched set",
            "enriched": "original was null/empty; enriched supplies a value",
            "removed": "original had a value but enriched is null (rare)",
            "still_null": "neither side has a value",
        },
        "totals": cat_counts,
        "total_attributes": sum(cat_counts.values()),
        "by_doc": per_doc,
    }
    diff_path = os.path.join(OUT_DIR, "ground_truth_diff.json")
    with open(diff_path, "w", encoding="utf-8") as f:
        json.dump(diff_out, f, ensure_ascii=False, indent=2)

    # ---- PART 2: re-score each model against the ORIGINAL ground truth ----
    approx_used = 0

    def original_verdict(tag: str, key: Key) -> Optional[bool]:
        """True/False if scorable against original GT, None if not scorable."""
        nonlocal approx_used
        d = diff_index.get(key)
        row = models[tag].get(key)
        if d is None or row is None:
            return None
        if d["orig_value"] in (None, ""):
            return None  # original GT null -> not scorable
        answer = row.get("answer") or ""
        if d["is_coord"]:
            is_match, _ = rag_evaluation.coordinate_verdict(str(d["orig_value"]), answer)
            return is_match
        # non-coordinate
        if d["category"] == "verified":
            return row.get("llm_hit") is True  # identical judge input
        # improved: original value differs from enriched
        if llm is not None:
            is_match, _ = rag_evaluation.judge_match_dispatch(
                llm, key[1], row.get("question", ""), str(d["orig_value"]), answer)
            return is_match
        approx_used += 1
        return rag_evaluation.llm_answer_is_correct(answer, d["orig_tokens"])

    def enriched_verdict(tag: str, key: Key) -> Optional[bool]:
        """Enriched-GT verdict (the stored llm_hit); None if enriched GT null."""
        d = diff_index.get(key)
        row = models[tag].get(key)
        if d is None or row is None:
            return None
        if d["enr_value"] in (None, ""):
            return None
        return row.get("llm_hit") is True

    model_stats: List[Dict[str, Any]] = []
    for tag in tags:
        keys = list(models[tag])
        # Row-level verdicts on both gold sets.
        orig_v = {k: original_verdict(tag, k) for k in keys}
        enr_v = {k: enriched_verdict(tag, k) for k in keys}

        # -- Comparison A: fair, restricted to rows scorable on ORIGINAL GT --
        fair_keys = [k for k in keys if orig_v[k] is not None]
        fair_orig_correct = sum(1 for k in fair_keys if orig_v[k])
        fair_enr_correct = sum(1 for k in fair_keys if enr_v[k])

        # -- Comparison B: all rows; null-original counts as not-answerable
        #    (wrong) for original, enriched scored on its own null set. --
        all_total = len(keys)
        allB_orig_correct = sum(1 for k in keys if orig_v[k] is True)
        allB_enr_correct = sum(1 for k in keys if enr_v[k] is True)

        model_stats.append({
            "model": tag,
            "total_questions": all_total,
            "comparison_A_fair_same_rows": {
                "rows": len(fair_keys),
                "enriched_correct": fair_enr_correct,
                "enriched_accuracy": (fair_enr_correct / len(fair_keys)) if fair_keys else None,
                "original_correct": fair_orig_correct,
                "original_accuracy": (fair_orig_correct / len(fair_keys)) if fair_keys else None,
            },
            "comparison_B_all_rows_null_counts": {
                "rows": all_total,
                "enriched_correct": allB_enr_correct,
                "enriched_accuracy": (allB_enr_correct / all_total) if all_total else None,
                "original_correct": allB_orig_correct,
                "original_accuracy": (allB_orig_correct / all_total) if all_total else None,
            },
        })

    acc_out = {
        "source": COMPARISON,
        "note": "Accuracy of each model's stored answers re-scored against the "
                "ORIGINAL OSDU ground truth. Rows whose original GT is null are "
                "excluded (not scorable), mirroring the enriched run.",
        "judge_mode": "live_llm" if llm is not None else "reuse+token_fallback",
        "approx_rows_token_fallback": approx_used,
        "models": model_stats,
    }
    acc_path = os.path.join(OUT_DIR, "accuracy_on_original.json")
    with open(acc_path, "w", encoding="utf-8") as f:
        json.dump(acc_out, f, ensure_ascii=False, indent=2)

    # ---- console report --------------------------------------------------
    print("\n" + "=" * 60)
    print("PART 1 - GROUND-TRUTH DIFF (enriched vs original)")
    print("=" * 60)
    total = diff_out["total_attributes"]
    for cat, n in cat_counts.items():
        pct = (n / total * 100) if total else 0
        print(f"  {cat:<11}: {n:>4}  ({pct:4.1f}%)")
    print(f"  {'TOTAL':<11}: {total:>4}")
    print(f"  saved -> {diff_path}")

    def _p(v: Optional[float]) -> str:
        return f"{v*100:.1f}%" if v is not None else "N/A"

    a = model_stats[0]["comparison_A_fair_same_rows"]["rows"]
    print("\n" + "=" * 66)
    print(f"PART 2A - FAIR COMPARISON (same {a} rows scorable on ORIGINAL GT)")
    print("=" * 66)
    hdr = f"  {'Model':<16} {'Enriched acc':>13} {'Original acc':>13}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for m in model_stats:
        ca = m["comparison_A_fair_same_rows"]
        print(f"  {m['model']:<16} "
              f"{ca['enriched_correct']:>3}/{ca['rows']:<3} {_p(ca['enriched_accuracy']):>6} "
              f"{ca['original_correct']:>3}/{ca['rows']:<3} {_p(ca['original_accuracy']):>6}")

    tot = model_stats[0]["comparison_B_all_rows_null_counts"]["rows"]
    print("\n" + "=" * 66)
    print(f"PART 2B - ALL {tot} ROWS (original null GT counts as not-answerable)")
    print("=" * 66)
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for m in model_stats:
        cb = m["comparison_B_all_rows_null_counts"]
        print(f"  {m['model']:<16} "
              f"{cb['enriched_correct']:>3}/{cb['rows']:<3} {_p(cb['enriched_accuracy']):>6} "
              f"{cb['original_correct']:>3}/{cb['rows']:<3} {_p(cb['original_accuracy']):>6}")
    if approx_used:
        print(f"\n  [note] {approx_used} 'improved' verdicts used the token-match "
              f"fallback (run with --judge for the live LLM judge).")
    print(f"  saved -> {acc_path}")


if __name__ == "__main__":
    main()
