"""Extract disagreement sets from the enriched embedding benchmark.

Reads the per-query results stored in
``output_table/benchmarking_enriched/embeddings/comparison.json`` (produced by
``benchmark_enriched_embeddings.py``) and emits two files next to it:

  * ``wrong_common.json``  - queries EVERY model got wrong (intersection of
    failures across all three configs).
  * ``wrong_unique.json``  - queries wrong for exactly ONE model (unique failure),
    grouped per model.

Correctness uses ``llm_hit`` - the headline accuracy flag set by the very same
pipeline the RAG model uses (``BatchRunner`` -> ``judge_match_dispatch``), so
``lat+long`` and other coordinate labels are graded by the deterministic
coordinate comparator ("calculator" path) rather than the LLM judge.

Run from the repo root:

    python -m benchmarking.analyze_wrong_queries
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from typing import Any, Dict, List, Tuple

COMPARISON = os.path.join(
    "output_table", "benchmarking_enriched", "embeddings", "comparison.json"
)
OUT_DIR = os.path.dirname(COMPARISON)

Key = Tuple[str, str]  # (doc_id, label)


def _is_correct(row: Dict[str, Any]) -> bool:
    """A query is "right" when the judge/coordinate comparator marked it a hit."""
    return row.get("llm_hit") is True


def load_models(path: str) -> "OrderedDict[str, Dict[Key, Dict[str, Any]]]":
    """Return {model_tag: {(doc_id, label): per_query_row}} for every ok config."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    models: "OrderedDict[str, Dict[Key, Dict[str, Any]]]" = OrderedDict()
    for res in data.get("results", []):
        if res.get("status") != "ok":
            continue
        rows = res.get("summary", {}).get("per_query", [])
        indexed: Dict[Key, Dict[str, Any]] = {}
        for r in rows:
            indexed[(r["doc_id"], r["label"])] = r
        models[res["tag"]] = indexed
    return models


def _entry(row: Dict[str, Any]) -> Dict[str, Any]:
    """Query / ground truth / answer view of one per-query row."""
    return {
        "query": row.get("question"),
        "ground_truth": row.get("ground_truth"),
        "answer": row.get("answer"),
        "retrieval_hit": row.get("retrieval_hit"),
        "token_match": row.get("token_match"),
        "judge_match": row.get("judge_match"),
    }


def build_common(models) -> List[Dict[str, Any]]:
    """Queries wrong across ALL models (per-model answer included)."""
    tags = list(models)
    # Only keys present in every model are comparable.
    common_keys = set.intersection(*(set(m) for m in models.values()))
    out: List[Dict[str, Any]] = []
    for key in sorted(common_keys):
        doc_id, label = key
        if all(not _is_correct(models[t][key]) for t in tags):
            out.append({
                "doc_id": doc_id,
                "label": label,
                "query": models[tags[0]][key].get("question"),
                "ground_truth": models[tags[0]][key].get("ground_truth"),
                "answers": {t: _entry(models[t][key]) for t in tags},
            })
    return out


def build_unique(models) -> "OrderedDict[str, List[Dict[str, Any]]]":
    """Queries wrong for exactly ONE model, grouped by that model."""
    tags = list(models)
    common_keys = set.intersection(*(set(m) for m in models.values()))
    unique: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict((t, []) for t in tags)
    for key in sorted(common_keys):
        doc_id, label = key
        wrong_tags = [t for t in tags if not _is_correct(models[t][key])]
        if len(wrong_tags) != 1:
            continue
        t = wrong_tags[0]
        row = models[t][key]
        unique[t].append({
            "doc_id": doc_id,
            "label": label,
            "query": row.get("question"),
            "ground_truth": row.get("ground_truth"),
            "answer": row.get("answer"),
            "retrieval_hit": row.get("retrieval_hit"),
            "token_match": row.get("token_match"),
            "judge_match": row.get("judge_match"),
            # What the other (correct) models answered, for contrast.
            "correct_models": {
                o: models[o][key].get("answer") for o in tags if o != t
            },
        })
    return unique


def main() -> None:
    models = load_models(COMPARISON)
    tags = list(models)
    print(f"Loaded {len(tags)} model(s): {', '.join(tags)}")

    common = build_common(models)
    common_out = {
        "source": COMPARISON,
        "models": tags,
        "criterion": "llm_hit is False for every model (same judge/coordinate "
                     "comparator the RAG pipeline uses)",
        "count": len(common),
        "queries": common,
    }
    common_path = os.path.join(OUT_DIR, "wrong_common.json")
    with open(common_path, "w", encoding="utf-8") as f:
        json.dump(common_out, f, ensure_ascii=False, indent=2)
    print(f"  wrote {len(common)} common-wrong queries -> {common_path}")

    unique = build_unique(models)
    unique_out = {
        "source": COMPARISON,
        "models": tags,
        "criterion": "llm_hit is False for exactly one model",
        "counts": {t: len(v) for t, v in unique.items()},
        "by_model": unique,
    }
    unique_path = os.path.join(OUT_DIR, "wrong_unique.json")
    with open(unique_path, "w", encoding="utf-8") as f:
        json.dump(unique_out, f, ensure_ascii=False, indent=2)
    counts = ", ".join(f"{t}={len(v)}" for t, v in unique.items())
    print(f"  wrote unique-wrong queries ({counts}) -> {unique_path}")


if __name__ == "__main__":
    main()
