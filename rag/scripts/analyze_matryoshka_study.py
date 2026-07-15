"""Analyze the Matryoshka embedding-dimension study.

Reads every ``output_table/matryoshka/summary_*.json`` produced by
``run_matryoshka_study.py`` and writes:

  * ``aggregate.json``      - machine-readable comparison across all runs.
  * ``analysis_report.md``  - detailed human-readable analysis (quality vs
                              dimension, cross-model crossover, diminishing
                              returns, per-label, cost/storage, MTEB curves,
                              pre/post rerank gap, distractors, escalation,
                              recommendation).

Run standalone (after the study has produced summaries):

    python analyze_matryoshka_study.py
    python analyze_matryoshka_study.py --output-dir output_table/matryoshka
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional

DEFAULT_OUTPUT_DIR = os.path.join("output_table", "matryoshka")
LABELS = ["facility name", "openv ID", "facilityevent spud date", "lat+long"]

# OpenAI list price per 1M input tokens (standard tier, verified Jul 2026).
# Embeddings are input-only. The study's stored embedding cost used a single
# flat rate for every model; we recompute per-model from the token counts here.
EMBEDDING_PRICE_PER_1M = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}
# gpt-5.2 chat pricing (drives generation + judge) - matches the pipeline config.
LLM_INPUT_PRICE_PER_1M = 1.75
LLM_OUTPUT_PRICE_PER_1M = 14.0

# Vectors stored per collection for this corpus (Chroma count; same for every
# model/dimension because chunking is fixed). Used for the memory comparison.
CORPUS_VECTORS = 33100
BYTES_PER_FLOAT32 = 4


# ---------------------------------------------------------------------------
# Loading + extraction
# ---------------------------------------------------------------------------


def _load_runs(output_dir: str) -> List[Dict[str, Any]]:
    runs = []
    for path in sorted(glob.glob(os.path.join(output_dir, "summary_*.json"))):
        try:
            data = json.load(open(path, encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] could not read {path}: {exc}")
            continue
        runs.append(_extract(data, path))
    return runs


def _per_label_e2e(per_query: List[Dict[str, Any]]) -> Dict[str, Dict[str, Optional[float]]]:
    """Per-label end-to-end accuracy (llm / token / retrieval) from per_query rows."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in per_query:
        buckets[row.get("label", "?")].append(row)

    def _acc(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
        if not rows:
            return None
        return sum(1 for r in rows if r.get(key) is True) / len(rows)

    out: Dict[str, Dict[str, Optional[float]]] = {}
    for label, rows in buckets.items():
        out[label] = {
            "n": len(rows),
            "llm": _acc(rows, "llm_hit"),
            "token": _acc(rows, "token_match"),
            "retrieval": _acc(rows, "retrieval_hit"),
        }
    return out


def _extract(data: Dict[str, Any], path: str) -> Dict[str, Any]:
    rc = data.get("run_config", {})
    s = data.get("summary", {})
    mteb = data.get("mteb_metrics", {})
    model = rc.get("openai_embedding_model", "?")
    dim = rc.get("openai_embedding_dimensions")

    tokens = s.get("tokens", {})
    stored_cost = s.get("cost_usd", {})
    embed_tokens = tokens.get("embed_input", 0) or 0
    # Recompute embedding cost with the correct per-model list price (the stored
    # value charged every model the 3-large rate). Generation/judge already use
    # gpt-5.2 pricing, so keep them as-is; total is rebuilt from the corrected embed.
    price_1m = EMBEDDING_PRICE_PER_1M.get(model)
    corrected_embed = (embed_tokens / 1_000_000.0 * price_1m) if price_1m is not None else stored_cost.get("embedding")
    gen_cost = stored_cost.get("generation", 0.0) or 0.0
    judge_cost = stored_cost.get("judge", 0.0) or 0.0
    corrected_total = (corrected_embed or 0.0) + gen_cost + judge_cost

    return {
        "path": path,
        "tag": data.get("tag", os.path.basename(path)),
        "model": model,
        "dim": dim,
        "total_questions": s.get("total_questions"),
        "llm_acc": s.get("combined_llm_accuracy"),
        "token_acc": s.get("combined_token_accuracy"),
        "retrieval_acc": s.get("combined_retrieval_accuracy"),
        "escalation": s.get("escalation", {}),
        "llm_meta": s.get("llm_meta", {}) or {},
        "cost": stored_cost,
        "embed_tokens": embed_tokens,
        "embed_price_1m": price_1m,
        "embed_cost": corrected_embed,
        "gen_cost": gen_cost,
        "judge_cost": judge_cost,
        "total_cost": corrected_total,
        "time_s": s.get("time_s", {}),
        "per_label_e2e": _per_label_e2e(s.get("per_query", [])),
        "mteb_pre": (mteb.get("pre_rerank", {}) or {}).get("macro", {}),
        "mteb_post": (mteb.get("post_rerank", {}) or {}).get("macro", {}),
        "mteb_pre_per_label": (mteb.get("pre_rerank", {}) or {}).get("per_label", {}),
    }


def _by_model(runs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in runs:
        grouped[r["model"]].append(r)
    for model in grouped:
        grouped[model].sort(key=lambda r: (r["dim"] is None, r["dim"]))
    return grouped


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _pct(v: Optional[float]) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) else "N/A"


def _f3(v: Optional[float]) -> str:
    return f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"


def _retention(v: Optional[float], base: Optional[float]) -> str:
    if not isinstance(v, (int, float)) or not isinstance(base, (int, float)) or base == 0:
        return "N/A"
    return f"{v / base * 100:.1f}%"


def _md_table(headers: List[str], rows: List[List[str]]) -> str:
    line = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join("---" for _ in headers) + " |"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([line, sep, body])


# ---------------------------------------------------------------------------
# Report sections
# ---------------------------------------------------------------------------


def _section_quality_curves(by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    out = ["## Quality vs dimension (end-to-end)\n",
           "Accuracy at each dimension, with retention relative to the model's full size.\n"]
    for model, runs in by_model.items():
        full = runs[-1]  # largest dimension
        out.append(f"### {model}\n")
        rows = []
        for r in runs:
            rows.append([
                str(r["dim"]),
                _pct(r["llm_acc"]), _retention(r["llm_acc"], full["llm_acc"]),
                _pct(r["token_acc"]),
                _pct(r["retrieval_acc"]),
            ])
        out.append(_md_table(
            ["Dim", "LLM acc", "LLM retention", "Token acc", "Retrieval acc"], rows))
        out.append("")
    return "\n".join(out)


def _section_llm_meta(by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    def _n(v: Any) -> str:
        return f"{int(v):,}" if isinstance(v, (int, float)) else "N/A"

    out = ["## LLM call metadata (OpenAI callback)\n",
           "Captured live from every generation/judge call: successful requests, "
           "prompt-cache hits and reasoning tokens straight from the OpenAI usage "
           "payload.\n"]
    for model, runs in by_model.items():
        out.append(f"### {model}\n")
        rows = []
        for r in runs:
            m = r.get("llm_meta", {}) or {}
            rows.append([
                str(r["dim"]),
                _n(m.get("successful_requests")),
                _n(m.get("prompt_tokens")),
                _n(m.get("cached_prompt_tokens")),
                _pct(m.get("cache_hit_rate")),
                _n(m.get("reasoning_tokens")),
                _n(m.get("completion_tokens")),
            ])
        out.append(_md_table(
            ["Dim", "Requests", "Prompt tok", "Cached tok", "Cache hit",
             "Reason tok", "Compl tok"], rows))
        out.append("")
    return "\n".join(out)


def _section_crossover(by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    out = ["## Cross-model crossover\n",
           "Does the larger model at a smaller dimension beat the smaller model at full size?\n"]
    rows = []
    for model, runs in by_model.items():
        for r in runs:
            rows.append([f"{model} @ {r['dim']}", _pct(r["llm_acc"]),
                         _pct(r["retrieval_acc"]),
                         _f3(r["mteb_pre"].get("ndcg@10"))])
    out.append(_md_table(
        ["Config", "LLM acc", "Retrieval acc", "pre-rerank nDCG@10"], rows))
    out.append("")
    return "\n".join(out)


def _section_diminishing(by_model: Dict[str, List[Dict[str, Any]]], tol: float = 0.02) -> str:
    out = [f"## Diminishing returns (within {tol * 100:.0f} pp of full-size LLM accuracy)\n"]
    rows = []
    for model, runs in by_model.items():
        full = runs[-1]
        base = full["llm_acc"]
        pick = None
        if isinstance(base, (int, float)):
            for r in runs:
                if isinstance(r["llm_acc"], (int, float)) and r["llm_acc"] >= base - tol:
                    pick = r
                    break
        if pick:
            rows.append([model, str(full["dim"]), _pct(base), str(pick["dim"]),
                         _pct(pick["llm_acc"])])
        else:
            rows.append([model, str(full["dim"]), _pct(base), "none", "N/A"])
    out.append(_md_table(
        ["Model", "Full dim", "Full LLM acc", "Smallest sufficient dim", "Its LLM acc"], rows))
    out.append("")
    return "\n".join(out)


def _section_per_label(by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    out = ["## Per-label end-to-end accuracy (LLM judge)\n",
           "Which question types degrade first as dimensions shrink.\n"]
    for model, runs in by_model.items():
        out.append(f"### {model}\n")
        headers = ["Dim"] + LABELS
        rows = []
        for r in runs:
            cells = [str(r["dim"])]
            for label in LABELS:
                info = r["per_label_e2e"].get(label, {})
                cells.append(_pct(info.get("llm")))
            rows.append(cells)
        out.append(_md_table(headers, rows))
        out.append("")
    return "\n".join(out)


def _section_cost_storage(by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    out = ["## Cost and storage tradeoff\n",
           "Embedding cost uses each model's real OpenAI list price "
           "(3-small $0.02/1M, 3-large $0.13/1M, input-only). Embedding cost does "
           "**not** vary with dimension (billing is per input token, not output "
           "vector width); only storage/search does. Generation + judge use gpt-5.2 "
           "pricing ($1.75/1M in, $14/1M out) and dominate total cost.\n"]
    for model, runs in by_model.items():
        price = runs[0].get("embed_price_1m")
        out.append(f"### {model}"
                   + (f"  (embed ${price:.2f}/1M tokens)" if price is not None else "")
                   + "\n")
        rows = []
        for r in runs:
            acc_per_dollar = None
            if isinstance(r["llm_acc"], (int, float)) and r["total_cost"]:
                acc_per_dollar = r["llm_acc"] / r["total_cost"]
            rows.append([
                str(r["dim"]),
                f"${r['embed_cost']:.3f}" if r["embed_cost"] is not None else "N/A",
                f"${r['gen_cost']:.2f}",
                f"${r['judge_cost']:.2f}",
                f"${r['total_cost']:.2f}",
                f"{acc_per_dollar:.3f}" if acc_per_dollar is not None else "N/A",
            ])
        out.append(_md_table(
            ["Dim", "Embed cost", "Gen cost", "Judge cost", "Total cost", "LLM acc / $"], rows))
        out.append("")
    return "\n".join(out)


def _section_memory(by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    dims = sorted({r["dim"] for runs in by_model.values() for r in runs if r["dim"]})
    full_dim = max(dims) if dims else None
    out = ["## Memory / storage comparison\n",
           f"Vectors are stored as float32 ({BYTES_PER_FLOAT32} bytes per dimension). "
           f"Footprint depends only on the dimension, so **both models use the same "
           f"memory at a given dimension** — the large model is not more expensive to "
           f"store than the small one at equal dims. Totals are for this corpus "
           f"({CORPUS_VECTORS:,} chunk vectors); the Chroma HNSW index adds a modest "
           f"constant overhead on top of the raw vectors below.\n"]
    rows = []
    for dim in dims:
        per_vec = dim * BYTES_PER_FLOAT32
        total_mb = CORPUS_VECTORS * per_vec / (1024 * 1024)
        rel = (dim / full_dim) if full_dim else None
        rows.append([
            str(dim),
            f"{per_vec:,} B",
            f"{per_vec / 1024:.1f} KB",
            f"{total_mb:.1f} MB",
            _retention(rel, 1.0) if rel is not None else "N/A",
        ])
    out.append(_md_table(
        ["Dim", "Bytes/vector", "Per vector", f"Index size ({CORPUS_VECTORS:,} vecs)",
         "Rel. to max dim"], rows))
    out.append("")
    out.append("Applies to both `text-embedding-3-small` and `text-embedding-3-large`. "
               "Example: shrinking large from 3072 to 512 dims cuts the raw index from "
               f"~{CORPUS_VECTORS * 3072 * 4 / (1024*1024):.0f} MB to "
               f"~{CORPUS_VECTORS * 512 * 4 / (1024*1024):.0f} MB (6x smaller) and speeds "
               "up cosine search proportionally.\n")
    return "\n".join(out)


def _section_mteb_curves(by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    out = ["## MTEB retrieval metrics vs dimension (pre-rerank)\n",
           "Ranking quality of the embedding alone, before the cross-encoder compensates.\n"]
    for model, runs in by_model.items():
        full = runs[-1]
        out.append(f"### {model}\n")
        rows = []
        for r in runs:
            m = r["mteb_pre"]
            rows.append([
                str(r["dim"]),
                _f3(m.get("hit@5")),
                _f3(m.get("recall@5")),
                _f3(m.get("mrr")),
                _f3(m.get("ndcg@10")),
                _retention(m.get("ndcg@10"), full["mteb_pre"].get("ndcg@10")),
            ])
        out.append(_md_table(
            ["Dim", "Hit@5", "Recall@5", "MRR", "nDCG@10", "nDCG@10 retention"], rows))
        out.append("")
    return "\n".join(out)


def _section_rerank_gap(by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    out = ["## Pre-rerank vs post-rerank gap\n",
           "How much the cross-encoder reranker compensates for weaker low-dimension embeddings "
           "(delta in nDCG@10).\n"]
    for model, runs in by_model.items():
        out.append(f"### {model}\n")
        rows = []
        for r in runs:
            pre = r["mteb_pre"].get("ndcg@10")
            post = r["mteb_post"].get("ndcg@10")
            delta = (post - pre) if isinstance(pre, (int, float)) and isinstance(post, (int, float)) else None
            rows.append([str(r["dim"]), _f3(pre), _f3(post),
                         f"{delta:+.3f}" if delta is not None else "N/A"])
        out.append(_md_table(["Dim", "pre nDCG@10", "post nDCG@10", "Δ (rerank gain)"], rows))
        out.append("")
    return "\n".join(out)


def _section_distractors(by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    out = ["## Distractor analysis (Distractor@5, pre-rerank)\n",
           "Fraction of top-5 slots occupied by confusable-but-wrong (grade-1) chunks. "
           "Higher = the embedding surfaces more look-alike distractors.\n"]
    for model, runs in by_model.items():
        out.append(f"### {model}\n")
        rows = [[str(r["dim"]), _f3(r["mteb_pre"].get("distractor@5"))] for r in runs]
        out.append(_md_table(["Dim", "Distractor@5"], rows))
        out.append("")
    return "\n".join(out)


def _section_escalation(by_model: Dict[str, List[Dict[str, Any]]]) -> str:
    out = ["## Hierarchical escalation\n",
           "Whether weaker low-dimension retrieval forces more LLM calls / deeper context climbing.\n"]
    for model, runs in by_model.items():
        out.append(f"### {model}\n")
        rows = []
        for r in runs:
            e = r["escalation"]
            rows.append([
                str(r["dim"]),
                str(e.get("total_llm_calls", "N/A")),
                _f3(e.get("avg_llm_calls_per_query")),
                _f3(e.get("avg_escalation_depth")),
            ])
        out.append(_md_table(
            ["Dim", "Total LLM calls", "Avg calls/query", "Avg escalation depth"], rows))
        out.append("")
    return "\n".join(out)


def _section_exec_summary(by_model: Dict[str, List[Dict[str, Any]]], tol: float = 0.02) -> str:
    out = ["## Executive summary\n"]
    bullets = []
    best_overall = None
    for model, runs in by_model.items():
        full = runs[-1]
        best = max(
            (r for r in runs if isinstance(r["llm_acc"], (int, float))),
            key=lambda r: r["llm_acc"], default=None)
        if best:
            bullets.append(
                f"- **{model}**: best LLM accuracy {_pct(best['llm_acc'])} at {best['dim']} dims; "
                f"full size ({full['dim']}) = {_pct(full['llm_acc'])}.")
            if best_overall is None or (best["llm_acc"] or 0) > (best_overall["llm_acc"] or 0):
                best_overall = best
        # smallest dim within tolerance
        base = full["llm_acc"]
        if isinstance(base, (int, float)):
            for r in runs:
                if isinstance(r["llm_acc"], (int, float)) and r["llm_acc"] >= base - tol:
                    bullets.append(
                        f"  - Diminishing returns: {r['dim']} dims already reaches "
                        f"{_pct(r['llm_acc'])} (within {tol * 100:.0f} pp of full).")
                    break
    if best_overall:
        bullets.append(
            f"- **Best overall accuracy**: {best_overall['model']} @ {best_overall['dim']} dims "
            f"= {_pct(best_overall['llm_acc'])}.")
    out.append("\n".join(bullets))
    out.append("")
    return "\n".join(out)


def _section_recommendation(by_model: Dict[str, List[Dict[str, Any]]], tol: float = 0.02) -> str:
    out = ["## Recommendation\n"]
    lines = []
    for model, runs in by_model.items():
        full = runs[-1]
        base = full["llm_acc"]
        pick = None
        if isinstance(base, (int, float)):
            for r in runs:
                if isinstance(r["llm_acc"], (int, float)) and r["llm_acc"] >= base - tol:
                    pick = r
                    break
        if pick and pick["dim"] and full["dim"]:
            storage_pct = pick["dim"] / full["dim"] * 100
            lines.append(
                f"- **{model}**: use **{pick['dim']} dims** — {_pct(pick['llm_acc'])} LLM accuracy "
                f"({_retention(pick['llm_acc'], base)} of full) at ~{storage_pct:.0f}% of full-size "
                f"storage/search cost.")
    lines.append(
        "\nBalance these against MTEB ranking quality (nDCG@10) and the pre/post-rerank gap: "
        "if the reranker fully closes the gap at a small dimension, that dimension is safe to ship.")
    out.append("\n".join(lines))
    out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Aggregate JSON
# ---------------------------------------------------------------------------


def _build_aggregate(runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "n_runs": len(runs),
        "runs": [
            {
                "tag": r["tag"],
                "model": r["model"],
                "dim": r["dim"],
                "total_questions": r["total_questions"],
                "llm_accuracy": r["llm_acc"],
                "token_accuracy": r["token_acc"],
                "retrieval_accuracy": r["retrieval_acc"],
                "escalation": r["escalation"],
                "llm_meta": r["llm_meta"],
                "cost_usd_stored": r["cost"],
                "cost_usd_corrected": {
                    "embed_price_per_1m": r["embed_price_1m"],
                    "embedding": r["embed_cost"],
                    "generation": r["gen_cost"],
                    "judge": r["judge_cost"],
                    "total": r["total_cost"],
                },
                "index_memory_mb": (r["dim"] * BYTES_PER_FLOAT32 * CORPUS_VECTORS / (1024 * 1024)) if r["dim"] else None,
                "time_s": r["time_s"],
                "per_label_e2e": r["per_label_e2e"],
                "mteb_pre_rerank_macro": r["mteb_pre"],
                "mteb_post_rerank_macro": r["mteb_post"],
            }
            for r in runs
        ],
    }


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run(output_dir: str = DEFAULT_OUTPUT_DIR) -> Optional[str]:
    """Generate aggregate.json + analysis_report.md from summary_*.json files."""
    runs = _load_runs(output_dir)
    if not runs:
        print(f"[warn] no summary_*.json found in {output_dir}; nothing to analyze.")
        return None

    aggregate = _build_aggregate(runs)
    agg_path = os.path.join(output_dir, "aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)

    by_model = _by_model(runs)
    parts = [
        "# Matryoshka Embedding Dimension Study — Analysis\n",
        f"Runs analyzed: {len(runs)}. Models: {', '.join(by_model.keys())}.\n",
        _section_exec_summary(by_model),
        _section_quality_curves(by_model),
        _section_crossover(by_model),
        _section_diminishing(by_model),
        _section_per_label(by_model),
        _section_cost_storage(by_model),
        _section_llm_meta(by_model),
        _section_memory(by_model),
        _section_mteb_curves(by_model),
        _section_rerank_gap(by_model),
        _section_distractors(by_model),
        _section_escalation(by_model),
        _section_recommendation(by_model),
    ]
    report = "\n".join(parts)
    report_path = os.path.join(output_dir, "analysis_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[analysis] wrote {agg_path}")
    print(f"[analysis] wrote {report_path}")
    return report_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze the Matryoshka embedding-dimension study.")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                   help="Directory holding summary_*.json (and where reports are written).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run(args.output_dir)


if __name__ == "__main__":
    main()
