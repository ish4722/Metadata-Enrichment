"""Score 8 retrieval configurations against the GRADED gold set.

Configurations (cross product):
  retrieval: dense, hybrid alpha=0.5, hybrid alpha=0.7, hybrid alpha=0.85
  query    : no-inject (plain question) vs inject (well name substituted in)

Relevance (from output_table/retrieval_goldset.json):
  * grade 2 ("correct") chunks  -> the relevant set for Hit@k / Recall@k / MRR /
    MAP / nDCG. These are chunks that actually yield the ground-truth answer.
  * grade 1 ("wrong") chunks    -> distractors. Tracked separately as
    "Distractor@k" (how often a config surfaces confusable-but-wrong chunks).

Only (doc, label) pairs with status == "answerable" are scored (a relevant
target exists). not_grounded / only_wrong / no_ground_truth pairs are skipped.

Efficiency: dense distances + BM25 are computed ONCE per (doc, label, arm); all
four retrieval rankings are derived from that single pass (no 8x re-embedding).

Run:
    python evaluate_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from rag.common.ssl_patch import apply_ssl_patches  # noqa: E402

apply_ssl_patches()

import rag_evaluation  # noqa: E402
from rag.config import RagConfig  # noqa: E402
from rag.indexing.build_pipeline import make_backend  # noqa: E402
from rag.indexing.chroma_store import _minmax_normalize, get_or_create_store  # noqa: E402
from rag.inference.reranker import get_reranker  # noqa: E402
import evaluate_all  # noqa: E402

GOLDSET_PATH = os.path.join("output_table", "retrieval_goldset.json")
OUT_JSON = os.path.join("output_table", "retrieval_metrics.json")
CHART_DIR = os.path.join("output_table", "retrieval_charts")
DOCX_PATH = os.path.join("output_table", "RAG_retrieval_metrics.docx")

KS = [1, 3, 5, 10]
TOP_N = 20  # ranking depth kept for MRR / MAP
LABELS = ["facility name", "openv ID", "facilityevent spud date", "lat+long"]
LABEL_SHORT = {"facility name": "facility", "openv ID": "openv",
               "facilityevent spud date": "spud", "lat+long": "lat+long"}
# alpha=None => dense
CONFIGS = [("dense", None), ("hybrid", 0.5), ("hybrid", 0.7), ("hybrid", 0.85)]
ARMS = [False, True]  # inject?


class _TransformersReranker:
    """Cross-encoder reranker via transformers directly.

    Fallback for environments where the sentence-transformers 5.x ``CrossEncoder``
    wrapper fails to load the tokenizer/processor. Uses the SAME model weights, so
    the relevance scores (and therefore the ranking) are equivalent. ``predict``
    mirrors the CrossEncoder API: list[(query, passage)] -> list[float] (higher = better).
    """

    def __init__(self, model_name, batch_size=32, max_length=512):
        # transformers 5.x makes a hub call (chat-template listing) that can return
        # empty JSON behind some proxies; force local cache (the model is already
        # downloaded) to avoid that network round-trip.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, local_files_only=True).eval()
        self.bs = batch_size
        self.max_length = max_length

    def predict(self, pairs):
        if not pairs:
            return []
        torch = self.torch
        scores = []
        with torch.no_grad():
            for i in range(0, len(pairs), self.bs):
                batch = pairs[i:i + self.bs]
                enc = self.tok([p[0] for p in batch], [p[1] for p in batch],
                               padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt")
                logits = self.model(**enc).logits.squeeze(-1)
                scores.extend(logits.reshape(-1).tolist())
        return scores


def config_name(mode, alpha, inject):
    base = "dense" if mode == "dense" else f"hyb{alpha}"
    return f"{base}_{'inj' if inject else 'noinj'}"


def inject_query(question, label, well_name):
    """Substitute the well name into the question (non-facility labels only)."""
    if label == "facility name" or not well_name or well_name == "the well":
        return question
    out = question
    for needle in ("this well", "the well"):
        if needle in out:
            return out.replace(needle, well_name)
    return out + f" for {well_name}"


def metrics_for(ranked_ids, rel_set, wrong_set):
    """All metrics for one ranked list against one relevant/wrong set."""
    rel_total = len(rel_set)
    rr = 0.0
    for i, cid in enumerate(ranked_ids, 1):
        if cid in rel_set:
            rr = 1.0 / i
            break
    seen = 0
    ap_sum = 0.0
    for i, cid in enumerate(ranked_ids, 1):
        if cid in rel_set:
            seen += 1
            ap_sum += seen / i
    ap = ap_sum / rel_total if rel_total else 0.0

    def idcg(k):
        return sum(1.0 / math.log2(i + 1) for i in range(1, min(k, rel_total) + 1))

    out = {"mrr": rr, "map": ap}
    for k in KS:
        topk = ranked_ids[:k]
        inter = sum(1 for c in topk if c in rel_set)
        dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(topk, 1) if c in rel_set)
        idc = idcg(k)
        out[f"hit@{k}"] = 1.0 if inter else 0.0
        out[f"recall@{k}"] = inter / rel_total if rel_total else 0.0
        out[f"ndcg@{k}"] = (dcg / idc) if idc > 0 else 0.0
        out[f"distractor@{k}"] = sum(1 for c in topk if c in wrong_set) / k
    return out


def rank_orders(dense, bm):
    """Return {config_key -> ranked index array} from per-chunk dense & bm arrays."""
    dn = _minmax_normalize(dense)
    bn = _minmax_normalize(bm) if bm is not None else np.zeros_like(dense)
    orders = {}
    for mode, alpha in CONFIGS:
        if mode == "dense":
            combined = dense
        else:
            combined = alpha * dn + (1 - alpha) * bn
        orders[(mode, alpha)] = np.argsort(combined)[::-1][:TOP_N]
    return orders


def main():
    rag_evaluation.configure_utf8_stdout()
    ap = argparse.ArgumentParser(description="Score 8 retrieval configs against the graded gold set.")
    ap.add_argument("--rerank", action="store_true",
                    help="Apply the cross-encoder reranker on top of each retriever before scoring.")
    ap.add_argument("--output", default=None, help="Output metrics JSON path.")
    args = ap.parse_args()
    out_json = args.output or (
        os.path.join("output_table", "retrieval_metrics_rerank.json") if args.rerank else OUT_JSON)

    t0 = time.perf_counter()
    gold = json.load(open(GOLDSET_PATH, encoding="utf-8"))
    docs_gold = gold["docs"]
    doc_ids = sorted(docs_gold.keys())
    print(f"Gold set: {len(doc_ids)} docs from {GOLDSET_PATH}")

    cfg = RagConfig.from_env()
    matched, _ = evaluate_all.discover_docs(
        evaluate_all.DEFAULT_DOCS_DIR, evaluate_all.DEFAULT_MATCHED_DIR, only=doc_ids)
    gt_path = {d["id"]: d["ground_truth"] for d in matched}

    backend = make_backend(cfg)
    store = get_or_create_store(cfg.chroma_dir, cfg.chroma_collection, backend, metric=cfg.faiss_metric)
    questions = {q["label"]: q["question"] for q in rag_evaluation.parse_queries_file("queries.txt")}

    reranker = None
    if args.rerank:
        reranker = get_reranker(cfg.reranker_model)
        if reranker is None:
            print("[info] sentence-transformers CrossEncoder unavailable; "
                  "using transformers fallback reranker (same weights).")
            reranker = _TransformersReranker(cfg.reranker_model)
        print(f"Reranker ON: {cfg.reranker_model}")

    # accumulator: acc[config_key][label] -> list of per-doc metric dicts
    acc = defaultdict(lambda: defaultdict(list))
    n_pairs = defaultdict(int)
    embed_calls = 0

    for di, doc_id in enumerate(doc_ids, 1):
        well_name = rag_evaluation.resolve_well_name(json.load(open(gt_path[doc_id], encoding="utf-8")))
        corpus = store._ensure_corpus(doc_id)
        ids = corpus["ids"]
        chunks = corpus["chunks"]
        bm25 = corpus["bm25"]
        entry = docs_gold[doc_id]

        for label in LABELS:
            ginfo = entry.get(label, {})
            if ginfo.get("status") != "answerable":
                continue
            rel_set = set(ginfo.get("correct_chunk_ids", []))
            wrong_set = {r["chunk_id"] for r in ginfo.get("relevant", []) if r["grade"] == 1}
            if not rel_set:
                continue
            n_pairs[label] += 1
            for inject in ARMS:
                q = inject_query(questions[label], label, well_name) if inject else questions[label]
                dense = store._dense_scores_full(q, ids, doc_id)
                embed_calls += 1
                bm = (np.array(bm25.get_scores(q.lower().split()), dtype=np.float32)
                      if bm25 is not None else None)
                orders = rank_orders(dense, bm)
                if reranker is not None:
                    # cross-encoder scores the UNION of the 4 candidate pools once,
                    # then each config's pool is re-ordered by those cached scores.
                    union = sorted({int(i) for order in orders.values() for i in order})
                    ce_scores = reranker.predict([(q, chunks[i].text) for i in union])
                    cross = {i: float(s) for i, s in zip(union, ce_scores)}
                    orders = {key: np.array(sorted((int(i) for i in order),
                                                   key=lambda i: cross[i], reverse=True))
                              for key, order in orders.items()}
                for (mode, alpha), order in orders.items():
                    ranked_ids = [ids[i] for i in order]
                    m = metrics_for(ranked_ids, rel_set, wrong_set)
                    acc[config_name(mode, alpha, inject)][label].append(m)
        print(f"  [{di}/{len(doc_ids)}] {doc_id} done")

    # ---- aggregate (mean across docs per config/label, then macro across labels) ----
    metric_keys = (["mrr", "map"]
                   + [f"hit@{k}" for k in KS] + [f"recall@{k}" for k in KS]
                   + [f"ndcg@{k}" for k in KS] + [f"distractor@{k}" for k in KS])
    results = {}
    for ckey, per_label in acc.items():
        per_label_means = {}
        for label, rows in per_label.items():
            per_label_means[label] = {mk: float(np.mean([r[mk] for r in rows])) for mk in metric_keys}
        macro = {mk: float(np.mean([per_label_means[l][mk] for l in per_label_means])) for mk in metric_keys}
        results[ckey] = {"per_label": per_label_means, "macro": macro}

    elapsed = time.perf_counter() - t0
    out = {
        "gold_set": GOLDSET_PATH,
        "n_docs": len(doc_ids),
        "evaluable_pairs_per_label": dict(n_pairs),
        "ks": KS,
        "reranker": cfg.reranker_model if reranker is not None else None,
        "configs": [config_name(m, a, inj) for (m, a) in CONFIGS for inj in ARMS],
        "embed_calls": embed_calls,
        "elapsed_s": round(elapsed, 1),
        "results": results,
    }
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    json.dump(out, open(out_json, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nSaved metrics -> {out_json}  ({elapsed:.1f}s, {embed_calls} query embeds)")

    # ---- console table: macro metrics per config ----
    show = ["hit@1", "hit@5", "recall@5", "recall@10", "mrr", "map", "ndcg@10", "distractor@5"]
    hdr = ["config"] + show
    rows = []
    for ckey in out["configs"]:
        macro = results[ckey]["macro"]
        rows.append([ckey] + [f"{macro[k]:.3f}" for k in show])
    w = [max(len(str(r[i])) for r in [hdr] + rows) for i in range(len(hdr))]
    line = lambda r: "  ".join(str(r[i]).ljust(w[i]) for i in range(len(r)))
    print("\nMACRO (mean over 4 labels):")
    print(line(hdr))
    print("-" * (sum(w) + 2 * len(w)))
    for r in rows:
        print(line(r))


if __name__ == "__main__":
    main()
