"""Reusable retrieval ranking metrics (MTEB-style) and gold-set scoring.

The pure ranking helpers (``metrics_for``, ``inject_query``) were factored out of
``evaluate_retrieval.py`` so both that script and the Matryoshka dimension study
compute Hit@k / Recall@k / nDCG@k / MRR / MAP / Distractor@k the same way.

``score_matryoshka_retrieval`` scores a single (embedding model, dimension)
configuration against the graded gold set, at two stages:

  * pre_rerank  - hybrid (dense + BM25) ranking only. Isolates the effect of the
                  embedding dimension on ranking quality.
  * post_rerank - after the cross-encoder rerank, matching what the LLM actually
                  sees in the batch_runner pipeline.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

# k cut-offs used for Hit@k / Recall@k / nDCG@k / Distractor@k.
KS = [1, 3, 5, 10]
# Ranking depth kept for MRR / MAP (and the reranker candidate pool default).
TOP_N = 20
# Labels scored in the study (align with queries.txt / the gold set). A label is
# only actually scored when it (a) is defined in queries.txt and (b) has answerable
# chunks in the gold set, so listing extras here is safe.
LABELS = [
    "facility name",
    "openv ID",
    "facilityevent spud date",
    "lat+long",
    "country name",
    "basin name",
    "field name",
    "default vertical measurement ID",
    "current operator ID",
    "marker names",
    "vertical measurement total depth",
    "completion date",
]


def inject_query(question: str, label: str, well_name: str) -> str:
    """Substitute the well name into the question (non-facility labels only)."""
    if label == "facility name" or not well_name or well_name == "the well":
        return question
    out = question
    for needle in ("this well", "the well"):
        if needle in out:
            return out.replace(needle, well_name)
    return out + f" for {well_name}"


def metrics_for(
    ranked_ids: Sequence[str],
    rel_set: set,
    wrong_set: set,
    ks: Sequence[int] = KS,
) -> Dict[str, float]:
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

    def idcg(k: int) -> float:
        return sum(1.0 / math.log2(i + 1) for i in range(1, min(k, rel_total) + 1))

    out = {"mrr": rr, "map": ap}
    for k in ks:
        topk = ranked_ids[:k]
        inter = sum(1 for c in topk if c in rel_set)
        dcg = sum(1.0 / math.log2(i + 1) for i, c in enumerate(topk, 1) if c in rel_set)
        idc = idcg(k)
        out[f"hit@{k}"] = 1.0 if inter else 0.0
        out[f"recall@{k}"] = inter / rel_total if rel_total else 0.0
        out[f"ndcg@{k}"] = (dcg / idc) if idc > 0 else 0.0
        out[f"distractor@{k}"] = sum(1 for c in topk if c in wrong_set) / k
    return out


def metric_keys(ks: Sequence[int] = KS) -> List[str]:
    """Ordered list of every metric key produced by :func:`metrics_for`."""
    keys = ["mrr", "map"]
    for prefix in ("hit", "recall", "ndcg", "distractor"):
        keys += [f"{prefix}@{k}" for k in ks]
    return keys


def _aggregate(acc: Dict[str, List[Dict[str, float]]], ks: Sequence[int]) -> Dict[str, object]:
    """Mean per label, then macro-average across labels."""
    import numpy as np

    keys = metric_keys(ks)
    per_label: Dict[str, Dict[str, float]] = {}
    for label, rows in acc.items():
        if not rows:
            continue
        per_label[label] = {mk: float(np.mean([r[mk] for r in rows])) for mk in keys}
    if per_label:
        macro = {mk: float(np.mean([per_label[l][mk] for l in per_label])) for mk in keys}
    else:
        macro = {mk: 0.0 for mk in keys}
    return {"macro": macro, "per_label": per_label}


def score_matryoshka_retrieval(
    config,
    goldset_path: str,
    queries_path: str = "queries.txt",
    ks: Sequence[int] = KS,
    top_n: int = TOP_N,
) -> Dict[str, object]:
    """Score one embedding configuration against the graded gold set.

    Returns a dict with ``pre_rerank`` and ``post_rerank`` sections, each holding
    macro and per-label MTEB metrics. The chunk ids in the gold set are the same
    Chroma-namespaced ids returned by the store, and chunking is deterministic, so
    the gold set transfers across every dimension without rebuilding it.
    """
    import json

    import numpy as np

    from rag.evaluation import batch as evaluate_all
    from rag.evaluation import core as rag_evaluation
    from rag.indexing.build_pipeline import make_backend
    from rag.indexing.chroma_store import _minmax_normalize, get_or_create_store
    from rag.inference.reranker import get_reranker, rerank

    gold = json.load(open(goldset_path, encoding="utf-8"))
    docs_gold = gold["docs"]
    doc_ids = sorted(docs_gold.keys())

    matched, _ = evaluate_all.discover_docs(
        evaluate_all.DEFAULT_DOCS_DIR, evaluate_all.DEFAULT_MATCHED_DIR, only=doc_ids
    )
    gt_path = {d["id"]: d["ground_truth"] for d in matched}

    backend = make_backend(config)
    store = get_or_create_store(
        config.chroma_dir, config.chroma_collection, backend, metric=config.faiss_metric
    )
    questions = {q["label"]: q["question"] for q in rag_evaluation.parse_queries_file(queries_path)}

    reranker_on = config.use_reranker and get_reranker(config.reranker_model) is not None

    pre_acc: Dict[str, List[Dict[str, float]]] = {label: [] for label in LABELS}
    post_acc: Dict[str, List[Dict[str, float]]] = {label: [] for label in LABELS}
    scored_pairs = 0

    for doc_id in doc_ids:
        if doc_id not in gt_path:
            continue
        try:
            corpus = store._ensure_corpus(doc_id)
        except Exception as exc:  # noqa: BLE001
            print(f"  [mteb] skip {doc_id}: {exc}")
            continue
        ids: List[str] = corpus["ids"]
        chunks = corpus["chunks"]
        bm25 = corpus["bm25"]
        if not ids:
            continue

        well_name = rag_evaluation.resolve_well_name(json.load(open(gt_path[doc_id], encoding="utf-8")))
        entry = docs_gold[doc_id]

        for label in LABELS:
            ginfo = entry.get(label, {})
            if ginfo.get("status") != "answerable":
                continue
            # Only score labels that the current queries file actually defines;
            # the gold set may cover more labels than queries.txt (which can be
            # trimmed to a subset for a focused benchmark).
            question_text = questions.get(label)
            if question_text is None:
                continue
            rel_set = set(ginfo.get("correct_chunk_ids", []))
            wrong_set = {r["chunk_id"] for r in ginfo.get("relevant", []) if r.get("grade") == 1}
            if not rel_set:
                continue

            q = inject_query(question_text, label, well_name)
            dense = store._dense_scores_full(q, ids, doc_id)
            bm = (
                np.array(bm25.get_scores(q.lower().split()), dtype=np.float32)
                if bm25 is not None
                else None
            )
            dn = _minmax_normalize(dense)
            if bm is not None:
                combined = config.hybrid_alpha * dn + (1 - config.hybrid_alpha) * _minmax_normalize(bm)
            else:
                combined = dense
            order = np.argsort(combined)[::-1]

            pre_ids = [ids[i] for i in order[:top_n]]
            pre_acc[label].append(metrics_for(pre_ids, rel_set, wrong_set, ks))

            cand_idx = [int(i) for i in order[: config.top_k]]
            matches = [(chunks[i], float(combined[i])) for i in cand_idx]
            # Recover each chunk's namespaced id (what the gold set uses) after
            # the reranker reorders the chunk objects; keyed by object identity.
            id_by_chunk = {id(chunks[i]): ids[i] for i in cand_idx}
            if reranker_on and matches:
                reranked = rerank(q, matches, config.reranker_model, top_k=config.rerank_top_k)
                post_ids = [id_by_chunk[id(c)] for c, _ in reranked]
            else:
                post_ids = [ids[i] for i in cand_idx[: config.rerank_top_k]]
            post_acc[label].append(metrics_for(post_ids, rel_set, wrong_set, ks))
            scored_pairs += 1

    return {
        "goldset": goldset_path,
        "reranker": config.reranker_model if reranker_on else None,
        "scored_pairs": scored_pairs,
        "hybrid_alpha": config.hybrid_alpha,
        "top_k": config.top_k,
        "rerank_top_k": config.rerank_top_k,
        "ks": list(ks),
        "pre_rerank": _aggregate(pre_acc, ks),
        "post_rerank": _aggregate(post_acc, ks),
    }
