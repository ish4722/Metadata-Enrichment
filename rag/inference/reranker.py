"""Step 2 - Cross-encoder reranking.

    >>>>>  SWAP THE RERANKER MODEL HERE  <<<<<

Re-scores the retrieved candidates with a cross-encoder (query, passage) model
for sharper ordering before context assembly. No-op (returns the input truncated
to ``top_k``) when sentence-transformers is unavailable or the model fails to load.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from rag.common.schema import Chunk


def _sigmoid(x: float) -> float:
    """Map a cross-encoder logit to a (0, 1) relevance probability.

    ms-marco cross-encoders emit raw, unbounded logits (often negative), which
    are not human-readable as a "score". A sigmoid turns them into an
    interpretable 0-1 relevance probability without changing the ranking.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)

try:
    from sentence_transformers import CrossEncoder
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

_reranker_cache = {}


def get_reranker(model_name: str) -> Optional["CrossEncoder"]:
    """Load (and cache) the cross-encoder reranker, or None if unavailable."""
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    if model_name not in _reranker_cache:
        try:
            print(f"Loading cross-encoder reranker ({model_name}) ...")
            _reranker_cache[model_name] = CrossEncoder(model_name)
            print("  [ok] reranker loaded")
        except Exception as e:
            print(f"[warn] Reranker unavailable ({e}) - skipping rerank step.")
            _reranker_cache[model_name] = None
    return _reranker_cache[model_name]


def rerank(
    query: str,
    matches: List[Tuple[Chunk, float]],
    model_name: str,
    top_k: int = 10,
) -> List[Tuple[Chunk, float]]:
    """Reorder ``matches`` by cross-encoder relevance and keep the top_k."""
    reranker = get_reranker(model_name)
    if reranker is None or not matches:
        return matches[:top_k]

    pairs = [(query, c.text) for c, _ in matches]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(matches, scores), key=lambda x: x[1], reverse=True)
    return [(chunk, _sigmoid(float(score))) for (chunk, _old), score in ranked[:top_k]]
