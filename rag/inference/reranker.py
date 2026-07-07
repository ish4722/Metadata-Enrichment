"""Step 2 - Cross-encoder reranking.

    >>>>>  SWAP THE RERANKER MODEL HERE  <<<<<

Re-scores the retrieved candidates with a cross-encoder (query, passage) model
for sharper ordering before context assembly. No-op (returns the input truncated
to ``top_k``) when sentence-transformers is unavailable or the model fails to load.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

from rag.common.schema import Chunk


def _force_hf_offline() -> None:
    """Force Hugging Face into offline mode so cached models load without a Hub call.

    transformers>=5 runs a Hub-driven "processing class" auto-resolution when it can
    reach huggingface.co online. The classic cross-encoder repos don't declare a
    processor class, so that online path raises "Unrecognized processing class" and the
    reranker silently disables itself. Loading from the local cache skips that path.
    We set the env vars *and* patch the already-imported constant, since huggingface_hub
    reads HF_HUB_OFFLINE at import time.
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        import huggingface_hub.constants as _hf_const

        _hf_const.HF_HUB_OFFLINE = True
    except Exception:
        pass


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
            _force_hf_offline()
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
