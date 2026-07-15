"""Per-model pricing and context-window lookup tables + cost helpers.

The pipeline's own cost accounting (``rag_evaluation.token_cost_usd`` /
``embedding_cost_usd``) uses a single fixed price for every model, so comparing
different embedding models (Task 1) or generator LLMs (Task 4) needs per-model
prices. The benchmark scripts therefore store raw token counts and recompute
cost here from these tables (the same pattern ``analyze_matryoshka_study.py``
uses for the embedding cost).

Prices are USD per 1,000,000 tokens; context windows are in tokens. LLM prices
and context windows were verified against OpenAI's published rate card
(July 2026). Embeddings are input-only (no output cost).
"""

from __future__ import annotations

from typing import Dict, Optional


# ---------------------------------------------------------------------------
# Embedding models
# ---------------------------------------------------------------------------

# ``price_per_1m``: USD per 1M input tokens (None => not metered per token, e.g.
# a self-hosted deployment where cost is infrastructure, not per-token).
# ``context_window``: max input tokens the encoder accepts.
EMBEDDING_SPECS: Dict[str, Dict[str, object]] = {
    "text-embedding-3-large": {
        "price_per_1m": 0.13,
        "context_window": 8191,
        "native_dim": 3072,
        "notes": "OpenAI; supports Matryoshka dimension truncation.",
    },
    "text-embedding-3-small": {
        "price_per_1m": 0.02,
        "context_window": 8191,
        "native_dim": 1536,
        "notes": "OpenAI; supports Matryoshka dimension truncation.",
    },
    # jina-embeddings-v3 served from the internal Cloud Run endpoint. It is
    # self-hosted, so there is no per-token list price (cost is infra/compute).
    "jina-api": {
        "price_per_1m": None,
        "context_window": 8192,
        "native_dim": 1024,
        "notes": "jina-embeddings-v3, self-hosted Cloud Run (cost is infra-only).",
    },
}

# Friendly benchmark-config keys map to the underlying embedding spec.
EMBEDDING_ALIAS_TO_SPEC = {
    "jina_latechunk": "jina-api",
    "jina_nolatechunk": "jina-api",
    "openai_large": "text-embedding-3-large",
    "openai_small": "text-embedding-3-small",
}


# ---------------------------------------------------------------------------
# Generator LLMs
# ---------------------------------------------------------------------------

# ``input_per_1m`` / ``output_per_1m``: USD per 1M tokens. ``context_window`` and
# ``max_output`` in tokens.
LLM_SPECS: Dict[str, Dict[str, object]] = {
    "gpt-5.2": {"input_per_1m": 1.75, "output_per_1m": 14.0, "context_window": 400_000, "max_output": 128_000},
    "gpt-5.1": {"input_per_1m": 1.25, "output_per_1m": 10.0, "context_window": 400_000, "max_output": 128_000},
    "gpt-5": {"input_per_1m": 1.25, "output_per_1m": 10.0, "context_window": 400_000, "max_output": 128_000},
    "gpt-5-mini": {"input_per_1m": 0.25, "output_per_1m": 2.0, "context_window": 400_000, "max_output": 128_000},
    "gpt-5-nano": {"input_per_1m": 0.05, "output_per_1m": 0.40, "context_window": 400_000, "max_output": 128_000},
    "gpt-4o": {"input_per_1m": 2.50, "output_per_1m": 10.0, "context_window": 128_000, "max_output": 16_384},
    "gpt-4o-mini": {"input_per_1m": 0.15, "output_per_1m": 0.60, "context_window": 128_000, "max_output": 16_384},
}

# Default generator sweep for Task 4 (order = user-requested order).
DEFAULT_LLM_SWEEP = [
    "gpt-5.2", "gpt-5.1", "gpt-5-nano", "gpt-5-mini", "gpt-5", "gpt-4o-mini", "gpt-4o",
]


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------


def embed_cost(model: str, tokens: int) -> Optional[float]:
    """USD to embed ``tokens`` input tokens with ``model`` (None if not metered)."""
    spec = EMBEDDING_SPECS.get(model)
    if spec is None:
        return None
    price = spec.get("price_per_1m")
    if price is None:
        return None
    return tokens / 1_000_000.0 * float(price)


def llm_cost(model: str, input_tokens: int, output_tokens: int) -> Optional[float]:
    """USD for ``input_tokens``/``output_tokens`` with generator ``model``."""
    spec = LLM_SPECS.get(model)
    if spec is None:
        return None
    return (
        input_tokens / 1_000_000.0 * float(spec["input_per_1m"])
        + output_tokens / 1_000_000.0 * float(spec["output_per_1m"])
    )


def embedding_context_window(model: str) -> Optional[int]:
    spec = EMBEDDING_SPECS.get(model)
    return int(spec["context_window"]) if spec else None


def llm_context_window(model: str) -> Optional[int]:
    spec = LLM_SPECS.get(model)
    return int(spec["context_window"]) if spec else None
