"""OpenAI usage/cost callback wrapper.

A LangChain callback handler that accumulates the token usage and the richer
metadata the OpenAI chat endpoint returns across every LLM call, so the
benchmarks can surface it in their comparison tables.

Why a custom handler instead of ``langchain_community`` ``get_openai_callback``:
the built-in one only prices a hard-coded list of public OpenAI model names, so
it reports ``$0`` for the SLB-internal ``gpt-5.x`` deployments. This tracker
records the raw counts (prompt / completion / cached / reasoning tokens,
successful requests, per-model split, finish reasons, model fingerprint) and
leaves pricing to :meth:`LLMUsageTracker.cost`, which takes explicit per-1M
prices (the benchmark scripts already keep those in ``benchmarking.model_specs``).

The handler is attached once to the ``ChatOpenAI`` client (see
``rag.inference.llm``), so it fires for both the generation and judge calls with
no changes at the call sites. It is thread-safe because the batch runner drives
generation / judging across a thread pool.
"""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any, Dict, Optional

try:
    from langchain_core.callbacks import BaseCallbackHandler
    _HAS_CALLBACK_BASE = True
except Exception:  # pragma: no cover - langchain missing / degraded install
    class BaseCallbackHandler:  # type: ignore
        """Fallback no-op base so imports never hard-fail without langchain."""

    _HAS_CALLBACK_BASE = False


def _as_int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


class LLMUsageTracker(BaseCallbackHandler):
    """Accumulate token usage + OpenAI response metadata across LLM calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.successful_requests = 0
        self.failed_requests = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0
        # Prompt tokens served from the provider prompt cache (billed cheaper).
        self.cached_prompt_tokens = 0
        # Reasoning tokens (o-series / gpt-5 reasoning) - counted in completion.
        self.reasoning_tokens = 0
        self.models: "Counter[str]" = Counter()          # model -> request count
        self.finish_reasons: "Counter[str]" = Counter()   # finish_reason -> count
        self.system_fingerprints: set[str] = set()

    # ── LangChain callback hooks ─────────────────────────────────────────────
    def on_llm_end(self, response: Any, **kwargs: Any) -> None:  # noqa: D401
        """Aggregate usage from a completed ``LLMResult``."""
        prompt = completion = total = cached = reasoning = 0
        model_name: Optional[str] = None

        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            model_name = llm_output.get("model_name") or llm_output.get("model")
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
            if isinstance(usage, dict):
                prompt = _as_int(usage.get("prompt_tokens"))
                completion = _as_int(usage.get("completion_tokens"))
                total = _as_int(usage.get("total_tokens"))
                cached = _as_int(
                    (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
                )
                reasoning = _as_int(
                    (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
                )
            fp = llm_output.get("system_fingerprint")
        else:
            fp = None

        # Fall back to / enrich from the standardized message.usage_metadata and
        # per-generation finish_reason on the chat generations.
        generations = getattr(response, "generations", None) or []
        finish_reason: Optional[str] = None
        for batch in generations:
            for gen in batch:
                info = getattr(gen, "generation_info", None) or {}
                finish_reason = finish_reason or info.get("finish_reason")
                msg = getattr(gen, "message", None)
                meta = getattr(msg, "usage_metadata", None) if msg is not None else None
                if isinstance(meta, dict) and not (prompt or completion):
                    prompt = _as_int(meta.get("input_tokens"))
                    completion = _as_int(meta.get("output_tokens"))
                    total = _as_int(meta.get("total_tokens"))
                    cached = cached or _as_int(
                        (meta.get("input_token_details") or {}).get("cache_read")
                    )
                    reasoning = reasoning or _as_int(
                        (meta.get("output_token_details") or {}).get("reasoning")
                    )
                if model_name is None and msg is not None:
                    model_name = (getattr(msg, "response_metadata", {}) or {}).get("model_name")

        if not total:
            total = prompt + completion

        with self._lock:
            self.successful_requests += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total
            self.cached_prompt_tokens += cached
            self.reasoning_tokens += reasoning
            if model_name:
                self.models[model_name] += 1
            if finish_reason:
                self.finish_reasons[finish_reason] += 1
            if fp:
                self.system_fingerprints.add(str(fp))

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        with self._lock:
            self.failed_requests += 1

    # ── consumers ────────────────────────────────────────────────────────────
    def cost(self, input_per_1m: Optional[float], output_per_1m: Optional[float]) -> Optional[float]:
        """USD for the accumulated tokens given per-1M input/output prices.

        Cached prompt tokens are billed at 25% of the input price (OpenAI's
        standard prompt-cache discount); everything else is full price. Returns
        ``None`` when either price is unknown.
        """
        if input_per_1m is None or output_per_1m is None:
            return None
        billable_prompt = max(self.prompt_tokens - self.cached_prompt_tokens, 0)
        return (
            billable_prompt / 1_000_000.0 * float(input_per_1m)
            + self.cached_prompt_tokens / 1_000_000.0 * float(input_per_1m) * 0.25
            + self.completion_tokens / 1_000_000.0 * float(output_per_1m)
        )

    def snapshot(self, *, input_per_1m: Optional[float] = None,
                 output_per_1m: Optional[float] = None) -> Dict[str, Any]:
        """Serializable view of the accumulated usage + metadata (+ cost if priced)."""
        with self._lock:
            data: Dict[str, Any] = {
                "successful_requests": self.successful_requests,
                "failed_requests": self.failed_requests,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "cached_prompt_tokens": self.cached_prompt_tokens,
                "reasoning_tokens": self.reasoning_tokens,
                "cache_hit_rate": (
                    self.cached_prompt_tokens / self.prompt_tokens
                    if self.prompt_tokens else None
                ),
                "models": dict(self.models),
                "finish_reasons": dict(self.finish_reasons),
                "system_fingerprints": sorted(self.system_fingerprints),
            }
        if input_per_1m is not None or output_per_1m is not None:
            data["cost_usd"] = self.cost(input_per_1m, output_per_1m)
        return data

    def reset(self) -> None:
        with self._lock:
            self.successful_requests = 0
            self.failed_requests = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.total_tokens = 0
            self.cached_prompt_tokens = 0
            self.reasoning_tokens = 0
            self.models.clear()
            self.finish_reasons.clear()
            self.system_fingerprints.clear()
