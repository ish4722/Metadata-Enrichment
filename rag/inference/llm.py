"""Step 5 - LLM client.

    >>>>>  SWAP THE LLM HERE  <<<<<

Thin wrapper around the internal SLB ChatOpenAI deployment. ``get_llm`` returns
None when the LLM stack (langchain-openai + sauth_token) is unavailable, so the
rest of the pipeline can degrade gracefully to retrieval-only.

The SLB token is a short-lived JWT, so the client is wrapped in
:class:`_RefreshingChatLLM`: on an auth/expired error it fetches a fresh token,
rebuilds the underlying ChatOpenAI, and retries once. This keeps long-running
processes (the web-app backend, batch runner) from dying mid-run with
"JWT expired" after the token they captured at startup ages out.
"""

from __future__ import annotations

from typing import Any, Optional

from rag.common.token_utils import is_auth_error, strip_bearer
from rag.config import RagConfig
from rag.inference.usage_callback import LLMUsageTracker

try:
    from langchain_openai import ChatOpenAI
    from rag.common.sauth_token import get_token
    HAS_LLM = True
except Exception:
    HAS_LLM = False


class _RefreshingChatLLM:
    """ChatOpenAI wrapper that refreshes the SLB token on auth failure.

    Exposes ``.invoke(...)`` (used everywhere in the pipeline) and transparently
    delegates any other attribute access to the live ChatOpenAI client.
    """

    def __init__(self, config: RagConfig, usage_tracker: Optional[LLMUsageTracker] = None):
        self._config = config
        self._client: Optional[Any] = None
        # May be disabled at runtime if the model rejects a non-default temperature.
        self._temperature: Optional[float] = getattr(config, "llm_temperature", 0.0)
        # Persist the usage callback across token refreshes so cost/metadata keep
        # accumulating even when the underlying ChatOpenAI is rebuilt.
        self.usage_tracker: LLMUsageTracker = usage_tracker or LLMUsageTracker()

    def _build(self) -> Any:
        token = strip_bearer(get_token())
        kwargs = dict(
            model=self._config.llm_model,
            base_url=self._config.llm_base_url,
            api_key=token,
            default_headers={"slb-partition-id": self._config.partition_id},
            callbacks=[self.usage_tracker],
        )
        if self._temperature is not None:
            kwargs["temperature"] = self._temperature
        return ChatOpenAI(**kwargs)

    def _ensure(self) -> Any:
        if self._client is None:
            self._client = self._build()
        return self._client

    def invoke(self, *args: Any, **kwargs: Any) -> Any:
        try:
            return self._ensure().invoke(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if is_auth_error(exc):
                print("[auth] LLM token expired - refreshing and retrying once ...")
                self._client = self._build()
                return self._client.invoke(*args, **kwargs)
            # Some reasoning models reject a non-default temperature - drop it and retry.
            if self._temperature is not None and "temperature" in str(exc).lower():
                print(f"[warn] model rejected temperature={self._temperature}; "
                      "retrying without it.")
                self._temperature = None
                self._client = self._build()
                return self._client.invoke(*args, **kwargs)
            raise

    async def ainvoke(self, *args: Any, max_retries: int = 5, **kwargs: Any) -> Any:
        """Async invoke with token-refresh, temperature fallback, and 429 backoff.

        Mirrors :meth:`invoke` but uses the client's ``ainvoke`` so many calls can
        run concurrently under an external semaphore. Token refresh matters here:
        a long async run can outlive the JWT captured at startup.
        """
        import asyncio

        delay = 2.0
        for attempt in range(max_retries):
            try:
                return await self._ensure().ainvoke(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                msg = str(exc).lower()
                if is_auth_error(exc):
                    self._client = self._build()
                    continue
                if self._temperature is not None and "temperature" in msg:
                    self._temperature = None
                    self._client = self._build()
                    continue
                # Rate limit / transient network: exponential backoff then retry.
                transient = (
                    "429" in msg or "rate limit" in msg or "too many requests" in msg
                    or "timeout" in msg or "timed out" in msg
                    or "connection error" in msg or "connecterror" in msg
                    or "apiconnectionerror" in msg or "getaddrinfo" in msg
                    or "temporarily" in msg or "service unavailable" in msg
                    or "502" in msg or "503" in msg or "504" in msg
                )
                if transient and attempt < max_retries - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 30.0)
                    continue
                raise
        # Final attempt without swallowing the error.
        return await self._ensure().ainvoke(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        # Delegate anything we don't define (e.g. .stream, .batch) to the client.
        return getattr(self._ensure(), name)


def get_llm(config: RagConfig,
            usage_tracker: Optional[LLMUsageTracker] = None) -> Optional[Any]:
    """Construct the (token-refreshing) ChatOpenAI client, or None if unavailable.

    The returned client always exposes a ``.usage_tracker`` (an
    :class:`LLMUsageTracker`) that accumulates token usage, cost inputs and the
    OpenAI response metadata for every call made through it. Pass an existing
    tracker to share accounting across clients.
    """
    if not HAS_LLM:
        return None
    try:
        llm = _RefreshingChatLLM(config, usage_tracker=usage_tracker)
        llm._ensure()  # build now so an unavailable stack returns None up front
        return llm
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] LLM unavailable ({exc}) - continuing without generation.")
        return None


def answer_with_llm(llm: Any, prompt: str) -> str:
    """Invoke the LLM with a fully-built prompt and return the answer text."""
    response = llm.invoke(prompt)
    return str(response.content) if hasattr(response, "content") else str(response)
