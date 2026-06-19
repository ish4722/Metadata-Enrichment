"""Step 5 - LLM client.

    >>>>>  SWAP THE LLM HERE  <<<<<

Thin wrapper around the internal SLB ChatOpenAI deployment. ``get_llm`` returns
None when the LLM stack (langchain-openai + sauth_token) is unavailable, so the
rest of the pipeline can degrade gracefully to retrieval-only.
"""

from __future__ import annotations

from typing import Any, Optional

from rag.config import RagConfig

try:
    from langchain_openai import ChatOpenAI
    from sauth_token import get_token
    HAS_LLM = True
except Exception:
    HAS_LLM = False


def get_llm(config: RagConfig) -> Optional[Any]:
    """Construct the ChatOpenAI client, or None if the LLM stack is missing."""
    if not HAS_LLM:
        return None
    token = get_token()
    if token.startswith("Bearer "):
        token = token[len("Bearer "):]
    return ChatOpenAI(
        model=config.llm_model,
        base_url=config.llm_base_url,
        api_key=token,
        default_headers={"slb-partition-id": config.partition_id},
    )


def answer_with_llm(llm: Any, prompt: str) -> str:
    """Invoke the LLM with a fully-built prompt and return the answer text."""
    response = llm.invoke(prompt)
    return str(response.content) if hasattr(response, "content") else str(response)
