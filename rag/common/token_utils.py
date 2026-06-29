"""Helpers for handling short-lived SLB auth tokens (JWTs).

The token returned by :func:`sauth_token.get_token` is a short-lived JWT. Any
client that bakes it in (the chat LLM and the OpenAI embeddings client) will
eventually fail with "JWT expired" once the token's ``exp`` passes. The helpers
here let those clients detect an auth failure and refresh the token + retry.
"""

from __future__ import annotations

# Substrings that indicate an authentication / expired-token failure (matched
# case-insensitively against the exception text).
_AUTH_ERROR_NEEDLES = (
    "jwt",
    "expired",
    "401",
    "unauthorized",
    "invalid_api_key",
    "invalid api key",
    "authentication",
    "could not validate credentials",
    "token is not valid",
)


def is_auth_error(exc: BaseException) -> bool:
    """True if ``exc`` looks like an expired/invalid auth-token error."""
    msg = str(exc).lower()
    return any(needle in msg for needle in _AUTH_ERROR_NEEDLES)


def strip_bearer(token: str) -> str:
    """Drop a leading 'Bearer ' prefix if present."""
    if token and token.startswith("Bearer "):
        return token[len("Bearer "):]
    return token
