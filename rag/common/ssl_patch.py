"""Corporate-proxy / self-signed-certificate SSL patches.

Some environments sit behind a proxy with a self-signed certificate, which
breaks HuggingFace / requests / httpx downloads. Importing this module (and
calling :func:`apply_ssl_patches`) disables SSL verification so model downloads
work. It MUST run before any ``transformers`` / ``huggingface_hub`` / ``requests``
import to take effect.

Set ``DISABLE_SSL_VERIFY=0`` in the environment to turn the patch off.
"""

from __future__ import annotations

import os
import ssl


def apply_ssl_patches() -> None:
    """Disable SSL verification globally when DISABLE_SSL_VERIFY != '0'."""
    if os.environ.get("DISABLE_SSL_VERIFY", "1") != "1":
        return

    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
    os.environ.setdefault("SSL_VERIFY", "0")

    _patch_hf_hub_ssl()


def _patch_hf_hub_ssl() -> None:
    """Monkey-patch httpx clients to skip SSL verification (huggingface_hub >= 0.23)."""
    try:
        import httpx

        os.environ["HTTPX_SSL_VERIFY"] = "0"

        _orig_client_init = httpx.Client.__init__

        def _patched_client_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            _orig_client_init(self, *args, **kwargs)

        httpx.Client.__init__ = _patched_client_init

        _orig_async_init = httpx.AsyncClient.__init__

        def _patched_async_init(self, *args, **kwargs):
            kwargs.setdefault("verify", False)
            _orig_async_init(self, *args, **kwargs)

        httpx.AsyncClient.__init__ = _patched_async_init
    except Exception as e:  # pragma: no cover - best effort
        print(f"[info] Could not patch httpx SSL: {e}")
