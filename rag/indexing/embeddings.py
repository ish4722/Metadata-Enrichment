"""Step 4 - Embedding backend.

    >>>>>  SWAP THE EMBEDDING MODEL HERE  <<<<<

A single :class:`EmbeddingBackend` exposes ``fit_transform`` (corpus, at build
time) and ``transform`` (queries, at inference time) returning dense float32
matrices that the FAISS store consumes.

Backend selection is driven by :class:`rag.config.RagConfig`:
  * ``embedding_model="tfidf"`` -> scikit-learn TF-IDF (no network).
  * any HF id (e.g. Jina v3 / MiniLM) -> a 3-level fallback chain:
        1. transformers AutoModel   (best - supports late chunking for Jina)
        2. transformers pipeline    (same model, different loader)
        3. SentenceTransformer      (lighter)
    falling back to TF-IDF if none load.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from transformers import pipeline as hf_pipeline, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


def _try_load_jina_automodel(model_id: str):
    if not HAS_TRANSFORMERS:
        return None
    try:
        print(f"  [strategy 1] AutoModel.from_pretrained('{model_id}') ...")
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        print("  [ok] loaded via AutoModel")
        return model
    except Exception as e:
        print(f"  [x] AutoModel failed: {e}")
        return None


def _try_load_pipeline(model_id: str):
    if not HAS_TRANSFORMERS:
        return None
    try:
        print(f"  [strategy 2] pipeline('feature-extraction', '{model_id}') ...")
        pipe = hf_pipeline("feature-extraction", model=model_id, trust_remote_code=True, device="cpu")
        print("  [ok] loaded via pipeline")
        return pipe
    except Exception as e:
        print(f"  [x] pipeline failed: {e}")
        return None


def _try_load_sentence_transformer(model_id: str):
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        print(f"  [strategy 3] SentenceTransformer('{model_id}') ...")
        model = SentenceTransformer(model_id)
        print(f"  [ok] loaded SentenceTransformer: {model_id}")
        return model
    except Exception as e:
        print(f"  [x] SentenceTransformer failed: {e}")
        return None


class EmbeddingBackend:
    _MODE_AUTOMODEL = "jina-automodel"
    _MODE_PIPELINE = "jina-pipeline"
    _MODE_ST = "sentence-transformer"
    _MODE_OPENAI = "openai"
    _MODE_TFIDF = "tfidf"

    def __init__(
        self,
        preferred_model: str = "tfidf",
        fallback_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        openai_model: str = "text-embedding-3-small",
        openai_base_url: Optional[str] = None,
        openai_partition_id: Optional[str] = None,
    ) -> None:
        self._mode: Optional[str] = None
        self._automodel = None
        self._hf_pipeline = None
        self._st_model = None
        self._tfidf = None
        self._openai = None

        # Parameters needed to rebuild the live (unpicklable) handles after the
        # backend is unpickled from a cached index / meta.pkl.
        self._preferred_model = preferred_model
        self._fallback_model = fallback_model
        self._openai_model = openai_model
        self._openai_base_url = openai_base_url
        self._openai_partition_id = openai_partition_id
        self._loaded_model_id: Optional[str] = None

        print(f"Initialising EmbeddingBackend (preferred: {preferred_model}) ...")

        if preferred_model == "tfidf":
            self._init_tfidf()
            return

        if preferred_model == "openai":
            self._init_openai(openai_model, openai_base_url, openai_partition_id)
            if self._mode is None:
                print("  [fallback] OpenAI embeddings unavailable - using TF-IDF")
                self._init_tfidf()
            print(f"EmbeddingBackend ready - using [{self._mode}]")
            return

        m = _try_load_jina_automodel(preferred_model)
        if m is not None:
            self._automodel, self._mode = m, self._MODE_AUTOMODEL
            self._loaded_model_id = preferred_model

        if self._mode is None:
            p = _try_load_pipeline(preferred_model)
            if p is not None:
                self._hf_pipeline, self._mode = p, self._MODE_PIPELINE
                self._loaded_model_id = preferred_model

        if self._mode is None:
            st = _try_load_sentence_transformer(preferred_model)
            if st is not None:
                self._st_model, self._mode = st, self._MODE_ST
                self._loaded_model_id = preferred_model

        if self._mode is None and preferred_model != fallback_model:
            print(f"  Trying lighter fallback model: {fallback_model}")
            st = _try_load_sentence_transformer(fallback_model)
            if st is not None:
                self._st_model, self._mode = st, self._MODE_ST
                self._loaded_model_id = fallback_model

        if self._mode is None:
            print("  [strategy 4] Falling back to TF-IDF (no network required)")
            self._init_tfidf()

        print(f"EmbeddingBackend ready - using [{self._mode}]")

    def _init_tfidf(self) -> None:
        if not HAS_SKLEARN:
            raise RuntimeError(
                "No embedding backend available. Install scikit-learn for TF-IDF "
                "(pip install scikit-learn) or fix HuggingFace network/SSL access."
            )
        self._tfidf = TfidfVectorizer(stop_words="english")
        self._mode = self._MODE_TFIDF

    def _init_openai(self, model: str, base_url: Optional[str], partition_id: Optional[str]) -> None:
        """Load an OpenAI-compatible embeddings client (SLB internal deployment)."""
        try:
            from langchain_openai import OpenAIEmbeddings
            from sauth_token import get_token

            print(f"  [strategy openai] OpenAIEmbeddings('{model}') ...")
            token = get_token()
            if token.startswith("Bearer "):
                token = token[len("Bearer "):]
            headers = {"slb-partition-id": partition_id} if partition_id else None
            self._openai = OpenAIEmbeddings(
                model=model,
                base_url=base_url,
                api_key=token,
                default_headers=headers,
            )
            self._mode = self._MODE_OPENAI
            print("  [ok] OpenAI embeddings ready")
        except Exception as e:
            print(f"  [x] OpenAI embeddings failed: {e}")
            self._openai = None

    # ── pickle support ─────────────────────────────────────────────────────
    # Live handles that hold unpicklable resources (e.g. an OpenAI/httpx client
    # with a thread RLock, or a loaded torch model). They are dropped on pickle
    # and rebuilt lazily by :meth:`_ensure_loaded`. The fitted TF-IDF vectorizer
    # is picklable and is therefore preserved as-is.
    _NON_PICKLABLE_HANDLES = ("_openai", "_automodel", "_hf_pipeline", "_st_model")

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        for key in self._NON_PICKLABLE_HANDLES:
            state[key] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        for key in self._NON_PICKLABLE_HANDLES:
            self.__dict__.setdefault(key, None)

    def _ensure_loaded(self) -> None:
        """Rebuild a live model/client handle that was dropped during pickling.

        TF-IDF needs nothing (the vectorizer round-trips through pickle). The
        network/model-backed modes are reconstructed from the parameters stored
        at construction time so a cached index can embed queries after reload.
        """
        if self._mode == self._MODE_TFIDF:
            return
        if self._mode == self._MODE_OPENAI and self._openai is None:
            self._init_openai(
                self._openai_model, self._openai_base_url, self._openai_partition_id
            )
            if self._openai is None:
                raise RuntimeError(
                    "OpenAI embeddings client could not be reconstructed after "
                    "loading the cached index. Check network/SSL and sauth_token."
                )
        elif self._mode == self._MODE_AUTOMODEL and self._automodel is None:
            self._automodel = _try_load_jina_automodel(
                self._loaded_model_id or self._preferred_model
            )
        elif self._mode == self._MODE_PIPELINE and self._hf_pipeline is None:
            self._hf_pipeline = _try_load_pipeline(
                self._loaded_model_id or self._preferred_model
            )
        elif self._mode == self._MODE_ST and self._st_model is None:
            self._st_model = _try_load_sentence_transformer(
                self._loaded_model_id or self._preferred_model
            )

    # ── public API ───────────────────────────────────────────────────────────
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit on the corpus and return its embedding matrix (build time)."""
        self._ensure_loaded()
        if self._mode == self._MODE_TFIDF:
            return self._tfidf.fit_transform(texts).toarray().astype(np.float32)
        if self._mode == self._MODE_OPENAI:
            return np.array(self._openai.embed_documents(texts), dtype=np.float32)
        return self._encode(texts, task="retrieval.passage")

    def transform(self, texts: List[str]) -> np.ndarray:
        """Encode query texts with the already-fitted backend (inference time)."""
        self._ensure_loaded()
        if self._mode == self._MODE_TFIDF:
            return self._tfidf.transform(texts).toarray().astype(np.float32)
        if self._mode == self._MODE_OPENAI:
            return np.array(self._openai.embed_documents(texts), dtype=np.float32)
        return self._encode(texts, task="retrieval.query")

    @property
    def backend_name(self) -> str:
        return self._mode or "unknown"

    # ── internal encoders ────────────────────────────────────────────────────
    def _encode(self, texts: List[str], task: str = "retrieval.passage") -> np.ndarray:
        if self._mode == self._MODE_AUTOMODEL:
            return self._encode_automodel(texts, task)
        if self._mode == self._MODE_PIPELINE:
            return self._encode_pipeline(texts)
        if self._mode == self._MODE_ST:
            return self._encode_st(texts)
        raise RuntimeError("TF-IDF mode must use fit_transform/transform, not _encode().")

    def _encode_automodel(self, texts: List[str], task: str) -> np.ndarray:
        with torch.no_grad():
            vecs = self._automodel.encode(texts, task=task, late_chunking=False)
        if HAS_TORCH and isinstance(vecs, torch.Tensor):
            vecs = vecs.cpu().numpy()
        return np.array(vecs, dtype=np.float32)

    def _encode_pipeline(self, texts: List[str]) -> np.ndarray:
        all_vecs = []
        for text in texts:
            out = self._hf_pipeline(text, return_tensors=False)
            arr = np.array(out[0], dtype=np.float32)  # (seq_len, hidden)
            all_vecs.append(arr.mean(axis=0))
        return np.stack(all_vecs).astype(np.float32)

    def _encode_st(self, texts: List[str]) -> np.ndarray:
        vecs = self._st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return np.array(vecs, dtype=np.float32)
