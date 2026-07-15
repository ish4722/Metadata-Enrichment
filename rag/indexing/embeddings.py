"""Step 4 - Embedding backend.

    >>>>>  SWAP THE EMBEDDING MODEL HERE  <<<<<

A single :class:`EmbeddingBackend` exposes ``fit_transform`` (corpus, at build
time) and ``transform`` (queries, at inference time) returning dense float32
matrices that the FAISS store consumes.

Backend selection is driven by :class:`rag.config.RagConfig`:
  * ``embedding_model="tfidf"`` -> scikit-learn TF-IDF (no network).
  * ``embedding_model="jina-api"`` -> a remotely hosted jina-embeddings-v3
        server (Cloud Run) reached over HTTP. Passages are embedded with LATE
        CHUNKING enabled (the server encodes each window in one pass and pools
        per chunk, so every chunk vector carries whole-window context); queries
        are embedded normally. This is the recommended dense backend - it needs
        no local weights, GPU, or the flash custom code.
  * any HF id (e.g. Jina v3 / MiniLM) -> a fallback chain:
        1. transformers AutoModel   (best - supports late chunking for Jina;
                                     needs the flash custom code, i.e. network)
        2. ONNX (Jina only)         (offline; uses local jina-v3/onnx/model.onnx)
        3. transformers pipeline    (non-Jina only; wrong arch for Jina LoRA)
        4. SentenceTransformer      (lighter)
    falling back to TF-IDF (with a loud warning) if none load.

For a local Jina v3 folder the AutoModel/SentenceTransformer paths require the
``xlm-roberta-flash-implementation`` custom modules (block.py / mha.py / ...),
which cannot be fetched offline. The ONNX strategy avoids them entirely.
"""

from __future__ import annotations

import json
import os
from typing import List, Optional

import numpy as np

try:
    import requests  # preferred HTTP client for the hosted Jina API
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

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

try:
    import onnxruntime as ort  # noqa: F401
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

# Default LoRA adapter order shipped in jina-embeddings-v3 config.json. Used to
# map a retrieval "task" name to the scalar task_id the ONNX graph expects.
_JINA_DEFAULT_TASKS = [
    "retrieval.query",
    "retrieval.passage",
    "separation",
    "classification",
    "text-matching",
]


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


class _JinaOnnxModel:
    """Offline Jina v3 encoder backed by the bundled ONNX graph.

    Uses ``<model_dir>/onnx/model.onnx`` plus the local tokenizer, so it needs
    none of the ``xlm-roberta-flash-implementation`` custom torch code that the
    AutoModel/SentenceTransformer paths require (and which cannot be fetched
    offline).
    """

    def __init__(
        self,
        model_dir: str,
        max_length: int = 8192,
        batch_size: int = 16,
        use_gpu: str = "auto",
    ):
        from transformers import AutoTokenizer

        self.model_dir = model_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)

        # Decide whether to run on the GPU. "auto" uses CUDA when onnxruntime
        # exposes a CUDA provider; "cuda"/"gpu" forces it; anything else = CPU.
        available = ort.get_available_providers()
        want_gpu = (
            str(use_gpu).lower() in {"auto", "cuda", "gpu", "1", "true"}
            and "CUDAExecutionProvider" in available
        )

        # Help onnxruntime-gpu find CUDA/cuDNN DLLs shipped as nvidia-* pip
        # wheels (no system-wide CUDA install needed). Safe no-op on CPU builds.
        if want_gpu and hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass

        onnx_dir = os.path.join(model_dir, "onnx")
        fp16_path = os.path.join(onnx_dir, "model_fp16.onnx")
        fp32_path = os.path.join(onnx_dir, "model.onnx")

        self.session = None
        # Attempt GPU first (fp16 graph: ~half the VRAM, faster on a 4 GB laptop
        # card). If CUDA cannot actually initialise (missing runtime DLLs, OOM,
        # or a graph-fusion failure) we fall back to a CPU session so the
        # pipeline keeps using real Jina embeddings instead of dropping to
        # TF-IDF.
        if want_gpu:
            gpu_weights = fp16_path if os.path.isfile(fp16_path) else fp32_path
            try:
                self.session = ort.InferenceSession(
                    gpu_weights,
                    ort.SessionOptions(),
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                )
                if "CUDAExecutionProvider" not in self.session.get_providers():
                    self.session = None  # CUDA silently rejected -> use CPU path
            except Exception as exc:
                print(f"  [onnx] CUDA init failed ({exc}); falling back to CPU")
                self.session = None

        if self.session is None:
            self.session = ort.InferenceSession(
                fp32_path,
                ort.SessionOptions(),
                providers=["CPUExecutionProvider"],
            )

        active = self.session.get_providers()
        on_cuda = "CUDAExecutionProvider" in active
        print(
            f"  [onnx] running on {'GPU/CUDA' if on_cuda else 'CPU'} "
            f"(providers={active})"
        )

        # GPU inference of a large transformer on 4 GB VRAM needs a smaller
        # batch than CPU to avoid out-of-memory; halve it when on CUDA.
        self.batch_size = max(1, batch_size // 2) if on_cuda else batch_size
        self._out_names = [o.name for o in self.session.get_outputs()]

        tasks = list(_JINA_DEFAULT_TASKS)
        cfg_path = os.path.join(model_dir, "config.json")
        try:
            with open(cfg_path, encoding="utf-8") as fh:
                adaptations = json.load(fh).get("lora_adaptations")
            if isinstance(adaptations, list) and adaptations:
                tasks = adaptations
        except Exception:
            pass
        self._task_to_id = {name: idx for idx, name in enumerate(tasks)}

    def _task_id(self, task: str) -> int:
        return self._task_to_id.get(task, self._task_to_id.get("retrieval.passage", 1))

    def encode(self, texts: List[str], task: str = "retrieval.passage") -> np.ndarray:
        task_id = np.array(self._task_id(task), dtype=np.int64)
        input_names = {i.name for i in self.session.get_inputs()}
        vecs: List[np.ndarray] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [str(t) for t in texts[start : start + self.batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="np",
            )
            attention_mask = enc["attention_mask"].astype(np.int64)
            feeds = {
                "input_ids": enc["input_ids"].astype(np.int64),
                "attention_mask": attention_mask,
            }
            if "task_id" in input_names:
                feeds["task_id"] = task_id
            outputs = self.session.run(self._out_names, feeds)

            # Jina v3 uses masked MEAN pooling over the token embeddings
            # (outputs[0], shape [batch, seq, hidden]) followed by L2-normalise.
            # The model's secondary pooled output is NOT the retrieval embedding
            # and yields wrong similarities, so we always pool outputs[0].
            token_emb = np.asarray(outputs[0], dtype=np.float32)
            mask = attention_mask[..., None].astype(np.float32)
            pooled = (token_emb * mask).sum(axis=1) / np.clip(mask.sum(axis=1), 1e-9, None)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            vecs.append((pooled / np.clip(norms, 1e-9, None)).astype(np.float32))
        return np.concatenate(vecs, axis=0).astype(np.float32)


def _try_load_jina_onnx(model_id: str):
    """Load the bundled ONNX Jina model from a local folder (offline-friendly)."""
    if not HAS_ONNXRUNTIME:
        return None
    if not (isinstance(model_id, str) and os.path.isdir(model_id)):
        return None
    onnx_path = os.path.join(model_id, "onnx", "model.onnx")
    if not os.path.isfile(onnx_path):
        return None
    try:
        print(f"  [strategy onnx] loading bundled ONNX weights from '{os.path.join(model_id, 'onnx')}' ...")
        model = _JinaOnnxModel(model_id)
        print("  [ok] loaded Jina via ONNX (offline)")
        return model
    except Exception as e:
        print(f"  [x] ONNX load failed: {e}")
        return None


class _JinaApiModel:
    """Client for a remotely hosted jina-embeddings-v3 server (``POST /embed``).

    The server owns the weights, GPU and (for late chunking) the sliding-window
    logic: given a list of ``texts`` it returns one L2-normalised vector per
    text. When ``late_chunking`` is on the server encodes each ~``window_tokens``
    window in a single pass and mean-pools per chunk, so each chunk vector
    carries whole-window context - exactly the late-chunking recipe, but without
    needing the flash custom code or a local model locally.
    """

    # Texts sent per HTTP request. Kept generous so a whole document usually
    # goes in one call (preserving the server's cross-chunk windows); very large
    # documents are split across requests, which only loses context at the seam.
    _HTTP_BATCH = 512
    _RETRIES = 4
    _TIMEOUT = 180

    def __init__(
        self,
        base_url: str,
        late_chunking: bool = True,
        window_tokens: int = 3072,
    ):
        self.base_url = base_url.rstrip("/")
        self.embed_url = f"{self.base_url}/embed"
        self.late_chunking = bool(late_chunking)
        self.window_tokens = int(window_tokens)
        self._dim: Optional[int] = None

    def _post(self, payload: dict) -> dict:
        """POST a JSON payload, retrying on cold-start / transient failures."""
        import time

        last_exc: Optional[Exception] = None
        for attempt in range(self._RETRIES):
            try:
                if HAS_REQUESTS:
                    resp = requests.post(self.embed_url, json=payload, timeout=self._TIMEOUT)
                    if resp.status_code >= 500:
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                    resp.raise_for_status()
                    return resp.json()
                # stdlib fallback so the backend works without `requests`.
                import urllib.request

                data = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.embed_url, data=data,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=self._TIMEOUT) as fh:
                    return json.loads(fh.read().decode("utf-8"))
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if attempt < self._RETRIES - 1:
                    wait = 2 ** attempt
                    print(f"  [jina-api] request failed ({exc}); retry in {wait}s ...")
                    time.sleep(wait)
        raise RuntimeError(f"Jina API request to {self.embed_url} failed: {last_exc}")

    def encode(
        self,
        texts: List[str],
        task: str = "retrieval.passage",
        late_chunking: Optional[bool] = None,
    ) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim or 1024), dtype=np.float32)
        # Late chunking only makes sense for multi-chunk passages, never for a
        # standalone query, so callers pass late_chunking=False for queries.
        use_late = self.late_chunking if late_chunking is None else bool(late_chunking)

        vecs: List[np.ndarray] = []
        for start in range(0, len(texts), self._HTTP_BATCH):
            batch = [str(t) for t in texts[start : start + self._HTTP_BATCH]]
            payload = {
                "texts": batch,
                "task": task,
                "late_chunking": use_late,
                "window_tokens": self.window_tokens,
            }
            body = self._post(payload)
            embeddings = body.get("embeddings")
            if not isinstance(embeddings, list) or len(embeddings) != len(batch):
                raise RuntimeError(
                    "Jina API returned an unexpected response "
                    f"(got {len(embeddings) if isinstance(embeddings, list) else 'no'} "
                    f"embeddings for {len(batch)} texts)."
                )
            arr = np.asarray(embeddings, dtype=np.float32)
            self._dim = arr.shape[1]
            vecs.append(arr)
        return np.concatenate(vecs, axis=0).astype(np.float32)


def _try_load_jina_api(base_url: str, late_chunking: bool, window_tokens: int):
    """Probe the hosted Jina server's /health and return a client (or None)."""
    if not base_url:
        return None
    try:
        print(f"  [strategy jina-api] connecting to '{base_url}' (late_chunking={late_chunking}) ...")
        model = _JinaApiModel(base_url, late_chunking=late_chunking, window_tokens=window_tokens)
        # Health probe wakes a scaled-to-zero Cloud Run instance and confirms
        # the endpoint is a real Jina v3 server before we commit to it.
        health_url = f"{model.base_url}/health"
        if HAS_REQUESTS:
            info = requests.get(health_url, timeout=_JinaApiModel._TIMEOUT).json()
        else:
            import urllib.request

            with urllib.request.urlopen(health_url, timeout=_JinaApiModel._TIMEOUT) as fh:
                info = json.loads(fh.read().decode("utf-8"))
        print(f"  [ok] hosted Jina reachable: {info}")
        return model
    except Exception as e:  # noqa: BLE001
        print(f"  [x] hosted Jina API unavailable: {e}")
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
    _MODE_ONNX = "jina-onnx"
    _MODE_JINA_API = "jina-api"
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
        openai_dimensions: Optional[int] = None,
        jina_api_url: Optional[str] = None,
        jina_api_late_chunking: bool = True,
        jina_api_window_tokens: int = 3072,
    ) -> None:
        self._mode: Optional[str] = None
        self._automodel = None
        self._onnx_model = None
        self._jina_api = None
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
        self._openai_dimensions = openai_dimensions
        self._jina_api_url = jina_api_url
        self._jina_api_late_chunking = bool(jina_api_late_chunking)
        self._jina_api_window_tokens = int(jina_api_window_tokens)
        self._loaded_model_id: Optional[str] = None

        print(f"Initialising EmbeddingBackend (preferred: {preferred_model}) ...")

        if preferred_model == "tfidf":
            self._init_tfidf()
            return

        # Remotely hosted Jina v3 server (late chunking done server-side).
        if preferred_model == "jina-api":
            self._jina_api = _try_load_jina_api(
                jina_api_url, jina_api_late_chunking, jina_api_window_tokens
            )
            if self._jina_api is not None:
                self._mode = self._MODE_JINA_API
                self._loaded_model_id = preferred_model
            else:
                print("  [fallback] hosted Jina API unavailable - using TF-IDF")
                self._init_tfidf()
            print(f"EmbeddingBackend ready - using [{self._mode}]")
            return

        if preferred_model == "openai":
            self._init_openai(openai_model, openai_base_url, openai_partition_id,
                              openai_dimensions)
            if self._mode is None:
                print("  [fallback] OpenAI embeddings unavailable - using TF-IDF")
                self._init_tfidf()
            print(f"EmbeddingBackend ready - using [{self._mode}]")
            return

        is_jina = "jina" in str(preferred_model).lower()

        # Offline-first for Jina: load the bundled ONNX graph from the local
        # weights folder. This needs neither the flash custom code nor any
        # network, so it is tried before AutoModel (which would try to fetch the
        # xlm-roberta-flash-implementation modules from HuggingFace).
        if is_jina:
            onnx_model = _try_load_jina_onnx(preferred_model)
            if onnx_model is not None:
                self._onnx_model, self._mode = onnx_model, self._MODE_ONNX
                self._loaded_model_id = preferred_model

        if self._mode is None:
            m = _try_load_jina_automodel(preferred_model)
            if m is not None:
                self._automodel, self._mode = m, self._MODE_AUTOMODEL
                self._loaded_model_id = preferred_model

        # NOTE: the generic feature-extraction pipeline cannot correctly load a
        # Jina v3 LoRA/flash checkpoint - it instantiates a vanilla
        # XLMRobertaModel with RANDOM weights (every real weight shows up as
        # "unexpected"), which silently yields garbage embeddings. So we only
        # use the pipeline strategy for non-Jina models.
        if self._mode is None and not is_jina:
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
            print(
                "  " + "!" * 70 + "\n"
                f"  [WARNING] Could not load requested embedding model "
                f"'{preferred_model}'.\n"
                "  Falling back to TF-IDF. The vectors are NOT semantic Jina "
                "embeddings.\n"
                "  For Jina v3 you need the 'xlm-roberta-flash-implementation' "
                "custom code\n"
                "  (block.py / mha.py / mlp.py / embedding.py / xlm_padding.py). "
                "Either:\n"
                "    * run once WITH network so transformers can fetch them, or\n"
                "    * use the bundled ONNX weights (pip install onnxruntime).\n"
                "  " + "!" * 70
            )
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

    def _init_openai(
        self,
        model: str,
        base_url: Optional[str],
        partition_id: Optional[str],
        dimensions: Optional[int] = None,
    ) -> None:
        """Load an OpenAI-compatible embeddings client (SLB internal deployment).

        ``dimensions`` requests a truncated (Matryoshka) output width from the
        embeddings endpoint; when None the model returns its native full width.
        """
        try:
            from langchain_openai import OpenAIEmbeddings
            from rag.common.sauth_token import get_token

            dim_note = f", dimensions={dimensions}" if dimensions else ""
            print(f"  [strategy openai] OpenAIEmbeddings('{model}'{dim_note}) ...")
            token = get_token()
            if token.startswith("Bearer "):
                token = token[len("Bearer "):]
            headers = {"slb-partition-id": partition_id} if partition_id else None
            kwargs = dict(
                model=model,
                base_url=base_url,
                api_key=token,
                default_headers=headers,
            )
            if dimensions:
                kwargs["dimensions"] = int(dimensions)
            self._openai = OpenAIEmbeddings(**kwargs)
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
    _NON_PICKLABLE_HANDLES = (
        "_openai",
        "_automodel",
        "_onnx_model",
        "_jina_api",
        "_hf_pipeline",
        "_st_model",
    )

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
                self._openai_model, self._openai_base_url, self._openai_partition_id,
                getattr(self, "_openai_dimensions", None),
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
        elif self._mode == self._MODE_ONNX and self._onnx_model is None:
            self._onnx_model = _try_load_jina_onnx(
                self._loaded_model_id or self._preferred_model
            )
            if self._onnx_model is None:
                raise RuntimeError(
                    "Jina ONNX model could not be reconstructed after loading the "
                    "cached index. Ensure onnxruntime is installed and the local "
                    "jina-v3/onnx/model.onnx is present."
                )
        elif self._mode == self._MODE_JINA_API and self._jina_api is None:
            self._jina_api = _try_load_jina_api(
                self._jina_api_url,
                self._jina_api_late_chunking,
                self._jina_api_window_tokens,
            )
            if self._jina_api is None:
                raise RuntimeError(
                    "Hosted Jina API client could not be reconstructed after "
                    "loading the cached index. Check JINA_API_URL / network access."
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
            return self._embed_openai(texts)
        return self._encode(texts, task="retrieval.passage")

    def transform(self, texts: List[str]) -> np.ndarray:
        """Encode query texts with the already-fitted backend (inference time)."""
        self._ensure_loaded()
        if self._mode == self._MODE_TFIDF:
            return self._tfidf.transform(texts).toarray().astype(np.float32)
        if self._mode == self._MODE_OPENAI:
            return self._embed_openai(texts)
        return self._encode(texts, task="retrieval.query")

    def _embed_openai(self, texts: List[str]) -> np.ndarray:
        """Embed via the OpenAI client, refreshing the token once on auth failure.

        The SLB token is a short-lived JWT; if it expires mid-build the embed call
        fails with "JWT expired". We then rebuild the client (which fetches a fresh
        token) and retry once so long builds don't die partway through.
        """
        try:
            return np.array(self._openai.embed_documents(texts), dtype=np.float32)
        except Exception as exc:  # noqa: BLE001
            from rag.common.token_utils import is_auth_error

            if not is_auth_error(exc):
                raise
            print("[auth] embedding token expired - refreshing and retrying once ...")
            self._init_openai(
                self._openai_model, self._openai_base_url, self._openai_partition_id,
                getattr(self, "_openai_dimensions", None),
            )
            if self._openai is None:
                raise
            return np.array(self._openai.embed_documents(texts), dtype=np.float32)

    @property
    def backend_name(self) -> str:
        return self._mode or "unknown"

    # ── internal encoders ────────────────────────────────────────────────────
    def _encode(self, texts: List[str], task: str = "retrieval.passage") -> np.ndarray:
        if self._mode == self._MODE_AUTOMODEL:
            return self._encode_automodel(texts, task)
        if self._mode == self._MODE_ONNX:
            return self._onnx_model.encode(texts, task=task)
        if self._mode == self._MODE_JINA_API:
            # Late chunking applies to multi-chunk passages only; a lone query
            # has no cross-chunk context to pool, so disable it for queries.
            late = self._jina_api_late_chunking and task != "retrieval.query"
            return self._jina_api.encode(texts, task=task, late_chunking=late)
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
