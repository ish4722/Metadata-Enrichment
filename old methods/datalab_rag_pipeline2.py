"""
Convert extracted PDF JSON into metadata-rich chunks and apply a RAG pipeline.

Improvements over v1:
  - Jina v3 embeddings with late chunking (falls back to TF-IDF if unavailable)
  - BM25 + dense hybrid retrieval with Reciprocal Rank Fusion (RRF)
  - Cross-encoder reranking before LLM call
  - Full-page context sent to LLM (not just raw chunks)
  - Metadata filtering (section / page range) before retrieval
  - Cross-page section merging for multi-page tables / sections
  - Fixed argparse bug on the query sub-command
  - Persistent chat history in interactive mode
  - Graceful fallbacks when optional deps are missing

Usage examples:
  python datalab_rag_pipeline.py build \\
    --input "datalab-output-PR3216.pdf.json" \\
    --chunks-output useful_chunks.jsonl \\
    --index-output rag_index.pkl

  python datalab_rag_pipeline.py query \\
    --index-output rag_index.pkl \\
    --query "What is the operator name?" \\
    --top-k 10

  python datalab_rag_pipeline.py query \\
    --index-output rag_index.pkl \\
    --query "What is the well depth?" \\
    --section-filter "Well Summary" \\
    --page-range 1 5

  # Interactive (no args):
  python datalab_rag_pipeline.py
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pickle
import re
import ssl
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import rag_evaluation

# ── SSL fix for corporate proxy / self-signed cert environments ───────────────
# Applied before any HuggingFace / requests / httpx import.

# Disable SSL verification globally for environments with self-signed certs.
# Set DISABLE_SSL_VERIFY=0 in your environment to turn this off if not needed.
if os.environ.get("DISABLE_SSL_VERIFY", "1") == "1":
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ.setdefault("CURL_CA_BUNDLE", "")
    os.environ.setdefault("REQUESTS_CA_BUNDLE", "")
    os.environ.setdefault("HF_HUB_DISABLE_SSL_VERIFICATION", "1")
    # httpx (used by huggingface_hub ≥ 0.23) also needs this
    os.environ.setdefault("SSL_VERIFY", "0")

# ── optional heavy deps ────────────────────────────────────────────────────────
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Try to import transformers pipeline + AutoModel
try:
    from transformers import pipeline as hf_pipeline, AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from sentence_transformers import CrossEncoder, SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

try:
    from langchain_openai import ChatOpenAI
    from sauth_token import get_token          # internal SLB helper
    HAS_LLM = True
except Exception:
    HAS_LLM = False

# ── constants ──────────────────────────────────────────────────────────────────
JINA_MODEL_ID   = "jinaai/jina-embeddings-v3"
# Fallback: lighter sentence-transformers model (also on HF, same SSL caveat)
FALLBACK_ST_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL     = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_TOP_K      = 10
RERANK_TOP_K       = 10      # final passages sent to LLM after reranking (full pages)
HYBRID_ALPHA       = 0.5     # weight: 0 = pure BM25, 1 = pure dense

# ── HuggingFace SSL patch for httpx (huggingface_hub ≥ 0.23 uses httpx) ───────
def _patch_hf_hub_ssl() -> None:
    """
    Monkey-patch huggingface_hub's httpx client to skip SSL verification.
    Required in corporate networks with self-signed certificates.
    """
    try:
        import huggingface_hub.utils._http as _hf_http
        import httpx

        _orig_build = getattr(_hf_http, "_build_hf_headers", None)

        # Patch the module-level http client creation
        original_client = getattr(_hf_http, "build_hf_headers", None)

        # Simpler: override the environment so httpx picks it up
        os.environ["HTTPX_SSL_VERIFY"] = "0"

        # Patch httpx.Client and httpx.AsyncClient defaults
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

    except Exception as e:
        print(f"[info] Could not patch httpx SSL: {e}")

if os.environ.get("DISABLE_SSL_VERIFY", "1") == "1":
    _patch_hf_hub_ssl()


# ══════════════════════════════════════════════════════════════════════════════
# Data model
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Chunk:
    id:           str
    source:       str
    page:         int
    section_path: str
    block_type:   str
    text:         str
    metadata:     Dict[str, Any] = field(default_factory=dict)


# ══════════════════════════════════════════════════════════════════════════════
# HTML → plain text
# ══════════════════════════════════════════════════════════════════════════════

def html_to_text(raw_html: str) -> str:
    raw_html = html.unescape(raw_html or "")
    raw_html = re.sub(r"<br\s*/?>",               "\n",  raw_html, flags=re.IGNORECASE)
    raw_html = re.sub(r"<style.*?>.*?</style>",   "",    raw_html, flags=re.IGNORECASE | re.DOTALL)
    raw_html = re.sub(r"<script.*?>.*?</script>", "",    raw_html, flags=re.IGNORECASE | re.DOTALL)
    raw_html = re.sub(r"<.*?>",                   "",    raw_html)
    raw_html = re.sub(r"\s+",                     " ",   raw_html)
    return raw_html.strip()


# ══════════════════════════════════════════════════════════════════════════════
# JSON → Chunk extraction
# ══════════════════════════════════════════════════════════════════════════════

def normalize_section_path(section_titles: List[str]) -> str:
    return " > ".join(t.strip() for t in section_titles if t and t.strip())


def flatten_blocks(
    node:           Dict[str, Any],
    page:           int            = -1,
    section_titles: List[str]      = None,
) -> List[Chunk]:
    if section_titles is None:
        section_titles = []

    blocks:         List[Chunk] = []
    node_page       = node.get("page", page)
    node_block_type = node.get("block_type", "Unknown")
    node_id         = node.get("id", "")
    node_html       = node.get("html", "")

    if node_block_type == "SectionHeader" and node_html:
        heading_text = html_to_text(node_html)
        if heading_text:
            section_titles = section_titles + [heading_text]

    if node_block_type in {"Text", "SectionHeader", "Table", "PageHeader"} and node_html:
        text = html_to_text(node_html)
        if text:
            blocks.append(
                Chunk(
                    id           = node_id,
                    source       = "",
                    page         = int(node_page) if isinstance(node_page, int) else -1,
                    section_path = normalize_section_path(section_titles),
                    block_type   = node_block_type,
                    text         = text,
                    metadata     = {
                        "block_id":          node_id,
                        "block_type":        node_block_type,
                        "page":              int(node_page) if isinstance(node_page, int) else -1,
                        "bbox":              node.get("bbox"),
                        "polygon":           node.get("polygon"),
                        "section_hierarchy": node.get("section_hierarchy"),
                        "raw_html":          node_html,
                    },
                )
            )

    for child in node.get("children", []):
        blocks.extend(flatten_blocks(child, page=node_page, section_titles=section_titles))

    return blocks


def load_extracted_json(path: str) -> List[Chunk]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    chunks: List[Chunk] = []
    for page_node in data.get("children", []):
        source      = Path(path).name
        page_index  = page_node.get("page", -1)
        for chunk in flatten_blocks(page_node, page=page_index):
            chunk.source = source
            chunks.append(chunk)
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Merging: within page + across pages
# ══════════════════════════════════════════════════════════════════════════════

def merge_blocks_by_section(chunks: List[Chunk]) -> List[Chunk]:
    """Merge blocks that share the same (source, page, section_path)."""
    grouped: Dict[Tuple[str, int, str], List[Chunk]] = {}
    for chunk in chunks:
        key = (chunk.source, chunk.page, chunk.section_path or chunk.block_type)
        grouped.setdefault(key, []).append(chunk)

    merged: List[Chunk] = []
    for idx, (key, group) in enumerate(grouped.items()):
        if len(group) == 1:
            merged.append(group[0])
            continue

        merged_text = "\n\n".join(c.text for c in group if c.text.strip()).strip()
        merged.append(
            Chunk(
                id           = f"{group[0].source}::{group[0].page}::{idx}",
                source       = group[0].source,
                page         = group[0].page,
                section_path = group[0].section_path,
                block_type   = group[0].block_type,
                text         = merged_text,
                metadata     = {
                    "source":          group[0].source,
                    "page":            group[0].page,
                    "section_path":    group[0].section_path,
                    "block_type":      group[0].block_type,
                    "block_ids":       [c.id for c in group],
                    "raw_html_blocks": [c.metadata.get("raw_html") for c in group],
                },
            )
        )
    return merged


def merge_across_pages(chunks: List[Chunk]) -> List[Chunk]:
    """
    Merge consecutive chunks that share the same section_path across adjacent
    pages — useful for multi-page tables or flowing paragraphs in O&G reports.
    """
    if not chunks:
        return chunks

    result: List[Chunk] = []
    # Sort by source, section_path, page so adjacency is meaningful
    sorted_chunks = sorted(chunks, key=lambda c: (c.source, c.section_path, c.page))

    prev = sorted_chunks[0]
    for curr in sorted_chunks[1:]:
        same_source  = curr.source       == prev.source
        same_section = curr.section_path == prev.section_path and curr.section_path != ""
        adjacent     = curr.page         == prev.page + 1

        if same_source and same_section and adjacent:
            # Extend prev across the page boundary
            prev = Chunk(
                id           = prev.id,
                source       = prev.source,
                page         = prev.page,           # keep the first page number
                section_path = prev.section_path,
                block_type   = prev.block_type,
                text         = prev.text + "\n\n" + curr.text,
                metadata     = {
                    **prev.metadata,
                    "merged_pages": prev.metadata.get("merged_pages", [prev.page]) + [curr.page],
                },
            )
        else:
            result.append(prev)
            prev = curr

    result.append(prev)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Sub-chunking for retrieval units
# ══════════════════════════════════════════════════════════════════════════════

def split_text_into_chunks(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    result, start = [], 0
    while start < len(words):
        end    = min(len(words), start + chunk_size)
        result.append(" ".join(words[start:end]).strip())
        if end == len(words):
            break
        start += chunk_size - overlap
    return result


def build_useful_chunks(
    chunks:     List[Chunk],
    chunk_size: int = 800,
    overlap:    int = 200,
) -> List[Chunk]:
    useful: List[Chunk] = []
    for ctr, chunk in enumerate(chunks):
        for part in split_text_into_chunks(chunk.text, chunk_size, overlap):
            if not part.strip():
                continue
            cid = f"chunk_{ctr}_{len(useful)}"
            useful.append(
                Chunk(
                    id           = cid,
                    source       = chunk.source,
                    page         = chunk.page,
                    section_path = chunk.section_path,
                    block_type   = chunk.block_type,
                    text         = part,
                    metadata     = {
                        **chunk.metadata,
                        "chunk_id":    cid,
                        "text_length": len(part),
                    },
                )
            )
    return useful


# ══════════════════════════════════════════════════════════════════════════════
# Serialisation helpers
# ══════════════════════════════════════════════════════════════════════════════

def save_chunks_jsonl(chunks: List[Chunk], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(
                {"id": c.id, "source": c.source, "page": c.page,
                 "section_path": c.section_path, "block_type": c.block_type,
                 "text": c.text, "metadata": c.metadata},
                ensure_ascii=False,
            ) + "\n")


def load_chunks_jsonl(path: str) -> List[Chunk]:
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            chunks.append(Chunk(
                id=d["id"], source=d["source"], page=d["page"],
                section_path=d.get("section_path", ""),
                block_type=d.get("block_type", ""),
                text=d["text"], metadata=d.get("metadata", {}),
            ))
    return chunks


# ══════════════════════════════════════════════════════════════════════════════
# Embeddings
# Priority: Jina v3 (AutoModel) → SentenceTransformer fallback → TF-IDF
#
# Corporate SSL fix: SSL verification is patched at module load time above.
# If you have a company CA bundle, point REQUESTS_CA_BUNDLE at it instead.
# ══════════════════════════════════════════════════════════════════════════════

def _try_load_jina_automodel(model_id: str):
    """
    Strategy 1: load Jina v3 via transformers AutoModel.
    This is the recommended path and supports true late chunking.
    Returns model or None.
    """
    if not HAS_TRANSFORMERS:
        return None
    try:
        print(f"  [strategy 1] AutoModel.from_pretrained('{model_id}') …")
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        model.eval()
        print("  ✓ Jina v3 loaded via AutoModel")
        return model
    except Exception as e:
        print(f"  ✗ AutoModel failed: {e}")
        return None


def _try_load_jina_pipeline(model_id: str):
    """
    Strategy 2: load via transformers pipeline("feature-extraction").
    Slightly more robust in some environments.
    Returns pipeline or None.
    """
    if not HAS_TRANSFORMERS:
        return None
    try:
        print(f"  [strategy 2] pipeline('feature-extraction', model='{model_id}') …")
        pipe = hf_pipeline(
            "feature-extraction",
            model=model_id,
            trust_remote_code=True,
            device="cpu",
        )
        print("  ✓ Jina v3 loaded via pipeline")
        return pipe
    except Exception as e:
        print(f"  ✗ pipeline failed: {e}")
        return None


def _try_load_sentence_transformer(model_id: str):
    """
    Strategy 3: lighter SentenceTransformer model.
    Falls back to all-MiniLM-L6-v2 if Jina is completely unreachable.
    Returns SentenceTransformer or None.
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    try:
        print(f"  [strategy 3] SentenceTransformer('{model_id}') …")
        model = SentenceTransformer(model_id)
        print(f"  ✓ SentenceTransformer loaded: {model_id}")
        return model
    except Exception as e:
        print(f"  ✗ SentenceTransformer failed: {e}")
        return None


class EmbeddingBackend:
    """
    Unified embedding interface with a 4-level fallback chain:
      1. Jina v3  via transformers AutoModel  (best — supports late chunking)
      2. Jina v3  via transformers pipeline   (same model, different loader)
      3. all-MiniLM-L6-v2 via SentenceTransformer  (fast, no trust_remote_code)
      4. TF-IDF                               (no internet required at all)

    SSL verification is disabled globally for corporate proxy environments.
    Set  DISABLE_SSL_VERIFY=0  to turn off the patch.
    """

    _MODE_AUTOMODEL   = "jina-automodel"
    _MODE_PIPELINE    = "jina-pipeline"
    _MODE_ST          = "sentence-transformer"
    _MODE_TFIDF       = "tfidf"

    def __init__(self, preferred_model: str = JINA_MODEL_ID) -> None:
        self._mode        = None
        self._automodel   = None   # transformers AutoModel
        self._hf_pipeline = None   # transformers pipeline
        self._st_model    = None   # SentenceTransformer
        self._tfidf       = None   # TfidfVectorizer

        print(f"Initialising EmbeddingBackend (preferred: {preferred_model}) …")

        # ── strategy 1: Jina via AutoModel ───────────────────────────────────
        m = _try_load_jina_automodel(preferred_model)
        if m is not None:
            self._automodel = m
            self._mode      = self._MODE_AUTOMODEL
        
        # ── strategy 2: Jina via HF pipeline ─────────────────────────────────
        if self._mode is None:
            p = _try_load_jina_pipeline(preferred_model)
            if p is not None:
                self._hf_pipeline = p
                self._mode        = self._MODE_PIPELINE

        # ── strategy 3a: Jina via SentenceTransformer ────────────────────────
        if self._mode is None:
            st = _try_load_sentence_transformer(preferred_model)
            if st is not None:
                self._st_model = st
                self._mode     = self._MODE_ST

        # ── strategy 3b: fallback ST model (lighter, more likely cached) ─────
        if self._mode is None and preferred_model != FALLBACK_ST_MODEL:
            print(f"  Trying lighter fallback model: {FALLBACK_ST_MODEL}")
            st = _try_load_sentence_transformer(FALLBACK_ST_MODEL)
            if st is not None:
                self._st_model = st
                self._mode     = self._MODE_ST

        # ── strategy 4: TF-IDF (no network needed) ───────────────────────────
        if self._mode is None:
            print("  [strategy 4] Falling back to TF-IDF (no network required)")
            if not HAS_SKLEARN:
                raise RuntimeError(
                    "No embedding backend available. "
                    "Install scikit-learn (pip install scikit-learn) for TF-IDF fallback, "
                    "or fix network/SSL access to HuggingFace."
                )
            self._tfidf = TfidfVectorizer(stop_words="english")
            self._mode  = self._MODE_TFIDF

        print(f"EmbeddingBackend ready — using [{self._mode}]")

    # ── public API ─────────────────────────────────────────────────────────────

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit on corpus texts and return embedding matrix. Call once at build time."""
        if self._mode == self._MODE_TFIDF:
            return self._tfidf.fit_transform(texts).toarray()
        return self._encode(texts, task="retrieval.passage")

    def transform(self, texts: List[str]) -> np.ndarray:
        """Encode query texts using already-fitted backend."""
        if self._mode == self._MODE_TFIDF:
            return self._tfidf.transform(texts).toarray()
        return self._encode(texts, task="retrieval.query")

    def encode_with_late_chunking(
        self,
        full_text:   str,
        chunk_texts: List[str],
    ) -> np.ndarray:
        """
        True late chunking (Jina AutoModel only).
        Embeds the full document with late_chunking=True so every chunk
        vector carries whole-document context, then maps model segments
        back to our chunks by nearest-cosine.

        Other backends fall through to independent chunk encoding.
        """
        if self._mode == self._MODE_AUTOMODEL:
            try:
                with torch.no_grad():
                    late_vecs_raw = self._automodel.encode(
                        [full_text],
                        task="retrieval.passage",
                        late_chunking=True,
                    )
                # late_vecs_raw: list of arrays, one per model-detected segment
                if isinstance(late_vecs_raw, (list, np.ndarray)) and len(late_vecs_raw):
                    late_vecs = np.array(
                        late_vecs_raw[0] if isinstance(late_vecs_raw[0], list)
                        else late_vecs_raw,
                        dtype=np.float32,
                    )
                    if late_vecs.ndim == 1:
                        late_vecs = late_vecs.reshape(1, -1)

                    # Independently encode chunks to use as query vectors for matching
                    chunk_vecs = self._encode(chunk_texts, task="retrieval.passage")
                    sims       = cosine_similarity(chunk_vecs, late_vecs)  # (n_chunks, n_segs)
                    best_seg   = sims.argmax(axis=1)
                    return late_vecs[best_seg]
            except Exception as e:
                print(f"[warn] Late chunking failed ({e}), using standard encoding.")

        # All other modes: standard independent chunk encoding
        return self._encode(chunk_texts, task="retrieval.passage")

    @property
    def backend_name(self) -> str:
        return self._mode or "unknown"

    # ── internal encode dispatcher ────────────────────────────────────────────

    def _encode(self, texts: List[str], task: str = "retrieval.passage") -> np.ndarray:
        if self._mode == self._MODE_AUTOMODEL:
            return self._encode_automodel(texts, task)
        elif self._mode == self._MODE_PIPELINE:
            return self._encode_pipeline(texts)
        elif self._mode == self._MODE_ST:
            return self._encode_st(texts)
        else:
            raise RuntimeError("TF-IDF mode should not call _encode(); use fit_transform/transform.")

    def _encode_automodel(self, texts: List[str], task: str) -> np.ndarray:
        with torch.no_grad():
            vecs = self._automodel.encode(texts, task=task, late_chunking=False)
        if isinstance(vecs, torch.Tensor):
            vecs = vecs.cpu().numpy()
        return np.array(vecs, dtype=np.float32)

    def _encode_pipeline(self, texts: List[str]) -> np.ndarray:
        """
        HF pipeline returns a list of tensors shaped (1, seq_len, hidden).
        We mean-pool across seq_len to get one vector per text.
        """
        all_vecs = []
        for text in texts:
            out = self._hf_pipeline(text, return_tensors=False)
            # out: list[list[list[float]]]  shape (1, seq_len, hidden)
            arr = np.array(out[0], dtype=np.float32)   # (seq_len, hidden)
            vec = arr.mean(axis=0)                      # (hidden,)
            all_vecs.append(vec)
        return np.stack(all_vecs)

    def _encode_st(self, texts: List[str]) -> np.ndarray:
        vecs = self._st_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return np.array(vecs, dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# Index  (dense + BM25)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class RetrievalIndex:
    chunks:    List[Chunk]
    backend:   EmbeddingBackend
    dense_mat: np.ndarray
    bm25:      Any              # BM25Okapi or None


def build_retrieval_index(chunks: List[Chunk], embedding_backend: EmbeddingBackend) -> RetrievalIndex:
    texts = [c.text for c in chunks]
    print(f"Building dense index with {embedding_backend.backend_name} …")
    dense_mat = embedding_backend.fit_transform(texts)

    bm25 = None
    if HAS_BM25:
        print("Building BM25 index …")
        tokenised = [t.lower().split() for t in texts]
        bm25      = BM25Okapi(tokenised)
    else:
        print("[info] rank_bm25 not installed — BM25 disabled.")

    return RetrievalIndex(chunks=chunks, backend=embedding_backend,
                          dense_mat=dense_mat, bm25=bm25)


def save_index(path: str, index: RetrievalIndex) -> None:
    with open(path, "wb") as f:
        pickle.dump({
            "chunks":    index.chunks,
            "backend":   index.backend,
            "dense_mat": index.dense_mat,
            "bm25":      index.bm25,
        }, f)
    print(f"Index saved → {path}")


def load_index(path: str, model_path: str = None) -> RetrievalIndex:
    """
    Load a saved retrieval index.

    Handles two formats:
      - New format (v2): has keys  chunks / backend / dense_mat / bm25
      - Old format (v1): has keys  chunks / vectorizer / matrix
        In this case the old TF-IDF vectorizer is re-wrapped in an
        EmbeddingBackend so the rest of the pipeline keeps working.
        You should rebuild the index with `build` to get Jina embeddings.
    """
    with open(path, "rb") as f:
        d = pickle.load(f)

    # ── new format ────────────────────────────────────────────────────────────
    if "backend" in d:
        return RetrievalIndex(
            chunks    = d["chunks"],
            backend   = d["backend"],
            dense_mat = d["dense_mat"],
            bm25      = d.get("bm25"),
        )

    # ── old format (v1 TF-IDF index) ─────────────────────────────────────────
    print("[warn] Old-format index detected (no 'backend' key).")
    print("       Using legacy TF-IDF backend for this session.")
    print("       Run  `python datalab_rag_pipeline.py build --input <json>`")
    print("       to rebuild with Jina v3 embeddings.\n")

    if not HAS_SKLEARN:
        raise RuntimeError(
            "Old index requires scikit-learn to load. "
            "pip install scikit-learn  OR  rebuild the index."
        )

    old_vectorizer = d["vectorizer"]   # TfidfVectorizer already fitted
    old_matrix     = d["matrix"]       # scipy sparse or np.ndarray

    # Wrap old TF-IDF state into a fresh EmbeddingBackend shell (tfidf mode)
    backend = EmbeddingBackend.__new__(EmbeddingBackend)
    backend._mode        = EmbeddingBackend._MODE_TFIDF
    backend._automodel   = None
    backend._hf_pipeline = None
    backend._st_model    = None
    backend._tfidf       = old_vectorizer

    dense_mat = old_matrix.toarray() if hasattr(old_matrix, "toarray") else np.array(old_matrix)

    # Rebuild BM25 from chunk texts (not stored in old format)
    chunks = d["chunks"]
    bm25   = None
    if HAS_BM25:
        print("Rebuilding BM25 index from loaded chunks …")
        tokenised = [c.text.lower().split() for c in chunks]
        bm25      = BM25Okapi(tokenised)

    return RetrievalIndex(
        chunks    = chunks,
        backend   = backend,
        dense_mat = dense_mat,
        bm25      = bm25,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Retrieval — hybrid BM25 + dense with RRF, then reranking
# ══════════════════════════════════════════════════════════════════════════════

def _normalize(arr: np.ndarray) -> np.ndarray:
    mn, mx = arr.min(), arr.max()
    if mx == mn:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def _rrf_fusion(ranks_list: List[np.ndarray], k: int = 60) -> np.ndarray:
    """Reciprocal Rank Fusion across multiple ranked lists."""
    n = len(ranks_list[0])
    scores = np.zeros(n)
    for ranks in ranks_list:
        for rank, idx in enumerate(ranks):
            scores[idx] += 1.0 / (k + rank + 1)
    return scores


def retrieve_top_chunks(
    query:          str,
    index:          RetrievalIndex,
    top_k:          int           = DEFAULT_TOP_K,
    section_filter: Optional[str] = None,
    page_range:     Optional[Tuple[int, int]] = None,
    alpha:          float         = HYBRID_ALPHA,
) -> List[Tuple[Chunk, float]]:
    """
    Hybrid retrieval:
      1. Optional metadata pre-filter (section / page range)
      2. BM25 lexical scores
      3. Dense cosine scores
      4. RRF fusion (or pure dense if BM25 unavailable)
    """
    chunks = index.chunks

    # ── 1. metadata filter ────────────────────────────────────────────────────
    mask = np.ones(len(chunks), dtype=bool)
    if section_filter:
        mask &= np.array(
            [section_filter.lower() in c.section_path.lower() for c in chunks]
        )
    if page_range:
        lo, hi = page_range
        mask &= np.array([lo <= c.page <= hi for c in chunks])

    active_idx  = np.where(mask)[0]
    if len(active_idx) == 0:
        print("[warn] Metadata filter matched 0 chunks — ignoring filter.")
        active_idx = np.arange(len(chunks))

    active_chunks    = [chunks[i] for i in active_idx]
    active_dense_mat = index.dense_mat[active_idx]

    # ── 2. dense retrieval ────────────────────────────────────────────────────
    query_vec    = index.backend.transform([query])
    dense_scores = cosine_similarity(query_vec, active_dense_mat)[0]

    # ── 3. BM25 retrieval ─────────────────────────────────────────────────────
    if index.bm25 is not None and HAS_BM25:
        # BM25 was built on *all* chunks; we need scores only for active subset
        all_bm25_scores   = np.array(index.bm25.get_scores(query.lower().split()))
        bm25_active       = all_bm25_scores[active_idx]
        norm_dense        = _normalize(dense_scores)
        norm_bm25         = _normalize(bm25_active)
        combined          = alpha * norm_dense + (1 - alpha) * norm_bm25
        ranked            = np.argsort(combined)[::-1]
        top_idx           = ranked[:top_k]
        results           = [(active_chunks[i], float(combined[i])) for i in top_idx
                             if combined[i] > 0]
    else:
        ranked  = np.argsort(dense_scores)[::-1]
        top_idx = ranked[:top_k]
        results = [(active_chunks[i], float(dense_scores[i])) for i in top_idx
                   if dense_scores[i] > 0]

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Cross-encoder reranking
# ══════════════════════════════════════════════════════════════════════════════

_reranker_cache = None


def get_reranker():
    global _reranker_cache
    if not HAS_SENTENCE_TRANSFORMERS:
        return None
    if _reranker_cache is None:
        try:
            print(f"Loading cross-encoder reranker ({RERANKER_MODEL}) …")
            _reranker_cache = CrossEncoder(RERANKER_MODEL)
            print("  ✓ Reranker loaded")
        except Exception as e:
            print(f"[warn] Reranker unavailable ({e}) — skipping rerank step.")
            return None
    return _reranker_cache


def rerank(
    query:   str,
    matches: List[Tuple[Chunk, float]],
    top_k:   int = RERANK_TOP_K,
) -> List[Tuple[Chunk, float]]:
    reranker = get_reranker()
    if reranker is None or not matches:
        return matches[:top_k]

    pairs  = [(query, c.text) for c, _ in matches]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(matches, scores), key=lambda x: x[1], reverse=True)
    return [m for m, _ in ranked[:top_k]]


# ══════════════════════════════════════════════════════════════════════════════
# Full-page context assembly
# ══════════════════════════════════════════════════════════════════════════════

def build_full_page_context(
    matches:    List[Tuple[Chunk, float]],
    all_chunks: List[Chunk],
) -> Dict[Tuple[str, int], str]:
    """
    For every (source, page) referenced by the matched chunks, assemble the
    complete page text from *all_chunks* — not just the retrieved snippets.
    """
    needed_pages = {(c.source, c.page) for c, _ in matches}
    page_buckets: Dict[Tuple[str, int], List[str]] = {p: [] for p in needed_pages}

    for chunk in all_chunks:
        key = (chunk.source, chunk.page)
        if key in page_buckets:
            page_buckets[key].append(chunk.text)

    return {key: "\n\n".join(texts) for key, texts in page_buckets.items()}


# ══════════════════════════════════════════════════════════════════════════════
# LLM helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_llm() -> Optional[Any]:
    if not HAS_LLM:
        return None
    token = get_token()
    if token.startswith("Bearer "):
        token = token[len("Bearer "):]
    return ChatOpenAI(
        model          = "gpt-5.2",
        base_url       = "https://datascience-services.app.weu-d1.delfi.slb-ds.com/ds/modelops/deployments/v1",
        api_key        = token,
        default_headers= {"slb-partition-id": "sandbox-weu-des-prod-testing-e"},
    )


def make_prompt(
    query:        str,
    matches:      List[Tuple[Chunk, float]],
    all_chunks:   List[Chunk],
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    # Build full-page context instead of raw chunk text
    page_context = build_full_page_context(matches, all_chunks)

    history_block = ""
    if chat_history:
        lines = []
        for turn in chat_history:
            lines.append(f"Previous question: {turn['query']}")
            lines.append(f"Previous answer:   {turn['answer']}")
        history_block = "\n".join(lines) + "\n\n"

    context_parts = []
    for (source, page), text in page_context.items():
        context_parts.append(
            f"[source={source}, page={page}]\n{text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    return (
        "You are an expert retrieval assistant for oil and gas reports. "
        "Use ONLY the provided page excerpts and cite the source page for every answer. "
        "If the answer is not in the excerpts, say the information is unavailable in the document.\n\n"
        # f"{history_block}"
        f"Question: {query}\n\n"
        "Relevant page excerpts:\n"
        f"{context}\n\n"
        "Answer concisely and cite page numbers."
    )


def answer_with_llm(
    query:        str,
    matches:      List[Tuple[Chunk, float]],
    all_chunks:   List[Chunk],
    llm:          Any,
    chat_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    prompt   = make_prompt(query, matches, all_chunks, chat_history)
    response = llm.invoke(prompt)
    return str(response.content) if hasattr(response, "content") else str(response)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline entry points
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_model_path(model_path: str = None) -> str:
    """
    Resolve which model to use, in priority order:
      1. Explicit --model-path CLI arg
      2. ./jina-v3  next to the script  (snapshot_download output)
      3. ./jina-v3  in current working directory
      4. HuggingFace hub ID  (requires internet / SSL fix)
    """
    if model_path:
        print(f"Using model from --model-path: {model_path}")
        return model_path

    local_candidates = [
        Path(__file__).resolve().parent / "jina-v3",
        Path("./jina-v3").resolve(),
    ]
    for candidate in local_candidates:
        if candidate.exists() and (candidate / "config.json").exists():
            print(f"Auto-detected local Jina v3 model at: {candidate}")
            return str(candidate)

    print(f"No local model found — will attempt HuggingFace download ({JINA_MODEL_ID})")
    return JINA_MODEL_ID


def build_pipeline(
    input_path:    str,
    chunks_output: str,
    index_output:  str,
    chunk_size:    int,
    overlap:       int,
    model_path:    str = None,
) -> None:
    print(f"Loading extracted JSON from {input_path}")
    base_chunks = load_extracted_json(input_path)
    print(f"  {len(base_chunks)} raw text blocks extracted")

    print("Merging blocks within same page/section …")
    merged = merge_blocks_by_section(base_chunks)
    print(f"  {len(merged)} section-level chunks after within-page merge")

    print("Merging consecutive same-section blocks across pages …")
    merged = merge_across_pages(merged)
    print(f"  {len(merged)} chunks after cross-page merge")

    print("Splitting long blocks into retrieval units …")
    useful = build_useful_chunks(merged, chunk_size=chunk_size, overlap=overlap)
    print(f"  {len(useful)} final retrieval chunks")

    print(f"Saving chunks → {chunks_output}")
    save_chunks_jsonl(useful, chunks_output)

    preferred = _resolve_model_path(model_path)
    backend   = EmbeddingBackend(preferred_model=preferred)
    index     = build_retrieval_index(useful, backend)
    save_index(index_output, index)


def _print_matches(matches: List[Tuple[Chunk, float]]) -> None:
    print(f"\nTop {len(matches)} chunks after reranking:")
    for rank, (chunk, score) in enumerate(matches, 1):
        print(f"\n[{rank}] score={score:.4f}  source={chunk.source}  "
              f"page={chunk.page}  section={chunk.section_path or chunk.block_type}")
        print(chunk.text[:600].strip())


def query_pipeline(
    index_path:     str,
    query:          str,
    top_k:          int,
    section_filter: Optional[str]           = None,
    page_range:     Optional[Tuple[int,int]]= None,
    chat_history:   Optional[List[Dict]]    = None,
) -> Optional[str]:
    index   = load_index(index_path)
    matches = retrieve_top_chunks(
        query, index, top_k=top_k,
        section_filter=section_filter, page_range=page_range,
    )

    if not matches:
        print("No matching chunks found.")
        return None

    matches = rerank(query, matches, top_k=RERANK_TOP_K)
    _print_matches(matches)

    if HAS_LLM:
        print("\nGenerating answer with LLM …")
        llm = get_llm()
        if llm is not None:
            answer = answer_with_llm(query, matches, index.chunks, llm, chat_history)
            print("\nLLM answer:\n", answer)
            return answer
    else:
        print("\n[info] LLM not available — showing retrieved chunks only.")
    return None


def interactive_query_loop(index_path: str, top_k: int) -> None:
    print(f"Loading index from {index_path} …")
    index        = load_index(index_path)
    chat_history: List[Dict[str, str]] = []
    llm          = get_llm() if HAS_LLM else None

    print("Interactive mode — type 'exit' or press Enter on blank line to quit.")
    print("Optional filters:  prefix query with  section:<name>  or  pages:<lo>-<hi>")
    print("  e.g.  section:Well Summary  What is the well depth?")
    print("  e.g.  pages:1-5  Who is the operator?\n")

    while True:
        raw = input("Query: ").strip()
        if not raw or raw.lower() in {"exit", "quit"}:
            print("Exiting.")
            break

        # Parse optional inline filters
        section_filter, page_range = None, None
        parts = raw

        sec_match = re.match(r"section:(.+?)\s{2,}(.+)", raw, re.IGNORECASE)
        if sec_match:
            section_filter = sec_match.group(1).strip()
            parts          = sec_match.group(2).strip()

        pg_match = re.match(r"pages:(\d+)-(\d+)\s{2,}(.+)", raw, re.IGNORECASE)
        if pg_match:
            page_range = (int(pg_match.group(1)), int(pg_match.group(2)))
            parts      = pg_match.group(3).strip()

        query   = parts
        matches = retrieve_top_chunks(
            query, index, top_k=top_k,
            section_filter=section_filter, page_range=page_range,
        )

        if not matches:
            print("No matching chunks found.\n")
            continue

        matches = rerank(query, matches, top_k=RERANK_TOP_K)
        _print_matches(matches)

        if llm is not None:
            print("\nGenerating answer …")
            answer = answer_with_llm(query, matches, index.chunks, llm, chat_history)
            print("\nLLM answer:\n", answer)
            chat_history.append({"query": query, "answer": answer})
        else:
            print("\n[info] LLM not available.")
        print()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_pipeline(
    input_path:        str,
    queries_path:      str,
    ground_truth_path: str,
    top_k:             int,
    chunk_size:        int,
    overlap:           int,
    model_path:        str = None,
    metrics_output:    Optional[str] = None,
    cache_root:        Optional[str] = rag_evaluation.DEFAULT_CACHE_ROOT,
    use_cache:         bool = True,
    force_rebuild:     bool = False,
    use_judge:         bool = True,
) -> None:
    """Adapter that wires this pipeline's steps into the shared evaluation driver."""
    llm = get_llm() if HAS_LLM else None
    if llm is None:
        print("LLM unavailable: accuracy will be measured against retrieved chunks only.")

    def build_index(useful: List[Chunk]) -> Dict[str, Any]:
        preferred = _resolve_model_path(model_path)
        backend   = EmbeddingBackend(preferred_model=preferred)
        index     = build_retrieval_index(useful, backend)
        return {"index": index, "chunks": useful}

    build_steps = [
        ("1. Load JSON",          load_extracted_json),
        ("2. Merge (section)",    merge_blocks_by_section),
        ("3. Merge (cross-page)", merge_across_pages),
        ("4. Chunking",           lambda c: build_useful_chunks(c, chunk_size=chunk_size, overlap=overlap)),
        ("5. Build index",        build_index),
    ]

    def retrieve_fn(ctx: Dict[str, Any], query: str, k: int):
        matches        = retrieve_top_chunks(query, ctx["index"], top_k=k)
        matches        = rerank(query, matches, top_k=RERANK_TOP_K)
        retrieved_text = "\n".join(c.text for c, _ in matches)
        return matches, retrieved_text

    def answer_fn(ctx: Dict[str, Any], question: str, matches):
        if llm is None or not matches:
            return None
        prompt = make_prompt(question, matches, ctx["chunks"])
        return rag_evaluation.invoke_llm_with_usage(llm, prompt)

    def judge_fn(question: str, ground_truth: str, answer: str):
        return rag_evaluation.llm_judge_match(llm, question, ground_truth, answer)

    def backend_mode_fn(ctx: Dict[str, Any]) -> str:
        try:
            return ctx["index"].backend.backend_name
        except Exception:
            return "unknown"

    rag_evaluation.run_evaluation(
        input_path        = input_path,
        queries_path      = queries_path,
        ground_truth_path = ground_truth_path,
        top_k             = top_k,
        build_steps       = build_steps,
        retrieve_fn       = retrieve_fn,
        answer_fn         = answer_fn if llm is not None else None,
        judge_fn          = judge_fn if (llm is not None and use_judge) else None,
        metrics_output    = metrics_output,
        pipeline_name     = "datalab_rag_pipeline2 (hybrid + rerank)",
        backend_name      = "jina",
        chunk_size        = chunk_size,
        overlap           = overlap,
        cache_root        = cache_root,
        use_cache         = use_cache,
        force_rebuild     = force_rebuild,
        backend_mode_fn   = backend_mode_fn,
    )


def parse_args() -> argparse.Namespace:
    # No sub-command → interactive mode
    if len(sys.argv) == 1:
        return argparse.Namespace(
            command        = "interactive",
            index_output   = "rag_index.pkl",
            top_k          = DEFAULT_TOP_K,
        )

    parser = argparse.ArgumentParser(
        description="Build and query a RAG pipeline from extracted PDF JSON."
    )
    sub = parser.add_subparsers(dest="command")

    # ── build ──────────────────────────────────────────────────────────────────
    bp = sub.add_parser("build", help="Build chunks and index from extracted JSON")
    bp.add_argument("--input",         required=True,                    help="Path to extracted JSON file")
    bp.add_argument("--chunks-output", default="useful_chunks.jsonl",    help="Output JSONL file for chunks")
    bp.add_argument("--index-output",  default="rag_index.pkl",          help="Output pickle file for index")
    bp.add_argument("--chunk-size",    type=int, default=800,            help="Max chunk size in words")
    bp.add_argument("--overlap",       type=int, default=200,            help="Chunk overlap in words")
    bp.add_argument("--model-path",    default=None,
                    help="Path to local Jina v3 model folder (e.g. ./jina-v3). "
                         "Auto-detected if omitted and ./jina-v3 exists.")

    # ── query ──────────────────────────────────────────────────────────────────
    qp = sub.add_parser("query", help="Query the retrieval index")
    qp.add_argument("--index-output",    default="rag_index.pkl", help="Path to index pickle")
    qp.add_argument("--query",           required=True,           help="Search query string")
    qp.add_argument("--top-k",           type=int, default=DEFAULT_TOP_K, help="Candidates to retrieve")
    qp.add_argument("--section-filter",  default=None,            help="Filter by section name substring")
    qp.add_argument("--page-range",      nargs=2, type=int,
                    metavar=("LO", "HI"), default=None,           help="Filter to page range (inclusive)")

    # ── interactive ───────────────────────────────────────────────────────────
    ip = sub.add_parser("interactive", help="Interactive query loop")
    ip.add_argument("--index-output", default="rag_index.pkl", help="Path to index pickle")
    ip.add_argument("--top-k",        type=int, default=DEFAULT_TOP_K)

    # ── evaluate ───────────────────────────────────────────────────────────────
    ev = sub.add_parser("evaluate", help="Run the pipeline and report per-step metrics")
    ev.add_argument("--input",          required=True,                   help="Path to extracted datalab JSON file")
    ev.add_argument("--queries",        default="queries.txt",           help="Path to the queries file")
    ev.add_argument("--ground-truth",   required=True,                   help="Path to the matched JSON file with ground-truth values")
    ev.add_argument("--top-k",          type=int, default=DEFAULT_TOP_K, help="Candidates to retrieve per query")
    ev.add_argument("--chunk-size",     type=int, default=800,           help="Max chunk size in words")
    ev.add_argument("--overlap",        type=int, default=200,           help="Chunk overlap in words")
    ev.add_argument("--model-path",     default=None,                    help="Path to local Jina v3 model folder")
    ev.add_argument("--metrics-output", default=None,                    help="Optional path to save metrics as JSON")
    ev.add_argument("--cache-root",     default=rag_evaluation.DEFAULT_CACHE_ROOT, help="Root folder for cached indexes (per-backend subfolders)")
    ev.add_argument("--no-cache",       action="store_true",             help="Disable reading/writing the index cache")
    ev.add_argument("--rebuild",        action="store_true",             help="Ignore any cached index and rebuild it")
    ev.add_argument("--no-judge",       action="store_true",             help="Disable the LLM judge (use literal token matching for accuracy)")

    return parser.parse_args()


def main() -> None:
    rag_evaluation.configure_utf8_stdout()
    args = parse_args()

    if args.command == "build":
        build_pipeline(
            input_path    = args.input,
            chunks_output = args.chunks_output,
            index_output  = args.index_output,
            chunk_size    = args.chunk_size,
            overlap       = args.overlap,
            model_path    = args.model_path,
        )

    elif args.command == "query":
        page_range = tuple(args.page_range) if args.page_range else None
        query_pipeline(
            index_path     = args.index_output,
            query          = args.query,
            top_k          = args.top_k,
            section_filter = args.section_filter,
            page_range     = page_range,
        )

    elif args.command == "interactive":
        interactive_query_loop(
            index_path = args.index_output,
            top_k      = args.top_k,
        )

    elif args.command == "evaluate":
        evaluate_pipeline(
            input_path        = args.input,
            queries_path      = args.queries,
            ground_truth_path = args.ground_truth,
            top_k             = args.top_k,
            chunk_size        = args.chunk_size,
            overlap           = args.overlap,
            model_path        = args.model_path,
            metrics_output    = args.metrics_output,
            cache_root        = args.cache_root,
            use_cache         = not args.no_cache,
            force_rebuild     = args.rebuild,
            use_judge         = not args.no_judge,
        )

    else:
        print("Unknown command. Run with --help for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()