"""Central control panel for the whole RAG system.

Every hyperparameter and component toggle lives here. Changing behaviour (e.g.
swapping the embedding model, turning the reranker off, switching to full-page
context) is done by editing this file or overriding via CLI flags in ``main.py``
- you should not need to touch the stage modules.

Reproducing the original three scripts:
  * v1  (datalab_rag_pipeline.py) : embedding_model="tfidf", retrieval_mode="dense",
                                    use_reranker=False, context_mode="chunk"
  * v2  (datalab_rag_pipeline2.py): embedding_model="jina",  retrieval_mode="hybrid",
                                    use_reranker=True,  context_mode="full_page"
  * v3  (datalab_rag_pipeline3.py): embedding_model="tfidf", retrieval_mode="dense",
                                    use_reranker=False, context_mode="full_page"
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Optional

# Repo root (rag/config.py -> rag/ -> repo root); used to auto-detect a local
# model snapshot such as ./jina-v3 that lives next to the package.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Friendly aliases accepted by ``embedding_model`` in addition to raw HF ids.
JINA_MODEL_ID = "jinaai/jina-embeddings-v3"
MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

# Folder names to probe when auto-detecting a local Jina v3 snapshot.
JINA_LOCAL_DIRS = ("jina-v3",)

EMBEDDING_ALIASES = {
    "jina": JINA_MODEL_ID,
    "minilm": MINILM_MODEL_ID,
    "tfidf": "tfidf",
    "openai": "openai",
}


@dataclass
class RagConfig:
    # ── chunking ────────────────────────────────────────────────────────────
    chunk_size: int = 800
    overlap: int = 200
    merge_across_pages: bool = True
    # When False, run the naive approach: skip section/cross-page merging and
    # chunk the raw datalab JSON blocks directly (for A/B comparison).
    merge_sections: bool = True
    # Chunking strategy:
    #   "section" - hierarchy-aware: group blocks by the leaf section resolved
    #               from datalab's section_hierarchy, prefix the parent
    #               breadcrumb, split oversize sections (recommended).
    #   "legacy"  - the old merge_by_section + merge_across_pages + window flow.
    chunk_strategy: str = "section"

    # ── embedding (folder 1)  ← SWAP EMBEDDING MODEL HERE ───────────────────
    # "jina" | "minilm" | "openai" | "tfidf" | any HuggingFace model id
    embedding_model: str = "openai"
    # OpenAI-compatible embeddings (SLB internal deployment). base_url defaults
    # to llm_base_url when left as None.
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_base_url: Optional[str] = "https://datascience-services.app.weu-d1.delfi.slb-ds.com/ds/modelops/deployments/v1"
    openai_embedding_partition_id: Optional[str] = "sandbox-weu-des-prod-testing-e"

    # ── vector store  ← SWAP DB HERE ────────────────────────────────────────
    # "chroma" - persistent ChromaDB collection that stores our PRECOMPUTED
    #            embeddings (embed-once; never re-embeds on reuse).
    # "faiss"  - the legacy in-memory FAISS index serialised to meta.pkl.
    vector_backend: str = "chroma"
    faiss_metric: str = "cosine"          # "cosine" | "l2"
    index_dir: str = "rag/faiss_store"    # holds index.faiss + meta.pkl (faiss)
    chroma_dir: str = "chroma_store"      # PersistentClient path (chroma)

    # ── parallelism (bounded, to respect API rate limits) ───────────────────
    embed_batch_size: int = 64            # texts per embedding request/batch
    embed_max_workers: int = 8            # concurrent embedding requests (API)
    llm_max_workers: int = 8              # concurrent LLM/judge calls
    doc_max_workers: int = 4              # docs processed concurrently
    load_max_workers: int = 4             # process-pool workers for load+chunk

    # ── retrieval (folder 2) ────────────────────────────────────────────────
    retrieval_mode: str = "hybrid"         # "dense" | "hybrid"
    hybrid_alpha: float = 0.85             # 0 = pure BM25, 1 = pure dense
    top_k: int = 25

    # ── reranker  ← SWAP RERANKER HERE ──────────────────────────────────────
    use_reranker: bool = True
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 10

    # ── context assembly ────────────────────────────────────────────────────
    # "chunk"        - send only the matched chunk texts.
    # "full_page"    - expand each matched chunk to its full page.
    # "hierarchical" - start from the matched chunks and add ancestor sections on
    #                  demand, re-prompting until the LLM can answer (see
    #                  rag.inference.hierarchical_answer).
    context_mode: str = "hierarchical"          # "chunk" | "full_page" | "hierarchical"
    # Max number of ancestor levels to climb in hierarchical mode.
    hierarchical_max_depth: int = 5
    save_pages_sent_to_llm: bool = False

    # ── LLM  ← SWAP LLM HERE ────────────────────────────────────────────────
    llm_model: str = "gpt-5.2"
    # Sampling temperature for generation + judge. 0.0 = deterministic (recommended
    # for reproducible evaluation). Set to None to omit it (some reasoning models
    # only accept the default); the client falls back automatically if rejected.
    llm_temperature: Optional[float] = 0.0
    llm_base_url: str = (
        "https://datascience-services.app.weu-d1.delfi.slb-ds.com/ds/modelops/deployments/v1"
    )
    partition_id: str = "sandbox-weu-des-prod-testing-e"

    # Optional local model folder for Jina (auto-detected if None)
    model_path: Optional[str] = None

    # ── resolved helpers ─────────────────────────────────────────────────────
    @property
    def resolved_embedding_model(self) -> str:
        """Resolve a friendly alias to a concrete model id (or 'tfidf')."""
        return EMBEDDING_ALIASES.get(self.embedding_model.lower(), self.embedding_model)

    @property
    def use_tfidf(self) -> bool:
        return self.resolved_embedding_model == "tfidf"

    @property
    def resolved_model_path(self) -> Optional[str]:
        """Explicit ``model_path``, or an auto-detected local Jina v3 folder.

        Mirrors the old pipeline's ``_resolve_model_path``: when the embedding
        model is Jina and no ``model_path`` was given, look for a local
        ``./jina-v3`` snapshot (next to the package or in the CWD) so we use the
        offline weights instead of trying to download from HuggingFace.
        """
        if self.model_path:
            return self.model_path
        is_jina = "jina" in self.embedding_model.lower() or "jina" in self.resolved_embedding_model.lower()
        if is_jina:
            for name in JINA_LOCAL_DIRS:
                for base in (_REPO_ROOT, os.getcwd()):
                    candidate = os.path.join(base, name)
                    if os.path.isdir(candidate):
                        return candidate
        return None

    @property
    def embedding_slug(self) -> str:
        """Filesystem-safe identifier that is unique per embedding model.

        Used to give each model its OWN vector-store folder (and cache tag) so
        building with a new model never overwrites or silently reuses another
        model's precomputed index. The concrete model id keeps OpenAI / Jina /
        MiniLM distinct from each other.
        """
        if self.use_tfidf:
            raw = "tfidf"
        elif self.embedding_model.lower() == "openai":
            raw = f"openai_{self.openai_embedding_model}"
        elif self.resolved_model_path:
            raw = os.path.basename(os.path.normpath(self.resolved_model_path))
        else:
            raw = self.resolved_embedding_model
        raw = str(raw).strip().strip("/\\")
        raw = raw.replace("/", "_").replace("\\", "_")
        raw = re.sub(r"[^A-Za-z0-9._-]", "_", raw)
        return raw.strip("._-") or "unknown"

    @property
    def resolved_index_dir(self) -> str:
        """Per-model index directory: ``<index_dir>/<embedding_slug>``.

        Both build (save) and query (load) resolve through this, so a model's
        ``index.faiss`` + ``meta.pkl`` live in their own subfolder.
        """
        return os.path.join(self.index_dir, self.embedding_slug)

    @property
    def chroma_collection(self) -> str:
        """Collection name unique per (embedding model, chunk strategy).

        Each (model, strategy) pair gets its own collection so vectors built
        with different settings never collide, and reusing the same pair serves
        the precomputed vectors straight from disk (embed-once).
        """
        name = f"{self.embedding_slug}__{self.chunk_strategy}"
        return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._-") or "default"

    def backend_tag(self) -> str:
        """Tag used to namespace cached indexes per (model, context, merge).

        Uses the per-model ``embedding_slug`` (not a coarse "dense") so that
        OpenAI / Jina / MiniLM each get their OWN cache and never reuse another
        model's pre-built index. TF-IDF keeps its existing ``tfidf_*`` tag.
        """
        merge = "merge" if self.merge_sections else "naive"
        return f"{self.embedding_slug}_{self.context_mode}_{merge}"

    @classmethod
    def from_env(cls) -> "RagConfig":
        """Build a config, letting environment variables override LLM settings."""
        cfg = cls()
        cfg.llm_model = os.getenv("LLM_MODEL", cfg.llm_model)
        cfg.llm_base_url = os.getenv("LLM_BASE_URL", cfg.llm_base_url)
        cfg.partition_id = os.getenv("SLB_PARTITION_ID", cfg.partition_id)
        return cfg


# Default singleton other modules can import directly.
DEFAULT_CONFIG = RagConfig.from_env()
