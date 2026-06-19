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
from dataclasses import dataclass, field
from typing import Optional

# Friendly aliases accepted by ``embedding_model`` in addition to raw HF ids.
JINA_MODEL_ID = "jinaai/jina-embeddings-v3"
MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_ALIASES = {
    "jina": JINA_MODEL_ID,
    "minilm": MINILM_MODEL_ID,
    "tfidf": "tfidf",
}


@dataclass
class RagConfig:
    # ── chunking ────────────────────────────────────────────────────────────
    chunk_size: int = 800
    overlap: int = 200
    merge_across_pages: bool = True

    # ── embedding (folder 1)  ← SWAP EMBEDDING MODEL HERE ───────────────────
    # "jina" | "minilm" | "tfidf" | any HuggingFace model id
    embedding_model: str = "tfidf"

    # ── vector store (FAISS)  ← SWAP DB HERE ────────────────────────────────
    faiss_metric: str = "cosine"          # "cosine" | "l2"
    index_dir: str = "rag/faiss_store"    # holds index.faiss + meta.pkl

    # ── retrieval (folder 2) ────────────────────────────────────────────────
    retrieval_mode: str = "dense"         # "dense" | "hybrid"
    hybrid_alpha: float = 0.5             # 0 = pure BM25, 1 = pure dense
    top_k: int = 10

    # ── reranker  ← SWAP RERANKER HERE ──────────────────────────────────────
    use_reranker: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    rerank_top_k: int = 10

    # ── context assembly ────────────────────────────────────────────────────
    context_mode: str = "chunk"           # "chunk" | "full_page"
    save_pages_sent_to_llm: bool = False

    # ── LLM  ← SWAP LLM HERE ────────────────────────────────────────────────
    llm_model: str = "gpt-5.2"
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

    def backend_tag(self) -> str:
        """Short tag used to namespace cached indexes per backend."""
        base = "tfidf" if self.use_tfidf else "dense"
        return f"{base}_{self.context_mode}"

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
