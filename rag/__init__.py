"""Modular RAG pipeline package.

Two pipelines:
  * ``indexing``  - PDF-extraction JSON  ->  chunks  ->  embeddings  ->  FAISS store
  * ``inference`` - query  ->  retrieval  ->  rerank  ->  context  ->  LLM  ->  answer

All hyperparameters live in :mod:`rag.config`; ``main.py`` is the single entry point.
"""
