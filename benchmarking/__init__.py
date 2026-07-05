"""Benchmarking suite for the Metadata-Enrichment RAG pipeline.

Four self-contained experiments, each with its own script, all reusing the
existing evaluation engines (``rag.inference.batch_runner`` for end-to-end
accuracy/cost/time and ``rag.evaluation.retrieval_metrics`` for MTEB-style
retrieval metrics) rather than re-implementing evaluation:

  * ``benchmark_embeddings``  - Task 1: compare embedding models (reranker off).
  * ``benchmark_matryoshka``  - Task 2: Matryoshka dimension study (reuse-only).
  * ``benchmark_reranker``    - Task 3: reranker impact on the best model.
  * ``benchmark_llms``        - Task 4: swap the generator LLM.

Shared helpers live in ``model_specs`` (pricing + context windows) and
``common`` (config building, run wrappers, cost recompute, report writers).
"""
