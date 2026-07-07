"""Optional FastAPI endpoints for schema-driven query generation.

Kept separate from the webapp so it stays opt-in: include it from the backend
with a single line (it does not run unless mounted)::

    # webapp/backend/main.py
    from query_generation.api import router as query_gen_router
    app.include_router(query_gen_router)

Endpoints (prefix ``/api/query-generation``):
  * POST /from-schema        - upload a schema/record file -> writes a queries
                               file and returns its path + generated queries.
  * POST /attributes         - upload a schema/record -> just the extracted
                               attributes (no LLM calls; cheap preview).

The generated file lands in ``generated_queries/<name>.txt`` and is ready to
feed to ``eval_all_queries.py --queries <path>``.
"""

from __future__ import annotations

import os
import tempfile
from typing import Any, Dict, Optional

try:
    from fastapi import APIRouter, HTTPException, UploadFile
    _HAS_FASTAPI = True
except Exception:  # pragma: no cover - fastapi optional
    _HAS_FASTAPI = False


def _build(source_path: str, name: Optional[str], include_null: bool, limit: Optional[int]) -> Dict[str, Any]:
    """Run the generation pipeline and return a JSON-serialisable summary."""
    from rag.config import RagConfig
    from rag.inference.llm import get_llm
    from query_generation.batch import build_query_file_from_schema

    config = RagConfig.from_env()
    llm = get_llm(config)
    if llm is None:
        raise RuntimeError("LLM unavailable (langchain-openai + sauth_token required).")

    result = build_query_file_from_schema(
        source_path, llm=llm, config=config, name=name,
        include_null=include_null, limit=limit,
    )
    return {
        "document_name": result.document_name,
        "output_path": result.output_path,
        "num_generated": result.num_generated,
        "attributes": result.attributes,
        "queries": [
            {"label": g.label, "query": g.query, "keywords": g.keywords}
            for g in result.generated
        ],
        "failures": result.failures,
    }


if _HAS_FASTAPI:
    router = APIRouter(prefix="/api/query-generation", tags=["query-generation"])

    @router.post("/attributes")
    async def extract(file: UploadFile, include_null: bool = True) -> Dict[str, Any]:
        """Preview the attributes in an uploaded schema/record (no LLM calls)."""
        from query_generation.schema_ingest import extract_attributes

        content = (await file.read()).decode("utf-8", errors="replace")
        try:
            attrs = extract_attributes(content, include_null=include_null)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=f"Could not parse schema/record: {exc}")
        return {
            "filename": file.filename,
            "count": len(attrs),
            "attributes": [
                {"raw_name": a.raw_name, "label": a.label, "is_null": a.is_null}
                for a in attrs
            ],
        }

    @router.post("/from-schema")
    async def from_schema(
        file: UploadFile,
        name: Optional[str] = None,
        include_null: bool = True,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Upload a schema/record; generate + save a per-document queries file."""
        content = await file.read()
        suffix = os.path.splitext(file.filename or "schema.json")[1] or ".json"
        derived_name = name or os.path.splitext(os.path.basename(file.filename or "generated"))[0]

        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        try:
            tmp.write(content)
            tmp.close()
            return _build(tmp.name, derived_name, include_null, limit)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc))
        finally:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
else:  # pragma: no cover
    router = None  # type: ignore
