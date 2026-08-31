"""Manufacturing Maintenance Assistant -- a retrieval-augmented Q&A service over
a machine-maintenance knowledge base. Loads a prebuilt vector index at startup
and answers questions with grounded, cited responses (offline-extractive by
default; synthesized by an LLM if API keys are configured)."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import rag, telemetry
from app.config import INDEX_DIR, TOP_K
from app.index import RagIndex

STATIC_DIR = Path(__file__).resolve().parent / "static"

_index: RagIndex | None = None
_load_error: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the vector index and configure tracing before serving traffic.

    Replaces the deprecated @app.on_event("startup") hook. Telemetry is set up
    after the index so that index-load failures are still reported by /health
    even if the exporter cannot be reached.
    """
    global _index, _load_error
    try:
        _index = RagIndex.load(INDEX_DIR)
    except Exception as exc:  # noqa: BLE001 -- degrade gracefully if the index isn't built
        _index = None
        _load_error = (f"No index at {INDEX_DIR} ({exc}). Run: python scripts/build_index.py")
    telemetry.setup_telemetry(app)
    yield


app = FastAPI(
    lifespan=lifespan,
    title="Manufacturing Maintenance Assistant",
    description="RAG over machine-maintenance guides: semantic retrieval (FAISS) + grounded, cited answers.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _gen_mode() -> str:
    if os.getenv("AZURE_OPENAI_API_KEY"):
        return "azure_openai"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "offline_extractive"


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def landing_page() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "index_loaded": _index is not None,
        "index_status": None if _index is not None else _load_error,
        "chunks": len(_index.chunks) if _index else 0,
        "generation_mode": _gen_mode(),
        "telemetry_configured": telemetry.is_configured(),
    }


@app.get("/corpus")
def corpus() -> dict:
    if _index is None:
        raise HTTPException(status_code=503, detail=_load_error or "Index not loaded")
    docs: dict[str, int] = {}
    for c in _index.chunks:
        docs[c.doc_title] = docs.get(c.doc_title, 0) + 1
    return {"documents": docs, "total_chunks": len(_index.chunks)}


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    k: int = Field(TOP_K, ge=1, le=8)


@app.post("/ask")
def ask(payload: AskIn) -> dict:
    if _index is None:
        raise HTTPException(status_code=503, detail=_load_error or "Index not loaded")
    result = rag.answer_question(_index, payload.question, k=payload.k)
    return {
        "answer": result.answer,
        "grounded": result.grounded,
        "mode": result.mode,
        "sources": [s.__dict__ for s in result.sources],
    }
