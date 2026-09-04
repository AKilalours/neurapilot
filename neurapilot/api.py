"""NeuraPilot REST API — FastAPI backend.

Provides programmatic access to all NeuraPilot features:
  - Course management (create, list)
  - Document ingestion (trigger pipeline)
  - Query endpoint (run full agentic pipeline)
  - Mastery data export
  - Interaction history export
  - Health check + metadata

Auto-generated OpenAPI docs at /docs and /redoc.
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from neurapilot.config import get_settings
from neurapilot.core import db as dbmod
from neurapilot.rag.ingest import ingest_course
from neurapilot.rag.llm import build_llm_bundle
from neurapilot.rag.store import get_retriever
from neurapilot.rag.agent_graph import build_pipeline
from neurapilot.storage import course_upload_dir, load_courses, save_courses


# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title="NeuraPilot API",
    version="1.0.0",
    description="Production-grade agentic RAG tutor REST API",
    contact={"name": "NeuraPilot", "email": "team@neurapilot.ai"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared resources (initialized once at startup) ────────────────────────────

_settings = get_settings()
_conn = dbmod.connect(_settings)
_bundle = build_llm_bundle(_settings)


# ── Request / Response Models ─────────────────────────────────────────────────


class CourseCreateRequest(BaseModel):
    course_id: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-zA-Z0-9_-]+$")
    title: str = Field(default="", max_length=200)
    description: str = Field(default="", max_length=1000)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    course_id: str = Field(..., min_length=1)
    strict: bool = Field(default=True)


class QueryResponse(BaseModel):
    intent: str
    topic: str
    output: str
    sources: list[dict[str, Any]]
    latency_ms: int


class MasteryResponse(BaseModel):
    course_id: str
    topics: list[dict[str, Any]]


# ── Endpoints ─────────────────────────────────────────────────────────────────


@app.get("/health", tags=["System"])
def health_check() -> dict[str, Any]:
    """Service health check and metadata."""
    return {
        "status": "ok",
        "version": "1.0.0",
        "provider": _bundle.provider,
        "model": _bundle.model_name,
        "embed_model": _bundle.embed_model_name,
        "ts": int(time.time()),
    }


@app.post("/courses", tags=["Courses"])
def create_course(req: CourseCreateRequest) -> dict[str, str]:
    """Create a new course workspace."""
    courses = load_courses(_settings)
    courses[req.course_id] = {"title": req.title, "description": req.description}
    save_courses(_settings, courses)
    course_upload_dir(_settings, req.course_id)
    dbmod.upsert_course(_conn, req.course_id, req.title, req.description)
    return {"status": "created", "course_id": req.course_id}


@app.get("/courses", tags=["Courses"])
def list_courses() -> dict[str, Any]:
    """List all available courses."""
    return {"courses": dbmod.list_courses(_conn)}


@app.post("/courses/{course_id}/upload", tags=["Ingestion"])
async def upload_file(
    course_id: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """Upload a document to a course workspace."""
    allowed_exts = {".pdf", ".txt", ".md"}
    import pathlib
    ext = pathlib.Path(file.filename or "").suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {allowed_exts}")

    dest = course_upload_dir(_settings, course_id) / (file.filename or "upload.txt")
    content = await file.read()
    dest.write_bytes(content)
    return {"status": "uploaded", "filename": file.filename, "size_bytes": len(content)}


@app.post("/courses/{course_id}/ingest", tags=["Ingestion"])
def trigger_ingest(
    course_id: str,
    clear_existing: bool = False,
) -> dict[str, Any]:
    """Trigger document ingestion for a course."""
    stats = ingest_course(
        course_id=course_id,
        settings=_settings,
        bundle=_bundle,
        clear_existing=clear_existing,
    )
    return {
        "status": "ok",
        "files_seen": stats.files_seen,
        "chunks_indexed": stats.chunks_indexed,
        "skipped_files": stats.skipped_files,
        "duration_s": stats.duration_s,
    }


@app.post("/query", tags=["Query"], response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Run the full agentic RAG pipeline on a question."""
    retriever = get_retriever(_settings, _bundle.embeddings, req.course_id)
    pipeline = build_pipeline(
        llm=_bundle.llm,
        retriever=retriever,
        strict_default=_settings.strict_grounding,
        top_k=_settings.top_k,
        hallucination_guard=_settings.hallucination_guard,
    )

    t0 = time.time()
    state = pipeline.invoke({"question": req.question, "strict": req.strict})
    latency_ms = int((time.time() - t0) * 1000)

    docs = state.get("docs", []) or []
    sources = [
        {
            "key": f"S{i}",
            "source": str(d.metadata.get("source", "unknown")),
            "page": d.metadata.get("page"),
            "chunk_id": d.metadata.get("chunk_id"),
        }
        for i, d in enumerate(docs, start=1)
    ]

    dbmod.log_interaction(
        _conn,
        course_id=req.course_id,
        strict=req.strict,
        user_text=req.question,
        intent=state.get("intent", "ask"),
        topic=state.get("topic", ""),
        output=state.get("output", ""),
        sources=sources,
        latency_ms=latency_ms,
    )

    return QueryResponse(
        intent=state.get("intent", "ask"),
        topic=state.get("topic", ""),
        output=state.get("output", ""),
        sources=sources,
        latency_ms=latency_ms,
    )


@app.get("/courses/{course_id}/mastery", tags=["Mastery"])
def get_mastery(course_id: str) -> MasteryResponse:
    """Get Bayesian mastery scores for all topics in a course."""
    return MasteryResponse(
        course_id=course_id,
        topics=dbmod.get_mastery(_conn, course_id),
    )


@app.get("/courses/{course_id}/interactions", tags=["Analytics"])
def get_interactions(course_id: str, limit: int = 50) -> dict[str, Any]:
    """Export recent interaction history for a course."""
    return {
        "course_id": course_id,
        "interactions": dbmod.get_recent_interactions(_conn, course_id, limit=limit),
    }


@app.get("/courses/{course_id}/flashcards/stats", tags=["Flashcards"])
def flashcard_stats(course_id: str) -> dict[str, Any]:
    """Get flashcard counts (total and due) for a course."""
    return {"course_id": course_id, **dbmod.count_flashcards(_conn, course_id)}
