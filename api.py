from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

# Load environment from .env if present (optional dependency).
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass

from advanced_rag import AdvancedRAG, RAGConfig
from config import build_rag_config, load_config_file
from ingest import ChunkingConfig, build_documents_from_dir


DEFAULT_PERSIST_DIR = Path(os.getenv("RAG_PERSIST_DIR", "chroma_db"))
DEFAULT_COLLECTION = os.getenv("RAG_COLLECTION", "rag_chunks")
DEFAULT_LLM_MODEL = os.getenv("RAG_LLM_MODEL", "qwen2.5:3b")
DEFAULT_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
DEFAULT_RERANKER_MODEL = os.getenv("RAG_RERANKER_MODEL", "BAAI/bge-reranker-base")
DEFAULT_CONFIG_PATH = Path(os.getenv("RAG_CONFIG_PATH", "rag.config.json"))

# Optional: protect your public API
API_KEY = os.getenv("RAG_API_KEY", "").strip()

_rag_lock = threading.RLock()
_rag: Optional[AdvancedRAG] = None


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    if not x_api_key or x_api_key.strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _get_rag() -> AdvancedRAG:
    global _rag
    with _rag_lock:
        if _rag is None:
            file_overrides = load_config_file(DEFAULT_CONFIG_PATH)
            base = RAGConfig(
                persist_dir=DEFAULT_PERSIST_DIR,
                collection_name=DEFAULT_COLLECTION,
                embedding_model=DEFAULT_EMBEDDING_MODEL,
                reranker_model=DEFAULT_RERANKER_MODEL,
                llm_model=DEFAULT_LLM_MODEL,
            )
            cfg = build_rag_config(base=base, overrides=file_overrides)
            _rag = AdvancedRAG.build_default(
                cfg
            )
            # No-op if empty; initializes BM25 from persistent store when present.
            _rag.load_index_from_vectorstore()
        return _rag


def _as_sources(chosen: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources: List[Dict[str, Any]] = []
    for c in chosen:
        meta = c.get("meta") or {}
        sources.append(
            {
                "doc_id": c.get("doc_id"),
                "score": c.get("score"),
                "source": meta.get("source"),
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
            }
        )
    return sources


class HealthResponse(BaseModel):
    status: str = "ok"
    collection: str
    persist_dir: str


class AskRequest(BaseModel):
    q: str = Field(..., min_length=1, description="Question utilisateur")
    source_contains: str = Field(default="", description="Filtre: substring sur metadata.source")
    page_min: Optional[int] = Field(default=None, ge=1, description="Filtre: page min (PDF)")
    page_max: Optional[int] = Field(default=None, ge=1, description="Filtre: page max (PDF)")
    debug: bool = Field(default=False, description="Inclure les traces de retrieval/rerank")


class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    debug: Optional[Dict[str, Any]] = None


class IngestRequest(BaseModel):
    data_dir: str = Field(default="data", description="Dossier à indexer (txt/md/pdf/docx)")
    chunk_size: int = Field(default=900, ge=200, le=10_000)
    chunk_overlap: int = Field(default=150, ge=0, le=5_000)


class IngestResponse(BaseModel):
    ingested_chunks: int
    collection: str
    persist_dir: str


class InspectIndexResponse(BaseModel):
    n: int
    rows: List[Dict[str, Any]]


app = FastAPI(title="Advanced RAG API", version="0.1.0")


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(collection=DEFAULT_COLLECTION, persist_dir=str(DEFAULT_PERSIST_DIR))


@app.post("/ask", dependencies=[Depends(_require_api_key)])
def ask(payload: AskRequest) -> AskResponse:
    rag = _get_rag()
    answer, debug = rag.answer(
        payload.q,
        source_contains=payload.source_contains,
        page_min=payload.page_min,
        page_max=payload.page_max,
    )
    resp = AskResponse(answer=answer, sources=_as_sources(debug.get("chosen") or []))
    if payload.debug:
        resp.debug = debug
    return resp


@app.post(
    "/ingest",
    dependencies=[Depends(_require_api_key)],
    responses={400: {"description": "Invalid data_dir"}},
)
def ingest(payload: IngestRequest) -> IngestResponse:
    data_dir = Path(payload.data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise HTTPException(status_code=400, detail=f"data_dir not found: {data_dir}")

    rag = _get_rag()
    with _rag_lock:
        docs, ids = build_documents_from_dir(
            data_dir,
            ChunkingConfig(chunk_size=payload.chunk_size, chunk_overlap=payload.chunk_overlap),
        )
        rag.add_documents(docs, ids=ids)

    return IngestResponse(
        ingested_chunks=len(docs),
        collection=DEFAULT_COLLECTION,
        persist_dir=str(DEFAULT_PERSIST_DIR),
    )


@app.get("/inspect-index", dependencies=[Depends(_require_api_key)])
def inspect_index(
    limit: int = 200,
    source_contains: str = "",
    no_preview: bool = False,
    preview_chars: int = 120,
) -> InspectIndexResponse:
    rag = _get_rag()
    rows = rag.list_index(
        limit=limit,
        source_contains=source_contains,
        with_preview=not no_preview,
        preview_chars=preview_chars,
    )
    return InspectIndexResponse(n=len(rows), rows=rows)

