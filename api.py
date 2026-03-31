from __future__ import annotations
import os
import tempfile
import threading
from pathlib import Path
from typing import Annotated
from typing import Any, Dict, List, Optional, Tuple
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field
from rag_engine import AdvancedRAG, RAGConfig
from config import build_rag_config, load_config_file, save_config_file
from ingest import ChunkingConfig, build_documents_from_dir, build_documents_from_file

# Load environment from .env if present (optional dependency).
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
except Exception:
    pass




def _require_env(name: str) -> str:
    value = os.getenv(name)
    if value is None or not value.strip():
        raise RuntimeError(f'Missing required environment variable "{name}" (set it in .env).')
    return value.strip()


PERSIST_DIR = Path(_require_env("RAG_PERSIST_DIR"))
COLLECTION = _require_env("RAG_COLLECTION")
LLM_MODEL = _require_env("RAG_LLM_MODEL")
EMBEDDING_MODEL = _require_env("RAG_EMBEDDING_MODEL")
RERANKER_MODEL = _require_env("RAG_RERANKER_MODEL")
CONFIG_PATH = Path(_require_env("RAG_CONFIG_PATH"))

# Optional: protect your public API
API_KEY = os.getenv("RAG_API_KEY", "").strip()

_rag_lock = threading.RLock()
_rag: Optional[AdvancedRAG] = None
_raw_file_config: Optional[Dict[str, Any]] = None


def _get_file_config() -> Dict[str, Any]:
    global _raw_file_config
    with _rag_lock:
        if _raw_file_config is None:
            _raw_file_config = load_config_file(CONFIG_PATH)
        return _raw_file_config


def _ingest_defaults() -> Tuple[int, int]:
    cfg = _get_file_config()
    if "ingest_chunk_size" not in cfg or "ingest_chunk_overlap" not in cfg:
        raise RuntimeError('Missing "ingest_chunk_size" / "ingest_chunk_overlap" in rag.config.json.')
    chunk_size = int(cfg["ingest_chunk_size"])
    chunk_overlap = int(cfg["ingest_chunk_overlap"])
    return chunk_size, chunk_overlap


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        return
    if not x_api_key or x_api_key.strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _get_rag() -> AdvancedRAG:
    global _rag
    with _rag_lock:
        if _rag is None:
            file_overrides = _get_file_config()
            base = RAGConfig(
                persist_dir=PERSIST_DIR,
                collection_name=COLLECTION,
                embedding_model=EMBEDDING_MODEL,
                reranker_model=RERANKER_MODEL,
                llm_model=LLM_MODEL,
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
    chunk_size: Optional[int] = Field(default=None, ge=200, le=10_000)
    chunk_overlap: Optional[int] = Field(default=None, ge=0, le=5_000)


class IngestResponse(BaseModel):
    ingested_chunks: int
    collection: str
    persist_dir: str


class InspectIndexResponse(BaseModel):
    n: int
    rows: List[Dict[str, Any]]


class ConfigUpdateRequest(BaseModel):
    updates: Dict[str, Any] = Field(
        ...,
        description="Clés/valeurs à fusionner dans rag.config.json (hyperparamètres RAG).",
    )


app = FastAPI(title="Advanced RAG API", version="0.1.0")


@app.get("/health")
def health() -> HealthResponse:
    return HealthResponse(collection=COLLECTION, persist_dir=str(PERSIST_DIR))


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
    default_chunk_size, default_chunk_overlap = _ingest_defaults()
    chunk_size = payload.chunk_size if payload.chunk_size is not None else default_chunk_size
    chunk_overlap = payload.chunk_overlap if payload.chunk_overlap is not None else default_chunk_overlap
    with _rag_lock:
        docs, ids = build_documents_from_dir(
            data_dir,
            ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap),
        )
        rag.add_documents(docs, ids=ids)

    return IngestResponse(
        ingested_chunks=len(docs),
        collection=COLLECTION,
        persist_dir=str(PERSIST_DIR),
    )


@app.post(
    "/ingest/file",
    dependencies=[Depends(_require_api_key)],
    responses={400: {"description": "Invalid upload"}},
)
async def ingest_file(
    file: Annotated[UploadFile, File(...)],
    chunk_size: Annotated[Optional[int], Form()] = None,
    chunk_overlap: Annotated[Optional[int], Form()] = None,
) -> IngestResponse:
    filename = (file.filename or "upload").strip()
    suffix = Path(filename).suffix.lower()
    if suffix not in (".txt", ".md", ".pdf", ".docx"):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use: .txt, .md, .pdf, .docx")

    content = await file.read()
    max_bytes = 25 * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(status_code=400, detail="File too large (max 25MB)")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / filename
        tmp_path.write_bytes(content)

        rag = _get_rag()
        default_chunk_size, default_chunk_overlap = _ingest_defaults()
        use_chunk_size = chunk_size if chunk_size is not None else default_chunk_size
        use_chunk_overlap = chunk_overlap if chunk_overlap is not None else default_chunk_overlap
        with _rag_lock:
            docs, ids = build_documents_from_file(
                tmp_path,
                ChunkingConfig(chunk_size=use_chunk_size, chunk_overlap=use_chunk_overlap),
                source_name=filename,
            )
            rag.add_documents(docs, ids=ids)

    return IngestResponse(
        ingested_chunks=len(docs),
        collection=COLLECTION,
        persist_dir=str(PERSIST_DIR),
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


@app.post(
    "/config",
    dependencies=[Depends(_require_api_key)],
    summary="Met à jour les hyperparamètres RAG",
)
def update_config(payload: ConfigUpdateRequest) -> Dict[str, Any]:
    """
    Met à jour rag.config.json avec un patch d'hyperparamètres.
    - Fusionne les clés fournies avec la config actuelle.
    - Sauvegarde le JSON.
    - Réinitialise le moteur RAG pour prendre en compte la nouvelle config.
    """
    global _rag, _raw_file_config
    with _rag_lock:
        current = _get_file_config().copy()
        current.update(payload.updates)
        save_config_file(CONFIG_PATH, current)
        _raw_file_config = current
        _rag = None
    return {"ok": True, "config": current}

