import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from langchain_core.documents import Document


@dataclass(frozen=True)
class ChunkingConfig:
    chunk_size: int = 900  # characters
    chunk_overlap: int = 150


def _safe_relpath(path: Path, base: Path) -> str:
    try:
        return str(path.relative_to(base))
    except Exception:
        return str(path)


def _load_pdf(path: Path) -> List[Document]:
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except Exception as e:  # pragma: no cover
        raise RuntimeError('Pour charger des PDF, installe "pypdf" et "langchain-community".') from e
    loader = PyPDFLoader(str(path))
    return loader.load()


def _load_docx(path: Path) -> List[Document]:
    try:
        from langchain_community.document_loaders import UnstructuredWordDocumentLoader
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            'Pour charger des DOCX, installe "unstructured" (+ dépendances) et "langchain-community".'
        ) from e
    loader = UnstructuredWordDocumentLoader(str(path))
    return loader.load()


def _load_file_as_documents(path: Path, *, data_dir: Path) -> List[Document]:
    ext = path.suffix.lower()
    rel = _safe_relpath(path, data_dir)

    if ext in (".txt", ".md"):
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            return []
        return [Document(page_content=text, metadata={"source": rel})]

    if ext == ".pdf":
        loaded = _load_pdf(path)
        out: List[Document] = []
        for d in loaded:
            meta = dict(d.metadata or {})
            meta.setdefault("source", rel)
            if "page" in meta and isinstance(meta["page"], int):
                meta["page"] = meta["page"] + 1
            out.append(Document(page_content=d.page_content, metadata=meta))
        return out

    if ext == ".docx":
        loaded = _load_docx(path)
        out = []
        for d in loaded:
            meta = dict(d.metadata or {})
            meta.setdefault("source", rel)
            out.append(Document(page_content=d.page_content, metadata=meta))
        return out

    return []


def read_documents(data_dir: Path) -> List[Document]:
    docs: List[Document] = []
    for p in sorted(data_dir.rglob("*")):
        if not p.is_file():
            continue
        docs.extend(_load_file_as_documents(p, data_dir=data_dir))

    return docs


def chunk_text(text: str, *, chunk_size: int, chunk_overlap: int) -> List[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >=0 and < chunk_size")

    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - chunk_overlap)
    return chunks


def make_doc_id(source: str, chunk_id: int, content: str) -> str:
    h = hashlib.sha1()
    h.update(source.encode("utf-8", errors="ignore"))
    h.update(b"\n")
    h.update(str(chunk_id).encode("utf-8"))
    h.update(b"\n")
    h.update(content.encode("utf-8", errors="ignore"))
    return h.hexdigest()[:20]


def build_documents_from_dir(data_dir: Path, cfg: ChunkingConfig) -> Tuple[List[Document], List[str]]:
    raw_docs = read_documents(data_dir)
    docs: List[Document] = []
    ids: List[str] = []
    for raw in raw_docs:
        text = raw.page_content or ""
        if not text.strip():
            continue
        source = str((raw.metadata or {}).get("source", "unknown"))
        page = (raw.metadata or {}).get("page")
        chunks = chunk_text(text, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        for i, chunk in enumerate(chunks):
            # include page in id generation so chunks from different pages don't collide
            source_key = f"{source}#p{page}" if page is not None else source
            doc_id = make_doc_id(source_key, i, chunk)
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "source": source,
                        **({"page": page} if page is not None else {}),
                        "chunk_id": i,
                    },
                )
            )
            ids.append(doc_id)
    return docs, ids


def build_documents_from_file(
    file_path: Path, cfg: ChunkingConfig, *, source_name: Optional[str] = None
) -> Tuple[List[Document], List[str]]:
    
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(str(file_path))

    base_dir = file_path.parent
    raw_docs = _load_file_as_documents(file_path, data_dir=base_dir)

    return _chunk_documents(raw_docs, cfg, default_source=file_path.name, source_name=source_name)


def _chunk_documents(
    raw_docs: List[Document],
    cfg: ChunkingConfig,
    *,
    default_source: str,
    source_name: Optional[str],
) -> Tuple[List[Document], List[str]]:
    docs: List[Document] = []
    ids: List[str] = []

    for raw in raw_docs:
        text = raw.page_content or ""
        if not text.strip():
            continue
        source = source_name or str((raw.metadata or {}).get("source", default_source))
        page = (raw.metadata or {}).get("page")
        chunks = chunk_text(text, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        for i, chunk in enumerate(chunks):
            source_key = f"{source}#p{page}" if page is not None else source
            doc_id = make_doc_id(source_key, i, chunk)
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "doc_id": doc_id,
                        "source": source,
                        **({"page": page} if page is not None else {}),
                        "chunk_id": i,
                    },
                )
            )
            ids.append(doc_id)

    return docs, ids

