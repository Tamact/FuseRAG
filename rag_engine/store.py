from typing import Any, Dict, List, Tuple
from langchain_core.documents import Document
from .text import tokenize


def load_from_chroma(vectorstore: Any, *, limit: int) -> Tuple[List[str], Dict[str, str], Dict[str, Dict[str, Any]], List[List[str]]]:
    col = vectorstore._collection
    n = col.count()
    got = 0
    offset = 0
    batch = 1000

    texts_by_id: Dict[str, str] = {}
    meta_by_id: Dict[str, Dict[str, Any]] = {}
    doc_ids: List[str] = []
    tokenized_docs: List[List[str]] = []

    while offset < n and got < limit:
        res = col.get(include=["documents", "metadatas"], limit=batch, offset=offset)
        ids = res.get("ids") or []
        docs = res.get("documents") or []
        metas = res.get("metadatas") or []
        for doc_id, text, meta in zip(ids, docs, metas):
            if not isinstance(doc_id, str) or not isinstance(text, str):
                continue
            texts_by_id[doc_id] = text
            meta_by_id[doc_id] = meta or {}
            doc_ids.append(doc_id)
            tokenized_docs.append(tokenize(text))
            got += 1
            if got >= limit:
                break
        offset += batch

    return doc_ids, texts_by_id, meta_by_id, tokenized_docs


def add_to_vectorstore(
    vectorstore: Any, docs: List[Document], *, ids: List[str]
) -> Tuple[List[str], List[str], List[Dict[str, Any]], List[List[str]]]:
    texts = [d.page_content for d in docs]
    metas = [d.metadata for d in docs]
    vectorstore.add_texts(texts=texts, metadatas=metas, ids=ids)

    added_ids: List[str] = []
    added_texts: List[str] = []
    added_metas: List[Dict[str, Any]] = []
    added_tokenized: List[List[str]] = []
    for doc_id, text, meta in zip(ids, texts, metas):
        if not isinstance(doc_id, str) or not isinstance(text, str):
            continue
        added_ids.append(doc_id)
        added_texts.append(text)
        added_metas.append(meta or {})
        added_tokenized.append(tokenize(text))

    return added_ids, added_texts, added_metas, added_tokenized

