import re
from typing import Any, Dict, List, Optional, Tuple
from langchain_core.documents import Document
from .text import rrf_fusion, tokenize


def route(query: str) -> str:
    q = query.strip()
    has_quotes = ('"' in q) or ("'" in q)
    has_digits = any(ch.isdigit() for ch in q)
    long_token = any(len(t) >= 12 for t in tokenize(q))
    if has_quotes or has_digits or long_token:
        return "bm25-heavy"
    return "hybrid"


def parse_self_query(query: str) -> Tuple[str, Dict[str, Any]]:
    q = query.strip()
    filters: Dict[str, Any] = {"source_contains": "", "page_min": None, "page_max": None}

    q, src = _extract_source(q)
    if src:
        filters["source_contains"] = src

    q, pmin, pmax = _extract_pages(q)
    if pmin is not None:
        filters["page_min"] = pmin
        filters["page_max"] = pmax

    q = re.sub(r"\s+", " ", q).strip()
    return q or query, filters


def _extract_source(q: str) -> Tuple[str, str]:
    m = re.search(r"\bsource\s*:\s*([^\s]+)", q, flags=re.IGNORECASE)
    if not m:
        return q, ""
    src = m.group(1).strip().strip('"').strip("'")
    q = re.sub(r"\bsource\s*:\s*[^\s]+", " ", q, flags=re.IGNORECASE).strip()
    return q, src


def _extract_pages(q: str) -> Tuple[str, Optional[int], Optional[int]]:
    m = re.search(r"\bpages?\s*(\d+)\s*-\s*(\d+)\b", q, flags=re.IGNORECASE)
    if m:
        p1, p2 = int(m.group(1)), int(m.group(2))
        q = re.sub(r"\bpages?\s*\d+\s*-\s*\d+\b", " ", q, flags=re.IGNORECASE).strip()
        return q, min(p1, p2), max(p1, p2)

    m = re.search(r"\b(?:page|p\.)\s*(\d+)\b", q, flags=re.IGNORECASE)
    if not m:
        return q, None, None
    p = int(m.group(1))
    q = re.sub(r"\b(?:page|p\.)\s*\d+\b", " ", q, flags=re.IGNORECASE).strip()
    return q, p, p


def multi_queries(query: str, *, enable: bool, max_queries: int) -> List[str]:
    if not enable:
        return [query]
    q = query.strip()
    variants = [q, re.sub(r"\s+", " ", q.lower()), re.sub(r"[^\w\s]", " ", q)]
    seen = set()
    out: List[str] = []
    for v in variants:
        v = v.strip()
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
        if len(out) >= max_queries:
            break
    return out


def dense_retrieve(vectorstore: Any, queries: List[str], *, top_k: int) -> List[str]:
    dense_docs: List[Document] = []
    for q in queries:
        dense_docs.extend(vectorstore.similarity_search(q, k=top_k))
    dense_ids: List[str] = []
    for d in dense_docs:
        doc_id = (d.metadata or {}).get("doc_id")
        if isinstance(doc_id, str):
            dense_ids.append(doc_id)
    seen = set()
    dense_top: List[str] = []
    for did in dense_ids:
        if did in seen:
            continue
        seen.add(did)
        dense_top.append(did)
        if len(dense_top) >= top_k:
            break
    return dense_top


def apply_metadata_filters(
    doc_ids: List[str],
    meta_by_id: Dict[str, Dict[str, Any]],
    *,
    source_contains: str = "",
    page_min: Optional[int] = None,
    page_max: Optional[int] = None,
) -> List[str]:
    out = _filter_by_source(doc_ids, meta_by_id, source_contains)
    return _filter_by_page(out, meta_by_id, page_min=page_min, page_max=page_max)


def _filter_by_source(doc_ids: List[str], meta_by_id: Dict[str, Dict[str, Any]], source_contains: str) -> List[str]:
    needle = source_contains.lower().strip()
    if not needle:
        return doc_ids
    return [
        doc_id
        for doc_id in doc_ids
        if needle in str(meta_by_id.get(doc_id, {}).get("source", "")).lower()
    ]


def _filter_by_page(
    doc_ids: List[str],
    meta_by_id: Dict[str, Dict[str, Any]],
    *,
    page_min: Optional[int],
    page_max: Optional[int],
) -> List[str]:
    if page_min is None and page_max is None:
        return doc_ids

    # If none of the candidate docs has page metadata (common for DOCX),
    # don't filter everything out.
    has_any_page = any(isinstance(meta_by_id.get(doc_id, {}).get("page"), int) for doc_id in doc_ids)
    if not has_any_page:
        return doc_ids

    out: List[str] = []
    for doc_id in doc_ids:
        page = meta_by_id.get(doc_id, {}).get("page")
        if not isinstance(page, int):
            continue
        if page_min is not None and page < page_min:
            continue
        if page_max is not None and page > page_max:
            continue
        out.append(doc_id)
    return out


def merge(route_name: str, bm25_ids: List[str], dense_ids: List[str], *, fused_k: int) -> List[str]:
    if route_name == "bm25-heavy":
        return rrf_fusion(bm25_ids, bm25_ids, dense_ids, limit=fused_k)
    return rrf_fusion(bm25_ids, dense_ids, limit=fused_k)

