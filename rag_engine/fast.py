from __future__ import annotations
from typing import Any, List


try:
    import rag_fast  # type: ignore
except Exception:
    rag_fast = None


def has_rust() -> bool:
    return rag_fast is not None


def tokenize_fast(text: str) -> List[str]:
    if rag_fast is None:
        raise RuntimeError("rag_fast is not installed")
    return rag_fast.tokenize(text)


def bm25_index_from_tokenized(tokenized_docs: List[List[str]], *, k1: float, b: float) -> Any:
    if rag_fast is None:
        raise RuntimeError("rag_fast is not installed")
    return rag_fast.BM25Index.from_tokenized(tokenized_docs, float(k1), float(b))

