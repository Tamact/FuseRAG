from typing import Any, Dict, List, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from .fast import bm25_index_from_tokenized, has_rust
from .text import tokenize


class SparseIndex:
    def __init__(
        self,
        *,
        enable_rust_bm25: bool,
        bm25_k1: float,
        bm25_b: float,
    ) -> None:
        self.enable_rust_bm25 = enable_rust_bm25
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b

        self._bm25_py: Optional[BM25Okapi] = None
        self._bm25_rust: Any = None

    def rebuild(self, tokenized_docs: List[List[str]]) -> None:
        self._bm25_py = BM25Okapi(tokenized_docs) if tokenized_docs else None
        self._bm25_rust = None
        if self.enable_rust_bm25 and has_rust() and tokenized_docs:
            self._bm25_rust = bm25_index_from_tokenized(tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)

    def ensure(self, tokenized_docs: List[List[str]]) -> None:
        if self.enable_rust_bm25 and has_rust():
            if self._bm25_rust is None:
                self._bm25_rust = bm25_index_from_tokenized(tokenized_docs, k1=self.bm25_k1, b=self.bm25_b)
            return
        if self._bm25_py is None:
            self._bm25_py = BM25Okapi(tokenized_docs)

    # Note: selection logic lives in bm25_topk_doc_ids(); this class only holds backends.


def bm25_topk_doc_ids(
    *,
    sparse: SparseIndex,
    doc_ids: List[str],
    tokenized_docs: List[List[str]],
    queries: List[str],
    top_k: int,
) -> List[str]:
    sparse.ensure(tokenized_docs)

    if sparse.enable_rust_bm25 and has_rust():
        ranked_lists: List[List[int]] = []
        for q in queries:
            idxs = sparse._bm25_rust.topk(tokenize(q), int(top_k))
            ranked_lists.append(list(idxs))
        merged: Dict[int, float] = {}
        for lst in ranked_lists:
            for rank, i in enumerate(lst):
                merged[i] = merged.get(i, 0.0) + 1.0 / (60.0 + rank + 1.0)
        top = [i for i, _ in sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]]
        return [doc_ids[i] for i in top]

    bm25_scores = np.zeros(len(doc_ids), dtype=np.float32)
    for q in queries:
        bm25_scores += sparse._bm25_py.get_scores(tokenize(q)).astype(np.float32)
    bm25_top_idx = np.argsort(bm25_scores)[::-1][:top_k].tolist()
    return [doc_ids[i] for i in bm25_top_idx]

