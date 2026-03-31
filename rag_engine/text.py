import re
from typing import List
from .fast import has_rust, tokenize_fast


def tokenize(text: str) -> List[str]:
    if has_rust():
        return tokenize_fast(text)
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def rrf_fusion(*ranked_lists: List[str], k: int = 60, limit: int = 20) -> List[str]:
    scores: dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]]

