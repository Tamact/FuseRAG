from typing import Any, Dict, List, Optional, Tuple
from .text import tokenize


def compress_context(query: str, context_lines: List[str], *, enabled: bool, max_chars: int) -> str:
    if not enabled:
        return "\n".join(context_lines)
    joined = "\n".join(context_lines)
    if len(joined) <= max_chars:
        return joined
    q_tokens = set(tokenize(query))
    kept = [line for line in context_lines if len(q_tokens & set(tokenize(line))) >= 1]
    out = "\n".join(kept) if kept else joined
    return out[:max_chars]


def build_prompt(query: str, context: str, *, system_prompt_template: str) -> str:
    try:
        return system_prompt_template.format(query=query, context=context)
    except Exception:
        # Fallback: never fail answering because of a bad template
        return system_prompt_template + f"\n\nQUESTION:\n{query}\n\nCONTEXTE:\n{context}\n"


def build_context_lines(
    picked: List[Tuple[float, str]],
    *,
    texts_by_id: Dict[str, str],
    meta_by_id: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[Dict[str, Any]]]:
    context_lines: List[str] = []
    chosen: List[Dict[str, Any]] = []
    for score, doc_id in picked:
        text = texts_by_id.get(doc_id, "")
        meta = meta_by_id.get(doc_id, {})
        src = meta.get("source", "unknown")
        chunk = meta.get("chunk_id", "?")
        page = meta.get("page")
        page_part = f", page={page}" if isinstance(page, int) else ""
        context_lines.append(f"[DOC {doc_id}] (source={src}{page_part}, chunk={chunk}) {text}")
        chosen.append({"doc_id": doc_id, "score": score, "meta": meta})
    return context_lines, chosen


def pick_answer_docs(ranked: List[Tuple[float, str]], *, threshold: float, k: int) -> List[Tuple[float, str]]:
    return [x for x in ranked if x[0] >= threshold][:k]

