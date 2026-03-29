import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from advanced_rag import AdvancedRAG


@dataclass(frozen=True)
class EvalCase:
    q: str
    must_contain: List[str]
    must_cite: bool = True
    gold_doc_ids: Optional[List[str]] = None


def load_eval_cases(path: Path) -> List[EvalCase]:
    cases: List[EvalCase] = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        q = str(obj["q"])
        must = [str(x) for x in (obj.get("must_contain") or [])]
        must_cite = bool(obj.get("must_cite", True))
        gold = [str(x) for x in (obj.get("gold_doc_ids") or [])]
        cases.append(EvalCase(q=q, must_contain=must, must_cite=must_cite, gold_doc_ids=gold))
    return cases


def recall_at_k(retrieved_texts: List[str], must_contain: List[str]) -> float:
    if not must_contain:
        return 1.0
    blob = "\n".join(retrieved_texts).lower()
    hits = 0
    for m in must_contain:
        if m.lower() in blob:
            hits += 1
    return hits / len(must_contain)


def _has_doc_citation(answer: str) -> bool:
    a = answer or ""
    return "[DOC " in a


def _faithfulness_simple(answer: str, context_texts: List[str]) -> float:
    """
    Very lightweight check: measures how much of the answer's content words
    appear in the provided context. Not a perfect metric, but useful as a guardrail.
    """
    a = (answer or "").lower()
    ctx = "\n".join(context_texts).lower()
    words = [w for w in a.replace("\n", " ").split() if w.isalpha() and len(w) >= 5]
    if not words:
        return 1.0
    hits = sum(1 for w in words if w in ctx)
    return hits / len(words)


def mrr_at_k(ranked_ids: List[str], gold_ids: List[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)
    for i, did in enumerate(ranked_ids[:k], start=1):
        if did in gold:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked_ids: List[str], gold_ids: List[str], k: int) -> float:
    if not gold_ids:
        return 0.0
    gold = set(gold_ids)

    def dcg(ids: List[str]) -> float:
        s = 0.0
        for i, did in enumerate(ids[:k], start=1):
            rel = 1.0 if did in gold else 0.0
            if rel > 0:
                s += rel / (math.log2(i + 1))
        return s

    import math

    ideal = dcg(list(gold_ids))
    if ideal <= 0.0:
        return 0.0
    return dcg(ranked_ids) / ideal

def run_eval(rag: AdvancedRAG, cases: List[EvalCase], *, k: int = 8) -> Dict[str, Any]:
    rows = []
    recalls = []
    faithful_scores = []
    cite_rates = []
    mrrs = []
    ndcgs = []
    for c in cases:
        ids, dbg = rag.retrieve(c.q)
        ranked, _ = rag.rerank(c.q, ids)
        chosen = [doc_id for _, doc_id in ranked[:k]]
        texts = [rag._texts_by_id.get(did, "") for did in chosen]  # pragmatic for local eval
        r = recall_at_k(texts, c.must_contain)
        recalls.append(r)
        answer, _debug = rag.answer(c.q)
        cite_ok = (not c.must_cite) or _has_doc_citation(answer)
        cite_rates.append(1.0 if cite_ok else 0.0)
        faith = _faithfulness_simple(answer, texts)
        faithful_scores.append(faith)

        mrr = mrr_at_k(chosen, c.gold_doc_ids or [], k)
        ndcg = ndcg_at_k(chosen, c.gold_doc_ids or [], k)
        mrrs.append(mrr)
        ndcgs.append(ndcg)
        rows.append(
            {
                "q": c.q,
                "recall": r,
                "faithfulness": faith,
                "cite_ok": cite_ok,
                "mrr": mrr,
                "ndcg": ndcg,
                "chosen": chosen[:5],
                "route": dbg.get("route"),
            }
        )
    avg = sum(recalls) / max(1, len(recalls))
    avg_faith = sum(faithful_scores) / max(1, len(faithful_scores))
    avg_cite = sum(cite_rates) / max(1, len(cite_rates))
    avg_mrr = sum(mrrs) / max(1, len(mrrs))
    avg_ndcg = sum(ndcgs) / max(1, len(ndcgs))
    return {
        "avg_recall": avg,
        "avg_mrr": avg_mrr,
        "avg_ndcg": avg_ndcg,
        "avg_faithfulness": avg_faith,
        "cite_rate": avg_cite,
        "n": len(cases),
        "rows": rows,
    }

