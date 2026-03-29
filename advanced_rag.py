import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def rrf_fusion(*ranked_lists: Sequence[str], k: int = 60, limit: int = 20) -> List[str]:
    scores: Dict[str, float] = {}
    for lst in ranked_lists:
        for rank, doc_id in enumerate(lst):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return [doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]]


@dataclass(frozen=True)
class RAGConfig:
    persist_dir: Path = Path("chroma_db")
    collection_name: str = "rag_chunks"

    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model: str = "BAAI/bge-reranker-base"

    # Retrieval
    top_k_bm25: int = 20
    top_k_dense: int = 20
    fused_k: int = 30
    rerank_k: int = 8
    answer_k: int = 4

    # Faithfulness controls
    rerank_threshold: float = 0.0  # set >0 to be stricter

    # Generation
    llm_model: str = "qwen2.5:3b"
    temperature: float = 0.1
    num_predict: int = 220

    # Routing / query expansion
    enable_multi_query: bool = True
    max_queries: int = 3

    # Self-query (LLM-based parsing of filters)
    enable_llm_self_query: bool = False

    # Context compression (reduce noise before final answer)
    enable_context_compression: bool = True
    compression_max_chars: int = 900

    # Performance: caches (set to 0 to disable)
    rerank_cache_size: int = 10_000
    answer_cache_size: int = 1_000


class AdvancedRAG:
    """
    Advanced-ish local RAG:
    - Persistent dense index (Chroma)
    - BM25 sparse index in-memory
    - RRF fusion
    - Cross-encoder rerank + threshold gating
    - Strict prompt with citations
    - Debug traces (scores, chosen docs)
    """

    def __init__(
        self,
        config: RAGConfig,
        *,
        vectorstore: Any,
        embeddings: Any,
        llm: Any,
    ) -> None:
        self.cfg = config
        self.vectorstore = vectorstore
        self.embeddings = embeddings
        self.llm = llm

        self.reranker = CrossEncoder(self.cfg.reranker_model)

        self._bm25: Optional[BM25Okapi] = None
        self._bm25_doc_ids: List[str] = []
        self._texts_by_id: Dict[str, str] = {}
        self._meta_by_id: Dict[str, Dict[str, Any]] = {}
        self._bm25_tokenized: List[List[str]] = []

        self._rerank_cache: Dict[Tuple[str, str], float] = {}
        self._answer_cache: Dict[Tuple[Any, ...], Tuple[str, Dict[str, Any]]] = {}

    @classmethod
    def build_default(cls, config: Optional[RAGConfig] = None) -> "AdvancedRAG":
        cfg = config or RAGConfig()

        # Imports here to keep this file usable even if optional deps missing.
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import OllamaLLM

        # Prefer the maintained Chroma package if available; fall back otherwise.
        try:
            from langchain_chroma import Chroma  # type: ignore
        except Exception:  # pragma: no cover
            from langchain_community.vectorstores import Chroma  # type: ignore

        embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model)
        vectorstore = Chroma(
            collection_name=cfg.collection_name,
            embedding_function=embeddings,
            persist_directory=str(cfg.persist_dir),
        )
        llm = OllamaLLM(model=cfg.llm_model, temperature=cfg.temperature, num_predict=cfg.num_predict)
        return cls(cfg, vectorstore=vectorstore, embeddings=embeddings, llm=llm)

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        if not self._bm25_tokenized:
            texts = [self._texts_by_id[doc_id] for doc_id in self._bm25_doc_ids]
            self._bm25_tokenized = [tokenize(t) for t in texts]
        self._bm25 = BM25Okapi(self._bm25_tokenized)

    def load_index_from_vectorstore(self, limit: int = 50_000) -> int:
        """
        Rebuilds in-memory BM25 + text/meta maps from the persistent vectorstore.
        """
        col = self.vectorstore._collection  # chromadb Collection
        n = col.count()
        got = 0
        offset = 0
        batch = 1000

        self._texts_by_id.clear()
        self._meta_by_id.clear()
        self._bm25_doc_ids.clear()
        self._bm25 = None
        self._bm25_tokenized.clear()

        while offset < n and got < limit:
            res = col.get(include=["documents", "metadatas"], limit=batch, offset=offset)
            ids = res.get("ids") or []
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            for doc_id, text, meta in zip(ids, docs, metas):
                if not isinstance(doc_id, str) or not isinstance(text, str):
                    continue
                self._texts_by_id[doc_id] = text
                self._meta_by_id[doc_id] = meta or {}
                self._bm25_doc_ids.append(doc_id)
                self._bm25_tokenized.append(tokenize(text))
                got += 1
                if got >= limit:
                    break
            offset += batch

        self._ensure_bm25()
        return got

    def _rebuild_bm25(self) -> None:
        self._bm25 = BM25Okapi(self._bm25_tokenized) if self._bm25_tokenized else None

    def _cache_put(self, cache: Dict[Any, Any], key: Any, value: Any, max_size: int) -> None:
        if max_size <= 0:
            return
        if key in cache:
            cache[key] = value
            return
        cache[key] = value
        if len(cache) > max_size:
            # pop oldest (insertion order)
            cache.pop(next(iter(cache)))

    def add_documents(self, docs: List[Document], *, ids: List[str]) -> None:
        texts = [d.page_content for d in docs]
        metas = [d.metadata for d in docs]
        self.vectorstore.add_texts(texts=texts, metadatas=metas, ids=ids)
        # Incremental in-memory update (avoid full reload from Chroma on each ingest).
        for doc_id, text, meta in zip(ids, texts, metas):
            if not isinstance(doc_id, str) or not isinstance(text, str):
                continue
            self._texts_by_id[doc_id] = text
            self._meta_by_id[doc_id] = meta or {}
            self._bm25_doc_ids.append(doc_id)
            self._bm25_tokenized.append(tokenize(text))
        self._rebuild_bm25()
        # Invalidate caches because corpus changed
        self._rerank_cache.clear()
        self._answer_cache.clear()

    def _filter_by_source(self, doc_ids: List[str], needle: str) -> List[str]:
        if not needle:
            return doc_ids
        n = needle.lower().strip()
        return [
            doc_id
            for doc_id in doc_ids
            if n in str(self._meta_by_id.get(doc_id, {}).get("source", "")).lower()
        ]

    def _filter_by_page(
        self, doc_ids: List[str], *, page_min: Optional[int], page_max: Optional[int]
    ) -> List[str]:
        if page_min is None and page_max is None:
            return doc_ids

        out: List[str] = []
        for doc_id in doc_ids:
            page = self._meta_by_id.get(doc_id, {}).get("page")
            if not isinstance(page, int):
                continue
            if page_min is not None and page < page_min:
                continue
            if page_max is not None and page > page_max:
                continue
            out.append(doc_id)
        return out

    def _apply_metadata_filters(
        self,
        doc_ids: List[str],
        *,
        source_contains: str = "",
        page_min: Optional[int] = None,
        page_max: Optional[int] = None,
    ) -> List[str]:
        out = self._filter_by_source(doc_ids, source_contains)
        return self._filter_by_page(out, page_min=page_min, page_max=page_max)

    def _route(self, query: str) -> str:
        """
        Simple router: favor BM25 when query contains many exact tokens.
        """
        q = query.strip()
        has_quotes = ('"' in q) or ("'" in q)
        has_digits = any(ch.isdigit() for ch in q)
        long_token = any(len(t) >= 12 for t in tokenize(q))
        if has_quotes or has_digits or long_token:
            return "bm25-heavy"
        return "hybrid"

    def _extract_source_filter(self, q: str) -> Tuple[str, str]:
        m = re.search(r"\bsource\s*:\s*([^\s]+)", q, flags=re.IGNORECASE)
        if not m:
            return q, ""
        source_contains = m.group(1).strip().strip('"').strip("'")
        q = re.sub(r"\bsource\s*:\s*[^\s]+", " ", q, flags=re.IGNORECASE).strip()
        return q, source_contains

    def _extract_page_filter(self, q: str) -> Tuple[str, Optional[int], Optional[int]]:
        m = re.search(r"\bpages?\s*(\d+)\s*-\s*(\d+)\b", q, flags=re.IGNORECASE)
        if m:
            p1, p2 = int(m.group(1)), int(m.group(2))
            q = re.sub(r"\bpages?\s*\d+\s*-\s*\d+\b", " ", q, flags=re.IGNORECASE).strip()
            return q, min(p1, p2), max(p1, p2)

        m = re.search(r"\b(?:page|p\.)\s*(\d+)\b", q, flags=re.IGNORECASE)
        if m:
            p = int(m.group(1))
            q = re.sub(r"\b(?:page|p\.)\s*\d+\b", " ", q, flags=re.IGNORECASE).strip()
            return q, p, p

        return q, None, None

    def _extract_trailing_source(self, q: str) -> Tuple[str, str]:
        m = re.search(r"\b(?:dans|fichier)\s+([^\n\r]+)$", q, flags=re.IGNORECASE)
        if not m:
            return q, ""
        source_contains = m.group(1).strip().strip('"').strip("'")
        q = re.sub(r"\b(?:dans|fichier)\s+[^\n\r]+$", " ", q, flags=re.IGNORECASE).strip()
        return q, source_contains

    def _parse_self_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Heuristic 'self-query' from natural language:
        - 'source:xxx' or 'dans xxx' or 'fichier xxx' => source_contains
        - 'page 3' or 'p. 3' or 'pages 3-5' => page range (PDF only)
        Returns (cleaned_query, filters)
        """
        q = query.strip()
        filters: Dict[str, Any] = {"source_contains": "", "page_min": None, "page_max": None}

        q, source = self._extract_source_filter(q)
        if source:
            filters["source_contains"] = source

        q, pmin, pmax = self._extract_page_filter(q)
        if pmin is not None:
            filters["page_min"] = pmin
            filters["page_max"] = pmax

        if not filters["source_contains"]:
            q, source2 = self._extract_trailing_source(q)
            if source2:
                filters["source_contains"] = source2

        q = re.sub(r"\s+", " ", q).strip()
        return q or query, filters

    def _llm_parse_self_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        """
        Uses the LLM to extract structured filters + cleaned query.
        Output schema:
          {"query": "...", "source_contains": "...", "page_min": 3, "page_max": 5}
        """
        schema = (
            '{ "query": string, "source_contains": string, "page_min": integer|null, "page_max": integer|null }'
        )
        prompt = f"""Tu es un assistant qui convertit une question utilisateur en JSON.
Retourne uniquement un JSON valide (sans texte autour) conforme au schéma:
{schema}

Règles:
- Si aucun filtre n'est présent, laisse source_contains vide et page_min/page_max à null.
- Détecte les patterns: source:xxx, pages 3-5, page 3, p. 3, "dans <fichier>".
- Le champ "query" doit contenir la question nettoyée (sans les filtres).

Question:
{query}
"""
        raw = str(self.llm.invoke(prompt)).strip()
        try:
            obj = json.loads(raw)
        except Exception:
            # Fallback: keep heuristic behavior if LLM output isn't parseable
            return self._parse_self_query(query)

        cleaned = str(obj.get("query") or query).strip() or query
        filters: Dict[str, Any] = {
            "source_contains": str(obj.get("source_contains") or "").strip(),
            "page_min": obj.get("page_min"),
            "page_max": obj.get("page_max"),
        }
        if not isinstance(filters["page_min"], int):
            filters["page_min"] = None
        if not isinstance(filters["page_max"], int):
            filters["page_max"] = None
        return cleaned, filters

    def _multi_queries(self, query: str) -> List[str]:
        if not self.cfg.enable_multi_query:
            return [query]

        # Cheap, deterministic expansions (no extra LLM call).
        q = query.strip()
        variants = [q]
        variants.append(re.sub(r"\s+", " ", q.lower()))
        variants.append(re.sub(r"[^\w\s]", " ", q))
        # Dedup while preserving order
        seen = set()
        out = []
        for v in variants:
            v = v.strip()
            if not v or v in seen:
                continue
            seen.add(v)
            out.append(v)
            if len(out) >= self.cfg.max_queries:
                break
        return out

    def _bm25_retrieve(self, queries: List[str]) -> List[str]:
        bm25_scores = np.zeros(len(self._bm25_doc_ids), dtype=np.float32)
        for q in queries:
            bm25_scores += self._bm25.get_scores(tokenize(q)).astype(np.float32)
        bm25_top_idx = np.argsort(bm25_scores)[::-1][: self.cfg.top_k_bm25].tolist()
        return [self._bm25_doc_ids[i] for i in bm25_top_idx]

    def _dense_retrieve(self, queries: List[str]) -> List[str]:
        dense_docs = []
        for q in queries:
            dense_docs.extend(self.vectorstore.similarity_search(q, k=self.cfg.top_k_dense))
        dense_ids: List[str] = []
        for d in dense_docs:
            doc_id = (d.metadata or {}).get("doc_id")
            if isinstance(doc_id, str):
                dense_ids.append(doc_id)

        # Dedup preserving order
        seen = set()
        dense_top_ids: List[str] = []
        for did in dense_ids:
            if did in seen:
                continue
            seen.add(did)
            dense_top_ids.append(did)
            if len(dense_top_ids) >= self.cfg.top_k_dense:
                break
        return dense_top_ids

    def _merge_retrieval(self, route: str, bm25_ids: List[str], dense_ids: List[str]) -> List[str]:
        if route == "bm25-heavy":
            return rrf_fusion(bm25_ids, bm25_ids, dense_ids, limit=self.cfg.fused_k)
        return rrf_fusion(bm25_ids, dense_ids, limit=self.cfg.fused_k)

    def retrieve(
        self,
        query: str,
        *,
        source_contains: str = "",
        page_min: Optional[int] = None,
        page_max: Optional[int] = None,
        enable_self_query: bool = True,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Returns ranked doc_ids and debug info.
        """
        self._ensure_bm25()
        if self._bm25 is None:
            raise RuntimeError("BM25 index not initialized. Did you ingest documents?")

        parsed_query = query
        parsed_filters = {"source_contains": source_contains, "page_min": page_min, "page_max": page_max}
        if enable_self_query:
            if self.cfg.enable_llm_self_query:
                parsed_query, parsed_filters = self._llm_parse_self_query(query)
            else:
                parsed_query, parsed_filters = self._parse_self_query(query)
            if source_contains:
                parsed_filters["source_contains"] = source_contains
            if page_min is not None:
                parsed_filters["page_min"] = page_min
            if page_max is not None:
                parsed_filters["page_max"] = page_max

        route = self._route(parsed_query)
        queries = self._multi_queries(parsed_query)

        bm25_top_ids = self._bm25_retrieve(queries)
        dense_top_ids = self._dense_retrieve(queries)
        fused = self._merge_retrieval(route, bm25_top_ids, dense_top_ids)

        fused = self._apply_metadata_filters(
            fused,
            source_contains=str(parsed_filters.get("source_contains") or ""),
            page_min=parsed_filters.get("page_min"),
            page_max=parsed_filters.get("page_max"),
        )

        debug = {
            "route": route,
            "queries": queries,
            "parsed_query": parsed_query,
            "filters": parsed_filters,
            "bm25_top_ids": bm25_top_ids[:10],
            "dense_top_ids": dense_top_ids[:10],
            "fused_top_ids": fused[:10],
        }
        return fused, debug

    def rerank(self, query: str, doc_ids: List[str]) -> Tuple[List[Tuple[float, str]], Dict[str, Any]]:
        pairs: List[List[str]] = []
        kept_ids: List[str] = []
        cached: List[Tuple[float, str]] = []
        for doc_id in doc_ids[: self.cfg.rerank_k]:
            text = self._texts_by_id.get(doc_id)
            if not text:
                continue
            key = (query, doc_id)
            if key in self._rerank_cache:
                cached.append((self._rerank_cache[key], doc_id))
            else:
                kept_ids.append(doc_id)
                pairs.append([query, text])

        scored: List[Tuple[float, str]] = []
        if pairs:
            scores = self.reranker.predict(pairs).tolist()
            for s, doc_id in zip(scores, kept_ids):
                self._cache_put(self._rerank_cache, (query, doc_id), float(s), self.cfg.rerank_cache_size)
                scored.append((float(s), doc_id))

        ranked = sorted(cached + scored, key=lambda x: x[0], reverse=True)
        debug = {
            "rerank_count": len(ranked),
            "top_scores": ranked[: min(5, len(ranked))],
            "cache_hits": len(cached),
            "cache_misses": len(scored),
        }
        return ranked, debug

    def _compress_context(self, query: str, context_lines: List[str]) -> str:
        if not self.cfg.enable_context_compression:
            return "\n".join(context_lines)

        joined = "\n".join(context_lines)
        if len(joined) <= self.cfg.compression_max_chars:
            return joined

        # Extractive compression: keep only the most query-relevant lines per DOC.
        q_tokens = set(tokenize(query))
        kept: List[str] = []
        for line in context_lines:
            tokens = set(tokenize(line))
            overlap = len(q_tokens & tokens)
            if overlap >= 1:
                kept.append(line)
        out = "\n".join(kept) if kept else joined
        return out[: self.cfg.compression_max_chars]

    def answer(
        self,
        query: str,
        *,
        source_contains: str = "",
        page_min: Optional[int] = None,
        page_max: Optional[int] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        cache_key = (
            query,
            source_contains,
            page_min,
            page_max,
            self.cfg.top_k_bm25,
            self.cfg.top_k_dense,
            self.cfg.fused_k,
            self.cfg.rerank_k,
            self.cfg.answer_k,
            self.cfg.rerank_threshold,
            self.cfg.enable_multi_query,
            self.cfg.max_queries,
            self.cfg.enable_llm_self_query,
            self.cfg.enable_context_compression,
            self.cfg.compression_max_chars,
            self.cfg.llm_model,
            self.cfg.temperature,
            self.cfg.num_predict,
        )
        if cache_key in self._answer_cache:
            ans, dbg = self._answer_cache[cache_key]
            dbg = {**dbg, "cache": "hit"}
            return ans, dbg

        retrieved_ids, dbg_retr = self.retrieve(
            query,
            source_contains=source_contains,
            page_min=page_min,
            page_max=page_max,
            enable_self_query=True,
        )
        ranked, dbg_rr = self.rerank(query, retrieved_ids)
        picked = self._pick_answer_docs(ranked)
        if not picked:
            return "Je ne sais pas.", {"retrieve": dbg_retr, "rerank": dbg_rr, "chosen": []}

        context_lines, chosen = self._build_context_lines(picked)
        context = self._compress_context(query, context_lines)

        prompt = f"""Tu es un assistant RAG strict.
Règles:
1) Utilise uniquement le CONTEXTE.
2) Si l'information n'y est pas, réponds exactement: "Je ne sais pas."
3) Si tu réponds, ajoute une ligne "Sources:" et cite au moins un identifiant [DOC ...].
4) Réponse courte, factuelle, en français.

QUESTION:
{query}

CONTEXTE:
{context}

RÉPONSE:
"""
        resp = self.llm.invoke(prompt)
        out = (str(resp), {"retrieve": dbg_retr, "rerank": dbg_rr, "chosen": chosen, "cache": "miss"})
        self._cache_put(self._answer_cache, cache_key, out, self.cfg.answer_cache_size)
        return out

    def _pick_answer_docs(self, ranked: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
        filtered = [x for x in ranked if x[0] >= self.cfg.rerank_threshold]
        return filtered[: self.cfg.answer_k]

    def _build_context_lines(self, picked: List[Tuple[float, str]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        context_lines: List[str] = []
        chosen: List[Dict[str, Any]] = []
        for score, doc_id in picked:
            text = self._texts_by_id.get(doc_id, "")
            meta = self._meta_by_id.get(doc_id, {})
            src = meta.get("source", "unknown")
            chunk = meta.get("chunk_id", "?")
            page = meta.get("page")
            page_part = f", page={page}" if isinstance(page, int) else ""
            context_lines.append(f"[DOC {doc_id}] (source={src}{page_part}, chunk={chunk}) {text}")
            chosen.append({"doc_id": doc_id, "score": score, "meta": meta})
        return context_lines, chosen

    def dump_debug(self, debug: Dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding="utf-8")

    def list_index(
        self,
        *,
        limit: int = 200,
        source_contains: str = "",
        with_preview: bool = True,
        preview_chars: int = 120,
    ) -> List[Dict[str, Any]]:
        """
        List documents currently loaded in the in-memory maps (after load_index_from_vectorstore()).
        """
        needle = source_contains.lower().strip()
        rows: List[Dict[str, Any]] = []
        for doc_id in self._bm25_doc_ids:
            meta = self._meta_by_id.get(doc_id, {})
            src = str(meta.get("source", ""))
            if needle and needle not in src.lower():
                continue
            row: Dict[str, Any] = {
                "doc_id": doc_id,
                "source": src,
                "page": meta.get("page"),
                "chunk_id": meta.get("chunk_id"),
            }
            if with_preview:
                text = self._texts_by_id.get(doc_id, "")
                row["preview"] = (text[:preview_chars] + "…") if len(text) > preview_chars else text
            rows.append(row)
            if len(rows) >= limit:
                break
        return rows

