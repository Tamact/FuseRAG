import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from .answering import build_context_lines, build_prompt, compress_context, pick_answer_docs
from .cache import cache_put
from .config import RAGConfig
from .retrieval import (
    apply_metadata_filters,
    dense_retrieve,
    merge,
    multi_queries,
    parse_self_query,
    route,
)
from .sparse import SparseIndex, bm25_topk_doc_ids
from .store import add_to_vectorstore, load_from_chroma
from .text import tokenize


class AdvancedRAG:
    _IDK = "Je ne sais pas."
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

        self._sparse = SparseIndex(
            enable_rust_bm25=self.cfg.enable_rust_bm25,
            bm25_k1=self.cfg.bm25_k1,
            bm25_b=self.cfg.bm25_b,
        )
        self._bm25_doc_ids: List[str] = []
        self._texts_by_id: Dict[str, str] = {}
        self._meta_by_id: Dict[str, Dict[str, Any]] = {}
        self._bm25_tokenized: List[List[str]] = []

        self._rerank_cache: Dict[Tuple[str, str], float] = {}
        self._answer_cache: Dict[Tuple[Any, ...], Tuple[str, Dict[str, Any]]] = {}

    @classmethod
    def build_default(cls, config: Optional[RAGConfig] = None) -> "AdvancedRAG":
        cfg = config or RAGConfig()

        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_ollama import OllamaLLM

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

    def _cache_put(self, cache: Dict[Any, Any], key: Any, value: Any, max_size: int) -> None:
        cache_put(cache, key, value, max_size)

    def _rebuild_sparse(self) -> None:
        self._sparse = SparseIndex(
            enable_rust_bm25=self.cfg.enable_rust_bm25,
            bm25_k1=self.cfg.bm25_k1,
            bm25_b=self.cfg.bm25_b,
        )
        self._sparse.rebuild(self._bm25_tokenized)

    def _ensure_sparse(self) -> None:
        self._sparse.ensure(self._bm25_tokenized)

    def load_index_from_vectorstore(self, limit: int = 50_000) -> int:
        doc_ids, texts_by_id, meta_by_id, tokenized_docs = load_from_chroma(self.vectorstore, limit=limit)
        self._texts_by_id = texts_by_id
        self._meta_by_id = meta_by_id
        self._bm25_doc_ids = doc_ids
        self._bm25_tokenized = tokenized_docs
        self._rebuild_sparse()
        return len(doc_ids)

    def add_documents(self, docs: List[Document], *, ids: List[str]) -> None:
        added_ids, added_texts, added_metas, added_tokenized = add_to_vectorstore(self.vectorstore, docs, ids=ids)
        for doc_id, text, meta, tok in zip(added_ids, added_texts, added_metas, added_tokenized):
            self._texts_by_id[doc_id] = text
            self._meta_by_id[doc_id] = meta
            self._bm25_doc_ids.append(doc_id)
            self._bm25_tokenized.append(tok)

        self._rebuild_sparse()
        self._rerank_cache.clear()
        self._answer_cache.clear()

    def _filter_by_source(self, doc_ids: List[str], needle: str) -> List[str]:
        return apply_metadata_filters(doc_ids, self._meta_by_id, source_contains=needle)

    def _filter_by_page(self, doc_ids: List[str], *, page_min: Optional[int], page_max: Optional[int]) -> List[str]:
        return apply_metadata_filters(doc_ids, self._meta_by_id, page_min=page_min, page_max=page_max)

    def _apply_metadata_filters(
        self, doc_ids: List[str], *, source_contains: str = "", page_min: Optional[int] = None, page_max: Optional[int] = None
    ) -> List[str]:
        return apply_metadata_filters(
            doc_ids,
            self._meta_by_id,
            source_contains=source_contains,
            page_min=page_min,
            page_max=page_max,
        )

    def _route(self, query: str) -> str:
        return route(query)

    def _parse_self_query(self, query: str) -> Tuple[str, Dict[str, Any]]:
        return parse_self_query(query)

    def _multi_queries(self, query: str) -> List[str]:
        return multi_queries(query, enable=self.cfg.enable_multi_query, max_queries=self.cfg.max_queries)

    def _bm25_retrieve(self, queries: List[str]) -> List[str]:
        self._ensure_sparse()
        return bm25_topk_doc_ids(
            sparse=self._sparse,
            doc_ids=self._bm25_doc_ids,
            tokenized_docs=self._bm25_tokenized,
            queries=queries,
            top_k=self.cfg.top_k_bm25,
        )

    def _dense_retrieve(self, queries: List[str]) -> List[str]:
        return dense_retrieve(self.vectorstore, queries, top_k=self.cfg.top_k_dense)

    def _merge_retrieval(self, route: str, bm25_ids: List[str], dense_ids: List[str]) -> List[str]:
        return merge(route, bm25_ids, dense_ids, fused_k=self.cfg.fused_k)

    def retrieve(
        self,
        query: str,
        *,
        source_contains: str = "",
        page_min: Optional[int] = None,
        page_max: Optional[int] = None,
        enable_self_query: bool = True,
    ) -> Tuple[List[str], Dict[str, Any]]:
        if enable_self_query:
            parsed_query, parsed_filters = self._parse_self_query(query)
        else:
            parsed_query, parsed_filters = query, {"source_contains": source_contains, "page_min": page_min, "page_max": page_max}

        if source_contains:
            parsed_filters["source_contains"] = source_contains
        if page_min is not None:
            parsed_filters["page_min"] = page_min
        if page_max is not None:
            parsed_filters["page_max"] = page_max

        route = self._route(parsed_query)
        queries = self._multi_queries(parsed_query)

        bm25_top = self._bm25_retrieve(queries)
        dense_top = self._dense_retrieve(queries)
        fused = self._merge_retrieval(route, bm25_top, dense_top)
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
            "bm25_top_ids": bm25_top[:10],
            "dense_top_ids": dense_top[:10],
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
        return compress_context(
            query,
            context_lines,
            enabled=self.cfg.enable_context_compression,
            max_chars=self.cfg.compression_max_chars,
        )

    def _pick_answer_docs(self, ranked: List[Tuple[float, str]]) -> List[Tuple[float, str]]:
        return pick_answer_docs(ranked, threshold=self.cfg.rerank_threshold, k=self.cfg.answer_k)

    def _build_context_lines(self, picked: List[Tuple[float, str]]) -> Tuple[List[str], List[Dict[str, Any]]]:
        return build_context_lines(picked, texts_by_id=self._texts_by_id, meta_by_id=self._meta_by_id)

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
            return ans, {**dbg, "cache": "hit"}

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
            out = (self._IDK, {"retrieve": dbg_retr, "rerank": dbg_rr, "chosen": [], "cache": "miss"})
            self._cache_put(self._answer_cache, cache_key, out, self.cfg.answer_cache_size)
            return out

        context_lines, chosen = self._build_context_lines(picked)
        context = self._compress_context(query, context_lines)
        prompt = build_prompt(query, context, system_prompt_template=self.cfg.system_prompt_template)
        try:
            resp = str(self.llm.invoke(prompt)).strip()
            llm_error: Optional[str] = None
        except Exception as e:
            resp = self._IDK
            llm_error = f"{type(e).__name__}: {e}"

        if llm_error is None and self.cfg.enable_idk_retry and resp == self._IDK and context.strip():
            retry_prompt = build_prompt(query, context, system_prompt_template=self.cfg.idk_retry_prompt_template)
            try:
                resp = str(self.llm.invoke(retry_prompt)).strip()
            except Exception as e:
                resp = self._IDK
                llm_error = f"{type(e).__name__}: {e}"

        if self.cfg.enable_idk_fallback and resp == self._IDK and context.strip():
            src_ids = [c.get("doc_id") for c in (chosen or []) if c.get("doc_id")]
            src_ids = src_ids[: max(1, int(self.cfg.idk_fallback_max_sources))]
            excerpt = context.strip()
            if len(excerpt) > 600:
                excerpt = excerpt[:600].rstrip() + "…"
            resp = "D’après le contexte, voici les éléments disponibles:\n" + excerpt + "\n\nSources: " + ", ".join(
                [f"[DOC {sid}]" for sid in src_ids]
            )

        debug: Dict[str, Any] = {"retrieve": dbg_retr, "rerank": dbg_rr, "chosen": chosen, "cache": "miss"}
        if llm_error is not None:
            debug["llm_error"] = llm_error
        out = (resp, debug)
        self._cache_put(self._answer_cache, cache_key, out, self.cfg.answer_cache_size)
        return out

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

