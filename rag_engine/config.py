from dataclasses import dataclass
from pathlib import Path


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

    # Sparse backend
    enable_rust_bm25: bool = True
    bm25_k1: float = 1.5
    bm25_b: float = 0.75

    # Faithfulness controls
    rerank_threshold: float = 0.0  # set >0 to be stricter

    # Generation
    llm_model: str = "qwen2.5:3b"
    temperature: float = 0.1
    num_predict: int = 220

    # Prompt template (use {query} and {context})
    system_prompt_template: str = (
        "Tu es un assistant RAG strict.\n"
        "Règles:\n"
        "1) Utilise uniquement le CONTEXTE.\n"
        "2) Si l'information n'y est pas, réponds exactement: \"Je ne sais pas.\"\n"
        "3) Si tu réponds, ajoute une ligne \"Sources:\" et cite au moins un identifiant [DOC ...].\n"
        "4) Réponse courte, factuelle, en français.\n\n"
        "QUESTION:\n"
        "{query}\n\n"
        "CONTEXTE:\n"
        "{context}\n\n"
        "RÉPONSE:\n"
    )

    # If the model replies exactly "Je ne sais pas." while context is non-empty,
    # optionally retry once with a slightly less strict prompt.
    enable_idk_retry: bool = True
    idk_retry_prompt_template: str = (
        "Tu es un assistant RAG strict.\n"
        "Règles:\n"
        "1) Utilise uniquement le CONTEXTE.\n"
        "2) Si le CONTEXTE contient des éléments partiels, réponds avec ce que tu peux et précise la limite.\n"
        "3) Si aucune information pertinente n'est présente, réponds exactement: \"Je ne sais pas.\"\n"
        "4) Ajoute une ligne \"Sources:\" et cite au moins un identifiant [DOC ...] si tu réponds.\n"
        "5) Réponse courte, factuelle, en français.\n\n"
        "QUESTION:\n"
        "{query}\n\n"
        "CONTEXTE:\n"
        "{context}\n\n"
        "RÉPONSE:\n"
    )

    # If the model still replies "Je ne sais pas." but context is non-empty,
    # return an extractive fallback instead of "Je ne sais pas.".
    enable_idk_fallback: bool = True
    idk_fallback_max_sources: int = 3

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

