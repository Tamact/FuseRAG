import numpy as np
import re
from typing import Iterable, List, Sequence

from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# LLM (instancié une seule fois)
llm = OllamaLLM(model="qwen2.5:3b", temperature=0.1, num_predict=180)


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


# 1. Préparation docs (tokenisés pour BM25)
docs = [
    "Qwen3-Embedding-8B excelle en multilingue MTEB.",
    "NV-Embed-v2 top anglais retrieval.",
    "RAG optimise avec reranking.",
    "Python pour IA locale Ollama.",
]
tokenized_docs = [tokenize(doc) for doc in docs]
bm25 = BM25Okapi(tokenized_docs)  # Sparse lexical

# Docs pour vector store
doc_objects = [Document(page_content=doc, metadata={"idx": i}) for i, doc in enumerate(docs)]
embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
vectorstore = Chroma.from_documents(doc_objects, embeddings)

# 2. Reranker
reranker = CrossEncoder('BAAI/bge-reranker-base')

# 3. Fusion RRF (Reciprocal Rank Fusion)
def rrf_fusion(*ranked_lists: Sequence[int], k: int = 60, limit: int = 10) -> List[int]:
    scores: dict[int, float] = {}
    for lst in ranked_lists:
        for rank, doc_idx in enumerate(lst):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    return [doc_idx for doc_idx, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:limit]]

# 4. Recherche hybride
def hybrid_search(query, top_k_bm25=5, top_k_dense=5):
    # BM25 sparse
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k_bm25].tolist()
    
    # Dense vectorielle
    dense_docs = vectorstore.similarity_search(query, k=top_k_dense)
    vector_top = [d.metadata["idx"] for d in dense_docs]
    
    # Fusion RRF
    fused_top = rrf_fusion(bm25_top, vector_top, limit=max(top_k_bm25, top_k_dense))
    return [doc_objects[i] for i in fused_top]

# 5. Reranking + Génération
def rag_pipeline(query):
    retrieved = hybrid_search(query)
    pairs = [[query, doc.page_content] for doc in retrieved]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)[:3]
    context_blocks = []
    used_docs = []
    for score, doc in reranked:
        idx = doc.metadata.get("idx", "?")
        used_docs.append(doc.page_content)
        context_blocks.append(f"[DOC {idx}] {doc.page_content}")
    context = "\n".join(context_blocks)

    prompt = f"""Tu es un assistant RAG strict.
Règles:
1) Réponds uniquement avec les informations présentes dans le CONTEXTE.
2) Si l'information manque, réponds exactement: "Je ne sais pas."
3) N'invente rien, même si tu "penses savoir".
4) Si tu réponds, cite au moins une source sous la forme [DOC n].
5) Réponse en français, courte.

QUESTION:
{query}

CONTEXTE:
{context}

RÉPONSE:
"""
    response = llm.invoke(prompt)
    return used_docs, response

# Test
query = "Meilleurs embeddings pour RAG multilingue ?"
docs_reranked, response = rag_pipeline(query)
print("Docs hybrides rerankés:", docs_reranked)
print("Réponse LLM:", response)
