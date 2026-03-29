import numpy as np
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# 1. Préparation docs (tokenisés pour BM25)
docs = [
    "Qwen3-Embedding-8B excelle en multilingue MTEB.",
    "NV-Embed-v2 top anglais retrieval.",
    "RAG optimise avec reranking.",
    "Python pour IA locale Ollama.",
]
tokenized_docs = [doc.split() for doc in docs]
bm25 = BM25Okapi(tokenized_docs)  # Sparse lexical

# Docs pour vector store
doc_objects = [Document(page_content=doc, metadata={"idx": i}) for i, doc in enumerate(docs)]
embeddings = HuggingFaceEmbeddings(model_name="Qwen/Qwen3-Embedding-0.6B")
vectorstore = Chroma.from_documents(doc_objects, embeddings)

# 2. Reranker
reranker = CrossEncoder('BAAI/bge-reranker-base')

# 3. Fusion RRF (Reciprocal Rank Fusion)
def rrf_fusion(bm25_results, vector_results, k=60):
    scores = {}
    for doc_idx in bm25_results + vector_results:
        scores[doc_idx] = 1 / (k + bm25_results.index(doc_idx)) if doc_idx in bm25_results else 0
        scores[doc_idx] += 1 / (k + vector_results.index(doc_idx)) if doc_idx in vector_results else 0
    fused = sorted(scores, key=scores.get, reverse=True)[:5]
    return [doc_objects[i] for i in fused]

# 4. Recherche hybride
def hybrid_search(query, top_k_bm25=5, top_k_dense=5):
    # BM25 sparse
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top = np.argsort(bm25_scores)[::-1][:top_k_bm25].tolist()
    
    # Dense vectorielle
    dense_docs = vectorstore.similarity_search(query, k=top_k_dense)
    vector_top = [d.metadata["idx"] for d in dense_docs]
    
    # Fusion RRF
    fused_docs = rrf_fusion(bm25_top, vector_top)
    return fused_docs

# 5. Reranking + Génération
def rag_pipeline(query):
    retrieved = hybrid_search(query)
    pairs = [[query, doc.page_content] for doc in retrieved]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, retrieved), key=lambda x: x[0], reverse=True)[:3]
    context = "\n".join([doc[1].page_content for doc in reranked])
    prompt = f"""Tu es un assistant RAG strict.
Règles:
1) Réponds uniquement avec les informations présentes dans le CONTEXTE.
2) Si le CONTEXTE ne contient pas la réponse, réponds exactement: "Je ne sais pas."
3) N'invente rien.

QUESTION:
{query}

CONTEXTE:
{context}
"""
    llm = OllamaLLM(model="qwen2.5:3b", temperature=0.1, num_predict=120)
    response = llm.invoke(prompt)
    return [doc[1].page_content for doc in reranked], response

# Test
query = "Meilleurs embeddings pour RAG multilingue ?"
docs_reranked, response = rag_pipeline(query)
print("Docs hybrides rerankés:", docs_reranked)
print("Réponse LLM:", response)
