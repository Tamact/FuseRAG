## RAG avancé (hybride + rerank) — API FastAPI

Ce repo fournit un moteur RAG local, prêt à l’emploi, avec :

- **Retrieval hybride** BM25 + dense (Chroma, embeddings Hugging Face)
- **Fusion RRF** des scores
- **Reranking cross-encoder** (sentence-transformers)
- Génération avec **Ollama** (LLM local) + prompt strict + **citations [DOC …]**
- Index **persistant** (Chroma)
- Pipeline modulaire (`rag_engine`) + bindings Rust (`rag_fast`) optionnels pour accélérer BM25

---

## 1. Installation

Dans un dossier de projet, installe d’abord les dépendances avec `uv` :

```powershell
uv sync
```

Assure‑toi d’avoir **Ollama** installé et le modèle voulu téléchargé (par ex. `qwen2.5:3b`).

---

## 2. Configuration (sans toucher au code)

- **Secrets & modèles** : via `.env` (voir `.env.example`)
  - `RAG_PERSIST_DIR`, `RAG_COLLECTION`
  - `RAG_LLM_MODEL`, `RAG_EMBEDDING_MODEL`, `RAG_RERANKER_MODEL`
  - `RAG_CONFIG_PATH` (chemin de `rag.config.json`)
  - `RAG_API_KEY` (optionnel, pour sécuriser l’API)
- **Hyperparamètres RAG** : via `rag.config.json`
  - top‑k BM25 / dense, taille de contexte, seuil de rerank, température, etc.

### Étapes

1. Copie `rag.config.json.example` → `rag.config.json`.
2. Copie `.env.example` → `.env` et remplis les variables (surtout les modèles et chemins).
3. (Optionnel) Change le chemin du fichier de config via `RAG_CONFIG_PATH` dans `.env`.

Tous les hyperparamètres (sauf les modèles/paths) **doivent** être définis dans `rag.config.json`.

---

## 3. Lancer l’API

```powershell
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Docs interactives (Swagger / OpenAPI) :

- `http://localhost:8000/docs`

### Clé API (optionnelle)

Pour activer l’authentification par clé :

```powershell
$env:RAG_API_KEY="change-me"
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Ensuite, ajoute le header `X-API-Key: change-me` à chaque requête.

---

## 4. Ingestion de documents

Deux façons d’indexer les documents (txt, md, pdf, docx) :

- **Dossier côté serveur** :  
  - `POST /ingest`  
    - body JSON :
    ```json
    {
      "data_dir": "data",
      "chunk_size": 900,
      "chunk_overlap": 150
    }
    ```
    - si `chunk_size` / `chunk_overlap` sont omis, les valeurs de `rag.config.json` sont utilisées.

- **Upload direct de fichier** :  
  - `POST /ingest/file` (multipart/form-data)  
    - champs :
      - `file` : le fichier à indexer
      - `chunk_size` (optionnel)
      - `chunk_overlap` (optionnel)

---

## 5. Poser des questions (RAG)

- Endpoint : `POST /ask`
- Body JSON minimal :

```json
{
  "q": "Ta question ici",
  "source_contains": "",
  "page_min": null,
  "page_max": null,
  "debug": false
}
```

- Réponse :
  - `answer` : réponse générée (en français, avec citations `[DOC ...]` si possible)
  - `sources` : liste des chunks utilisés (doc_id, source, page, chunk_id, score)
  - `debug` : détails du retrieval/rerank (si `debug=true`)

Le système applique un **prompt strict** : s’il n’y a pas assez de contexte, il répond exactement **"Je ne sais pas."**, avec un mécanisme de retry et de fallback extractif configurable dans `rag.config.json`.

---

## 6. Inspection de l’index

- Endpoint : `GET /inspect-index`
- Paramètres :
  - `limit` : nombre max de lignes (par défaut 200)
  - `source_contains` : filtre par nom de source (substring)
  - `no_preview` : si `true`, supprime l’aperçu de texte
  - `preview_chars` : longueur de l’aperçu texte

Permet de vérifier **ce qui est réellement indexé** (sources, pages, aperçus de chunks).

---

## 7. Mise à jour dynamique des hyperparamètres

- Endpoint : `POST /config` (protégé par la clé API s’il y en a une)
- Body :

```json
{
  "updates": {
    "top_k_bm25": 30,
    "temperature": 0.2
  }
}
```

Effet :

- fusionne ces valeurs dans `rag.config.json`  
- recharge la config en mémoire  
- réinitialise le moteur RAG (il sera reconstruit à la prochaine requête)

---

## 8. Stack technique

- **Python** + **FastAPI**
- **Chroma** pour les embeddings (Hugging Face) et le stockage persistant
- **rank-bm25** + **rag_fast** (Rust/PyO3) pour un BM25 accéléré
- **sentence-transformers** pour le reranking
- **Ollama** comme LLM local

Le projet est pensé pour être **open source et générique** : n’importe qui peut cloner, configurer `.env` + `rag.config.json`, ingérer ses propres docs et bénéficier d’un RAG hybride robuste sans modifier le code.
