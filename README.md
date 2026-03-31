# RAG (hybride + rerank) — version avancée

Ce repo contient un pipeline RAG local :

- Retrieval hybride **BM25 + dense (Chroma)**
- Fusion **RRF**
- **Reranking cross-encoder**
- Génération avec **Ollama** + prompt strict + **citations**
- Index **persistant** (Chroma)

## API (FastAPI)

Installer:

```powershell
uv add fastapi uvicorn
```

## Configuration (sans toucher au code)

- Secrets & modèles: via `.env` (voir `.env.example`)
- Paramètres RAG (temperature, top_k, seuils…): via `rag.config.json`

Démarrage rapide:

1) Copie `rag.config.json.example` → `rag.config.json`
2) Modifie `rag.config.json`
3) (Optionnel) Change le chemin via `RAG_CONFIG_PATH` dans `.env`

Lancer en local:

```powershell
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Docs Swagger:

- `http://localhost:8000/docs`

Sécuriser avec une clé API (optionnel):

```powershell
$env:RAG_API_KEY="change-me"
uv run uvicorn main:app --host 0.0.0.0 --port 8000
```

Puis ajoute le header `X-API-Key` sur chaque requête.

## Ingestion

- Dossier (serveur):
  - `POST /ingest`
- Upload direct:
  - `POST /ingest/file` (multipart/form-data: `file`, `chunk_size`, `chunk_overlap`)
