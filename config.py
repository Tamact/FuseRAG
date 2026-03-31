import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional
from rag_engine import RAGConfig


def load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return {}
    return json.loads(raw)


def save_config_file(path: Path, data: Dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# Champs pilotés par le .env (non obligatoires dans le JSON)
_ENV_DRIVEN_FIELDS = {
    "persist_dir",
    "collection_name",
    "embedding_model",
    "reranker_model",
    "llm_model",
}

# Tous les autres champs du dataclass doivent venir du JSON
_REQUIRED_JSON_FIELDS = {
    name
    for name in RAGConfig.__dataclass_fields__.keys()
    if name not in _ENV_DRIVEN_FIELDS
}


def build_rag_config(*, base: RAGConfig, overrides: Optional[Dict[str, Any]] = None) -> RAGConfig:
    if not overrides:
        raise RuntimeError(
            "rag.config.json est manquant ou vide : tous les hyperparamètres RAG doivent y être définis."
        )

    missing = _REQUIRED_JSON_FIELDS - set(overrides.keys())
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise RuntimeError(
            f"Clés manquantes dans rag.config.json pour RAGConfig: {missing_str}"
        )

    data = asdict(base)
    for k, v in overrides.items():
        if k not in data:
            continue
        data[k] = v
    # persist_dir may be provided as string
    if isinstance(data.get("persist_dir"), str):
        data["persist_dir"] = Path(data["persist_dir"])
    return RAGConfig(**data)

