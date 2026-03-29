import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Optional

from advanced_rag import RAGConfig


def load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    raw = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not raw:
        return {}
    return json.loads(raw)


def build_rag_config(*, base: RAGConfig, overrides: Optional[Dict[str, Any]] = None) -> RAGConfig:
    data = asdict(base)
    for k, v in (overrides or {}).items():
        if k not in data:
            continue
        data[k] = v
    # persist_dir may be provided as string
    if isinstance(data.get("persist_dir"), str):
        data["persist_dir"] = Path(data["persist_dir"])
    return RAGConfig(**data)

