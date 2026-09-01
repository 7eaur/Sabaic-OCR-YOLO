from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_classes(path: str | Path = "config/classes.json") -> dict:
    cfg = load_json(path)
    classes = cfg["classes"]
    ids = [c["id"] for c in classes]
    if ids != list(range(len(classes))):
        raise ValueError("Class IDs must be contiguous and start at 0.")
    if cfg["num_classes"] != len(classes):
        raise ValueError("num_classes does not match classes length.")
    return cfg
