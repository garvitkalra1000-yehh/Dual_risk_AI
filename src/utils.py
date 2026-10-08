from __future__ import annotations

from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_CLEAN = ROOT / "data" / "clean"
DATA_OUTPUTS = ROOT / "data" / "outputs"
MODELS_DIR = ROOT / "models"


def ensure_dirs() -> None:
    for d in [DATA_RAW, DATA_CLEAN, DATA_OUTPUTS, MODELS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_json(path: Path, default=None):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
