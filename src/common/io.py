from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

def read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    """Read a CSV file safely, returning an empty DataFrame if it doesn't exist."""
    if not path.exists():
        return pd.DataFrame()
    kwargs.setdefault("low_memory", False)
    return pd.read_csv(path, **kwargs)

def write_csv(df: pd.DataFrame, path: Path, **kwargs: Any) -> None:
    """Write a DataFrame to CSV with standard UTF-8-sig encoding and directory creation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("index", False)
    kwargs.setdefault("encoding", "utf-8-sig")
    df.to_csv(path, **kwargs)

def read_json(path: Path) -> dict[str, Any]:
    """Read and parse a JSON file safely."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(data: Any, path: Path, **kwargs: Any) -> None:
    """Serialize data to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    kwargs.setdefault("ensure_ascii", False)
    kwargs.setdefault("indent", 2)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, **kwargs)

def load_model(path: Path) -> Any:
    """Load a serialized joblib model."""
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")
    return joblib.load(path)

def save_model(model: Any, path: Path) -> None:
    """Serialize a model using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)

def copy_file(src: Path, dst: Path) -> None:
    """Copy a file safely, creating parent folders if necessary."""
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
