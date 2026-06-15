from __future__ import annotations

import os
from pathlib import Path


def load_api_keys() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    env_path = root / ".env"
    values: dict[str, str] = {}
    if env_path.exists():
        for raw_line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key.startswith("export const "):
                key = key.replace("export const ", "", 1).strip()
            elif key.startswith("export "):
                key = key.replace("export ", "", 1).strip()
            value = value.strip().rstrip(";").strip().strip('"').strip("'")
            if key:
                values[key] = value
    aliases = {
        "kosisApiKey": "KOSIS_API_KEY",
        "datagokrApiKey": "DATAGOKR_API_KEY",
        "schoolinfoApiKey": "SCHOOLINFO_API_KEY",
        "neisApiKey": "NEIS_API_KEY",
    }
    for src, dst in aliases.items():
        if src in values and dst not in values:
            values[dst] = values[src]
    for key, value in os.environ.items():
        if key.endswith("_API_KEY") or key.endswith("_SERVICE_KEY"):
            values[key] = value
    return values
