from __future__ import annotations

import ast
import re
from pathlib import Path


KEY_ALIASES = {
    "schoolinfoApiKey": "SCHOOLINFO_API_KEY",
    "neisApiKey": "NEIS_API_KEY",
    "kosisApiKey": "KOSIS_API_KEY",
    "datagokrApiKey": "DATAGOKR_API_KEY",
    "naverClientId": "NAVER_CLIENT_ID",
    "naverClientSecret": "NAVER_CLIENT_SECRET",
    "vworldapiKey": "VWORLD_API_KEY",
    "vworldApiKey": "VWORLD_API_KEY",
    "vworldKey": "VWORLD_API_KEY",
    "sgisServiceId": "SGIS_SERVICE_ID",
    "sgisSecureId": "SGIS_SECURE_ID",
    "sgisConsumerKey": "SGIS_SERVICE_ID",
    "sgisConsumerSecret": "SGIS_SECURE_ID",
}


def _normalize_key(name: str) -> str:
    return KEY_ALIASES.get(name, name).upper()


def _strip_quotes(value: str) -> str:
    value = value.strip().rstrip(";").strip()
    if "#" in value:
        in_single = False
        in_double = False
        for idx, char in enumerate(value):
            if char == "'" and not in_double:
                in_single = not in_single
            elif char == '"' and not in_single:
                in_double = not in_double
            elif char == "#" and not in_single and not in_double:
                value = value[:idx].strip()
                break
    value = value.rstrip(";").strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    return value


def _parse_object_literal(text: str) -> dict[str, str]:
    match = re.search(
        r"export\s+const\s+\w+\s*=\s*(\{.*?\})\s*;?\s*$",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return {}

    body = match.group(1)
    body = re.sub(r"([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:", r'\1"\2":', body)
    body = body.replace("'", '"')

    try:
        parsed = ast.literal_eval(body)
    except (SyntaxError, ValueError):
        return {}

    if not isinstance(parsed, dict):
        return {}

    return {
        _normalize_key(str(key)): str(value)
        for key, value in parsed.items()
        if value is not None
    }


def load_api_keys(env_path: str | Path = ".env") -> dict[str, str]:
    """Load API keys from either dotenv, export-const, or JS object style files.

    Supported examples:

    KEY=value
    export const neisApiKey = "..."
    export const apiKeys = { neisApiKey: "...", kosisApiKey: "..." }
    """

    path = Path(env_path)
    if not path.exists():
        return {}

    text = path.read_text(encoding="utf-8")
    keys: dict[str, str] = {}

    keys.update(_parse_object_literal(text))

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        export_const = re.match(
            r"^export\s+const\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$",
            line,
        )
        if export_const and not export_const.group(2).lstrip().startswith("{"):
            keys[_normalize_key(export_const.group(1))] = _strip_quotes(
                export_const.group(2)
            )
            continue

        dotenv = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+?)\s*$", line)
        if dotenv:
            keys[_normalize_key(dotenv.group(1))] = _strip_quotes(dotenv.group(2))

    return keys


if __name__ == "__main__":
    loaded = load_api_keys()
    for name, value in sorted(loaded.items()):
        print(f"{name}: {'set' if value else 'empty'} ({len(value)} chars)")
