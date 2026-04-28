from pathlib import Path
from typing import Any

import yaml


def dump(meta: dict[str, Any], body: str) -> str:
    yml = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False).strip()
    return f"---\n{yml}\n---\n\n{body.rstrip()}\n"


def write_note(path: Path, meta: dict[str, Any], body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump(meta, body), encoding="utf-8")
