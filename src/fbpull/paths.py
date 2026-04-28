import os
from pathlib import Path


def vault_root() -> Path:
    p = os.environ.get("OBSIDIAN_VAULT")
    if not p:
        raise RuntimeError(
            "OBSIDIAN_VAULT env var not set. Copy .env.example to .env and edit."
        )
    return Path(p).expanduser().resolve()


def fb_root() -> Path:
    return vault_root() / "vault" / "Private" / "Facebook"


def raw_dir() -> Path:
    return fb_root() / "_raw"


def intermediate_dir() -> Path:
    return fb_root() / "_intermediate"


def archive_dir() -> Path:
    return fb_root() / "Archive"


def synthesized_dir() -> Path:
    return fb_root() / "Synthesized"


def index_path() -> Path:
    return fb_root() / "_index.md"


def ensure_dirs() -> None:
    for d in (raw_dir(), intermediate_dir(), archive_dir(), synthesized_dir()):
        d.mkdir(parents=True, exist_ok=True)
