import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .paths import intermediate_dir, raw_dir


def fix_mojibake(s: str) -> str:
    """FB DYI exports often double-encode UTF-8 as Latin-1. Restore."""
    if not s:
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def stable_id(timestamp: int, text: str) -> str:
    h = hashlib.sha256(text[:200].encode("utf-8", errors="ignore")).hexdigest()[:10]
    return f"{timestamp}-{h}"


def normalize_one(raw: dict, source_path: str) -> dict | None:
    timestamp = raw.get("timestamp")
    if not isinstance(timestamp, int) or timestamp <= 0:
        return None

    text_parts: list[str] = []
    for d in raw.get("data") or []:
        post = d.get("post") if isinstance(d, dict) else None
        if post:
            text_parts.append(fix_mojibake(post))
    title = raw.get("title")
    if title and not text_parts:
        text_parts.append(fix_mojibake(title))
    text = "\n".join(text_parts).strip()

    links: list[str] = []
    for att in raw.get("attachments") or []:
        for d in (att or {}).get("data") or []:
            if not isinstance(d, dict):
                continue
            ec = d.get("external_context") or {}
            url = ec.get("url")
            if url:
                links.append(url)

    post_id = stable_id(timestamp, text or title or "")
    date = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d")

    return {
        "post_id": post_id,
        "date": date,
        "timestamp": timestamp,
        "text": text,
        "links": links,
        "source_path": source_path,
    }


def iter_raw_files() -> Iterator[Path]:
    """Recursively find all *.json under _raw/, so users can drop unzipped
    DYI archives (which contain subfolders like posts/, your_posts_*) directly.
    """
    yield from sorted(raw_dir().rglob("*.json"))


def run() -> int:
    intermediate_dir().mkdir(parents=True, exist_ok=True)
    out_path = intermediate_dir() / "01_parsed.jsonl"

    seen: set[str] = set()
    records: list[dict] = []

    for f in iter_raw_files():
        data = json.loads(f.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            entries = data.get("posts") or data.get("data") or [data]
        elif isinstance(data, list):
            entries = data
        else:
            continue

        rel_path = str(f.relative_to(raw_dir()))
        for e in entries:
            if not isinstance(e, dict):
                continue
            r = normalize_one(e, rel_path)
            if r is None or r["post_id"] in seen:
                continue
            seen.add(r["post_id"])
            records.append(r)

    records.sort(key=lambda r: r["post_id"])

    with out_path.open("w", encoding="utf-8") as out:
        for r in records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[parse] {len(records)} unique posts → {out_path.name}")
    return len(records)
