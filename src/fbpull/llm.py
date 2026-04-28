import hashlib
import json
import re
from pathlib import Path
from typing import Any

from anthropic import Anthropic

_client: Anthropic | None = None


def client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def cache_get(cache_dir: Path, key: str) -> dict[str, Any] | None:
    f = cache_dir / f"{key}.json"
    if f.exists():
        return json.loads(f.read_text(encoding="utf-8"))
    return None


def cache_put(cache_dir: Path, key: str, value: dict[str, Any]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(
        json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8"
    )


_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def _extract_json(text: str) -> str:
    text = text.strip()
    m = _JSON_FENCE.search(text)
    if m:
        return m.group(1).strip()
    # Fall back: first {...} block
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text


def call_json(model: str, system: str, user: str, max_tokens: int = 1024) -> dict[str, Any]:
    msg = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text")
    return json.loads(_extract_json(text))
