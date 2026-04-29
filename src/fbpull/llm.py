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


def call_json(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
    cache_system: bool = False,
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call Anthropic, return a dict.

    If `schema` is provided, uses Anthropic's tool-use to force a
    structured response that matches the schema (no JSON parsing
    errors possible). Otherwise falls back to parsing JSON from the
    text response, which is fragile with smaller models.

    cache_system: when True, mark the system prompt for ephemeral prompt
    caching. Useful when calling many times in a row with the same long
    system prompt (classify a few thousand posts) — first call pays full
    price, subsequent within the cache TTL pay 10% on the system tokens.
    """
    if cache_system:
        system_arg = [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    else:
        system_arg = system

    if schema is not None:
        tool = {
            "name": "submit",
            "description": "Submit the structured result.",
            "input_schema": schema,
        }
        msg = client().messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_arg,
            tools=[tool],
            tool_choice={"type": "tool", "name": "submit"},
            messages=[{"role": "user", "content": user}],
        )
        for block in msg.content:
            if getattr(block, "type", None) == "tool_use":
                return dict(block.input)
        raise RuntimeError("Model returned no tool_use block")

    msg = client().messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_arg,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(getattr(b, "text", "") for b in msg.content if getattr(b, "type", None) == "text")
    return json.loads(_extract_json(text))
