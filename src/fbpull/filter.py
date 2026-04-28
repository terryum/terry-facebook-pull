import json
import re
from collections import Counter

from .paths import intermediate_dir

_URL_ONLY = re.compile(r"^\s*(https?://\S+\s*)+$")

# Facebook system messages start with the user's name (1-5 title-case
# English words) followed by an action verb. Examples:
#   "Terry Taewoong Um shared a post."
#   "Terry Taewoong Um Shared from Instagram"
#   "Jane Doe added 3 new photos."
_SYSTEM_PREFIX = re.compile(
    r"^[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+){0,4}\s+"
    r"(shared|Shared|posted|liked|commented|added|wrote|changed|tagged|updated)\b"
)


def reason_for(rec: dict) -> str | None:
    text = (rec.get("text") or "").strip()
    if not text:
        return "empty"
    if _URL_ONLY.match(text):
        return "link_only"
    if _SYSTEM_PREFIX.match(text):
        return "system"
    if len(text) < 10:
        return "too_short"
    return None


def run() -> tuple[int, int]:
    in_path = intermediate_dir() / "01_parsed.jsonl"
    out_path = intermediate_dir() / "02_filtered.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"Run `fbpull parse` first; missing {in_path}")

    kept = 0
    dropped = 0
    reasons: Counter[str] = Counter()

    with in_path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as out:
        for line in f:
            rec = json.loads(line)
            reason = reason_for(rec)
            rec["kept"] = reason is None
            rec["drop_reason"] = reason
            if rec["kept"]:
                kept += 1
            else:
                dropped += 1
                reasons[reason] += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[filter] kept={kept} dropped={dropped}")
    for r, n in reasons.most_common():
        print(f"  {r}: {n}")
    return kept, dropped
