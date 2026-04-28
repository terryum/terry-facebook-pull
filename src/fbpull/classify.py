import json
import os

from . import llm
from .paths import intermediate_dir

_SYSTEM = """당신은 한 사람의 페이스북 글을 분석합니다. 각 글에 대해 다음 JSON 만 출력하세요. 다른 텍스트나 마크다운 펜스 없이.

{
  "type": "thought | lesson | event | quote | announcement",
  "primary_topic": "한국어 단어 1-3개 (예: '학습', '연구 방법론', '인간관계')",
  "keep_for_synthesis": true | false
}

기준:
- thought: 자신의 사고/철학/의견 → keep_for_synthesis=true
- lesson: 경험에서 얻은 깨달음 → true
- event: 단순 사건/일상 보고 → false
- quote: 인용 → false
- announcement: 공지/홍보 → false
- 의미 없는 단편 → false"""

_VALID = {"thought", "lesson", "event", "quote", "announcement"}


def _stub(text: str) -> dict:
    n = len(text)
    if n < 100:
        return {"type": "event", "primary_topic": "일상", "keep_for_synthesis": False}
    if n < 200:
        return {"type": "thought", "primary_topic": "사고", "keep_for_synthesis": True}
    return {"type": "lesson", "primary_topic": "성찰", "keep_for_synthesis": True}


def classify_one(model: str, post_id: str, text: str, no_llm: bool) -> dict:
    if no_llm:
        return _stub(text)

    cache_dir = intermediate_dir() / "llm_cache" / model
    key = f"{post_id}_{llm.text_hash(text)}"
    cached = llm.cache_get(cache_dir, key)
    if cached:
        return cached

    result = llm.call_json(model, _SYSTEM, text, max_tokens=200)
    if result.get("type") not in _VALID:
        result["type"] = "thought"
    if "keep_for_synthesis" not in result:
        result["keep_for_synthesis"] = False
    if "primary_topic" not in result:
        result["primary_topic"] = ""

    llm.cache_put(cache_dir, key, result)
    return result


def run(no_llm: bool = False) -> dict[str, int]:
    in_path = intermediate_dir() / "02_filtered.jsonl"
    out_path = intermediate_dir() / "03_classified.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"Run `fbpull filter` first; missing {in_path}")

    model = os.environ.get("FBPULL_CLASSIFY_MODEL", "claude-haiku-4-5")
    type_counts: dict[str, int] = {}
    keep_count = 0
    total = 0

    with in_path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as out:
        for line in f:
            rec = json.loads(line)
            if not rec.get("kept"):
                continue
            total += 1
            cls = classify_one(model, rec["post_id"], rec["text"], no_llm)
            rec.update(cls)
            type_counts[cls["type"]] = type_counts.get(cls["type"], 0) + 1
            if cls.get("keep_for_synthesis"):
                keep_count += 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[classify] total={total} keep_for_synthesis={keep_count}")
    for t, n in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"  {t}: {n}")
    return type_counts
