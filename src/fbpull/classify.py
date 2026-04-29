import json
import os
from datetime import datetime, timezone

from . import llm
from . import taxonomy as taxonomy_mod
from .paths import intermediate_dir


_BASE_SYSTEM = """당신은 한 사용자의 페이스북 글들을 분석합니다. 각 글에 대해 submit 도구를 호출해 구조화된 결과를 반환하세요.

# 사용자 정보
{bio}

# 카테고리 (반드시 아래 중 정확히 한 이름)
{categories}

기준:
- type=thought: 자신의 사고/철학/의견 → keep_for_synthesis=true
- type=lesson: 경험에서 얻은 깨달음 → true
- type=event: 단순 사건/일상 보고 → false
- type=quote: 인용 → false
- type=announcement: 공지/홍보 → false
- 의미 없는 단편 → false"""

_LEGACY_SYSTEM = """당신은 한 사람의 페이스북 글을 분석합니다. submit 도구를 호출해 구조화된 결과를 반환하세요.

기준:
- type=thought: 자신의 사고/철학/의견 → keep_for_synthesis=true
- type=lesson: 경험에서 얻은 깨달음 → true
- type=event: 단순 사건/일상 보고 → false
- type=quote: 인용 → false
- type=announcement: 공지/홍보 → false
- 의미 없는 단편 → false"""

_VALID_TYPES = {"thought", "lesson", "event", "quote", "announcement"}


def _stub(text: str, tax: taxonomy_mod.Taxonomy | None) -> dict:
    n = len(text)
    cat = tax.fallback_category().name if tax else ""
    if n < 100:
        return {"type": "event", "primary_topic": "일상", "category": cat, "keep_for_synthesis": False}
    if n < 200:
        return {"type": "thought", "primary_topic": "사고", "category": cat, "keep_for_synthesis": True}
    return {"type": "lesson", "primary_topic": "성찰", "category": cat, "keep_for_synthesis": True}


def _build_system_prompt(tax: taxonomy_mod.Taxonomy | None) -> str:
    if tax is None:
        return _LEGACY_SYSTEM
    return _BASE_SYSTEM.format(bio=tax.bio.strip(), categories=tax.category_names_for_prompt())


def _classify_schema(tax: taxonomy_mod.Taxonomy | None) -> dict:
    base: dict = {
        "type": "object",
        "properties": {
            "type": {
                "type": "string",
                "enum": list(_VALID_TYPES),
                "description": "분류 타입",
            },
            "primary_topic": {
                "type": "string",
                "description": "한국어 단어 1-3개",
            },
            "keep_for_synthesis": {
                "type": "boolean",
                "description": "합성 후보 여부",
            },
        },
        "required": ["type", "primary_topic", "keep_for_synthesis"],
    }
    if tax is not None:
        base["properties"]["category"] = {
            "type": "string",
            "enum": [c.name for c in tax.categories],
            "description": "카테고리 (위 enum 중 하나)",
        }
        base["required"].append("category")
    return base


def _validate(result: dict, tax: taxonomy_mod.Taxonomy | None) -> dict:
    if result.get("type") not in _VALID_TYPES:
        result["type"] = "thought"
    result.setdefault("keep_for_synthesis", False)
    result.setdefault("primary_topic", "")
    if tax is not None:
        cat_name = result.get("category", "")
        if not tax.category_by_name(cat_name):
            result["category"] = tax.fallback_category().name
    return result


def classify_one(
    model: str,
    post_id: str,
    text: str,
    tax: taxonomy_mod.Taxonomy | None,
    no_llm: bool,
) -> dict:
    if no_llm:
        return _stub(text, tax)

    cache_dir = intermediate_dir() / "llm_cache" / model
    tax_hash = tax.hash if tax else "no-tax"
    key = f"{post_id}_{llm.text_hash(text)}_{tax_hash}"
    cached = llm.cache_get(cache_dir, key)
    if cached:
        return cached

    system = _build_system_prompt(tax)
    schema = _classify_schema(tax)
    result = llm.call_json(
        model, system, text, max_tokens=400, cache_system=True, schema=schema
    )
    result = _validate(result, tax)

    llm.cache_put(cache_dir, key, result)
    return result


def run(no_llm: bool = False) -> dict[str, int]:
    in_path = intermediate_dir() / "02_filtered.jsonl"
    out_path = intermediate_dir() / "03_classified.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"Run `fbpull filter` first; missing {in_path}")

    tax = taxonomy_mod.load()
    if tax is None:
        print("[classify] no _taxonomy.md found in vault — running in legacy mode (no category/era)")
    else:
        print(f"[classify] taxonomy: {len(tax.categories)} categories, {len(tax.eras)} eras (hash {tax.hash})")

    model = os.environ.get("FBPULL_CLASSIFY_MODEL", "claude-haiku-4-5")
    type_counts: dict[str, int] = {}
    cat_counts: dict[str, int] = {}
    era_counts: dict[str, int] = {}
    keep_count = 0
    total = 0

    with in_path.open(encoding="utf-8") as f, out_path.open("w", encoding="utf-8") as out:
        for line in f:
            rec = json.loads(line)
            if not rec.get("kept"):
                continue
            total += 1
            cls = classify_one(model, rec["post_id"], rec["text"], tax, no_llm)
            rec.update(cls)

            # Era is deterministic — derived from timestamp, not LLM
            if tax is not None:
                year = datetime.fromtimestamp(rec["timestamp"], tz=timezone.utc).year
                rec["era"] = tax.era_for_year(year)
                era_counts[rec["era"]] = era_counts.get(rec["era"], 0) + 1

            type_counts[cls["type"]] = type_counts.get(cls["type"], 0) + 1
            if cls.get("keep_for_synthesis"):
                keep_count += 1
            if cls.get("category"):
                cat_counts[cls["category"]] = cat_counts.get(cls["category"], 0) + 1
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[classify] total={total} keep_for_synthesis={keep_count}")
    if cat_counts:
        print("  Categories:")
        for c, n in sorted(cat_counts.items(), key=lambda x: -x[1]):
            print(f"    {c}: {n}")
    if era_counts:
        print("  Eras:")
        for e, n in sorted(era_counts.items(), key=lambda x: -x[1]):
            print(f"    {e}: {n}")
    return type_counts
