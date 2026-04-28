import json
import os
from collections import Counter

from slugify import slugify

from . import llm
from . import taxonomy as taxonomy_mod
from .paths import intermediate_dir


_SYSTEM_TEMPLATE = """당신은 사용자의 과거 페이스북 글들을 받아, 이들에서 반복되는 핵심 사고를 한 단락으로 압축한 노트를 작성합니다.

# 사용자 정보
{bio}

# 출력 형식 (JSON 만, 코드 펜스 없이)
{{
  "title": "한국어 제목 10-30자",
  "slug": "english-kebab-case-slug",
  "body": "300-600자 마크다운 본문. 사용자 1인칭 ('나는', '내가'). 여러 글의 공통점을 추상화하되 구체성을 잃지 마세요. 시기·맥락이 다양하면 그 변화를 인지하세요.",
  "primary_tag": "한국어 또는 영어 단어 1개"
}}"""


def _stub(cluster_id: str, members: list[dict]) -> dict:
    first = members[0]["text"][:30] if members else ""
    return {
        "title": f"개념 {cluster_id}",
        "slug": f"concept-{slugify(cluster_id, allow_unicode=False)}",
        "body": "이 클러스터에 묶인 글들의 발췌:\n\n"
        + "\n\n".join(f"> {m['text'][:150]}" for m in members[:3]),
        "primary_tag": "stub",
    }


def synth_one(
    model: str,
    cluster_id: str,
    category_name: str,
    members: list[dict],
    tax: taxonomy_mod.Taxonomy | None,
    no_llm: bool,
) -> dict:
    if no_llm:
        return _stub(cluster_id, members)

    cache_dir = intermediate_dir() / "llm_cache" / model
    tax_hash = tax.hash if tax else "no-tax"
    key_src = "|".join(sorted(m["post_id"] for m in members))
    key = f"cluster_{slugify(cluster_id, allow_unicode=False)}_{tax_hash}_{llm.text_hash(key_src)}"
    cached = llm.cache_get(cache_dir, key)
    if cached:
        return cached

    bio = tax.bio.strip() if tax else ""
    system = _SYSTEM_TEMPLATE.format(bio=bio or "(unspecified)")

    # Era distribution gives the model a sense of when these thoughts were written
    era_counts = Counter(m.get("era", "unknown") for m in members)
    era_summary = ", ".join(f"{era} ({n}건)" for era, n in era_counts.most_common())

    user = (
        f"# 카테고리\n{category_name}\n\n"
        f"# 시기 분포\n{era_summary}\n\n"
        f"# 글들 ({len(members)}편)\n\n"
    )
    for m in members:
        user += f"## [{m['date']}]\n{m['text']}\n\n"

    result = llm.call_json(model, system, user, max_tokens=2000, cache_system=True)
    if not result.get("title"):
        result["title"] = f"클러스터 {cluster_id}"
    if not result.get("slug"):
        result["slug"] = (
            slugify(result["title"], allow_unicode=False) or f"concept-{slugify(cluster_id, allow_unicode=False)}"
        )
    if not result.get("body"):
        result["body"] = ""
    if not result.get("primary_tag"):
        result["primary_tag"] = "facebook"

    llm.cache_put(cache_dir, key, result)
    return result


def run(no_llm: bool = False, include_sensitive: bool = False) -> int:
    clusters_path = intermediate_dir() / "05_clusters.json"
    classified_path = intermediate_dir() / "03_classified.jsonl"
    if not clusters_path.exists():
        raise FileNotFoundError("Run `fbpull cluster` first")

    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))

    posts: dict[str, dict] = {}
    if classified_path.exists():
        with classified_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                posts[rec["post_id"]] = rec

    tax = taxonomy_mod.load()
    cat_by_slug: dict[str, taxonomy_mod.Category] = {}
    if tax:
        for c in tax.categories:
            cat_by_slug[c.slug] = c

    model = os.environ.get("FBPULL_SYNTHESIZE_MODEL", "claude-sonnet-4-6")
    out_path = intermediate_dir() / "06_synthesized.jsonl"

    used_slugs: set[str] = set()
    n_written = 0
    n_skipped_strict = 0
    n_skipped_sensitive = 0
    n_skipped_noise = 0

    with out_path.open("w", encoding="utf-8") as out:
        for cid_str, member_ids in clusters.items():
            cat_slug, _, num = cid_str.rpartition("/")
            try:
                cluster_num = int(num)
            except ValueError:
                cluster_num = -1
            if cluster_num < 0:
                n_skipped_noise += len(member_ids)
                continue

            cat = cat_by_slug.get(cat_slug)
            cat_name = cat.name if cat else cat_slug
            if cat and cat.strict:
                n_skipped_strict += 1
                continue
            if cat and cat.sensitive and not include_sensitive:
                n_skipped_sensitive += 1
                continue

            members = [posts[pid] for pid in member_ids if pid in posts]
            if not members:
                continue
            result = synth_one(model, cid_str, cat_name, members, tax, no_llm)
            base = result["slug"]
            slug = base
            i = 2
            while slug in used_slugs:
                slug = f"{base}-{i}"
                i += 1
            used_slugs.add(slug)
            result["slug"] = slug
            result["cluster_id"] = cid_str
            result["category"] = cat_name
            result["category_slug"] = cat_slug
            result["sensitive"] = bool(cat and cat.sensitive)
            result["member_post_ids"] = sorted(member_ids)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            n_written += 1

    msg = f"[synthesize] {n_written} concept notes"
    if n_skipped_strict:
        msg += f" (skipped {n_skipped_strict} strict)"
    if n_skipped_sensitive:
        msg += f" (skipped {n_skipped_sensitive} sensitive — use --include-sensitive)"
    print(msg)
    return n_written
