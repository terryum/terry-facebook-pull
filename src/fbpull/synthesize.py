import json
import os

from slugify import slugify

from . import llm
from .paths import intermediate_dir

_SYSTEM = """당신은 사용자의 과거 페이스북 글들을 받아, 이들에서 반복되는 핵심 사고를 한 단락으로 압축한 노트를 작성합니다. 출력은 다음 JSON 만 (코드 펜스 없이):

{
  "title": "한국어 제목 10-30자",
  "slug": "english-kebab-case-slug",
  "body": "300-600자 마크다운 본문. 사용자 1인칭 ('나는', '내가'). 여러 글의 공통점을 추상화하되 구체성을 잃지 마세요.",
  "primary_tag": "한국어 또는 영어 단어 1개"
}"""


def _stub(cluster_id: str, members: list[dict]) -> dict:
    first = members[0]["text"][:30] if members else ""
    return {
        "title": f"개념 {cluster_id}",
        "slug": f"concept-{cluster_id}",
        "body": "이 클러스터에 묶인 글들의 발췌:\n\n"
        + "\n\n".join(f"> {m['text'][:150]}" for m in members[:3]),
        "primary_tag": "stub",
    }


def synth_one(model: str, cluster_id: str, members: list[dict], no_llm: bool) -> dict:
    if no_llm:
        return _stub(cluster_id, members)

    cache_dir = intermediate_dir() / "llm_cache" / model
    key_src = "|".join(sorted(m["post_id"] for m in members))
    key = f"cluster_{cluster_id}_{llm.text_hash(key_src)}"
    cached = llm.cache_get(cache_dir, key)
    if cached:
        return cached

    user = "다음은 한 클러스터로 묶인 글들입니다.\n\n"
    for m in members:
        user += f"## [{m['date']}]\n{m['text']}\n\n"

    result = llm.call_json(model, _SYSTEM, user, max_tokens=2000)
    if not result.get("title"):
        result["title"] = f"개념 {cluster_id}"
    if not result.get("slug"):
        result["slug"] = (
            slugify(result["title"], allow_unicode=False) or f"concept-{cluster_id}"
        )
    if not result.get("body"):
        result["body"] = ""
    if not result.get("primary_tag"):
        result["primary_tag"] = "facebook"

    llm.cache_put(cache_dir, key, result)
    return result


def run(no_llm: bool = False) -> int:
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

    model = os.environ.get("FBPULL_SYNTHESIZE_MODEL", "claude-sonnet-4-6")
    out_path = intermediate_dir() / "06_synthesized.jsonl"

    used_slugs: set[str] = set()
    n = 0
    with out_path.open("w", encoding="utf-8") as out:
        for cid_str, member_ids in clusters.items():
            cid = int(cid_str)
            if cid < 0:
                continue
            members = [posts[pid] for pid in member_ids if pid in posts]
            if not members:
                continue
            result = synth_one(model, cid_str, members, no_llm)
            # Dedupe slugs
            base = result["slug"]
            slug = base
            i = 2
            while slug in used_slugs:
                slug = f"{base}-{i}"
                i += 1
            used_slugs.add(slug)
            result["slug"] = slug
            result["cluster_id"] = cid
            result["member_post_ids"] = sorted(member_ids)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            n += 1

    print(f"[synthesize] {n} concept notes")
    return n
