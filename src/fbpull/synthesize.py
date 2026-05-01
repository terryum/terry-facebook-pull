import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from slugify import slugify

from . import llm
from . import taxonomy as taxonomy_mod
from .paths import fb_root, intermediate_dir


_SYNTH_WORKERS = int(os.environ.get("FBPULL_SYNTH_WORKERS", "6"))


_SYSTEM_TEMPLATE = """당신은 사용자의 과거 페이스북 글들을 받아, 이들에서 반복되는 핵심 사고를 한 단락으로 압축한 노트를 작성합니다.

# 사용자 정보
{bio}

submit 도구를 호출해 구조화된 결과를 반환하세요. body 필드는 300–600자, 사용자 1인칭 ("나는", "내가"), 여러 글의 공통점을 추상화하되 구체성을 잃지 마세요. 시기·맥락이 다양하면 그 변화를 인지하세요.

# leaf 의 가치 등급
이 leaf 의 멤버 글들은 importance 라벨이 부여돼 있습니다. core 글은 universal lesson (시간 무관 통찰), topic 글은 특정 주제 chapter 자료 (자서전·사회·산업 등), noise 글은 의미 약함. 합성 시 core 글의 통찰을 중심으로 추상화하되, topic 글이 그 통찰의 구체적 사례·맥락을 보강한다면 함께 활용하세요. noise 글은 톤 잡는 보조 자료로만 참고."""

_SYNTH_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "한국어 제목 10-30자"},
        "slug": {
            "type": "string",
            "description": "english kebab-case slug (lowercase letters, numbers, hyphens)",
        },
        "body": {"type": "string", "description": "300-600자 마크다운 본문"},
        "primary_tag": {
            "type": "string",
            "description": "한국어 또는 영어 단어 1개",
        },
    },
    "required": ["title", "slug", "body", "primary_tag"],
}


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
    leaf_tier: str,
    leaf_scope: list[str],
    tier_dist: dict[str, int],
    themes: list[dict],
    post_importance: dict[str, dict],
    no_llm: bool,
) -> dict:
    if no_llm:
        return _stub(cluster_id, members)

    cache_dir = intermediate_dir() / "llm_cache" / model
    tax_hash = tax.hash if tax else "no-tax"
    key_src = "|".join(sorted(m["post_id"] for m in members))
    # Include tier/theme/scope info in cache key so prompt changes invalidate cache
    extra = json.dumps(
        {
            "tier": leaf_tier,
            "scope": sorted(leaf_scope),
            "tier_dist": dict(sorted(tier_dist.items())),
            "themes": [t.get("name", "") for t in themes],
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    key = f"cluster_{slugify(cluster_id, allow_unicode=False)}_{tax_hash}_{llm.text_hash(key_src + '|' + extra)}"
    cached = llm.cache_get(cache_dir, key)
    if cached:
        return cached

    bio = tax.bio.strip() if tax else ""
    system = _SYSTEM_TEMPLATE.format(bio=bio or "(unspecified)")

    era_counts = Counter(m.get("era", "unknown") for m in members)
    era_summary = ", ".join(f"{era} ({n}건)" for era, n in era_counts.most_common())

    tier_summary = ", ".join(f"{t}={n}" for t, n in tier_dist.items() if n > 0)
    scope_summary = ", ".join(leaf_scope) if leaf_scope else "—"
    themes_summary = ", ".join(
        f"{t['name']} ({t.get('score', 0):.2f})" for t in themes
    ) if themes else "—"

    user = (
        f"# 카테고리\n{category_name}\n\n"
        f"# Leaf 가치 정보\n"
        f"- leaf 등급: {leaf_tier}\n"
        f"- topic scope: {scope_summary}\n"
        f"- 멤버 tier 분포: {tier_summary}\n"
        f"- cross-category 주제 (theme axis): {themes_summary}\n\n"
        f"# 시기 분포\n{era_summary}\n\n"
        f"# 글들 ({len(members)}편 — 각 글에 [tier] 표시)\n\n"
    )
    for m in members:
        pid = m["post_id"]
        t = post_importance.get(pid, {}).get("tier", "?")
        user += f"## [{m['date']} · tier={t}]\n{m['text']}\n\n"

    result = llm.call_json(
        model, system, user, max_tokens=2000, cache_system=True, schema=_SYNTH_SCHEMA
    )
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


def _load_overrides() -> dict:
    p = fb_root() / "_classify_overrides.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _load_leaf_themes() -> dict:
    """Return {leaf_id: [{theme_id, name, score}, ...]}."""
    p = intermediate_dir() / "leaf_themes.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    theme_meta = data.get("theme_meta", {})
    leaf_themes_raw = data.get("leaf_themes", {})
    out: dict[str, list[dict]] = {}
    for lid, tids in leaf_themes_raw.items():
        out[lid] = [
            {
                "theme_id": tid,
                "name": theme_meta.get(tid, {}).get("name", tid),
                "score": float(score),
            }
            for tid, score in tids
        ]
    return out


def _load_leaf_decisions() -> dict[str, str]:
    """Return {leaf_id: 'core'|'mixed'|'noise'|'topic_uniform'}."""
    p = intermediate_dir() / "leaf_label_threshold.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("leaf_decision", {})


def _load_labels_5a() -> dict[str, dict]:
    p = intermediate_dir() / "leaf_label_5a.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


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

    # Load tier / scope / theme metadata
    overrides = _load_overrides()
    post_importance: dict[str, dict] = overrides.get("post_importance", {})
    leaf_themes = _load_leaf_themes()
    leaf_decisions = _load_leaf_decisions()
    labels_5a = _load_labels_5a()

    model = os.environ.get("FBPULL_SYNTHESIZE_MODEL", "claude-sonnet-4-6")
    out_path = intermediate_dir() / "06_synthesized.jsonl"

    n_skipped_strict = 0
    n_skipped_sensitive = 0
    n_skipped_noise_unclustered = 0
    n_skipped_noise_leaf = 0

    # Build list of leaves to synthesize
    todo: list[dict] = []  # each: {cid_str, cat_slug, cat_name, members, member_ids, leaf_dec, ...}
    for cid_str, member_ids in clusters.items():
        parts = cid_str.split("/")
        cat_slug = parts[0]
        leaf_path = "/".join(parts[1:]) if len(parts) > 1 else ""
        if leaf_path == "-1":
            n_skipped_noise_unclustered += len(member_ids)
            continue

        cat = cat_by_slug.get(cat_slug)
        cat_name = cat.name if cat else cat_slug
        if cat and cat.strict:
            n_skipped_strict += 1
            continue
        if cat and cat.sensitive and not include_sensitive:
            n_skipped_sensitive += 1
            continue

        leaf_dec = leaf_decisions.get(cid_str, "topic_uniform")
        if leaf_dec == "noise":
            n_skipped_noise_leaf += 1
            continue

        members = [posts[pid] for pid in member_ids if pid in posts]
        if not members:
            continue

        tier_dist = Counter(
            post_importance.get(pid, {}).get("tier", "?") for pid in member_ids
        )
        leaf_5a = labels_5a.get(cid_str, {})
        leaf_scope = list(leaf_5a.get("topic_scope", []))
        themes = leaf_themes.get(cid_str, [])

        todo.append({
            "cid_str": cid_str,
            "cat_slug": cat_slug,
            "cat_name": cat_name,
            "cat_sensitive": bool(cat and cat.sensitive),
            "members": members,
            "member_ids": member_ids,
            "leaf_dec": leaf_dec,
            "leaf_scope": leaf_scope,
            "tier_dist": dict(tier_dist),
            "themes": themes,
        })

    print(f"[synthesize] {len(todo)} leaves to synthesize ({_SYNTH_WORKERS} workers, model={model})")

    def _work(item: dict) -> dict:
        result = synth_one(
            model, item["cid_str"], item["cat_name"], item["members"], tax,
            leaf_tier=item["leaf_dec"],
            leaf_scope=item["leaf_scope"],
            tier_dist=item["tier_dist"],
            themes=item["themes"],
            post_importance=post_importance,
            no_llm=no_llm,
        )
        return {"item": item, "result": result}

    used_slugs: set[str] = set()
    results: list[dict] = []
    n_done = 0
    n_total = len(todo)
    with ThreadPoolExecutor(max_workers=_SYNTH_WORKERS) as ex:
        futures = {ex.submit(_work, item): item for item in todo}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                results.append(r)
                n_done += 1
                if n_done % 10 == 0 or n_done == n_total:
                    print(f"  [{n_done}/{n_total}] {r['item']['cid_str']}: {r['result'].get('title', '?')[:40]}")
            except Exception as e:
                item = futures[fut]
                print(f"  [ERROR] {item['cid_str']}: {e}")

    # Deterministic write ordering: by cluster_id
    results.sort(key=lambda r: r["item"]["cid_str"])
    n_written = 0
    with out_path.open("w", encoding="utf-8") as out:
        for r in results:
            item = r["item"]
            result = r["result"]
            base = result["slug"]
            slug = base
            i = 2
            while slug in used_slugs:
                slug = f"{base}-{i}"
                i += 1
            used_slugs.add(slug)
            result["slug"] = slug
            result["cluster_id"] = item["cid_str"]
            result["category"] = item["cat_name"]
            result["category_slug"] = item["cat_slug"]
            result["sensitive"] = item["cat_sensitive"]
            result["member_post_ids"] = sorted(item["member_ids"])
            result["leaf_tier"] = item["leaf_dec"]
            result["leaf_scope"] = item["leaf_scope"]
            result["member_tier_distribution"] = {
                k: item["tier_dist"].get(k, 0) for k in ("core", "topic", "noise")
            }
            result["themes"] = item["themes"]
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            n_written += 1

    msg = f"[synthesize] {n_written} concept notes"
    if n_skipped_noise_leaf:
        msg += f" (skipped {n_skipped_noise_leaf} noise-tier leaves)"
    if n_skipped_strict:
        msg += f" (skipped {n_skipped_strict} strict)"
    if n_skipped_sensitive:
        msg += f" (skipped {n_skipped_sensitive} sensitive — use --include-sensitive)"
    print(msg)
    return n_written
