"""Step 5b: Per-post Haiku pass on uncertain leaves.

5a 에서 'uncertain' 으로 라벨된 leaf 들에 대해, 각 글마다 Haiku 가 include/exclude
판정. 글 단위 결정으로 leaf 가 정말 mixed 인지 (post-level 분포가 어떤지) 확인.

산출: `_intermediate/post_label_5b.json` = {post_id: {tier, confidence, reason, leaf_id}}
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbpull import llm  # noqa: E402
from fbpull.cli import _bootstrap  # noqa: E402

_bootstrap(no_llm=False)

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
INT = VAULT / "_intermediate"
OUT_PATH = INT / "post_label_5b.json"
MODEL = "claude-haiku-4-5"
WORKERS = 8


SYSTEM = """당신은 한 사용자의 페이스북 글을 평가합니다. 사용자는 향후 **자기계발서** 를 집필합니다 — 시간·시기에 구애받지 않는 보편적 레슨이 1순위. 자서전·회고록은 후순위. 따라서 **개인성 ≠ 가치, 보편성 = 가치**.

# 사용자
엄태웅 (Terry, 1983년생). 서울대 기계항공 → KIST/LIG넥스원 연구원 → Waterloo 박사 (딥러닝) → ART Lab 창업·대표 → 코스맥스 AI혁신본부장.

# 책 후보 (우선순위 순)
1. **창업 레슨, 인생 레슨, 산업 레슨** — universal 레슨을 정리한 자기계발서 (최우선)
2. 특정 주제 책·챕터 — 사회·정치 비평 / 자서전 / 테크·산업 회고 등

# 평가 대상
**한 글 (post)** + 그 글이 속한 leaf cluster 컨텍스트.

# 평가 schema (3-tier + multi-scope)

## tier (단일 값)
- **"core"** = 1순위 자기계발서에 그대로 쓰일 universal lesson. 시간·맥락 무관 통찰.
- **"topic"** = 특정 주제 챕터·책에서만 가치. 일반 자기계발서엔 부적합.
- **"noise"** = 어떤 책에도 안 쓰일 (즉흥 사건·감정·분노·잡담·외국어 단편·사진 stub).

## topic_scope (tier="topic" 일 때만, multi-label)
- **personal-family**: 가족 관계·시간
- **personal-life**: 자서전적 일상 narrative (캐나다 적응, 일과, 감정 단편, 취미·hobby)
- **society-politics**: 정당·정치인·정부 비평, 정치 가치관
- **society-issues**: 사회 비평 (미디어·젠더·노동·계급·교육·사법)
- **industry-tech**: 테크·산업 동향
- **industry-academic**: 연구·학계·박사·논문 구체 사건
- **industry-management**: 구체 회사·경영·창업·조직 사례 (보편 원칙은 core 후보)

# 판정 원칙
1. **개인성 ≠ 가치**: 개인 경험 + 보편 lesson = core. 개인 narrative 만 = topic+personal-life.
2. **시간 의존도**: 시간 무관 → core 후보. 그 시기 사건만 → topic.
3. **보수적**: 의심스러우면 topic. core 와 noise 는 명확할 때만.
4. **multi-scope**: tier=topic 이면 topic_scope 1개 이상.

# 출력
submit 도구:
- tier: "core" | "topic" | "noise"
- topic_scope: array (tier=topic 일 때만; 그 외 [])
- confidence: 0.0–1.0
- reason: 1 문장 한국어
"""


SCHEMA = {
    "type": "object",
    "properties": {
        "tier": {"type": "string", "enum": ["core", "topic", "noise"]},
        "topic_scope": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [
                    "personal-family", "personal-life",
                    "society-politics", "society-issues",
                    "industry-tech", "industry-academic", "industry-management",
                ],
            },
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["tier", "topic_scope", "confidence", "reason"],
}


def main() -> None:
    labels_5a = json.loads((INT / "leaf_label_5a.json").read_text(encoding="utf-8"))
    clusters = json.loads((INT / "05_clusters.json").read_text(encoding="utf-8"))
    posts: dict[str, dict] = {}
    with (INT / "03_classified.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            posts[r["post_id"]] = r

    # leaf cohesion + tax mapping
    stats = json.loads((INT / "05_cluster_stats.json").read_text(encoding="utf-8"))
    cohesion_by_leaf: dict[str, float] = {}
    for leaf in stats.get("leaves", []):
        cohesion_by_leaf[leaf["id"]] = leaf.get("mean_cohesion", 0.0)

    from fbpull import taxonomy as taxonomy_mod
    tax = taxonomy_mod.load()
    cats = tax.categories if tax else []
    slug_to_name = {c.slug: c.name for c in cats}

    # Reuse 5a's TF-IDF — recompute (fast)
    import re
    from sklearn.feature_extraction.text import TfidfVectorizer
    by_leaf_texts: dict[str, list[str]] = {}
    for cid, members in clusters.items():
        if cid.endswith("/-1"):
            continue
        by_leaf_texts[cid] = [posts[p].get("text", "") for p in members if p in posts]

    leaf_ids = list(by_leaf_texts.keys())
    docs = [re.sub(r"https?://\S+", " ", " ".join(t)) for t in by_leaf_texts.values()]
    keywords_by_leaf = {lid: [] for lid in leaf_ids}
    try:
        vec = TfidfVectorizer(token_pattern=r"[가-힣]{2,}", max_features=20000,
                              max_df=0.6, min_df=2, sublinear_tf=True)
        m = vec.fit_transform(docs)
        feats = vec.get_feature_names_out()
        for i, lid in enumerate(leaf_ids):
            s = m[i].toarray().ravel()
            top = s.argsort()[::-1][:8]
            keywords_by_leaf[lid] = [feats[j] for j in top if s[j] > 0]
    except ValueError:
        pass

    # 'topic' leaves — drill in to find core/topic/noise distribution per leaf
    target_leaves = [lid for lid, r in labels_5a.items() if r["tier"] == "topic"]
    print(f"[5b] {len(target_leaves)} 'topic' leaves to dive into (post-level core/topic/noise)")

    todo: list[tuple[str, str]] = []  # (post_id, leaf_id)
    for lid in target_leaves:
        for pid in clusters.get(lid, []):
            todo.append((pid, lid))
    print(f"[5b] {len(todo)} posts total to label (Haiku, {WORKERS} workers)")

    # Existing results (resume support)
    existing: dict[str, dict] = {}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        print(f"[5b] resuming — {len(existing)} posts already labeled")
    todo = [(p, l) for p, l in todo if p not in existing]
    print(f"[5b] {len(todo)} posts to label this run")

    cache_dir = INT / "llm_cache" / MODEL / "label_5b"

    def label_one(pid: str, lid: str) -> tuple[str, dict]:
        post = posts.get(pid, {})
        text = post.get("text", "") or ""
        if not text.strip():
            text = "(empty)"
        date = post.get("date", "")[:10]
        cat_slug = lid.split("/", 1)[0]
        cat_name = slug_to_name.get(cat_slug, cat_slug)
        kws = keywords_by_leaf.get(lid, [])
        cohesion = cohesion_by_leaf.get(lid, 0.0)

        user = (
            f"# Leaf 컨텍스트\n"
            f"- leaf id: `{lid}`\n"
            f"- 카테고리: {cat_name}\n"
            f"- leaf 의 주제어: {', '.join(kws) if kws else '(없음)'}\n"
            f"- leaf cohesion: {cohesion:.2f}\n"
            f"- leaf 크기: {len(clusters.get(lid, []))}\n\n"
            f"# 평가 대상 글 [{date}]\n\n{text}"
        )

        cache_key = f"{llm.text_hash(pid)}_{llm.text_hash(SYSTEM)}_{llm.text_hash(user)}"
        cached = llm.cache_get(cache_dir, cache_key)
        if cached:
            cached["leaf_id"] = lid
            return pid, cached

        result = llm.call_json(
            MODEL, SYSTEM, user, max_tokens=300, cache_system=True, schema=SCHEMA
        )
        llm.cache_put(cache_dir, cache_key, result)
        result["leaf_id"] = lid
        return pid, result

    results: dict[str, dict] = dict(existing)
    n_done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(label_one, p, l): (p, l) for p, l in todo}
        for fut in as_completed(futures):
            p, l = futures[fut]
            try:
                _, r = fut.result()
                results[p] = r
                n_done += 1
                if n_done % 50 == 0 or n_done == len(todo):
                    print(f"  [{n_done}/{len(todo)}] last: {p[:24]} → {r['tier']}")
                if n_done % 25 == 0:
                    OUT_PATH.write_text(
                        json.dumps(results, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
            except Exception as e:
                print(f"  [ERROR] {p} ({l}): {e}")

    OUT_PATH.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Summary
    print(f"\n[5b] tier distribution ({len(results)} posts):")
    ctr = Counter(r["tier"] for r in results.values())
    for t, n in ctr.most_common():
        print(f"  {t:12s} {n:4d}  ({100 * n / len(results):.1f}%)")

    # Per-leaf core/noise rate distribution (within each topic leaf)
    print(f"\n[5b] per-leaf composition (within 'topic' leaves):")
    by_leaf: dict[str, list[str]] = {}
    for pid, r in results.items():
        by_leaf.setdefault(r["leaf_id"], []).append(r["tier"])
    print(f"  leaves with mostly core (>=60% core posts): "
          f"{sum(1 for l in by_leaf.values() if sum(1 for t in l if t == 'core') / len(l) >= 0.6)}")
    print(f"  leaves with mostly noise (>=60% noise posts): "
          f"{sum(1 for l in by_leaf.values() if sum(1 for t in l if t == 'noise') / len(l) >= 0.6)}")
    print(f"  leaves with mostly topic (>=60% topic posts): "
          f"{sum(1 for l in by_leaf.values() if sum(1 for t in l if t == 'topic') / len(l) >= 0.6)}")
    rate_buckets = Counter()
    for lid, tiers in by_leaf.items():
        rate = 100 * sum(1 for t in tiers if t == "noise") / len(tiers)
        bucket = f"{int(rate // 10) * 10}-{int(rate // 10) * 10 + 10}%"
        rate_buckets[bucket] += 1
    for b, n in sorted(rate_buckets.items(), key=lambda kv: int(kv[0].split("-")[0])):
        print(f"  noise rate {b:8s} : {n} leaves")

    print(f"\n[5b] saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
