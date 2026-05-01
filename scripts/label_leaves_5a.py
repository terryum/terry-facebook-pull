"""Step 5a: Per-leaf 보수적 importance LLM pass (3-tier + multi-scope).

각 leaf 의 요약 정보 (TF-IDF + sample posts + 메타) 를 보고 Haiku 가:
- tier: "core" (universal lesson, 대부분의 자기계발서에 사용) / "topic" (특정 주제
  챕터에서만 유효) / "noise" (어떤 책에도 부적합)
- topic_scope (tier=topic 일 때만, multi-label): personal-family / personal-life
  / society-politics / society-issues / industry-tech / industry-academic
  / industry-management

산출: `_intermediate/leaf_label_5a.json` = {leaf_id: {tier, topic_scope, confidence, reason}}
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from fbpull import llm  # noqa: E402
from fbpull.cli import _bootstrap  # noqa: E402

_bootstrap(no_llm=False)

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
INT = VAULT / "_intermediate"
OUT_PATH = INT / "leaf_label_5a.json"
MODEL = "claude-haiku-4-5"
WORKERS = 6
N_SAMPLE_POSTS = 5
SAMPLE_CHARS = 220


SYSTEM = """당신은 한 사용자의 페이스북 글 클러스터를 평가합니다. 사용자는 향후 **자기계발서** 를 집필합니다 — 시간·시기에 구애받지 않는 보편적 레슨이 1순위. 자서전·회고록은 후순위. 따라서 **개인성 ≠ 가치, 보편성 = 가치**.

# 사용자
엄태웅 (Terry, 1983년생). 서울대 기계항공 → KIST/LIG넥스원 연구원 → Waterloo 박사 (딥러닝) → ART Lab 창업·대표 → 코스맥스 AI혁신본부장.

# 책 후보 (우선순위 순)
1. **창업 레슨, 인생 레슨, 산업 레슨** — universal 레슨을 정리한 자기계발서 (최우선)
2. 특정 주제 책·챕터 — 사회·정치 비평 / 자서전 / 테크·산업 회고 등
3. 그 외 미정

# 평가 schema (3-tier + multi-scope)

## tier (단일 값)
- **"core"** = 1순위 자기계발서에 그대로 쓰일 universal lesson. 시간·맥락 무관 통찰.
  - 예: "스타트업이 성공하는 법 — 미래를 그리고 그 곳에 먼저 가 있으면 된다", "리더는 의사결정 3개만 잘하면 된다", "공부는 정직과 인내가 90%다"
- **"topic"** = 특정 주제 챕터·책에서만 가치. 일반 자기계발서엔 부적합.
  - 예: "캐나다에서 박사과정 시절" (자서전·회고), "박근혜 정부 사태에 대한 분노" (정치 챕터), "코스맥스 인수 과정" (특정 산업·경영 회고)
- **"noise"** = 어떤 책에도 안 쓰일. 즉흥 사건·감정·분노·잡담·외국어 단편·사진 stub·외부글 공유.

## topic_scope (tier="topic" 일 때만, multi-label — 여러 개 가능)
- **personal-family**: 부모·형제·자녀·결혼·가족 관계 / 명절·여행 가족 시간
- **personal-life**: 자서전적 일상 narrative — 캐나다 적응, 회사 생활, 학교/박사 일과, 일상 감정 단편, 취미·hobby (운동·축구·게임), 음식·소비
- **society-politics**: 정당·정치인·정부 비평, 선거, 정치적 가치관·태도
- **society-issues**: 사회 비평 — 미디어·젠더·노동·계급·교육·사법, 본인의 사회를 보는 시선·기준
- **industry-tech**: 테크·산업 동향 — AI·스타트업·플랫폼·블록체인 산업 트렌드, 그 시기 기술 관심사
- **industry-academic**: 연구·학계 — 박사과정·논문·심사·교수·진로·학술 메타 (구체 일상·사건; 보편 학습 노하우는 core 후보)
- **industry-management**: 경영·창업·조직·리더십 — 구체 회사 사례·인물 (보편 경영 원칙은 core 후보)

# 판정 원칙
1. **개인성 ≠ 가치**: 개인 경험에서 출발해도 보편 lesson 담으면 core. 개인 narrative 만이고 lesson 없으면 topic+personal-life.
2. **시간 의존도**: 시간 무관 = core 후보. 그 시기 사건만 의미 = topic.
3. **재사용성**: 다양한 책에 쓸 수 있으면 core. 한 챕터에서만 쓸 수 있으면 topic.
4. **보수적**: 의심스러우면 topic (≠ noise). core 와 noise 는 명확할 때만.
5. **multi-scope**: tier=topic 일 때 topic_scope 는 1개 이상 (보통 1–2개). 안 맞는 scope 는 빼라.

# 출력
submit 도구로 다음 schema 의 JSON 반환:
- tier: "core" | "topic" | "noise"
- topic_scope: array of scope strings (tier=topic 일 때만; 그 외엔 빈 배열 [])
- confidence: 0.0–1.0
- reason: 1–2 문장 한국어 설명 (어느 tier·scope 인지 / 왜)
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


_URL_RE = re.compile(r"https?://\S+")


def _clean(t: str) -> str:
    return _URL_RE.sub(" ", t or "")


def _tfidf_kw(by_leaf_texts: dict[str, list[str]], n_top: int = 8) -> dict[str, list[str]]:
    leaf_ids = list(by_leaf_texts.keys())
    docs = [_clean(" ".join(texts)) for texts in by_leaf_texts.values()]
    if not docs:
        return {lid: [] for lid in leaf_ids}
    try:
        vec = TfidfVectorizer(
            token_pattern=r"[가-힣]{2,}",
            max_features=20000,
            max_df=0.6,
            min_df=2,
            sublinear_tf=True,
        )
        m = vec.fit_transform(docs)
        feats = vec.get_feature_names_out()
    except ValueError:
        return {lid: [] for lid in leaf_ids}
    out = {}
    for i, lid in enumerate(leaf_ids):
        s = m[i].toarray().ravel()
        top = s.argsort()[::-1][:n_top]
        out[lid] = [feats[j] for j in top if s[j] > 0]
    return out


def _category_label_for(slug: str, tax_categories: list) -> str:
    for c in tax_categories:
        if c.slug == slug:
            return c.name
    return slug


def _build_leaf_input(
    leaf_id: str,
    member_pids: list[str],
    posts: dict,
    keywords: list[str],
    cohesion: float,
    cat_name: str,
) -> str:
    samples_pids = list(member_pids)
    n = len(samples_pids)
    if n > N_SAMPLE_POSTS:
        step = n / N_SAMPLE_POSTS
        samples_pids = [samples_pids[int(i * step)] for i in range(N_SAMPLE_POSTS)]
    samples = []
    for pid in samples_pids:
        p = posts.get(pid, {})
        text = (p.get("text") or "").replace("\n", " ").strip()[:SAMPLE_CHARS]
        date = p.get("date", "")[:10]
        samples.append(f"[{date}] {text}")

    parts = [
        f"# Leaf 식별자\n`{leaf_id}`",
        f"\n# 카테고리\n{cat_name}",
        f"\n# 메타\n- 글 수: {len(member_pids)}\n- Cohesion: {cohesion:.2f}",
        f"\n# TF-IDF 주제어\n{', '.join(keywords) if keywords else '(없음)'}",
        f"\n# Sample 글 ({len(samples)}개 / 전체 {len(member_pids)}편 중 균등 추출)\n",
    ]
    for s in samples:
        parts.append(s)
    return "\n".join(parts)


def main() -> None:
    from fbpull import taxonomy as taxonomy_mod
    tax = taxonomy_mod.load()
    cats = tax.categories if tax else []

    clusters = json.loads((INT / "05_clusters.json").read_text(encoding="utf-8"))
    posts: dict[str, dict] = {}
    with (INT / "03_classified.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            posts[r["post_id"]] = r

    stats = json.loads((INT / "05_cluster_stats.json").read_text(encoding="utf-8"))
    cohesion_by_leaf: dict[str, float] = {}
    for leaf in stats.get("leaves", []):
        cohesion_by_leaf[leaf["id"]] = leaf.get("mean_cohesion", leaf.get("cohesion", 0.0))

    by_leaf_texts: dict[str, list[str]] = {}
    for cid, members in clusters.items():
        if cid.endswith("/-1"):
            continue
        by_leaf_texts[cid] = [posts[pid].get("text", "") for pid in members if pid in posts]
    keywords_by_leaf = _tfidf_kw(by_leaf_texts, n_top=8)

    existing: dict[str, dict] = {}
    if OUT_PATH.exists():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
        print(f"[5a] resuming — {len(existing)} leaves already labeled")

    cache_dir = INT / "llm_cache" / MODEL / "label_5a"

    leaf_ids = sorted(by_leaf_texts.keys())
    todo = [lid for lid in leaf_ids if lid not in existing]
    print(f"[5a] {len(todo)} leaves to label (total {len(leaf_ids)}, model={MODEL})")

    def label_one(leaf_id: str) -> tuple[str, dict]:
        member_pids = clusters[leaf_id]
        slug = leaf_id.split("/", 1)[0]
        cat_name = _category_label_for(slug, cats)
        kws = keywords_by_leaf.get(leaf_id, [])
        cohesion = cohesion_by_leaf.get(leaf_id, 0.0)

        user = _build_leaf_input(leaf_id, member_pids, posts, kws, cohesion, cat_name)

        cache_key = f"{llm.text_hash(leaf_id)}_{llm.text_hash(SYSTEM)}_{llm.text_hash(user)}"
        cached = llm.cache_get(cache_dir, cache_key)
        if cached:
            return leaf_id, cached

        result = llm.call_json(
            MODEL, SYSTEM, user, max_tokens=500, cache_system=True, schema=SCHEMA
        )
        llm.cache_put(cache_dir, cache_key, result)
        return leaf_id, result

    results: dict[str, dict] = dict(existing)
    n_done = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(label_one, lid): lid for lid in todo}
        for fut in as_completed(futures):
            lid = futures[fut]
            try:
                _, r = fut.result()
                results[lid] = r
                n_done += 1
                if n_done % 25 == 0 or n_done == len(todo):
                    scope = ",".join(r.get("topic_scope", [])) or "-"
                    print(f"  [{n_done}/{len(todo)}] {lid}: {r['tier']}/{scope} (conf={r['confidence']:.2f})")
                OUT_PATH.write_text(
                    json.dumps(results, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                print(f"  [ERROR] {lid}: {e}")

    tier_dist = Counter(r["tier"] for r in results.values())
    print(f"\n[5a] tier distribution ({len(results)} leaves):")
    for t, n in tier_dist.most_common():
        print(f"  {t:8s} {n:3d}  ({100 * n / len(results):.1f}%)")

    scope_dist = Counter()
    for r in results.values():
        for s in r.get("topic_scope", []):
            scope_dist[s] += 1
    print(f"\n[5a] topic_scope distribution (multi-label):")
    for s, n in scope_dist.most_common():
        print(f"  {s:25s} {n}")

    print(f"\n[5a] saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
