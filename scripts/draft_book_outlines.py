"""Step 7: 다방향 책 개요 초안 (Sonnet, 3 테마).

각 테마에 대해 Sonnet 이 leaf inventory 를 보고 챕터 outline 초안 작성.
출력: chapters / gaps / orphans.

테마 (사용자 1순위 — universal lesson 자기계발서):
1. 인생 레슨 (Life lessons) — 시간·관계·성장·존재
2. 창업·경영 레슨 (Management lessons) — 스타트업·조직·리더십·시장
3. 산업·기술 레슨 (Industry/tech lessons) — AI·테크·연구

Leaf inventory 는 system prompt 에 cache (3 호출에 재사용).

산출: `_reports/<date>/book_outlines/{name}.md`
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from fbpull import llm  # noqa: E402
from fbpull.cli import _bootstrap  # noqa: E402

_bootstrap(no_llm=False)

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
INT = VAULT / "_intermediate"
MODEL = "claude-sonnet-4-6"


THEMES = [
    {
        "name": "인생 레슨",
        "filename": "lesson-life.md",
        "intent": (
            "보편적 인생 레슨 자기계발서. 시간·관계·성장·존재·자기관리·태도에 대한 reflection."
            " 누구에게나 통찰 주는 universal lesson. 개인 narrative 가 아닌 추상화된 깨달음 우선."
        ),
        "scope_hint": (
            "core leaf 중심. 일부 mixed leaf 의 core 포션도 chapter 자료로 유효."
            " 카테고리: 삶의 철학·성장·자기·학습·메타인지·일상 감정 (의미 있는 reflection 만)."
        ),
    },
    {
        "name": "창업·경영 레슨",
        "filename": "lesson-management.md",
        "intent": (
            "보편적 창업·경영·리더십 레슨 자기계발서. 스타트업 운영·시장 분석·팀 빌딩·"
            "의사결정·조직 문화. 구체 사례에서 추출된 universal 원칙 우선."
        ),
        "scope_hint": (
            "core leaf 중 창업·경영·조직·리더십 관련 + topic leaf 중 industry-management,"
            " industry-tech (시장·산업) scope. 학습·메타인지의 leadership 적 통찰도 포함."
        ),
    },
    {
        "name": "산업·기술 레슨",
        "filename": "lesson-industry.md",
        "intent": (
            "테크·산업·연구의 universal 통찰. AI 발전사, 기술이 사회·일·사람을 바꾸는 방식,"
            " 학계 시스템·연구자 정체성, 산업 트렌드 분석. 구체 사례 → 보편 패턴 추출."
        ),
        "scope_hint": (
            "industry-tech, industry-academic scope. core leaf 중 테크·산업·연구 관련."
            " 단순 제품 리뷰·인터뷰 기록은 배제, 산업 구조·기술 진화·연구자 정체성 위주."
        ),
    },
]


SYSTEM_TEMPLATE = """당신은 한 사용자의 페이스북 글 cluster inventory 를 보고 **자기계발서** 챕터 outline 초안을 작성합니다.

# 사용자
엄태웅 (Terry, 1983년생). 서울대 기계항공 → KIST/LIG넥스원 연구원 → Waterloo 박사 (딥러닝) → ART Lab 창업·대표 → 코스맥스 AI혁신본부장.

# 작업 내용
사용자는 **자기계발서** (시간·시기 무관 universal lesson) 를 집필합니다. 자서전·회고록은 후순위.
당신은 사용자가 지정한 책 테마에 부합하는 **leaf 들을 골라 챕터로 묶고**, 챕터 outline 을 작성합니다.

# 평가 원칙
1. **개인성 ≠ 가치, 보편성 = 가치**. 개인 narrative 라도 universal lesson 추출 가능하면 chapter 후보.
2. **시간 무관**: 그 시기에만 의미 있는 사건은 noise. 시간 지나도 통하는 통찰만.
3. **추상화**: 구체 사례 → 패턴 → 통찰 순서. 챕터 제목은 통찰 (사례 X).
4. **응집**: 한 챕터에 들어가는 leaf 들은 동일 통찰을 다양한 사례로 보강.

# 출력 schema (submit tool)
{
  "book_title": "책 제목 후보 (한국어, 부드러운 톤)",
  "book_thesis": "책 전체의 thesis 1–2 문장 (이 책이 무엇을 말하는가)",
  "chapters": [
    {
      "title": "챕터 제목 (통찰 중심, 12–25자)",
      "key_argument": "이 챕터에서 사용자가 말하려는 핵심 주장 (2–4 문장)",
      "supporting_leaves": ["leaf_id", ...],  // chapter source leaves (3–10개)
      "n_posts_estimate": 0,  // 합산 post 수 (rough)
      "outline_subsections": ["subsection 1", "subsection 2", ...]  // 4–8개
    },
    ...
  ],  // 5–9 챕터
  "gaps": [
    "이 책에 있어야 할 챕터/주제인데 적절한 leaf 가 부족한 영역 (1–2 문장 each)"
  ],
  "orphans": [
    "leaf_id"  // 책 테마에 잘 안 맞는 leaf — 다른 책에 더 어울림
  ]
}

# Leaf inventory (동일 inventory, 모든 테마 호출에 재사용)

각 leaf 항목 형식:
`leaf_id | 카테고리 | size(core/topic/noise) | cohesion | tier | scope | 주제어 | 샘플 글들`

(noise 비율 너무 높은 leaf 는 inventory 에서 제외 — 가치 낮음)

{leaf_inventory}
"""


SCHEMA = {
    "type": "object",
    "properties": {
        "book_title": {"type": "string"},
        "book_thesis": {"type": "string"},
        "chapters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "key_argument": {"type": "string"},
                    "supporting_leaves": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "n_posts_estimate": {"type": "integer"},
                    "outline_subsections": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": [
                    "title", "key_argument", "supporting_leaves",
                    "n_posts_estimate", "outline_subsections",
                ],
            },
        },
        "gaps": {"type": "array", "items": {"type": "string"}},
        "orphans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["book_title", "book_thesis", "chapters", "gaps", "orphans"],
}


_URL_RE = re.compile(r"https?://\S+")


def _build_leaf_inventory(clusters, posts, post_importance, labels_5a,
                           cohesion_by_leaf, slug_to_name) -> str:
    """Build a compact text inventory of synthesizable (non-noise-leaf) leaves."""
    # TF-IDF keywords (use clean text, skip noise leaves)
    by_leaf_texts: dict[str, list[str]] = {}
    for cid, members in clusters.items():
        if cid.endswith("/-1"):
            continue
        by_leaf_texts[cid] = [posts[p].get("text", "") for p in members if p in posts]
    leaf_ids = list(by_leaf_texts.keys())
    docs = [_URL_RE.sub(" ", " ".join(t)) for t in by_leaf_texts.values()]
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

    lines = []
    for lid in sorted(leaf_ids):
        members = clusters[lid]
        # Tier counts
        tiers = Counter(post_importance.get(p, {}).get("tier", "?") for p in members)
        n_core, n_topic, n_noise = tiers.get("core", 0), tiers.get("topic", 0), tiers.get("noise", 0)

        # Skip leaves where all members are noise (no synth value)
        if n_noise == len(members):
            continue
        # Skip leaves where noise ratio > 70% (too low signal)
        if n_noise / max(1, len(members)) > 0.70:
            continue

        cat_slug = lid.split("/", 1)[0]
        cat_name = slug_to_name.get(cat_slug, cat_slug)
        cohesion = cohesion_by_leaf.get(lid, 0)
        leaf_5a = labels_5a.get(lid, {})
        tier_5a = leaf_5a.get("tier", "?")
        scope_5a = ",".join(leaf_5a.get("topic_scope", [])) or "-"
        kws = ", ".join(keywords_by_leaf.get(lid, [])[:8]) or "-"

        # Sample posts: prefer core members, then topic, then any
        scored = []
        for pid in members:
            t = post_importance.get(pid, {}).get("tier", "?")
            score = 0 if t == "core" else (1 if t == "topic" else 2)
            scored.append((score, pid))
        scored.sort()
        sample_pids = [pid for _, pid in scored[:3]]
        samples = []
        for pid in sample_pids:
            p = posts.get(pid, {})
            text = (p.get("text") or "").replace("\n", " ").strip()[:140]
            date = p.get("date", "")[:10]
            samples.append(f"  [{date}] {text}")

        lines.append(
            f"`{lid}` | {cat_name} | {len(members)}({n_core}/{n_topic}/{n_noise}) | "
            f"coh={cohesion:.2f} | 5a={tier_5a}/{scope_5a} | 주제어: {kws}"
        )
        lines.extend(samples)
        lines.append("")

    return "\n".join(lines)


def _format_outline_md(theme_name: str, intent: str, result: dict) -> str:
    md = []
    md.append(f"# {theme_name} — 책 개요 초안")
    md.append("")
    md.append(f"**테마 의도**: {intent}")
    md.append("")
    md.append(f"**책 제목 후보**: {result.get('book_title', '')}")
    md.append("")
    md.append(f"**책 thesis**: {result.get('book_thesis', '')}")
    md.append("")
    md.append("---")
    md.append("")
    md.append("## 챕터 outline")
    md.append("")
    for i, ch in enumerate(result.get("chapters", []), 1):
        md.append(f"### Chapter {i}. {ch['title']}")
        md.append("")
        md.append(f"**핵심 주장**: {ch['key_argument']}")
        md.append("")
        md.append(f"**Source leaves** (n_posts ≈ {ch.get('n_posts_estimate', 0)}):")
        for lid in ch.get("supporting_leaves", []):
            md.append(f"- `{lid}`")
        md.append("")
        md.append("**Outline**:")
        for sub in ch.get("outline_subsections", []):
            md.append(f"- {sub}")
        md.append("")

    md.append("---")
    md.append("")
    md.append("## Gaps (글 부족 영역)")
    md.append("")
    if not result.get("gaps"):
        md.append("_(없음)_")
    else:
        for g in result["gaps"]:
            md.append(f"- {g}")
    md.append("")
    md.append("## Orphans (이 책에 안 어울리는 leaves)")
    md.append("")
    orphans = result.get("orphans", [])
    if not orphans:
        md.append("_(없음)_")
    else:
        for lid in orphans:
            md.append(f"- `{lid}`")
    md.append("")
    return "\n".join(md)


def main() -> None:
    clusters = json.loads((INT / "05_clusters.json").read_text(encoding="utf-8"))
    labels_5a = json.loads((INT / "leaf_label_5a.json").read_text(encoding="utf-8"))
    ovr = json.loads((VAULT / "_classify_overrides.json").read_text(encoding="utf-8"))
    post_importance = ovr["post_importance"]

    posts: dict[str, dict] = {}
    with (INT / "03_classified.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            posts[r["post_id"]] = r

    stats = json.loads((INT / "05_cluster_stats.json").read_text(encoding="utf-8"))
    cohesion_by_leaf = {leaf["id"]: leaf.get("mean_cohesion", 0.0) for leaf in stats.get("leaves", [])}

    from fbpull import taxonomy as taxonomy_mod
    tax = taxonomy_mod.load()
    slug_to_name = {c.slug: c.name for c in (tax.categories if tax else [])}

    leaf_inventory = _build_leaf_inventory(
        clusters, posts, post_importance, labels_5a, cohesion_by_leaf, slug_to_name
    )

    n_leaves_in_inventory = leaf_inventory.count("\n`")
    print(f"[draft] leaf inventory: ~{n_leaves_in_inventory} leaves "
          f"({len(leaf_inventory):,} chars)")

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = VAULT / "_reports" / today / "book_outlines"
    out_dir.mkdir(parents=True, exist_ok=True)

    system = SYSTEM_TEMPLATE.replace("{leaf_inventory}", leaf_inventory)
    cache_dir = INT / "llm_cache" / MODEL / "book_outlines"

    for theme in THEMES:
        print(f"\n[draft] {theme['name']} ...")
        user = (
            f"# 책 테마\n{theme['name']}\n\n"
            f"# 의도\n{theme['intent']}\n\n"
            f"# Scope hint\n{theme['scope_hint']}\n\n"
            "위 inventory 와 의도에 맞춰 책 개요 초안을 작성하세요. "
            "한 leaf 가 여러 챕터에 등장해도 괜찮습니다. "
            "supporting_leaves 는 leaf_id 그대로 쓰세요."
        )

        cache_key = f"{llm.text_hash(theme['name'])}_{llm.text_hash(system)}_{llm.text_hash(user)}"
        cached = llm.cache_get(cache_dir, cache_key)
        if cached:
            print(f"  [cache hit]")
            result = cached
        else:
            result = llm.call_json(
                MODEL, system, user, max_tokens=8000,
                cache_system=True, schema=SCHEMA,
            )
            llm.cache_put(cache_dir, cache_key, result)

        out_path = out_dir / theme["filename"]
        md = _format_outline_md(theme["name"], theme["intent"], result)
        out_path.write_text(md, encoding="utf-8")
        n_chap = len(result.get("chapters", []))
        n_orph = len(result.get("orphans", []))
        n_gaps = len(result.get("gaps", []))
        print(f"  → {out_path.name}: {n_chap} chapters, {n_gaps} gaps, {n_orph} orphans")

    print(f"\n[draft] all outlines → {out_dir}")


if __name__ == "__main__":
    main()
