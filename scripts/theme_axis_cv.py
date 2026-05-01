"""Step 8 (revision): CV-driven theme axis search.

목적: 현재 category-axis cluster 위에 theme-axis 를 추가. 4 topic 책 outline 의
gaps/orphans 를 CV signal 로 → overfitting 안 하면서 generalization 좋은 변형 선정.

Phase 1: Leaf centroid + 3 variants (k=10/15/20) agglomerative + Haiku 가 theme 명명
Phase 2: 4 topic × 4 variant = 16 outline (V0 의 3 개는 caching)
Phase 3: Score grid + 4-fold CV → winner
Phase 4: Persist + held-out 전환점 final outline

Outputs:
- `_intermediate/leaf_themes_V*.json` (variant 별 theme axis)
- `_intermediate/cv_scores.json` (grid)
- `_intermediate/leaf_themes.json` (winner)
- `_reports/<date>/book_outlines_cv/{V*}/{topic}.md` (16 outlines)
- `_reports/<date>/book_outlines_cv/cv_report.md` (summary)
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sklearn.cluster import AgglomerativeClustering  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402

from fbpull import llm  # noqa: E402
from fbpull.cli import _bootstrap  # noqa: E402

_bootstrap(no_llm=False)

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
INT = VAULT / "_intermediate"

THEME_LABEL_MODEL = "claude-haiku-4-5"
OUTLINE_MODEL = "claude-sonnet-4-6"


VARIANTS = [
    {"name": "V0_baseline", "k": None},
    {"name": "V1_k10", "k": 10, "multi_top": 3},
    {"name": "V2_k15", "k": 15, "multi_top": 3},
    {"name": "V3_k20", "k": 20, "multi_top": 3},
]


TOPICS = [
    {
        "key": "life",
        "name": "인생 레슨",
        "intent": (
            "보편적 인생 레슨 자기계발서. 시간·관계·성장·존재·자기관리·태도에 대한 reflection."
            " 누구에게나 통찰 주는 universal lesson. 개인 narrative 가 아닌 추상화된 깨달음 우선."
        ),
        "scope_hint": "core leaf 중심. 카테고리: 삶의 철학·성장·자기·학습·메타인지·일상 감정 (의미 있는 reflection 만).",
    },
    {
        "key": "management",
        "name": "창업·경영 레슨",
        "intent": (
            "보편적 창업·경영·리더십 레슨 자기계발서. 스타트업 운영·시장 분석·팀 빌딩·"
            "의사결정·조직 문화. 구체 사례에서 추출된 universal 원칙 우선."
        ),
        "scope_hint": "core + industry-management/industry-tech scope topic.",
    },
    {
        "key": "industry",
        "name": "산업·기술 레슨",
        "intent": (
            "테크·산업·연구의 universal 통찰. AI 발전사, 기술이 사회·일·사람을 바꾸는 방식,"
            " 학계 시스템·연구자 정체성, 산업 트렌드 분석. 구체 사례 → 보편 패턴 추출."
        ),
        "scope_hint": "industry-tech, industry-academic scope. core leaf 중 테크·산업·연구.",
    },
    {
        "key": "transitions",
        "name": "전환점에 서서",
        "intent": (
            "결정과 후회의 순간들 — 인생의 전환점에서 우리는 무엇을 바라보고 무엇을 선택했는가."
            " 박사 진학·창업 결정·임원 전직 등 era-cross 한 결정 narrative 와 그로부터 추출한"
            " universal 의사결정·후회·책임 원칙. 구체 사건 → 보편 패턴."
        ),
        "scope_hint": "era-cross. 창업·경영(결정), 인생(reflection), 산업(직업 전환), 학계(진학·진로) 융합.",
    },
]


SYSTEM_TEMPLATE = """당신은 한 사용자의 페이스북 글 cluster inventory 를 보고 **자기계발서** 챕터 outline 초안을 작성합니다.

# 사용자
엄태웅 (Terry, 1983년생). 서울대 기계항공 → KIST/LIG넥스원 연구원 → Waterloo 박사 (딥러닝) → ART Lab 창업·대표 → 코스맥스 AI혁신본부장.

# 작업 내용
사용자는 자기계발서 (시간·시기 무관 universal lesson) 를 집필합니다. 자서전·회고록은 후순위.
당신은 사용자가 지정한 책 테마에 부합하는 leaf 들을 골라 챕터로 묶고, 챕터 outline 을 작성합니다.

# 평가 원칙
1. **개인성 ≠ 가치, 보편성 = 가치**.
2. **시간 무관**: 그 시기에만 의미 있는 사건은 noise.
3. **추상화**: 구체 사례 → 패턴 → 통찰.
4. **응집**: 한 챕터에 들어가는 leaf 들은 동일 통찰을 다양한 사례로 보강.

# 출력 schema (submit tool)
{
  "book_title": "...",
  "book_thesis": "...",
  "chapters": [
    {"title": "...", "key_argument": "...", "supporting_leaves": [...], "n_posts_estimate": 0, "outline_subsections": [...]}
  ],
  "gaps": ["..."],
  "orphans": ["leaf_id"]
}

# Leaf inventory
각 항목 형식: `leaf_id | 카테고리 | size(core/topic/noise) | cohesion | tier | scope | THEMES | 주제어 | 샘플`
THEMES 는 cluster 들이 카테고리를 가로지르는 보조 axis. 한 leaf 가 여러 theme 멤버일 수 있음.

(noise 비율 70% 초과 leaf 는 inventory 제외)

{leaf_inventory}
"""

OUTLINE_SCHEMA = {
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
                    "supporting_leaves": {"type": "array", "items": {"type": "string"}},
                    "n_posts_estimate": {"type": "integer"},
                    "outline_subsections": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["title", "key_argument", "supporting_leaves", "n_posts_estimate", "outline_subsections"],
            },
        },
        "gaps": {"type": "array", "items": {"type": "string"}},
        "orphans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["book_title", "book_thesis", "chapters", "gaps", "orphans"],
}


THEME_LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "한국어 theme 이름 (5-15자, 추상적 통찰 keyword)"},
        "description": {"type": "string", "description": "1 문장 한국어 설명 — 이 theme 의 통찰 정의"},
    },
    "required": ["name", "description"],
}


_URL_RE = re.compile(r"https?://\S+")


def _load_data():
    clusters = json.loads((INT / "05_clusters.json").read_text(encoding="utf-8"))
    embeddings = np.load(INT / "04_embeddings.npy")
    post_ids = json.loads((INT / "04_post_ids.json").read_text(encoding="utf-8"))
    pid_to_row = {pid: i for i, pid in enumerate(post_ids)}

    posts: dict[str, dict] = {}
    with (INT / "03_classified.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            posts[r["post_id"]] = r

    stats = json.loads((INT / "05_cluster_stats.json").read_text(encoding="utf-8"))
    cohesion_by_leaf = {leaf["id"]: leaf.get("mean_cohesion", 0.0) for leaf in stats.get("leaves", [])}

    labels_5a = json.loads((INT / "leaf_label_5a.json").read_text(encoding="utf-8"))
    ovr = json.loads((VAULT / "_classify_overrides.json").read_text(encoding="utf-8"))
    post_importance = ovr["post_importance"]

    from fbpull import taxonomy as taxonomy_mod
    tax = taxonomy_mod.load()
    slug_to_name = {c.slug: c.name for c in (tax.categories if tax else [])}

    return {
        "clusters": clusters, "embeddings": embeddings, "pid_to_row": pid_to_row,
        "posts": posts, "cohesion_by_leaf": cohesion_by_leaf,
        "labels_5a": labels_5a, "post_importance": post_importance,
        "slug_to_name": slug_to_name,
    }


def _compute_leaf_centroids(data) -> tuple[list[str], np.ndarray]:
    """Mean of L2-normalized post embeddings per leaf, then re-normalize."""
    leaf_ids = []
    centroids = []
    for cid, members in data["clusters"].items():
        if cid.endswith("/-1"):
            continue
        rows = [data["pid_to_row"][p] for p in members if p in data["pid_to_row"]]
        if not rows:
            continue
        c = data["embeddings"][rows].mean(axis=0)
        n = float(np.linalg.norm(c))
        c = c / n if n > 0 else c
        leaf_ids.append(cid)
        centroids.append(c)
    return leaf_ids, np.stack(centroids)


def _agglomerative_themes(leaf_ids: list[str], centroids: np.ndarray, k: int) -> dict:
    """Agglomerative clustering → k themes. Returns {theme_id: [leaf_id, ...]}."""
    clf = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="average")
    labels = clf.fit_predict(centroids)
    themes = defaultdict(list)
    for lid, lab in zip(leaf_ids, labels):
        themes[f"T{lab:02d}"].append(lid)
    return dict(themes)


def _theme_centroids(themes: dict[str, list[str]], leaf_ids_to_idx: dict[str, int],
                     leaf_centroids: np.ndarray) -> dict[str, np.ndarray]:
    out = {}
    for tid, lids in themes.items():
        rows = [leaf_ids_to_idx[l] for l in lids if l in leaf_ids_to_idx]
        if not rows:
            continue
        c = leaf_centroids[rows].mean(axis=0)
        n = float(np.linalg.norm(c))
        out[tid] = c / n if n > 0 else c
    return out


def _multi_membership(leaf_ids: list[str], leaf_centroids: np.ndarray,
                       theme_ids: list[str], theme_centroids_arr: np.ndarray,
                       top_n: int = 3) -> dict[str, list[tuple[str, float]]]:
    """Each leaf → top-N theme by cosine."""
    sims = leaf_centroids @ theme_centroids_arr.T  # (L, T)
    out = {}
    for i, lid in enumerate(leaf_ids):
        s = sims[i]
        top_idx = s.argsort()[::-1][:top_n]
        out[lid] = [(theme_ids[j], float(s[j])) for j in top_idx]
    return out


def _label_theme(theme_id: str, member_leaves: list[str], leaf_keywords: dict[str, list[str]],
                 leaf_categories: dict[str, str], data) -> dict:
    """Haiku names a theme based on its member leaves' keywords + samples."""
    cache_dir = INT / "llm_cache" / THEME_LABEL_MODEL / "theme_label"
    # Build context
    lines = [f"# Theme {theme_id} 의 멤버 leaves ({len(member_leaves)}개)"]
    for lid in member_leaves[:12]:  # cap context
        kws = ", ".join(leaf_keywords.get(lid, [])[:5]) or "-"
        cat = leaf_categories.get(lid, "?")
        size = len(data["clusters"].get(lid, []))
        lines.append(f"- `{lid}` ({cat}, n={size}) — {kws}")
        # 1 sample
        members = data["clusters"].get(lid, [])
        if members:
            p = data["posts"].get(members[0], {})
            t = (p.get("text") or "").replace("\n", " ").strip()[:120]
            if t:
                lines.append(f"  e.g. [{p.get('date','')[:10]}] {t}")
    if len(member_leaves) > 12:
        lines.append(f"_(+{len(member_leaves) - 12} more leaves)_")

    user = "\n".join(lines)
    system = (
        "당신은 한 사용자의 글 cluster 들의 묶음을 보고 그 묶음을 관통하는 한 가지 통찰·주제를 명명합니다.\n"
        "이 주제 명명은 카테고리 분류와 다른 axis — 카테고리를 가로지를 수 있습니다.\n"
        "예: '의사결정의 순간들', '관계와 이타성', '기술이 만든 사회 변화', '메타인지·자기 관찰'.\n"
        "submit 도구로 {name, description} 반환."
    )

    cache_key = f"{theme_id}_{llm.text_hash(system)}_{llm.text_hash(user)}"
    cached = llm.cache_get(cache_dir, cache_key)
    if cached:
        return cached

    result = llm.call_json(THEME_LABEL_MODEL, system, user, max_tokens=300,
                           cache_system=False, schema=THEME_LABEL_SCHEMA)
    llm.cache_put(cache_dir, cache_key, result)
    return result


def _build_inventory_text(leaf_ids: list[str], data, leaf_keywords: dict[str, list[str]],
                          variant: dict, leaf_themes: dict | None,
                          theme_meta: dict | None) -> str:
    """Build leaf inventory string for outline prompt. Optional theme axis."""
    lines = []
    for lid in sorted(leaf_ids):
        members = data["clusters"][lid]
        tiers = Counter(data["post_importance"].get(p, {}).get("tier", "?") for p in members)
        n_core, n_topic, n_noise = tiers.get("core", 0), tiers.get("topic", 0), tiers.get("noise", 0)

        if n_noise == len(members):
            continue
        if n_noise / max(1, len(members)) > 0.70:
            continue

        cat_slug = lid.split("/", 1)[0]
        cat_name = data["slug_to_name"].get(cat_slug, cat_slug)
        cohesion = data["cohesion_by_leaf"].get(lid, 0)
        leaf_5a = data["labels_5a"].get(lid, {})
        tier_5a = leaf_5a.get("tier", "?")
        scope_5a = ",".join(leaf_5a.get("topic_scope", [])) or "-"
        kws = ", ".join(leaf_keywords.get(lid, [])[:8]) or "-"

        themes_part = "-"
        if leaf_themes and theme_meta:
            tids = leaf_themes.get(lid, [])
            theme_strs = []
            for tid, score in tids:
                tname = theme_meta.get(tid, {}).get("name", tid)
                theme_strs.append(f"{tname}({score:.2f})")
            themes_part = " | ".join(theme_strs) if theme_strs else "-"

        # Sample posts (prefer core, then topic)
        scored = []
        for pid in members:
            t = data["post_importance"].get(pid, {}).get("tier", "?")
            score = 0 if t == "core" else (1 if t == "topic" else 2)
            scored.append((score, pid))
        scored.sort()
        sample_pids = [pid for _, pid in scored[:3]]

        lines.append(
            f"`{lid}` | {cat_name} | {len(members)}({n_core}/{n_topic}/{n_noise}) | "
            f"coh={cohesion:.2f} | 5a={tier_5a}/{scope_5a} | THEMES: {themes_part} | "
            f"주제어: {kws}"
        )
        for pid in sample_pids:
            p = data["posts"].get(pid, {})
            text = (p.get("text") or "").replace("\n", " ").strip()[:140]
            date = p.get("date", "")[:10]
            lines.append(f"  [{date}] {text}")
        lines.append("")

    return "\n".join(lines)


def _draft_outline(model: str, system: str, user: str, cache_subdir: str) -> dict:
    cache_dir = INT / "llm_cache" / model / cache_subdir
    cache_key = f"{llm.text_hash(system)}_{llm.text_hash(user)}"
    cached = llm.cache_get(cache_dir, cache_key)
    if cached:
        return cached
    result = llm.call_json(model, system, user, max_tokens=8000, cache_system=True,
                            schema=OUTLINE_SCHEMA)
    llm.cache_put(cache_dir, cache_key, result)
    return result


def _score_outline(result: dict, inventory_size: int) -> dict:
    chapters = result.get("chapters", [])
    n_chap = len(chapters)
    n_rich = sum(1 for c in chapters if len(c.get("supporting_leaves", [])) >= 5)
    n_gaps = len(result.get("gaps", []))
    n_orph = len(result.get("orphans", []))
    orphan_ratio = n_orph / max(1, inventory_size)
    score = n_rich * 2.0 + (n_chap * 0.5) - (n_gaps * 0.5) + (1 - orphan_ratio) * 5
    return {
        "score": round(score, 2),
        "n_chapters": n_chap,
        "n_rich_chapters": n_rich,
        "n_gaps": n_gaps,
        "n_orphans": n_orph,
        "orphan_ratio": round(orphan_ratio, 3),
    }


def _format_outline_md(theme_name: str, intent: str, result: dict) -> str:
    md = [f"# {theme_name} — 책 개요 초안\n"]
    md.append(f"**테마 의도**: {intent}\n")
    md.append(f"**책 제목 후보**: {result.get('book_title', '')}\n")
    md.append(f"**책 thesis**: {result.get('book_thesis', '')}\n")
    md.append("---\n")
    md.append("## 챕터 outline\n")
    for i, ch in enumerate(result.get("chapters", []), 1):
        md.append(f"### Chapter {i}. {ch['title']}\n")
        md.append(f"**핵심 주장**: {ch['key_argument']}\n")
        md.append(f"**Source leaves** (n_posts ≈ {ch.get('n_posts_estimate', 0)}):")
        for lid in ch.get("supporting_leaves", []):
            md.append(f"- `{lid}`")
        md.append("\n**Outline**:")
        for sub in ch.get("outline_subsections", []):
            md.append(f"- {sub}")
        md.append("")
    md.append("---\n## Gaps\n")
    for g in result.get("gaps", []):
        md.append(f"- {g}")
    md.append("\n## Orphans\n")
    for lid in result.get("orphans", []):
        md.append(f"- `{lid}`")
    return "\n".join(md)


def main() -> None:
    print("[Phase 0] Loading data ...")
    data = _load_data()

    # Pre-compute TF-IDF keywords once
    by_leaf_texts = {
        cid: [data["posts"][p].get("text", "") for p in members if p in data["posts"]]
        for cid, members in data["clusters"].items() if not cid.endswith("/-1")
    }
    leaf_keywords: dict[str, list[str]] = {lid: [] for lid in by_leaf_texts}
    docs = [_URL_RE.sub(" ", " ".join(t)) for t in by_leaf_texts.values()]
    try:
        vec = TfidfVectorizer(token_pattern=r"[가-힣]{2,}", max_features=20000,
                               max_df=0.6, min_df=2, sublinear_tf=True)
        m = vec.fit_transform(docs)
        feats = vec.get_feature_names_out()
        for i, lid in enumerate(by_leaf_texts.keys()):
            s = m[i].toarray().ravel()
            top = s.argsort()[::-1][:8]
            leaf_keywords[lid] = [feats[j] for j in top if s[j] > 0]
    except ValueError:
        pass

    leaf_categories = {lid: data["slug_to_name"].get(lid.split("/", 1)[0], lid.split("/", 1)[0])
                       for lid in by_leaf_texts}

    # Phase 1: leaf centroids
    print("[Phase 1] Computing leaf centroids ...")
    leaf_ids, leaf_cents = _compute_leaf_centroids(data)
    leaf_idx = {l: i for i, l in enumerate(leaf_ids)}
    print(f"  → {len(leaf_ids)} leaves")

    # Phase 2: variants
    print("[Phase 2] Building variants ...")
    variant_themes_data: dict[str, dict] = {}
    for v in VARIANTS:
        if v["k"] is None:
            variant_themes_data[v["name"]] = None
            continue
        print(f"  Variant {v['name']} (k={v['k']}) ...")
        themes = _agglomerative_themes(leaf_ids, leaf_cents, v["k"])
        # theme centroids
        tcents = _theme_centroids(themes, leaf_idx, leaf_cents)
        tids = sorted(tcents.keys())
        tcents_arr = np.stack([tcents[t] for t in tids])
        # multi-membership
        leaf_themes = _multi_membership(leaf_ids, leaf_cents, tids, tcents_arr,
                                          top_n=v["multi_top"])
        # LLM theme labels
        theme_meta = {}
        for tid in tids:
            label = _label_theme(tid, themes[tid], leaf_keywords, leaf_categories, data)
            theme_meta[tid] = {
                "name": label["name"],
                "description": label["description"],
                "n_leaves": len(themes[tid]),
            }
            print(f"    {tid}: {label['name']} ({len(themes[tid])} leaves)")
        # save
        variant_path = INT / f"leaf_themes_{v['name']}.json"
        variant_path.write_text(
            json.dumps({
                "k": v["k"], "multi_top": v["multi_top"],
                "themes": themes, "theme_meta": theme_meta,
                "leaf_themes": leaf_themes,
            }, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        variant_themes_data[v["name"]] = {
            "themes": themes, "theme_meta": theme_meta, "leaf_themes": leaf_themes,
        }

    # Phase 3: outline drafting per (variant, topic)
    print("\n[Phase 3] Drafting outlines (4 variants × 4 topics = 16) ...")
    today = datetime.now().strftime("%Y-%m-%d")
    out_root = VAULT / "_reports" / today / "book_outlines_cv"
    out_root.mkdir(parents=True, exist_ok=True)

    grid: dict = {}  # variant → topic → {result, score}
    for v in VARIANTS:
        vname = v["name"]
        grid[vname] = {}
        # build inventory once per variant (theme info baked in)
        vd = variant_themes_data.get(vname)
        leaf_themes = vd["leaf_themes"] if vd else None
        theme_meta = vd["theme_meta"] if vd else None
        inventory = _build_inventory_text(leaf_ids, data, leaf_keywords, v, leaf_themes, theme_meta)
        inventory_size = inventory.count("\n`")

        # If variant has theme axis, prepend a theme legend
        if vd:
            legend_lines = ["# Theme axis (보조 — 카테고리를 가로지르는 추상 통찰 axis)\n"]
            for tid, meta in sorted(theme_meta.items()):
                legend_lines.append(f"- **{meta['name']}** ({tid}, {meta['n_leaves']} leaves) — {meta['description']}")
            legend_lines.append("")
            legend = "\n".join(legend_lines)
        else:
            legend = ""

        system = SYSTEM_TEMPLATE.replace("{leaf_inventory}", legend + inventory)
        out_dir = out_root / vname
        out_dir.mkdir(parents=True, exist_ok=True)

        for topic in TOPICS:
            print(f"  {vname} × {topic['key']} ...")
            user = (
                f"# 책 테마\n{topic['name']}\n\n"
                f"# 의도\n{topic['intent']}\n\n"
                f"# Scope hint\n{topic['scope_hint']}\n\n"
                "위 inventory 와 의도에 맞춰 책 개요 초안을 작성하세요. "
                "한 leaf 가 여러 챕터에 등장해도 OK. supporting_leaves 는 leaf_id 그대로."
            )
            result = _draft_outline(OUTLINE_MODEL, system, user, f"book_outlines_cv/{vname}")
            score_info = _score_outline(result, inventory_size)
            grid[vname][topic["key"]] = {**score_info, "result": result}

            md = _format_outline_md(topic["name"], topic["intent"], result)
            (out_dir / f"{topic['key']}.md").write_text(md, encoding="utf-8")
            print(f"    score={score_info['score']} (chap={score_info['n_chapters']}, "
                  f"rich={score_info['n_rich_chapters']}, gaps={score_info['n_gaps']}, "
                  f"orph={score_info['n_orphans']})")

    # Save score grid
    cv_scores_path = INT / "cv_scores.json"
    score_only_grid = {v: {k: {kk: vv for kk, vv in t.items() if kk != "result"}
                          for k, t in topics.items()}
                       for v, topics in grid.items()}
    cv_scores_path.write_text(json.dumps(score_only_grid, ensure_ascii=False, indent=2),
                                encoding="utf-8")

    # Phase 4: 4-fold CV
    print("\n[Phase 4] CV (4-fold leave-one-out) ...")
    topics_list = [t["key"] for t in TOPICS]
    fold_winners = {}
    held_out_scores: dict[str, list[float]] = {v["name"]: [] for v in VARIANTS}
    for held_out in topics_list:
        train_topics = [t for t in topics_list if t != held_out]
        # variant ranking by mean train score
        v_scores = {}
        for v in VARIANTS:
            vname = v["name"]
            train_score = np.mean([grid[vname][t]["score"] for t in train_topics])
            v_scores[vname] = train_score
        winner = max(v_scores.items(), key=lambda x: x[1])
        fold_winners[held_out] = {"winner": winner[0], "train_mean": round(winner[1], 2),
                                   "held_out_score": round(grid[winner[0]][held_out]["score"], 2)}
        for v in VARIANTS:
            held_out_scores[v["name"]].append(grid[v["name"]][held_out]["score"])

    print("  Fold winners:")
    for ho, info in fold_winners.items():
        print(f"    held_out={ho}: winner={info['winner']} (train_mean={info['train_mean']}, held_out={info['held_out_score']})")

    # Aggregate: best variant by avg held-out + variance
    print("\n  Variant aggregate scores:")
    aggregate = {}
    for v in VARIANTS:
        scores = held_out_scores[v["name"]]
        aggregate[v["name"]] = {
            "mean": round(float(np.mean(scores)), 2),
            "std": round(float(np.std(scores)), 2),
            "scores": [round(s, 2) for s in scores],
        }
        print(f"    {v['name']}: mean={aggregate[v['name']]['mean']} std={aggregate[v['name']]['std']}")

    # winner: highest mean (tiebreak by lower std)
    winner = sorted(aggregate.items(), key=lambda x: (-x[1]["mean"], x[1]["std"]))[0][0]
    print(f"\n  → Winner: {winner}")

    # Phase 5: persist winner
    if winner != "V0_baseline":
        winner_path = INT / f"leaf_themes_{winner}.json"
        target_path = INT / "leaf_themes.json"
        target_path.write_text(winner_path.read_text(), encoding="utf-8")
        print(f"  persisted theme axis → {target_path}")

    # CV report
    cv_report = []
    cv_report.append(f"# CV Theme Axis Search — {today}\n")
    cv_report.append("## Variants\n")
    for v in VARIANTS:
        cv_report.append(f"- {v['name']}: k={v.get('k')}, multi_top={v.get('multi_top', '-')}")
    cv_report.append("")
    cv_report.append("## Score grid (variant × topic)\n")
    cv_report.append("| Variant | " + " | ".join([t["name"] for t in TOPICS]) + " | mean |")
    cv_report.append("|---|" + "|".join(["---:"] * (len(TOPICS) + 1)) + "|")
    for v in VARIANTS:
        vname = v["name"]
        scores = [grid[vname][t["key"]]["score"] for t in TOPICS]
        mean = float(np.mean(scores))
        cv_report.append(f"| {vname} | " + " | ".join([f"{s:.2f}" for s in scores]) + f" | {mean:.2f} |")
    cv_report.append("")
    cv_report.append("## CV folds (leave-one-out)\n")
    cv_report.append("| Held out | Winner (best train mean) | Train mean | Held-out score |")
    cv_report.append("|---|---|---:|---:|")
    for ho, info in fold_winners.items():
        cv_report.append(f"| {ho} | {info['winner']} | {info['train_mean']} | {info['held_out_score']} |")
    cv_report.append("")
    cv_report.append("## Aggregate held-out\n")
    cv_report.append("| Variant | held-out mean | std | held-out scores |")
    cv_report.append("|---|---:|---:|---|")
    for v in VARIANTS:
        a = aggregate[v["name"]]
        cv_report.append(f"| {v['name']} | {a['mean']} | {a['std']} | {a['scores']} |")
    cv_report.append("")
    cv_report.append(f"**Winner: `{winner}`**\n")
    cv_report.append("(이 variant 의 theme axis 가 `_intermediate/leaf_themes.json` 에 저장됨)")
    (out_root / "cv_report.md").write_text("\n".join(cv_report), encoding="utf-8")

    print(f"\n[done] CV report → {out_root / 'cv_report.md'}")
    print(f"[done] outlines → {out_root}/<variant>/<topic>.md")


if __name__ == "__main__":
    main()
