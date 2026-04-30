"""Cluster analysis report — reads 05_*.json and produces a markdown + charts.

Output at `<vault>/Private/Facebook/_reports/<date>/cluster_analysis/`. Re-run
after any cluster reconfiguration.
"""

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.font_manager as fm  # noqa: E402
import numpy as np  # noqa: E402

from .paths import fb_root, intermediate_dir


def _setup_korean_font() -> None:
    candidates = [
        "AppleSDGothicNeo-Regular",
        "Apple SD Gothic Neo",
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False


def _save(fig, path: Path, dpi: int = 130) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _chart_posts_per_category(stats: dict, img_dir: Path) -> str:
    cats = stats["categories"]
    items = sorted(cats.items(), key=lambda x: -x[1]["posts"])
    names = [v["name"] for _, v in items]
    posts = [v["posts"] for _, v in items]
    leaves = [v["leaves"] for _, v in items]
    strict = [v["is_strict"] for _, v in items]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = ["#999" if s else "#3b82f6" for s in strict]
    bars = ax.barh(names, posts, color=colors)
    for bar, n_leaves, n_posts in zip(bars, leaves, posts):
        ax.text(n_posts + 10, bar.get_y() + bar.get_height() / 2,
                f"{n_leaves} leaves", va="center", fontsize=9, color="#444")
    ax.invert_yaxis()
    ax.set_xlabel("keep_for_synthesis posts")
    ax.set_title("카테고리별 글 수 + leaf 수 (회색 = strict, unclustered)")
    name = "01_posts_per_category.png"
    _save(fig, img_dir / name)
    return name


def _chart_leaf_size_histogram(stats: dict, img_dir: Path) -> str:
    leaves = [l for l in stats["leaves"] if not l["is_strict"]]
    sizes = [l["size"] for l in leaves]
    leftover_sizes = [l["size"] for l in leaves if l["is_leftover"]]
    bins = list(range(0, 45, 2))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(sizes, bins=bins, color="#3b82f6", alpha=0.85, label="all leaves")
    ax.hist(leftover_sizes, bins=bins, color="#f59e0b", alpha=0.85, label="leftover")
    params = stats["global"]["params"]
    ax.axvline(params["leaf_max"], color="red", linestyle="--", linewidth=1, label=f"LEAF_MAX={params['leaf_max']}")
    ax.axvline(params["leaf_min"], color="gray", linestyle=":", linewidth=1, label=f"LEAF_MIN={params['leaf_min']}")
    ax.set_xlabel("leaf size (post 수)")
    ax.set_ylabel("leaf 개수")
    ax.set_title(f"Leaf 크기 분포 ({len(leaves)} leaves, mean={np.mean(sizes):.0f}, median={np.median(sizes):.0f})")
    ax.legend()
    name = "02_leaf_size_histogram.png"
    _save(fig, img_dir / name)
    return name


def _chart_depth_distribution(stats: dict, img_dir: Path) -> str:
    cats = stats["categories"]
    leaves_by_cat_by_depth: dict[str, Counter] = defaultdict(Counter)
    for leaf in stats["leaves"]:
        leaves_by_cat_by_depth[leaf["category_slug"]][leaf["depth"]] += 1

    items = sorted(cats.items(), key=lambda x: -x[1]["posts"])
    names = [v["name"] for _, v in items]
    slugs = [k for k, _ in items]
    max_depth = max(int(d) for d in stats["global"]["depth_distribution"].keys())
    depth_levels = list(range(max_depth + 1))

    counts = np.array([
        [leaves_by_cat_by_depth[s].get(d, 0) for d in depth_levels]
        for s in slugs
    ])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bottoms = np.zeros(len(slugs))
    cmap = plt.get_cmap("viridis")
    for i, d in enumerate(depth_levels):
        ax.barh(names, counts[:, i], left=bottoms, color=cmap(i / max(1, max_depth)),
                label=f"depth {d}")
        bottoms += counts[:, i]
    ax.invert_yaxis()
    ax.set_xlabel("leaf 개수")
    ax.set_title("카테고리별 트리 깊이 분포 (가변 depth 시각화)")
    ax.legend(title="depth", loc="lower right", fontsize=9)
    name = "03_depth_distribution.png"
    _save(fig, img_dir / name)
    return name


def _chart_cohesion_histogram(stats: dict, img_dir: Path) -> str:
    leaves = [l for l in stats["leaves"] if not l["is_strict"]]
    cohs = [l["mean_cohesion"] for l in leaves]
    leftover = [l["mean_cohesion"] for l in leaves if l["is_leftover"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0.1, 1.0, 19)
    ax.hist(cohs, bins=bins, color="#3b82f6", alpha=0.85, label="all leaves")
    ax.hist(leftover, bins=bins, color="#f59e0b", alpha=0.85, label="leftover")
    ax.axvline(0.45, color="green", linestyle="--", linewidth=1, label="θ=0.45 (tight)")
    ax.set_xlabel("mean within-leaf cosine")
    ax.set_ylabel("leaf 개수")
    ax.set_title(f"Leaf 응집도 분포 (mean={np.mean(cohs):.3f})")
    ax.legend()
    name = "04_cohesion_histogram.png"
    _save(fig, img_dir / name)
    return name


def _chart_time_series_categories(ts: dict, img_dir: Path) -> str:
    """Stacked area: posts per (year × category)."""
    by_cat = ts["by_category"]
    all_years_set: set[str] = set()
    for cat, yrs in by_cat.items():
        all_years_set.update(yrs.keys())
    years = sorted(int(y) for y in all_years_set)
    cats_sorted = sorted(by_cat.keys(), key=lambda c: -sum(by_cat[c].values()))

    matrix = np.zeros((len(cats_sorted), len(years)), dtype=int)
    for i, cat in enumerate(cats_sorted):
        for j, y in enumerate(years):
            matrix[i, j] = by_cat[cat].get(str(y), 0)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.get_cmap("tab20")
    ax.stackplot(years, matrix, labels=cats_sorted,
                 colors=[cmap(i / max(1, len(cats_sorted)-1)) for i in range(len(cats_sorted))])
    ax.set_xlabel("년도")
    ax.set_ylabel("posts")
    ax.set_title("연도별 카테고리 분포 (stacked area)")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    name = "05_timeseries_category_stacked.png"
    _save(fig, img_dir / name)
    return name


def _chart_time_series_categories_lines(ts: dict, img_dir: Path) -> str:
    """Line chart: each category's trajectory."""
    by_cat = ts["by_category"]
    all_years_set: set[str] = set()
    for cat, yrs in by_cat.items():
        all_years_set.update(yrs.keys())
    years = sorted(int(y) for y in all_years_set)
    cats_sorted = sorted(by_cat.keys(), key=lambda c: -sum(by_cat[c].values()))[:8]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for cat in cats_sorted:
        vals = [by_cat[cat].get(str(y), 0) for y in years]
        ax.plot(years, vals, marker="o", markersize=3, linewidth=1.5, label=cat)
    ax.set_xlabel("년도")
    ax.set_ylabel("posts")
    ax.set_title("주요 카테고리별 시간 추이 (top 8)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    name = "06_timeseries_category_lines.png"
    _save(fig, img_dir / name)
    return name


def _chart_top_leaves_heatmap(ts: dict, stats: dict, img_dir: Path) -> str:
    """Heatmap: top-30 leaves × years."""
    leaves_sorted = sorted(stats["leaves"], key=lambda x: -x["size"])[:30]
    by_leaf = ts["by_leaf"]
    all_years_set: set[str] = set()
    for leaf_id in [l["id"] for l in leaves_sorted]:
        all_years_set.update(by_leaf.get(leaf_id, {}).keys())
    years = sorted(int(y) for y in all_years_set)

    matrix = np.zeros((len(leaves_sorted), len(years)), dtype=int)
    for i, leaf in enumerate(leaves_sorted):
        for j, y in enumerate(years):
            matrix[i, j] = by_leaf.get(leaf["id"], {}).get(str(y), 0)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right")
    labels = [f"{l['id']} (n={l['size']}, c={l['mean_cohesion']:.2f})" for l in leaves_sorted]
    ax.set_yticks(range(len(leaves_sorted)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title("Top-30 Leaves × Year (post count)")
    fig.colorbar(im, ax=ax, label="posts")
    name = "07_top_leaves_heatmap.png"
    _save(fig, img_dir / name)
    return name


def _chart_per_category_summary(stats: dict, img_dir: Path) -> str:
    """Posts vs leaves per category, scatter with size = leaf_size_mean."""
    cats = stats["categories"]
    items = [(v["name"], v) for _, v in cats.items() if not v["is_strict"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    for name, v in items:
        x = v["posts"]
        y = v["leaves"]
        s = max(20, v["leaf_size_mean"] * 8)
        ax.scatter(x, y, s=s, alpha=0.7)
        ax.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("총 글 수")
    ax.set_ylabel("leaf 수")
    ax.set_title("카테고리별 글 수 vs leaf 수 (점 크기 = mean leaf size)")
    ax.grid(True, alpha=0.3)
    name = "08_category_scatter.png"
    _save(fig, img_dir / name)
    return name


def _build_topic_preview(clusters: dict, stats: dict, posts: dict, top_n: int = 30) -> str:
    """Top-N leaves by size, with first post as topic preview."""
    leaves_sorted = sorted([l for l in stats["leaves"] if not l["is_strict"]],
                          key=lambda x: -x["size"])[:top_n]
    lines = []
    for leaf in leaves_sorted:
        lid = leaf["id"]
        members = clusters[lid]
        sample = posts.get(members[0], {})
        text = (sample.get("text") or "").replace("\n", " ").strip()[:100]
        date = sample.get("date", "")
        cat = leaf["category_slug"]
        cohesion = leaf["mean_cohesion"]
        depth = leaf["depth"]
        lines.append(f"- **`{lid}`** (n={leaf['size']}, cohesion={cohesion:.2f}, depth={depth})")
        lines.append(f"  - [{date}] {text}…")
    return "\n".join(lines)


def _build_per_category_section(stats: dict, clusters: dict, posts: dict) -> str:
    """For each category, list top 5 leaves with first-post sample."""
    out = []
    cats = stats["categories"]
    for slug, c in sorted(cats.items(), key=lambda x: -x[1]["posts"]):
        if c["is_strict"]:
            out.append(f"\n### {c['name']} (strict, {c['posts']} posts, unclustered)\n")
            continue
        out.append(
            f"\n### {c['name']} ({c['posts']} posts → {c['leaves']} leaves, "
            f"depth {c['depth_min']}–{c['depth_max']}, mean leaf {c['leaf_size_mean']:.0f})\n"
        )
        leaves = sorted(
            [l for l in stats["leaves"] if l["category_slug"] == slug and not l["is_leftover"]],
            key=lambda x: -x["size"],
        )[:5]
        for leaf in leaves:
            members = clusters.get(leaf["id"], [])
            if not members:
                continue
            sample = posts.get(members[0], {})
            text = (sample.get("text") or "").replace("\n", " ").strip()[:120]
            date = sample.get("date", "")
            out.append(
                f"- **`{leaf['id']}`** (n={leaf['size']}, cohesion={leaf['mean_cohesion']:.2f}, "
                f"depth={leaf['depth']})  \n"
                f"  [{date}] {text}…"
            )
    return "\n".join(out)


def run() -> Path:
    _setup_korean_font()

    int_dir = intermediate_dir()
    stats = json.loads((int_dir / "05_cluster_stats.json").read_text(encoding="utf-8"))
    ts = json.loads((int_dir / "05_time_series.json").read_text(encoding="utf-8"))
    hierarchy = json.loads((int_dir / "05_hierarchy.json").read_text(encoding="utf-8"))
    clusters = json.loads((int_dir / "05_clusters.json").read_text(encoding="utf-8"))

    posts: dict[str, dict] = {}
    classified_path = int_dir / "03_classified.jsonl"
    if classified_path.exists():
        with classified_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                posts[rec["post_id"]] = rec

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = fb_root() / "_reports" / today / "cluster_analysis"
    img_dir = out_dir / "img"
    img_dir.mkdir(parents=True, exist_ok=True)

    img1 = _chart_posts_per_category(stats, img_dir)
    img2 = _chart_leaf_size_histogram(stats, img_dir)
    img3 = _chart_depth_distribution(stats, img_dir)
    img4 = _chart_cohesion_histogram(stats, img_dir)
    img5 = _chart_time_series_categories(ts, img_dir)
    img6 = _chart_time_series_categories_lines(ts, img_dir)
    img7 = _chart_top_leaves_heatmap(ts, stats, img_dir)
    img8 = _chart_per_category_summary(stats, img_dir)

    g = stats["global"]
    params = g["params"]
    md = []
    md.append(f"# 클러스터 분석 리포트 — {today}\n")
    md.append("이 리포트는 `fbpull cluster` 산출물 (`05_*.json`) 을 자동으로 분석한 결과다. "
              "본 결과를 보고 synthesize 진행 여부를 결정한다.\n")
    md.append("## 실행 메타데이터\n")
    md.append(f"- 총 글: **{g['total_posts']}** (keep_for_synthesis)")
    md.append(f"- 총 leaf: **{g['total_leaves']}**")
    md.append(f"- 카테고리: {g['total_categories']}")
    md.append(f"- 깊이 분포: {g['depth_distribution']}")
    md.append(f"- 응집도 histogram: {g['leaf_cohesion_histogram']}")
    md.append(f"- 알고리즘 파라미터: `{params}`\n")

    md.append("## 1. 분포 개요\n")
    md.append(f"![]({Path('img') / img1})\n")
    md.append("→ 사회비평·자기관리·취미·학습·메타인지가 큰 카테고리. 정치는 strict 라 unclustered (회색).\n")

    md.append(f"![]({Path('img') / img2})\n")
    md.append(f"→ Leaf 크기는 LEAF_MIN={params['leaf_min']} 이상, LEAF_MAX={params['leaf_max']} 이하로 모두 bounded. "
              "leftover (주황) 는 sparse stragglers — 응집 못 이룬 글들의 보존 버킷.\n")

    md.append(f"![]({Path('img') / img4})\n")
    md.append("→ 대부분 leaf 가 cohesion 0.3–0.5 구간. 0.45+ 는 'tight thematic group'. "
              "1.0 근처는 거의 singleton (글이 1-2개).\n")

    md.append("## 2. 트리 구조\n")
    md.append(f"![]({Path('img') / img3})\n")
    md.append("→ 가변 depth: 응집 sub-theme 풍부한 카테고리는 깊이, sparse 한 카테고리는 얕음. "
              "창업·경영·학습·메타인지가 가장 다층, 대기업·임원·기타·미분류는 단일 leaf.\n")

    md.append(f"![]({Path('img') / img8})\n")
    md.append("→ 글 수 vs leaf 수 의 관계. 점 크기는 mean leaf size. 일반적으로 글 많을수록 "
              "leaf 도 많지만 카테고리별 응집 패턴에 따라 달라짐.\n")

    md.append("## 3. Top 30 Leaves (글 수 기준)\n")
    md.append(_build_topic_preview(clusters, stats, posts, top_n=30))
    md.append("\n")

    md.append("## 4. 카테고리별 상위 5 leaves (sample post)\n")
    md.append(_build_per_category_section(stats, clusters, posts))
    md.append("\n")

    md.append("## 5. 시간 추이 — 카테고리\n")
    md.append(f"![]({Path('img') / img5})\n")
    md.append("→ 연도별 카테고리 분포 (stacked area). 페이스북 활발기·잠잠기 + 어떤 시기에 어떤 주제가 우세했는지.\n")
    md.append(f"![]({Path('img') / img6})\n")
    md.append("→ 같은 데이터 line chart. 각 카테고리가 언제 정점·하강하는지.\n")

    md.append("## 6. 시간 추이 — Top 30 Leaves\n")
    md.append(f"![]({Path('img') / img7})\n")
    md.append("→ 각 leaf 가 언제 어떤 강도로 등장했는지. 짙은 색 = 그 해에 그 leaf 의 글이 많았음.\n")

    md.append("## 점검 포인트\n")
    md.append("- [ ] 1번 분포: 카테고리 비율이 직관과 맞나? (사회비평 1022 가 1위인데 잡탕 가능성)")
    md.append("- [ ] 2번 트리: 깊이 5 가 정당한가? (창업·경영) 너무 깊으면 MAX_DEPTH 줄이는 것 고려")
    md.append("- [ ] 3번 Top leaves: 각 leaf 의 첫 글이 그 leaf 주제를 대표하는가?")
    md.append("- [ ] 4번 카테고리별: 실제 책 챕터 outline 으로 쓸 만한 묶음인지")
    md.append("- [ ] 5번 시간추이: 시기별 큰 흐름이 본인 인생 시기와 일치하는가? (era 매핑 정합성)")
    md.append("- [ ] 6번 leaf 시간추이: 단발성 leaf vs 장기 지속 leaf 패턴")
    md.append("\n이상 없으면 `fbpull synthesize` 진행.\n")

    out_path = out_dir / "README.md"
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"[report] {out_path}")
    print(f"  charts: {img_dir} ({len(list(img_dir.glob('*.png')))} files)")
    return out_path
