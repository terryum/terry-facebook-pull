import json
from collections import Counter, defaultdict
from datetime import date
from typing import Any

from slugify import slugify

from . import taxonomy as taxonomy_mod
from .frontmatter import write_note
from .paths import (
    archive_dir,
    fb_root,
    index_path,
    intermediate_dir,
    synthesized_dir,
)


def _archive_basename(rec: dict) -> str:
    d = rec["date"]  # YYYY-MM-DD
    yymmdd = d[2:4] + d[5:7] + d[8:10]
    text = rec.get("text") or ""
    slug = slugify(text[:60], allow_unicode=False, max_length=40) or "post"
    return f"{yymmdd}-{slug}"


def _unique_name(base: str, used: set[str]) -> str:
    name = base
    i = 2
    while name in used:
        name = f"{base}-{i}"
        i += 1
    used.add(name)
    return name


def _cluster_num(cid_str: str) -> int:
    if "/" in cid_str:
        _, _, num = cid_str.rpartition("/")
    else:
        num = cid_str
    try:
        return int(num)
    except ValueError:
        return -1


def run() -> dict[str, int]:
    fb_root().mkdir(parents=True, exist_ok=True)
    archive_dir().mkdir(parents=True, exist_ok=True)
    synthesized_dir().mkdir(parents=True, exist_ok=True)

    filtered_path = intermediate_dir() / "02_filtered.jsonl"
    classified_path = intermediate_dir() / "03_classified.jsonl"
    clusters_path = intermediate_dir() / "05_clusters.json"
    neighbors_path = intermediate_dir() / "05_neighbors.json"
    synth_path = intermediate_dir() / "06_synthesized.jsonl"

    if not filtered_path.exists():
        raise FileNotFoundError("Run pipeline (parse → filter → ...) before export")

    tax = taxonomy_mod.load()
    cat_by_name: dict[str, taxonomy_mod.Category] = {
        c.name: c for c in (tax.categories if tax else [])
    }

    # Kept posts (Archive candidates)
    kept: list[dict] = []
    with filtered_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kept"):
                kept.append(rec)

    # Classify output: category, era, primary_topic, type
    cls_map: dict[str, dict] = {}
    if classified_path.exists():
        with classified_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cls_map[rec["post_id"]] = rec

    # Cluster assignment per post (string id or None for noise)
    cluster_for: dict[str, str | None] = {}
    cluster_members: dict[str, list[str]] = {}
    if clusters_path.exists():
        clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
        for cid_str, members in clusters.items():
            n = _cluster_num(cid_str)
            cluster_members[cid_str] = members
            for pid in members:
                cluster_for[pid] = None if n < 0 else cid_str

    neighbors: dict[str, list[dict]] = {}
    if neighbors_path.exists():
        neighbors = json.loads(neighbors_path.read_text(encoding="utf-8"))

    # Synthesized notes — keyed by cluster_id for upstream link lookup
    synth_records: list[dict] = []
    synth_slug_for: dict[str, str] = {}
    if synth_path.exists():
        with synth_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                synth_records.append(rec)
                synth_slug_for[rec["cluster_id"]] = rec["slug"]

    # Stable Archive filenames
    kept.sort(key=lambda r: (r["date"], r["post_id"]))
    used_names: set[str] = set()
    name_for: dict[str, str] = {}
    for rec in kept:
        name_for[rec["post_id"]] = _unique_name(_archive_basename(rec), used_names)

    # Write Archive notes
    n_archive = 0
    for rec in kept:
        name = name_for[rec["post_id"]]
        cls = cls_map.get(rec["post_id"], {})
        cid_str = cluster_for.get(rec["post_id"])
        cat_name = cls.get("category") or ""
        cat_obj = cat_by_name.get(cat_name)
        era = cls.get("era") or ""
        topic = cls.get("primary_topic") or ""

        tags = ["facebook", "archive"]
        if cat_obj:
            tags.append(cat_obj.slug)
        if topic:
            stopic = slugify(topic, allow_unicode=False, max_length=30)
            if stopic and stopic not in tags:
                tags.append(stopic)
        if cat_obj and cat_obj.strict:
            tags.append("strict")
        if cat_obj and cat_obj.sensitive:
            tags.append("sensitive")

        meta: dict[str, Any] = {
            "type": "archive",
            "source": "facebook",
            "visibility": "private",
            "created_at": rec["date"],
            "fb_post_id": str(rec.get("post_id", "")),
            "source_path": rec.get("source_path", ""),
            "category": cat_name,
            "era": era,
            "cluster_id": cid_str,
            "tags": tags,
        }
        if cat_obj and cat_obj.strict:
            meta["strict"] = True
        if cat_obj and cat_obj.sensitive:
            meta["sensitive"] = True

        body = rec["text"]
        if cid_str and cid_str in synth_slug_for:
            body += f"\n\n## 속한 컨셉\n- [[{synth_slug_for[cid_str]}]]\n"

        ns = neighbors.get(rec["post_id"], [])
        if ns:
            body += "\n\n## 비슷한 글\n"
            for n in ns:
                neighbor_name = name_for.get(n["id"])
                if neighbor_name:
                    body += f"- [[{neighbor_name}]]\n"

        write_note(archive_dir() / f"{name}.md", meta, body)
        n_archive += 1

    # Write Synthesized notes
    n_synth = 0
    synth_index: list[dict] = []
    for rec in synth_records:
        slug = rec["slug"]
        tag = rec.get("primary_tag") or ""
        cat_name = rec.get("category", "")
        cat_obj = cat_by_name.get(cat_name)
        tags = ["facebook", "synthesized"]
        if cat_obj:
            tags.append(cat_obj.slug)
        if tag:
            slug_tag = slugify(tag, allow_unicode=False, max_length=30)
            if slug_tag and slug_tag not in tags:
                tags.append(slug_tag)
        if rec.get("sensitive"):
            tags.append("sensitive")
        meta = {
            "type": "synthesized",
            "source": "facebook",
            "visibility": "private",
            "created_at": str(date.today()),
            "cluster_id": rec["cluster_id"],
            "category": cat_name,
            "member_count": len(rec["member_post_ids"]),
            "tags": tags,
        }
        if rec.get("sensitive"):
            meta["sensitive"] = True
        body = f"# {rec['title']}\n\n{rec['body']}\n\n## 멤버 글\n"
        for pid in rec["member_post_ids"]:
            if pid in name_for:
                body += f"- [[{name_for[pid]}]]\n"
        write_note(synthesized_dir() / f"{slug}.md", meta, body)
        synth_index.append(
            {
                "slug": slug,
                "title": rec["title"],
                "cluster_id": rec["cluster_id"],
                "category": cat_name,
                "member_count": len(rec["member_post_ids"]),
            }
        )
        n_synth += 1

    # Build coverage matrix (era × category) and write _coverage.md
    if tax is not None:
        _write_coverage(tax, kept, cls_map, cluster_members)

    # _index.md
    _write_index(n_archive, n_synth, synth_index, tax)

    print(f"[export] archive={n_archive} synthesized={n_synth} index=1 coverage={'1' if tax else '0'}")
    return {"archive": n_archive, "synthesized": n_synth}


def _write_coverage(
    tax: taxonomy_mod.Taxonomy,
    kept: list[dict],
    cls_map: dict[str, dict],
    cluster_members: dict[str, list[str]],
) -> None:
    """Generate _coverage.md: era × category matrix + sparse cells."""

    # Counts: posts and clusters per (era, category_name)
    posts_per_cell: dict[tuple[str, str], int] = defaultdict(int)
    for rec in kept:
        cls = cls_map.get(rec["post_id"], {})
        era = cls.get("era") or "unknown"
        cat = cls.get("category") or "기타·미분류"
        posts_per_cell[(era, cat)] += 1

    clusters_per_cell: dict[tuple[str, str], int] = defaultdict(int)
    for cid_str, members in cluster_members.items():
        n = _cluster_num(cid_str)
        if n < 0:
            continue
        cat_slug, _, _ = cid_str.rpartition("/")
        # Find category name by slug
        cat_name = next(
            (c.name for c in tax.categories if c.slug == cat_slug),
            cat_slug,
        )
        # Era of this cluster: take majority era of members
        era_counts = Counter(cls_map.get(pid, {}).get("era", "unknown") for pid in members)
        era = era_counts.most_common(1)[0][0] if era_counts else "unknown"
        clusters_per_cell[(era, cat_name)] += 1

    # Build matrix
    era_labels = [e.label for e in tax.eras]
    lines = ["| Category | " + " | ".join(era_labels) + " |"]
    lines.append("|" + "---|" * (len(era_labels) + 1))

    # Active categories/eras = those with ≥1 post somewhere. Cells outside
    # this product are uninteresting (no signal that user ever cared).
    active_cats = {cat for (era, cat), n in posts_per_cell.items() if n > 0}
    active_eras = {era for (era, cat), n in posts_per_cell.items() if n > 0}

    sparse_cells: list[tuple[str, str, int]] = []  # (era, cat, posts) where 1 ≤ posts < 5
    blackout_cells: list[tuple[str, str]] = []     # (era, cat) where 0 posts BUT cat & era both active

    for cat in tax.categories:
        flag = ""
        if cat.strict:
            flag = " 🚫"
        elif cat.sensitive:
            flag = " ⚠️"
        row = [f"**{cat.name}**{flag}"]
        for era in era_labels:
            p = posts_per_cell.get((era, cat.name), 0)
            c = clusters_per_cell.get((era, cat.name), 0)
            row.append(f"{p} ({c})")
            if cat.strict:
                continue
            if 0 < p < 5 and cat.name in active_cats and era in active_eras:
                sparse_cells.append((era, cat.name, p))
            elif p == 0 and cat.name in active_cats and era in active_eras:
                blackout_cells.append((era, cat.name))
        lines.append("| " + " | ".join(row) + " |")

    body = "# Facebook Coverage Matrix\n\n"
    body += "각 셀: `<글 수> (<클러스터 수>)`. 클러스터 0 = 합성 안 됨 (글 부족 또는 STRICT/SENSITIVE).\n"
    body += "🚫 = STRICT (합성 절대 제외)  ⚠️ = SENSITIVE (기본 제외, opt-in 가능)\n\n"
    body += "\n".join(lines) + "\n\n"

    if sparse_cells:
        body += "## Sparse cells (1–4 글)\n\n"
        body += "글이 적게 있는 (era × category) 조합. 클러스터로 묶이기엔 부족하지만 시드는 있음:\n\n"
        for era, cat, p in sorted(sparse_cells, key=lambda t: (-t[2], t[0], t[1])):
            body += f"- **{cat}** × _{era}_ — {p}개 글\n"
        body += "\n"

    if blackout_cells:
        body += "## Blackout cells (0 글)\n\n"
        body += "사용자가 다른 시기엔 썼지만 이 시기엔 쓰지 못한/않은 카테고리.\n"
        body += "책 완성 위해 신규 집필을 고려할 수 있는 후보:\n\n"
        for era, cat in sorted(blackout_cells, key=lambda t: (t[0], t[1])):
            body += f"- **{cat}** × _{era}_\n"
        body += "\n"

    if tax.coverage_gradient:
        body += "## Coverage gradient (taxonomy 에서)\n\n"
        body += tax.coverage_gradient + "\n"

    meta = {
        "type": "coverage",
        "source": "facebook",
        "visibility": "private",
        "created_at": str(date.today()),
    }
    write_note(fb_root() / "_coverage.md", meta, body)


def _write_index(
    n_archive: int,
    n_synth: int,
    synth_index: list[dict],
    tax: taxonomy_mod.Taxonomy | None,
) -> None:
    body = "# Facebook Memory Index\n\n"
    body += f"- Archive: {n_archive} 글\n"
    body += f"- Synthesized: {n_synth} 컨셉\n"
    body += f"- Generated: {date.today()}\n\n"

    if tax is not None:
        body += "관련: [[_coverage]] (era × category 매트릭스), [[_taxonomy]] (분류 정의)\n\n"

    if synth_index:
        # Group by category
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for s in synth_index:
            by_cat[s.get("category", "기타")].append(s)
        body += "## Synthesized Concepts (by category)\n\n"
        for cat in sorted(by_cat.keys()):
            body += f"### {cat}\n\n"
            for s in sorted(by_cat[cat], key=lambda x: -x["member_count"]):
                body += f"- [[{s['slug']}]] — {s['title']} ({s['member_count']} 글)\n"
            body += "\n"
    else:
        body += "## Synthesized Concepts\n\n_(none yet)_\n"

    meta = {
        "type": "index",
        "source": "facebook",
        "visibility": "private",
        "created_at": str(date.today()),
    }
    write_note(index_path(), meta, body)
