import json
from datetime import date
from typing import Any

from slugify import slugify

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

    kept: list[dict] = []
    with filtered_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kept"):
                kept.append(rec)

    cls_map: dict[str, dict] = {}
    if classified_path.exists():
        with classified_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                cls_map[rec["post_id"]] = rec

    cluster_for: dict[str, int | None] = {}
    if clusters_path.exists():
        clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
        for cid_str, members in clusters.items():
            cid = int(cid_str)
            for pid in members:
                cluster_for[pid] = None if cid < 0 else cid

    neighbors: dict[str, list[dict]] = {}
    if neighbors_path.exists():
        neighbors = json.loads(neighbors_path.read_text(encoding="utf-8"))

    # Pre-build cluster_id → synthesized slug map so each Archive note
    # can wiki-link UP to its representative (Synthesized) note.
    synth_slug_for: dict[int, str] = {}
    synth_records: list[dict] = []
    if synth_path.exists():
        with synth_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                synth_records.append(rec)
                synth_slug_for[rec["cluster_id"]] = rec["slug"]

    kept.sort(key=lambda r: (r["date"], r["post_id"]))
    used_names: set[str] = set()
    name_for: dict[str, str] = {}
    for rec in kept:
        name_for[rec["post_id"]] = _unique_name(_archive_basename(rec), used_names)

    n_archive = 0
    for rec in kept:
        name = name_for[rec["post_id"]]
        cls = cls_map.get(rec["post_id"], {})
        cid = cluster_for.get(rec["post_id"])
        topic = cls.get("primary_topic") or ""

        tags = ["facebook", "archive"]
        if topic:
            slug_topic = slugify(topic, allow_unicode=False, max_length=30)
            if slug_topic:
                tags.append(slug_topic)

        meta: dict[str, Any] = {
            "type": "archive",
            "source": "facebook",
            "visibility": "private",
            "created_at": rec["date"],
            "fb_post_id": str(rec.get("post_id", "")),
            "source_path": rec.get("source_path", ""),
            "cluster_id": cid,
            "tags": tags,
        }

        body = rec["text"]

        # Upstream link: Archive → Synthesized (cluster representative)
        if cid is not None and cid in synth_slug_for:
            body += f"\n\n## 속한 컨셉\n- [[{synth_slug_for[cid]}]]\n"

        # Sideways links: Archive ↔ Archive (top-N similar)
        ns = neighbors.get(rec["post_id"], [])
        if ns:
            body += "\n\n## 비슷한 글\n"
            for n in ns:
                neighbor_name = name_for.get(n["id"])
                if neighbor_name:
                    body += f"- [[{neighbor_name}]]\n"

        write_note(archive_dir() / f"{name}.md", meta, body)
        n_archive += 1

    n_synth = 0
    synth_index: list[dict] = []
    for rec in synth_records:
        slug = rec["slug"]
        tag = rec.get("primary_tag") or ""
        tags = ["facebook", "synthesized"]
        if tag:
            slug_tag = slugify(tag, allow_unicode=False, max_length=30)
            if slug_tag:
                tags.append(slug_tag)
        meta = {
            "type": "synthesized",
            "source": "facebook",
            "visibility": "private",
            "created_at": str(date.today()),
            "cluster_id": rec["cluster_id"],
            "member_count": len(rec["member_post_ids"]),
            "tags": tags,
        }
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
                "member_count": len(rec["member_post_ids"]),
            }
        )
        n_synth += 1

    idx_meta = {
        "type": "index",
        "source": "facebook",
        "visibility": "private",
        "created_at": str(date.today()),
    }
    idx_body = (
        "# Facebook Memory Index\n\n"
        f"- Archive: {n_archive} 글\n"
        f"- Synthesized: {n_synth} 컨셉\n"
        f"- Generated: {date.today()}\n\n"
        "## Synthesized Concepts\n\n"
    )
    for s in sorted(synth_index, key=lambda x: -x["member_count"]):
        idx_body += f"- [[{s['slug']}]] — {s['title']} ({s['member_count']} 글)\n"

    write_note(index_path(), idx_meta, idx_body)

    print(f"[export] archive={n_archive} synthesized={n_synth} index=1")
    return {"archive": n_archive, "synthesized": n_synth}
