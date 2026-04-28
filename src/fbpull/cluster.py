import json

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from . import taxonomy as taxonomy_mod
from .paths import intermediate_dir


def _hdbscan_within(arr: np.ndarray, min_cluster_size: int) -> list[int]:
    if len(arr) < max(2, min_cluster_size):
        return [-1] * len(arr)
    actual_min = max(2, min(min_cluster_size, max(2, len(arr) // 4)))
    hdb = HDBSCAN(min_cluster_size=actual_min, metric="euclidean")
    return hdb.fit_predict(arr).tolist()


def run(
    min_cluster_size: int = 4,
    top_n: int = 5,
    neighbor_threshold: float = 0.55,
) -> dict:
    embed_path = intermediate_dir() / "04_embeddings.npy"
    ids_path = intermediate_dir() / "04_post_ids.json"
    classified_path = intermediate_dir() / "03_classified.jsonl"
    if not embed_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Run `fbpull embed` first")

    arr: np.ndarray = np.load(embed_path)
    post_ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))
    pid_to_row = {pid: i for i, pid in enumerate(post_ids)}

    # Load category for each post (from classified jsonl)
    post_cat: dict[str, str] = {}
    if classified_path.exists():
        with classified_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                post_cat[rec["post_id"]] = rec.get("category", "")

    tax = taxonomy_mod.load()
    strict_categories = {c.name for c in (tax.categories if tax else []) if c.strict}
    cat_slug = {c.name: c.slug for c in (tax.categories if tax else [])}

    # Group posts by category
    by_cat: dict[str, list[str]] = {}
    for pid in post_ids:
        cat = post_cat.get(pid, "") or "기타·미분류"
        by_cat.setdefault(cat, []).append(pid)

    # Cluster within each non-strict category. Strict categories: every post is noise.
    clusters: dict[str, list[str]] = {}
    for cat, pids in sorted(by_cat.items()):
        slug = cat_slug.get(cat) or cat
        if cat in strict_categories:
            # No clustering for strict (정치) — every post stays unclustered.
            cid_str = f"{slug}/-1"
            clusters[cid_str] = sorted(pids)
            continue

        rows = np.array([arr[pid_to_row[p]] for p in pids])
        labels = _hdbscan_within(rows, min_cluster_size)
        for pid, lbl in zip(pids, labels):
            cid_str = f"{slug}/{lbl}"
            clusters.setdefault(cid_str, []).append(pid)

    for k in clusters:
        clusters[k].sort()
    clusters = {k: clusters[k] for k in sorted(clusters.keys())}

    (intermediate_dir() / "05_clusters.json").write_text(
        json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Neighbors: global cosine top-N. Strict-category posts are excluded both
    # as sources (no neighbors computed for them — keeps Archive notes clean)
    # and as candidates (other posts won't link into politics).
    is_strict = {pid for pid in post_ids if post_cat.get(pid, "") in strict_categories}
    neighbors: dict[str, list[dict]] = {}
    if len(post_ids) >= 2:
        sim = cosine_similarity(arr)
        np.fill_diagonal(sim, -1.0)
        # Mask strict columns so they never appear as neighbors of others
        strict_rows = [pid_to_row[p] for p in is_strict]
        if strict_rows:
            sim[:, strict_rows] = -1.0
        for i, pid in enumerate(post_ids):
            if pid in is_strict:
                neighbors[pid] = []
                continue
            row = sim[i]
            top_idx = np.argsort(-row)[:top_n]
            ns = [
                {"id": post_ids[j], "score": float(row[j])}
                for j in top_idx
                if row[j] >= neighbor_threshold
            ]
            neighbors[pid] = ns
    else:
        neighbors = {pid: [] for pid in post_ids}

    (intermediate_dir() / "05_neighbors.json").write_text(
        json.dumps(neighbors, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Stats
    cat_stats: dict[str, dict[str, int]] = {}
    for cid_str, pids in clusters.items():
        cat_slug_part, _, num = cid_str.rpartition("/")
        try:
            n = int(num)
        except ValueError:
            n = -1
        s = cat_stats.setdefault(cat_slug_part, {"clusters": 0, "noise": 0, "posts": 0})
        s["posts"] += len(pids)
        if n >= 0:
            s["clusters"] += 1
        else:
            s["noise"] += len(pids)

    n_clusters_total = sum(s["clusters"] for s in cat_stats.values())
    avg_n = sum(len(v) for v in neighbors.values()) / max(1, len(neighbors))
    print(f"[cluster] {n_clusters_total} clusters across {len(cat_stats)} categories, avg_neighbors={avg_n:.1f}")
    for cs, s in sorted(cat_stats.items()):
        print(f"  {cs}: posts={s['posts']} clusters={s['clusters']} noise={s['noise']}")
    return {"clusters": n_clusters_total}
