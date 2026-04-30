import json
import math
import os
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from . import taxonomy as taxonomy_mod
from .paths import intermediate_dir


def _mean_cohesion(rows: np.ndarray) -> float:
    """Mean pairwise cosine within a set of L2-normalized vectors. 1.0 if singleton."""
    n = len(rows)
    if n < 2:
        return 1.0
    sim = rows @ rows.T
    iu = np.triu_indices(n, k=1)
    return float(sim[iu].mean())


def _split_recursive(
    rows: np.ndarray,
    indices: list[int],
    *,
    depth: int,
    prefix: str,
    leaf_max: int,
    leaf_min: int,
    target: int,
    lift: float,
    floor: float,
    min_grad: float,
    max_depth: int,
    nodes: dict[str, dict],
) -> list[tuple[str, list[int]]]:
    """Split a set of points recursively. Cohesion-driven, with size bounds.

    Returns list of (leaf_id, [global_indices]). Also writes intermediate node
    metadata into `nodes` for hierarchy export.
    """
    n = len(rows)
    parent_coh = _mean_cohesion(rows)
    nodes[prefix] = {
        "size": n,
        "mean_cohesion": parent_coh,
        "depth": depth,
        "is_leaf": False,
        "children": [],
        "is_leftover": prefix.endswith("/leftover"),
    }

    # Stop: small enough → leaf
    if n <= leaf_max:
        nodes[prefix]["is_leaf"] = True
        return [(prefix, indices)]

    # Depth cap → force size-split into leaves at this level
    if depth >= max_depth:
        forced_k = max(2, math.ceil(n / target))
        labels = KMeans(n_clusters=forced_k, random_state=0, n_init=3).fit_predict(rows)
        return _split_with_labels(
            rows, indices, labels, forced_k, depth, prefix, nodes,
            leaf_max=leaf_max, leaf_min=leaf_min, target=target,
            lift=lift, floor=floor, min_grad=min_grad, max_depth=max_depth,
            forced_at_cap=True,
        )

    # Cohesion-driven k search
    target_θ = max(floor, parent_coh + lift)
    best_k, best_score, best_labels = None, 0.0, None
    max_k_try = min(40, n // leaf_min)
    for k in range(2, max_k_try + 1):
        if n < k + 2:
            break
        labels = KMeans(n_clusters=k, random_state=0, n_init=3).fit_predict(rows)
        graduated = 0
        for i in range(k):
            sub = rows[labels == i]
            if len(sub) >= leaf_min and _mean_cohesion(sub) >= target_θ:
                graduated += len(sub)
        score = graduated / n
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels

    # Decision: cohesion or size-fallback (still semantic via KMeans)
    if best_score < min_grad:
        forced_k = max(2, math.ceil(n / target))
        best_labels = KMeans(n_clusters=forced_k, random_state=0, n_init=3).fit_predict(rows)
        best_k = forced_k

    return _split_with_labels(
        rows, indices, best_labels, best_k, depth, prefix, nodes,
        leaf_max=leaf_max, leaf_min=leaf_min, target=target,
        lift=lift, floor=floor, min_grad=min_grad, max_depth=max_depth,
        forced_at_cap=False,
    )


def _split_with_labels(
    rows, indices, labels, k, depth, prefix, nodes, *,
    leaf_max, leaf_min, target, lift, floor, min_grad, max_depth,
    forced_at_cap: bool,
):
    leaves: list[tuple[str, list[int]]] = []
    leftover_local: list[int] = []
    for i in range(k):
        sub_local = [j for j, lbl in enumerate(labels) if lbl == i]
        if len(sub_local) < leaf_min:
            leftover_local.extend(sub_local)
            continue
        sub_rows = rows[sub_local]
        sub_global = [indices[j] for j in sub_local]
        sub_prefix = f"{prefix}/{i}"
        if forced_at_cap:
            # at depth cap, treat each sub directly as leaf
            nodes[sub_prefix] = {
                "size": len(sub_global),
                "mean_cohesion": _mean_cohesion(sub_rows),
                "depth": depth + 1,
                "is_leaf": True,
                "children": [],
                "is_leftover": False,
            }
            nodes[prefix]["children"].append(sub_prefix)
            leaves.append((sub_prefix, sub_global))
        else:
            sub_leaves = _split_recursive(
                sub_rows, sub_global,
                depth=depth + 1, prefix=sub_prefix,
                leaf_max=leaf_max, leaf_min=leaf_min, target=target,
                lift=lift, floor=floor, min_grad=min_grad, max_depth=max_depth,
                nodes=nodes,
            )
            nodes[prefix]["children"].append(sub_prefix)
            leaves.extend(sub_leaves)

    if leftover_local:
        sub_global = [indices[j] for j in leftover_local]
        sub_rows = rows[leftover_local]
        sub_prefix = f"{prefix}/leftover"
        nodes[sub_prefix] = {
            "size": len(sub_global),
            "mean_cohesion": _mean_cohesion(sub_rows),
            "depth": depth + 1,
            "is_leaf": True,
            "children": [],
            "is_leftover": True,
        }
        nodes[prefix]["children"].append(sub_prefix)
        leaves.append((sub_prefix, sub_global))

    return leaves


def run(top_n: int = 5, neighbor_threshold: float = 0.55) -> dict:
    embed_path = intermediate_dir() / "04_embeddings.npy"
    ids_path = intermediate_dir() / "04_post_ids.json"
    classified_path = intermediate_dir() / "03_classified.jsonl"
    if not embed_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Run `fbpull embed` first")

    arr: np.ndarray = np.load(embed_path)
    post_ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))
    pid_to_row = {pid: i for i, pid in enumerate(post_ids)}

    # Per-post classify metadata (category, era, year)
    post_cat: dict[str, str] = {}
    post_era: dict[str, str] = {}
    post_year: dict[str, int] = {}
    if classified_path.exists():
        with classified_path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                pid = rec["post_id"]
                post_cat[pid] = rec.get("category", "")
                post_era[pid] = rec.get("era", "unknown")
                ts = rec.get("timestamp")
                if ts:
                    post_year[pid] = datetime.fromtimestamp(ts, tz=timezone.utc).year
                elif rec.get("date"):
                    post_year[pid] = int(rec["date"][:4])

    tax = taxonomy_mod.load()
    strict_categories = {c.name for c in (tax.categories if tax else []) if c.strict}
    cat_slug = {c.name: c.slug for c in (tax.categories if tax else [])}

    # F3' tunable parameters via env vars
    leaf_max = int(os.environ.get("FBPULL_LEAF_MAX", "40"))
    leaf_min = int(os.environ.get("FBPULL_LEAF_MIN", "8"))
    target = int(os.environ.get("FBPULL_TARGET", "30"))
    lift = float(os.environ.get("FBPULL_LIFT", "0.10"))
    floor = float(os.environ.get("FBPULL_FLOOR", "0.35"))
    min_grad = float(os.environ.get("FBPULL_MIN_GRAD", "0.20"))
    max_depth = int(os.environ.get("FBPULL_MAX_DEPTH", "4"))

    # Group posts by category
    by_cat: dict[str, list[str]] = {}
    for pid in post_ids:
        cat = post_cat.get(pid, "") or "기타·미분류"
        by_cat.setdefault(cat, []).append(pid)

    # Run F3' per category
    clusters: dict[str, list[str]] = {}
    nodes: dict[str, dict] = {}  # hierarchy
    for cat, pids in sorted(by_cat.items()):
        slug = cat_slug.get(cat) or cat
        if cat in strict_categories:
            # Strict (정치) — keep as single unclustered bucket, key matches prior format.
            cid = f"{slug}/-1"
            clusters[cid] = sorted(pids)
            nodes[slug] = {
                "size": len(pids), "mean_cohesion": 0.0, "depth": 0,
                "is_leaf": False, "children": [cid],
                "is_strict": True, "category_name": cat,
            }
            nodes[cid] = {
                "size": len(pids), "mean_cohesion": 0.0, "depth": 1,
                "is_leaf": True, "children": [], "is_leftover": False,
                "is_strict": True,
            }
            continue

        rows_idx = [pid_to_row[p] for p in pids]
        rows = arr[rows_idx]
        leaves = _split_recursive(
            rows, list(range(len(pids))),
            depth=0, prefix=slug,
            leaf_max=leaf_max, leaf_min=leaf_min, target=target,
            lift=lift, floor=floor, min_grad=min_grad, max_depth=max_depth,
            nodes=nodes,
        )
        nodes[slug]["category_name"] = cat
        for leaf_id, local_indices in leaves:
            clusters[leaf_id] = sorted(pids[i] for i in local_indices)

    # Sort clusters dict
    for k in clusters:
        clusters[k].sort()
    clusters = {k: clusters[k] for k in sorted(clusters.keys())}

    (intermediate_dir() / "05_clusters.json").write_text(
        json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (intermediate_dir() / "05_hierarchy.json").write_text(
        json.dumps(nodes, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Neighbors: post-level cosine top-N, strict masked.
    is_strict = {pid for pid in post_ids if post_cat.get(pid, "") in strict_categories}
    neighbors: dict[str, list[dict]] = {}
    if len(post_ids) >= 2:
        sim = cosine_similarity(arr)
        np.fill_diagonal(sim, -1.0)
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

    # === Analysis outputs (for report graphs) ===
    _write_stats_and_timeseries(
        clusters=clusters, nodes=nodes, by_cat=by_cat,
        post_year=post_year, post_era=post_era,
        cat_slug=cat_slug, strict_categories=strict_categories,
        params={
            "leaf_max": leaf_max, "leaf_min": leaf_min, "target": target,
            "lift": lift, "floor": floor, "min_grad": min_grad,
            "max_depth": max_depth,
        },
    )

    # Console summary
    leaf_count = sum(1 for n in nodes.values() if n.get("is_leaf"))
    leaf_sizes = [n["size"] for n in nodes.values() if n.get("is_leaf") and not n.get("is_strict")]
    avg_n = sum(len(v) for v in neighbors.values()) / max(1, len(neighbors))
    print(
        f"[cluster] {leaf_count} leaves across {len(by_cat)} categories "
        f"(non-strict mean={np.mean(leaf_sizes):.0f}, max={max(leaf_sizes) if leaf_sizes else 0}), "
        f"avg_neighbors={avg_n:.1f}"
    )

    # Per-category summary
    for cat, pids in sorted(by_cat.items()):
        slug = cat_slug.get(cat) or cat
        cat_leaves = [
            n for nid, n in nodes.items()
            if n.get("is_leaf") and nid.split("/")[0] == slug
        ]
        sizes = [n["size"] for n in cat_leaves]
        depths = [n["depth"] for n in cat_leaves]
        leftover = sum(1 for n in cat_leaves if n.get("is_leftover"))
        if cat in strict_categories:
            print(f"  {slug}: posts={len(pids)} (strict, unclustered)")
        else:
            print(
                f"  {slug}: posts={len(pids)} leaves={len(cat_leaves)} "
                f"depth={min(depths)}-{max(depths)} mean_size={np.mean(sizes):.0f} "
                f"leftover={leftover}"
            )

    return {"leaves": leaf_count}


def _write_stats_and_timeseries(
    *, clusters, nodes, by_cat, post_year, post_era,
    cat_slug, strict_categories, params,
):
    """Write 05_cluster_stats.json and 05_time_series.json for later report graphs."""
    # Per-category stats
    cat_stats = {}
    for cat, pids in by_cat.items():
        slug = cat_slug.get(cat) or cat
        cat_leaves = [
            (nid, n) for nid, n in nodes.items()
            if n.get("is_leaf") and nid.split("/")[0] == slug
        ]
        sizes = [n["size"] for _, n in cat_leaves]
        depths = [n["depth"] for _, n in cat_leaves]
        leftover_count = sum(1 for _, n in cat_leaves if n.get("is_leftover"))
        tight_leaves = sum(1 for _, n in cat_leaves if n.get("mean_cohesion", 0) >= 0.45)
        cat_stats[slug] = {
            "name": cat,
            "is_strict": cat in strict_categories,
            "posts": len(pids),
            "leaves": len(cat_leaves),
            "leftover_leaves": leftover_count,
            "tight_leaves": tight_leaves,
            "depth_min": min(depths) if depths else 0,
            "depth_max": max(depths) if depths else 0,
            "depth_mean": float(np.mean(depths)) if depths else 0.0,
            "leaf_size_min": min(sizes) if sizes else 0,
            "leaf_size_max": max(sizes) if sizes else 0,
            "leaf_size_mean": float(np.mean(sizes)) if sizes else 0.0,
        }

    # Per-leaf stats with era distribution
    leaves_stats = []
    for cid, members in clusters.items():
        node = nodes.get(cid, {})
        cat_slug_part = cid.split("/")[0]
        era_dist = defaultdict(int)
        year_dist = defaultdict(int)
        for pid in members:
            era_dist[post_era.get(pid, "unknown")] += 1
            y = post_year.get(pid)
            if y:
                year_dist[str(y)] += 1
        leaves_stats.append({
            "id": cid,
            "category_slug": cat_slug_part,
            "size": len(members),
            "mean_cohesion": node.get("mean_cohesion", 0.0),
            "depth": node.get("depth", 0),
            "is_leftover": node.get("is_leftover", False),
            "is_strict": node.get("is_strict", False),
            "era_distribution": dict(era_dist),
            "year_distribution": dict(year_dist),
        })

    # Global summary
    all_depths = [n["depth"] for n in nodes.values() if n.get("is_leaf")]
    depth_dist = defaultdict(int)
    for d in all_depths:
        depth_dist[d] += 1
    all_cohesions = [n.get("mean_cohesion", 0.0) for n in nodes.values() if n.get("is_leaf") and not n.get("is_strict")]
    coh_bins = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    coh_hist = {}
    for i in range(len(coh_bins) - 1):
        lo, hi = coh_bins[i], coh_bins[i + 1]
        coh_hist[f"{lo:.1f}-{hi:.1f}"] = sum(1 for c in all_cohesions if lo <= c < hi)
    coh_hist[f">={coh_bins[-1]:.1f}"] = sum(1 for c in all_cohesions if c >= coh_bins[-1])

    global_stats = {
        "params": params,
        "total_posts": sum(len(v) for v in by_cat.values()),
        "total_leaves": sum(1 for n in nodes.values() if n.get("is_leaf")),
        "total_categories": len(by_cat),
        "depth_distribution": {str(k): v for k, v in sorted(depth_dist.items())},
        "leaf_cohesion_histogram": coh_hist,
    }

    (intermediate_dir() / "05_cluster_stats.json").write_text(
        json.dumps({
            "global": global_stats,
            "categories": cat_stats,
            "leaves": leaves_stats,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Time series: by_category and by_leaf (year × count)
    by_cat_time: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_leaf_time: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for cid, members in clusters.items():
        cat_slug_part = cid.split("/")[0]
        # Map slug back to category name
        cat_name = next((c for c, s in cat_slug.items() if s == cat_slug_part), cat_slug_part)
        for pid in members:
            y = post_year.get(pid)
            if not y:
                continue
            ys = str(y)
            by_cat_time[cat_name][ys] += 1
            by_leaf_time[cid][ys] += 1

    # Sort years inside each map
    by_cat_time_sorted = {
        cat: {y: by_cat_time[cat][y] for y in sorted(by_cat_time[cat].keys())}
        for cat in sorted(by_cat_time.keys())
    }
    by_leaf_time_sorted = {
        leaf: {y: by_leaf_time[leaf][y] for y in sorted(by_leaf_time[leaf].keys())}
        for leaf in sorted(by_leaf_time.keys())
    }

    (intermediate_dir() / "05_time_series.json").write_text(
        json.dumps({
            "by_category": by_cat_time_sorted,
            "by_leaf": by_leaf_time_sorted,
        }, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
