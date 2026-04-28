import json

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics.pairwise import cosine_similarity

from .paths import intermediate_dir


def run(
    min_cluster_size: int = 4,
    top_n: int = 5,
    neighbor_threshold: float = 0.55,
) -> dict:
    embed_path = intermediate_dir() / "04_embeddings.npy"
    ids_path = intermediate_dir() / "04_post_ids.json"
    if not embed_path.exists() or not ids_path.exists():
        raise FileNotFoundError("Run `fbpull embed` first")

    arr: np.ndarray = np.load(embed_path)
    post_ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))

    # Cluster
    if len(post_ids) < min_cluster_size:
        labels = [-1] * len(post_ids)
    else:
        actual_min = max(2, min(min_cluster_size, max(2, len(post_ids) // 4)))
        hdb = HDBSCAN(min_cluster_size=actual_min, metric="euclidean")
        labels = hdb.fit_predict(arr).tolist()

    clusters: dict[str, list[str]] = {}
    for pid, lbl in zip(post_ids, labels):
        clusters.setdefault(str(int(lbl)), []).append(pid)
    for k in clusters:
        clusters[k].sort()
    clusters = {k: clusters[k] for k in sorted(clusters.keys(), key=lambda s: int(s))}

    (intermediate_dir() / "05_clusters.json").write_text(
        json.dumps(clusters, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Neighbors
    neighbors: dict[str, list[dict]] = {}
    if len(post_ids) >= 2:
        sim = cosine_similarity(arr)
        np.fill_diagonal(sim, -1.0)
        for i, pid in enumerate(post_ids):
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

    n_clusters = sum(1 for k in clusters if int(k) >= 0)
    n_noise = len(clusters.get("-1", []))
    avg_n = sum(len(v) for v in neighbors.values()) / max(1, len(neighbors))
    print(f"[cluster] clusters={n_clusters} noise={n_noise} avg_neighbors={avg_n:.1f}")
    return {"clusters": n_clusters, "noise": n_noise}
