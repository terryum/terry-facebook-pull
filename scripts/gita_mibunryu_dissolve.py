"""2026-05-01: 기타·미분류 카테고리 폐지 — 53 posts 를 nearest-leaf 카테고리로 분산.

검색·생성 차원에서 의미 X. 각 gita post 를 embedding 공간에서 가장 가까운 non-gita
leaf 의 카테고리로 post-level override. 결정적·LLM 없음.

Run once. _classify_overrides.json 만 수정. _taxonomy.md 에서 ## 기타·미분류 섹션은
별도로 제거.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
INT = VAULT / "_intermediate"
OVR_PATH = VAULT / "_classify_overrides.json"
TAX_PATH = VAULT / "_taxonomy.md"

GITA_SLUG = "gita-mibunryu"


def _load_taxonomy_slug_to_name() -> dict[str, str]:
    """Parse _taxonomy.md for slug → name mapping. Reuses simple regex; no deps."""
    import re
    from slugify import slugify

    out: dict[str, str] = {}
    in_categories = False
    for line in TAX_PATH.read_text(encoding="utf-8").splitlines():
        if line.strip() == "# Categories":
            in_categories = True
            continue
        if not in_categories:
            continue
        m = re.match(r"^##\s+(.+?)\s*$", line)
        if not m:
            continue
        name = re.sub(r"\s*\[(SENSITIVE|STRICT)\]\s*", "", m.group(1)).strip()
        slug = slugify(name, allow_unicode=False, max_length=40) or name
        out[slug] = name
    return out


def main() -> None:
    clusters = json.loads((INT / "05_clusters.json").read_text())
    embeddings = np.load(INT / "04_embeddings.npy")
    post_ids = json.loads((INT / "04_post_ids.json").read_text())
    pid_to_row = {pid: i for i, pid in enumerate(post_ids)}
    slug_to_name = _load_taxonomy_slug_to_name()

    # Build non-gita leaf centroids
    leaf_centroids: dict[str, np.ndarray] = {}
    for leaf_id, member_pids in clusters.items():
        if leaf_id.startswith(GITA_SLUG):
            continue
        rows = [pid_to_row[p] for p in member_pids if p in pid_to_row]
        if not rows:
            continue
        c = embeddings[rows].mean(axis=0)
        n = float(np.linalg.norm(c))
        leaf_centroids[leaf_id] = c / n if n > 0 else c
    leaf_ids = list(leaf_centroids.keys())
    centroid_matrix = np.stack([leaf_centroids[lid] for lid in leaf_ids])  # (L, D)

    # Collect gita posts
    gita_pids: list[str] = []
    for leaf_id, member_pids in clusters.items():
        if leaf_id.startswith(GITA_SLUG):
            gita_pids.extend(member_pids)

    # For each gita post, find nearest leaf
    moves: list[tuple[str, str, str, float]] = []  # (pid, leaf, cat, score)
    for pid in gita_pids:
        if pid not in pid_to_row:
            print(f"  [skip] {pid} not in embeddings")
            continue
        v = embeddings[pid_to_row[pid]]
        scores = centroid_matrix @ v  # cosine (vectors L2-normalized)
        idx = int(scores.argmax())
        nearest_leaf = leaf_ids[idx]
        cat_slug = nearest_leaf.split("/", 1)[0]
        cat_name = slug_to_name.get(cat_slug, cat_slug)
        moves.append((pid, nearest_leaf, cat_name, float(scores[idx])))

    # Update _classify_overrides.json
    data = json.loads(OVR_PATH.read_text(encoding="utf-8"))
    overrides: dict[str, str] = data["overrides"]
    for pid, _, cat_name, _ in moves:
        overrides[pid] = cat_name

    # Per-target distribution
    cat_dist = Counter(cat for _, _, cat, _ in moves)
    leaf_dist = Counter(leaf for _, leaf, _, _ in moves)

    rule_entry = {
        "date": "2026-05-01",
        "action": "기타·미분류 폐지 — 53 posts → nearest-leaf 카테고리 (embedding cosine)",
        "detail": (
            f"각 gita-mibunryu post 의 embedding vs non-gita leaf centroid cosine 최대값으로 "
            f"카테고리 재할당. {len(moves)} posts redistributed. "
            f"분포: {', '.join(f'{c}({n})' for c, n in cat_dist.most_common())}. "
            f"_taxonomy.md 의 ## 기타·미분류 섹션 별도 제거. "
            f"clustering 재실행 시 자동 반영."
        ),
    }
    data.setdefault("_rules", []).append(rule_entry)

    note = (
        f"\n2026-05-01: 기타·미분류 폐지 — {len(moves)} posts → nearest-leaf 카테고리. "
        f"분포: {', '.join(f'{c}({n})' for c, n in cat_dist.most_common(5))} ..."
    )
    data["_comment"] = (data.get("_comment", "") + note).strip()
    data["_generated"] = "2026-05-01"

    OVR_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[gita_dissolve] {len(moves)} posts redistributed (mean cosine={np.mean([s for _,_,_,s in moves]):.3f})\n")
    print("[gita_dissolve] per-category:")
    for cat, n in cat_dist.most_common():
        print(f"  +{n:3d}  {cat}")
    print()
    print("[gita_dissolve] top destination leaves:")
    for leaf, n in leaf_dist.most_common(8):
        print(f"  {n:3d}  {leaf}")


if __name__ == "__main__":
    main()
