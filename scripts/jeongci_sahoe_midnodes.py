"""정치 + 사회 mid-node 정비 (B) + 사회 경계 재분류 (C).

C: 사회 leaf 중 결이 다른 leaf 2개를 다른 카테고리로 이동
   - sahoe/18 (32p, 한글·외국어·표준어 학습) → 학습·메타인지
   - sahoe/1  (19p, IT자본주의·구글-오라클 macro) → 테크 커뮤니티·산업 동향

B: 사회 + 정치 카테고리 안에 mid-node 도입 (leaf_groups 사용)
   사회 (잔여 13 leaves) → 7 mid-node:
     media (9,8,26) / gender (25) / labor-class (3,22) / rights-law (16,10) /
     memorial (21) / violence-protest (28,leftover) / education (13)
     unassigned: sahoe/2 (25p, 이두희+학계 mixed)
   정치 (20 children) → 4 mid-node:
     election-party (1,3,10,14) / power-critique (5,6,8,11,15,19) /
     ideology-history (9,13,20) / meta-attitude (2,7,12,16,17,18)
     unassigned: jeongci/leftover (6p)

Updates _classify_overrides.json (overrides + leaf_groups).
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
INT = VAULT / "_intermediate"
OVERRIDE_PATH = VAULT / "_classify_overrides.json"

# === C: 사회 leaf → 다른 카테고리 ===
RECLASS_LEAVES: dict[str, str] = {
    "sahoe/18": "학습·메타인지",
    "sahoe/1":  "테크 커뮤니티·산업 동향",
}

# === B: leaf_groups (post-id 기반 mid-node) ===
# 각 직속 자식(leaf 또는 mid-node)을 mid-name 으로 묶음
MIDNODE_PLAN: dict[str, dict[str, list[str]]] = {
    "sahoe": {
        "media":            ["sahoe/9", "sahoe/8", "sahoe/26"],
        "gender":           ["sahoe/25"],
        "labor-class":      ["sahoe/3", "sahoe/22"],
        "rights-law":       ["sahoe/16", "sahoe/10"],
        "memorial":         ["sahoe/21"],
        "violence-protest": ["sahoe/28", "sahoe/leftover"],
        "education":        ["sahoe/13"],
        # sahoe/2 (25p, 이두희+학계 mixed) → unassigned, root 직속 유지
    },
    "jeongci": {
        "election-party":   ["jeongci/1", "jeongci/3", "jeongci/10", "jeongci/14"],
        "power-critique":   ["jeongci/5", "jeongci/6", "jeongci/8", "jeongci/11", "jeongci/15", "jeongci/19"],
        "ideology-history": ["jeongci/9", "jeongci/13", "jeongci/20"],
        "meta-attitude":    ["jeongci/2", "jeongci/7", "jeongci/12", "jeongci/16", "jeongci/17", "jeongci/18"],
        # jeongci/leftover (6p) → unassigned
    },
}


def _expand(prefix: str, clusters: dict) -> list[str]:
    """Return all leaf cluster IDs under prefix (inclusive)."""
    out: list[str] = []
    if prefix in clusters:
        out.append(prefix)
    sep = prefix + "/"
    for cid in clusters:
        if cid.startswith(sep) and cid not in out:
            out.append(cid)
    return out


def main() -> None:
    clusters = json.loads((INT / "05_clusters.json").read_text(encoding="utf-8"))
    doc = json.loads(OVERRIDE_PATH.read_text(encoding="utf-8"))

    bak = OVERRIDE_PATH.with_name(
        OVERRIDE_PATH.name + ".bak-2026-05-01-jeongci-sahoe-midnodes"
    )
    shutil.copy(OVERRIDE_PATH, bak)
    print(f"[backup] {bak.name}")

    # --- C: post-level overrides ---
    overrides = doc.setdefault("overrides", {})
    for prefix, new_cat in RECLASS_LEAVES.items():
        leaves = _expand(prefix, clusters)
        n = 0
        for leaf in leaves:
            for pid in clusters.get(leaf, []):
                overrides[pid] = new_cat
                n += 1
        print(f"[C] {prefix} ({n} posts) → {new_cat}  (covers {leaves})")

    # --- B: leaf_groups ---
    lg = doc.setdefault("leaf_groups", {})
    for cat, groups in MIDNODE_PLAN.items():
        lg[cat] = {}
        total_in_cat = 0
        for mid, prefixes in groups.items():
            pids: list[str] = []
            covered_leaves: list[str] = []
            for pre in prefixes:
                leaves = _expand(pre, clusters)
                covered_leaves.extend(leaves)
                for leaf in leaves:
                    pids.extend(clusters.get(leaf, []))
            pids = sorted(set(pids))
            lg[cat][mid] = pids
            total_in_cat += len(pids)
            print(f"[B] {cat}/{mid}: {len(prefixes)} child(ren) {covered_leaves} → {len(pids)} posts")
        print(f"[B] {cat} total in groups: {total_in_cat}")

    note = (
        "2026-05-01 정치+사회 mid-node + 사회 경계 재분류: "
        "sahoe/18 (32, 한글·외국어 학습) → 학습·메타인지; "
        "sahoe/1 (19, IT 자본주의 macro) → 테크 커뮤니티·산업 동향. "
        "leaf_groups[sahoe] = 7 mid-node "
        "(media/gender/labor-class/rights-law/memorial/violence-protest/education); "
        "leaf_groups[jeongci] = 4 mid-node "
        "(election-party/power-critique/ideology-history/meta-attitude). "
        "Unassigned (root 직속): sahoe/2 (25, 이두희+학계 mixed), jeongci/leftover (6)."
    )
    doc.setdefault("_rules", []).append({"date": "2026-05-01", "action": note})
    doc["_generated"] = datetime.utcnow().isoformat() + "Z"

    OVERRIDE_PATH.write_text(
        json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[done] overrides total={len(overrides)}")
    print(f"[saved] {OVERRIDE_PATH}")


if __name__ == "__main__":
    main()
