"""2026-05-01: 창업·경영 ↔ 조직·리더십 boundary 재정의 (옵션 B).

새 boundary:
- 창업·경영 = 스타트업 일반론·시장 분석·사업 의사결정·ART Lab 사업 전략 (사람 빼고)
- 조직·리더십 = 팀·조직·리더십·직장생활·인력 영입·면담 (회사원·창업가 모두 포괄)

Leaf 이동 (창업·경영 537 posts 중 136 redirect):
- /1 (운동·풋살, 21p)         → 취미·문화
- /3 (유튜브·별풍·스타, 13p)   → 테크 커뮤니티·산업 동향
- /10 (우버·카카오 규제, 23p)  → 테크 커뮤니티·산업 동향
- /17 (슬럼프·번아웃, 19p)    → 일상 감정
- /19 (회사원 근무·일상, 25p) → 일상 사건
- /22 (ART Lab 멤버 모집, 35p) → 조직·리더십  (boundary 재정의 핵심)

Leaf 이동 (조직·리더십 283 posts 중 34 redirect):
- /4 (박종우 교수님·학부, 34p) → 학계

Run once. _classify_overrides.json 만 수정. 클러스터링은 다른 카테고리 조정 끝난 뒤 한꺼번에.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
OVR_PATH = VAULT / "_classify_overrides.json"
CLUSTERS_PATH = VAULT / "_intermediate" / "05_clusters.json"


MOVES: list[tuple[str, str]] = [
    ("cangeob-gyeongyeong/1",  "취미·문화"),
    ("cangeob-gyeongyeong/3",  "테크 커뮤니티·산업 동향"),
    ("cangeob-gyeongyeong/10", "테크 커뮤니티·산업 동향"),
    ("cangeob-gyeongyeong/17", "일상 감정"),
    ("cangeob-gyeongyeong/19", "일상 사건"),
    ("cangeob-gyeongyeong/22", "조직·리더십"),
    ("jojig-rideosib/4",       "학계"),
]


def main() -> None:
    data = json.loads(OVR_PATH.read_text(encoding="utf-8"))
    clusters = json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))
    overrides: dict[str, str] = data["overrides"]

    per_target: Counter[str] = Counter()
    per_leaf: list[tuple[str, str, int]] = []

    for leaf, target in MOVES:
        pids = clusters.get(leaf, [])
        if not pids:
            print(f"[WARN] leaf {leaf} not found in 05_clusters.json — skipping")
            continue
        for pid in pids:
            overrides[pid] = target
        per_target[target] += len(pids)
        per_leaf.append((leaf, target, len(pids)))

    total = sum(n for _, _, n in per_leaf)

    rule_entry = {
        "date": "2026-05-01",
        "action": "창업·경영 ↔ 조직·리더십 boundary 재정의 (옵션 B)",
        "detail": (
            "새 boundary — 창업·경영=사업·시장·전략 (사람 빼고), "
            "조직·리더십=팀·문화·리더십·직장생활·인력. "
            f"창업·경영 6 leaves ({sum(n for l,_,n in per_leaf if l.startswith('cangeob-gyeongyeong/'))} posts) 다른 카테고리로 redirect: "
            "/1→취미·문화, /3+/10→테크 커뮤니티·산업 동향, /17→일상 감정, /19→일상 사건, /22→조직·리더십. "
            "조직·리더십 1 leaf (/4 박종우 교수님·학부, "
            f"{sum(n for l,_,n in per_leaf if l.startswith('jojig-rideosib/'))} posts) → 학계. "
            f"총 {total} posts redirected. "
            "clustering 은 다른 카테고리 세부조정 끝난 뒤 한꺼번에."
        ),
    }
    data.setdefault("_rules", []).append(rule_entry)

    note = (
        f"\n2026-05-01: 창업·경영 ↔ 조직·리더십 boundary 재정의 (옵션 B). "
        f"총 {total} posts redirected: "
        + ", ".join(f"{leaf}→{tgt}({n})" for leaf, tgt, n in per_leaf)
        + "."
    )
    data["_comment"] = (data.get("_comment", "") + note).strip()
    data["_generated"] = "2026-05-01"

    OVR_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[cangeob_jojig] {len(per_leaf)} leaves moved, {total} posts redirected\n")
    for leaf, tgt, n in per_leaf:
        print(f"  {leaf:36s} → {tgt:24s} {n:4d} posts")
    print()
    print("[cangeob_jojig] per-target counts:")
    for tgt, n in per_target.most_common():
        print(f"  +{n:4d}  {tgt}")


if __name__ == "__main__":
    main()
