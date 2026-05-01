"""2026-05-01: Split 학습·학계 back into 학계 + 학습·메타인지 (3-way taxonomy
for research/study area). Plus: move 4 yeongu-hagsul leaves that drifted into
researcher-life territory back to 학계, and the noisy /35 leaf to 일상 사건.

Run once. Edits _classify_overrides.json in place.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
OVR_PATH = VAULT / "_classify_overrides.json"
CLUSTERS_PATH = VAULT / "_intermediate" / "05_clusters.json"


YEONGU_TO_HAGGYE_LEAVES = [
    "yeongu-hagsul/26",        # Graham Taylor 방문, 딥러닝 거성 패널, 테뉴어
    "yeongu-hagsul/28",        # ICRA 심사·h-index·논문 motivation 메타
    "yeongu-hagsul/37",        # 창의적 로봇 CEO·진로·엉터리 개발자
    "yeongu-hagsul/leftover/2",  # 연구 중단·렉쳐 자료
]
YEONGU_TO_ILSANG_LEAVES = [
    "yeongu-hagsul/35",        # 호칭·허세·짧은 일상 노이즈
]


def main() -> None:
    data = json.loads(OVR_PATH.read_text(encoding="utf-8"))
    clusters = json.loads(CLUSTERS_PATH.read_text(encoding="utf-8"))

    overrides: dict[str, str] = data["overrides"]

    # Step 1: bulk rename "학계·진로" → "학계" inside overrides values
    rename_count = 0
    for pid, cat in list(overrides.items()):
        if cat == "학계·진로":
            overrides[pid] = "학계"
            rename_count += 1

    # Step 2: drop the two collapsing renames so 학습·메타인지 / 학계 stay separate
    data["category_renames"] = {}

    # Step 3: per-post overrides for the 4 yeongu leaves drifting into 학계
    moved_to_haggye = 0
    for leaf in YEONGU_TO_HAGGYE_LEAVES:
        for pid in clusters.get(leaf, []):
            overrides[pid] = "학계"
            moved_to_haggye += 1

    # Step 4: yeongu /35 (호칭·허세 noise) → 일상 사건
    moved_to_ilsang = 0
    for leaf in YEONGU_TO_ILSANG_LEAVES:
        for pid in clusters.get(leaf, []):
            overrides[pid] = "일상 사건"
            moved_to_ilsang += 1

    # Step 5: append a _rules entry documenting this change
    rule_entry = {
        "date": "2026-05-01",
        "action": "학습·학계 → 학계 + 학습·메타인지 분리 (3-way for 연구·학계·학습)",
        "detail": (
            f"category_renames 두 항목 제거 (학계·진로/학습·메타인지 → 학습·학계). "
            f"overrides 의 {rename_count} 개 학계·진로 값을 학계로 일괄 변경. "
            f"yeongu-hagsul off-topic leaf 4개 ({len(YEONGU_TO_HAGGYE_LEAVES)} leaves, "
            f"{moved_to_haggye} posts) → 학계 (Graham Taylor 방문, ICRA 심사·h-index, "
            f"창의적 로봇 CEO 진로, 연구 중단·렉쳐 자료). "
            f"yeongu-hagsul/35 (호칭·허세 노이즈, {moved_to_ilsang} posts) → 일상 사건. "
            f"taxonomy.md 에서 ## 학습·학계 섹션을 ## 학계 + ## 학습·메타인지 두 섹션으로 분리. "
            f"clustering 은 다른 카테고리 세부조정 끝난 뒤 한꺼번에."
        ),
    }
    data.setdefault("_rules", []).append(rule_entry)

    # Step 6: append note to _comment for human readers
    note = (
        "\n2026-05-01: 학습·학계 분해 — 학계 + 학습·메타인지 두 카테고리로 다시 분리. "
        f"category_renames 비움. overrides 학계·진로 → 학계 일괄 ({rename_count}). "
        f"yeongu-hagsul /26,/28,/37,/leftover/2 → 학계 ({moved_to_haggye}); /35 → 일상 사건 ({moved_to_ilsang})."
    )
    data["_comment"] = (data.get("_comment", "") + note).strip()
    data["_generated"] = "2026-05-01"

    OVR_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[haggye_split] renamed 학계·진로 → 학계 in overrides: {rename_count} posts")
    print(f"[haggye_split] yeongu → 학계 (4 leaves): {moved_to_haggye} posts")
    print(f"[haggye_split] yeongu → 일상 사건 (1 leaf): {moved_to_ilsang} posts")
    print(f"[haggye_split] category_renames cleared (was 2 entries)")

    # Summary of effective override targets after change
    ctr = Counter(overrides.values())
    print("\n[haggye_split] effective override targets after change:")
    for c, n in ctr.most_common():
        print(f"  {n:5d}  {c}")


if __name__ == "__main__":
    main()
