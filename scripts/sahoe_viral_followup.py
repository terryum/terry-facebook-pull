"""2026-05-01: 사회 viral 3 posts → media mid-node 보정 (deferred follow-up #1).

leaf_groups majority-vote 메커니즘은 leaf 단위 → 14p violence-protest 통째 D1-α 흡수
때 leaf 안 viral pid 가 같이 rights-law/labor-class 로 끌려갔음. 재클러스터링
후 잔여 viral pid 3개를 leaf_groups[sahoe][media] 에 직접 추가해 다음 cluster
에서 majority-vote 가 media 쪽으로 끌리도록 한다.

Run once. _classify_overrides.json 만 수정. 1 viral 14p 중 1개 (유튜브 키워드)
는 이미 sahoe/media/6 → OK; 나머지 3개 보정.
"""

from __future__ import annotations

import json
from pathlib import Path

OVR_PATH = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook/_classify_overrides.json")

VIRAL_PIDS = [
    "1491928964-21d52c7415",  # #boycottunited (united airlines passenger)
    "1541894501-48cb5627f3",  # BTS 원폭 티셔츠
    "1550064727-e983093c1f",  # 블랙핑크 미국
]


def main() -> None:
    data = json.loads(OVR_PATH.read_text(encoding="utf-8"))

    media = data["leaf_groups"]["sahoe"]["media"]
    before = len(media)
    added = 0
    for pid in VIRAL_PIDS:
        if pid not in media:
            media.append(pid)
            added += 1
    media.sort()
    after = len(media)

    rule_entry = {
        "date": "2026-05-01",
        "action": "사회 viral 3 posts → media mid-node 보정 (deferred follow-up #1)",
        "detail": (
            f"leaf_groups[sahoe][media] 에 {added} pid 추가 ({before} → {after}). "
            "재클러스터링 시 majority-vote 가 #boycottunited / BTS 원폭 / 블랙핑크 "
            "leaf 를 media mid-node 로 끌어당기도록. 1 viral (유튜브 키워드) 은 이미 "
            "media/6 에 있어 보정 불요."
        ),
    }
    data.setdefault("_rules", []).append(rule_entry)

    note = (
        f"\n2026-05-01: 사회 viral 3 pid → leaf_groups[sahoe][media] 추가 "
        f"({before}→{after}). deferred follow-up #1 처리."
    )
    data["_comment"] = (data.get("_comment", "") + note).strip()
    data["_generated"] = "2026-05-01"

    OVR_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"[sahoe_viral_followup] leaf_groups[sahoe][media]: {before} → {after} (+{added})")


if __name__ == "__main__":
    main()
