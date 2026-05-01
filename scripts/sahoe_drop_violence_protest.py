"""사회 violence-protest mid-node 폐지 (D1-α).

현재 sahoe/violence-protest/leftover 의 14p (war 10 + viral 4) 를
rights-law mid-node 로 통째 흡수. leaf 단위 majority vote 라 split 못 함.

follow-up 후보 (post-level override 로 media 로 옮기면 정확):
  - #boycottunited (2017)
  - BTS 원폭 티셔츠 (2018)
  - 블랙핑크 미국 (2019)
  - 유튜브 키워드 (2019)
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
INT = VAULT / "_intermediate"
OVERRIDE = VAULT / "_classify_overrides.json"


def main() -> None:
    clusters = json.loads((INT / "05_clusters.json").read_text(encoding="utf-8"))
    doc = json.loads(OVERRIDE.read_text(encoding="utf-8"))

    bak = OVERRIDE.with_name(OVERRIDE.name + ".bak-2026-05-01-violence-protest-drop")
    shutil.copy(OVERRIDE, bak)
    print(f"[backup] {bak.name}")

    vp_leaf = "sahoe/violence-protest/leftover"
    vp_pids = list(clusters.get(vp_leaf, []))
    if not vp_pids:
        print(f"[warn] {vp_leaf} not found or empty — nothing to do")
        return
    print(f"[scan] {vp_leaf}: {len(vp_pids)} posts")

    sahoe_groups = doc["leaf_groups"]["sahoe"]
    if "violence-protest" in sahoe_groups:
        del sahoe_groups["violence-protest"]
        print("[edit] removed leaf_groups[sahoe][violence-protest]")
    else:
        print("[warn] violence-protest already absent from leaf_groups[sahoe]")

    rl = set(sahoe_groups.get("rights-law", []))
    rl.update(vp_pids)
    sahoe_groups["rights-law"] = sorted(rl)
    print(f"[edit] rights-law: +{len(vp_pids)} → {len(rl)} posts")

    doc.setdefault("_rules", []).append({
        "date": "2026-05-01",
        "action": "violence-protest mid-node 폐지 (D1-α)",
        "detail": (
            f"sahoe/violence-protest/leftover 14p (war 10 + viral 4) → rights-law 통째 흡수. "
            f"leaf 단위 majority-vote 제약. follow-up: 4 viral pids "
            f"(#boycottunited, BTS 원폭, 블랙핑크, 유튜브 키워드) → media 로 "
            f"post-level override 검토."
        ),
    })
    doc["_generated"] = datetime.utcnow().isoformat() + "Z"

    OVERRIDE.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] {OVERRIDE}")


if __name__ == "__main__":
    main()
