"""Step 5e: audit.md 의 사용자 답변을 읽어 post_importance 를 보강.

사용자가 `_reports/<date>/labeling/audit.md` 의 체크박스에 답변한 결과를 파싱해서
`_classify_overrides.json` 의 `post_importance` 필드를 갱신.

Audit.md 형식:
- `- [x] LEAF=<leaf_id> core` / `topic` / `noise`  (체크된 줄만 적용)
- `- [x] POST=<post_id> core` / `topic` / `noise`

체크 안 된 줄 (`- [ ] ...`) 은 LLM 판정 그대로 유지.

Usage:
    python scripts/label_audit_apply.py [audit.md path]

audit.md 경로 지정 안 하면 가장 최근 _reports/*/labeling/audit.md 자동 선택.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
INT = VAULT / "_intermediate"
OVR_PATH = VAULT / "_classify_overrides.json"

VALID_TIERS = {"core", "topic", "noise"}

LEAF_LINE = re.compile(r"-\s*\[(x|X)\]\s+LEAF=(\S+)\s+(core|topic|noise)\b")
POST_LINE = re.compile(r"-\s*\[(x|X)\]\s+POST=(\S+)\s+(core|topic|noise)\b")


def _find_latest_audit() -> Path:
    reports = VAULT / "_reports"
    candidates = sorted(reports.glob("*/labeling/audit.md"), reverse=True)
    if not candidates:
        raise FileNotFoundError(f"no audit.md found under {reports}")
    return candidates[0]


def main() -> None:
    if len(sys.argv) > 1:
        audit_path = Path(sys.argv[1])
    else:
        audit_path = _find_latest_audit()
    print(f"[5e] reading {audit_path}")

    text = audit_path.read_text(encoding="utf-8")
    leaf_overrides: dict[str, str] = {}
    post_overrides: dict[str, str] = {}

    for line in text.splitlines():
        m = LEAF_LINE.search(line)
        if m:
            leaf_id = m.group(2)
            tier = m.group(3)
            leaf_overrides[leaf_id] = tier
            continue
        m = POST_LINE.search(line)
        if m:
            post_id = m.group(2)
            tier = m.group(3)
            post_overrides[post_id] = tier

    print(f"[5e] checked entries: leaf={len(leaf_overrides)}, post={len(post_overrides)}")
    if not leaf_overrides and not post_overrides:
        print("[5e] no checked entries — nothing to apply.")
        return

    ovr = json.loads(OVR_PATH.read_text(encoding="utf-8"))
    post_importance = ovr.setdefault("post_importance", {})
    clusters = json.loads((INT / "05_clusters.json").read_text(encoding="utf-8"))
    labels_5a = json.loads((INT / "leaf_label_5a.json").read_text(encoding="utf-8"))

    n_post_changed = 0
    n_leaf_changed = 0

    # Apply leaf-level overrides — set tier for ALL members of that leaf
    for lid, tier in leaf_overrides.items():
        members = clusters.get(lid, [])
        scope = labels_5a.get(lid, {}).get("topic_scope", []) if tier == "topic" else []
        for pid in members:
            cur = post_importance.get(pid, {})
            if cur.get("tier") != tier or cur.get("topic_scope") != scope:
                post_importance[pid] = {"tier": tier, "topic_scope": list(scope)}
                n_post_changed += 1
        n_leaf_changed += 1

    # Apply post-level overrides — only the specified post
    for pid, tier in post_overrides.items():
        cur = post_importance.get(pid, {})
        if cur.get("tier") != tier:
            # Preserve existing scope when tier changes (best-effort)
            scope = cur.get("topic_scope", []) if tier == "topic" else []
            post_importance[pid] = {"tier": tier, "topic_scope": list(scope)}
            n_post_changed += 1

    # Audit log
    rule_entry = {
        "date": Path(__file__).stat().st_mtime_ns,  # marker; replaced below
    }
    from datetime import datetime
    rule_entry["date"] = datetime.now().strftime("%Y-%m-%d")
    rule_entry["action"] = "Step 5e — user audit answers applied to post_importance"
    rule_entry["detail"] = (
        f"audit.md = {audit_path.name}. leaf overrides={n_leaf_changed} ({sum(len(clusters.get(l, [])) for l in leaf_overrides)} posts), "
        f"post overrides={len(post_overrides)}. Total post_importance changes={n_post_changed}."
    )
    ovr.setdefault("_rules", []).append(rule_entry)

    OVR_PATH.write_text(json.dumps(ovr, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[5e] applied: {n_leaf_changed} leaf overrides → {n_post_changed} post tier changes "
          f"(+{len(post_overrides)} direct post overrides). Saved to {OVR_PATH.name}.")


if __name__ == "__main__":
    main()
