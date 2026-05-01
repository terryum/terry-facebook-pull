"""Step 5c+5d: 3-tier (core/topic/noise) + multi-scope post_importance + stats/audit.

Inputs:
- `_intermediate/leaf_label_5a.json` (per-leaf {tier, topic_scope, confidence, reason})
- `_intermediate/post_label_5b.json` (per-post within 'topic' leaves)

Outputs:
- `_classify_overrides.json` 의 `post_importance` 필드:
  {post_id: {"tier": "core"|"topic"|"noise", "topic_scope": [...]}}
- `_intermediate/leaf_label_threshold.json` — 채택된 threshold + collapsed leaves
- `_reports/<date>/labeling/{stats.md, audit.md}`

Threshold 로직 (3-way collapse for 'topic' leaves drilled in 5b):
- 5b 결과 noise rate ≥ T_NOISE (기본 80%) → leaf 통째 noise
- 5b 결과 core rate ≥ T_CORE (기본 80%) → leaf 통째 core
- 그 외 → mixed (post-level 라벨 보존, 5a 의 topic_scope 를 mixed leaf 의 topic 멤버에 inherit)
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbpull.cli import _bootstrap  # noqa: E402

_bootstrap(no_llm=True)

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
INT = VAULT / "_intermediate"
OVR_PATH = VAULT / "_classify_overrides.json"

T_NOISE = 60   # ≥60% noise → leaf 통째 noise
T_CORE = 60    # ≥60% core → leaf 통째 core

VALID_SCOPES = {
    "personal-family", "personal-life",
    "society-politics", "society-issues",
    "industry-tech", "industry-academic", "industry-management",
}
VALID_TIERS = {"core", "topic", "noise"}

AUDIT_LEAF_NOISE_LO = 40   # noise rate 40-70% → audit leaf
AUDIT_LEAF_NOISE_HI = 70
AUDIT_POST_CONF = 0.55


def main() -> None:
    labels_5a = json.loads((INT / "leaf_label_5a.json").read_text(encoding="utf-8"))
    labels_5b = json.loads((INT / "post_label_5b.json").read_text(encoding="utf-8"))

    # Normalize 5b schema enum violations: Haiku occasionally returned a scope
    # value (e.g. "personal-life") in the tier field. Repair: tier="topic", and
    # add the misplaced scope value into topic_scope.
    n_normalized = 0
    for pid, r in labels_5b.items():
        t = r.get("tier")
        if t in VALID_TIERS:
            continue
        if t in VALID_SCOPES:
            scope = list(r.get("topic_scope") or [])
            if t not in scope:
                scope.append(t)
            r["tier"] = "topic"
            r["topic_scope"] = scope
            n_normalized += 1
        else:
            # Unknown — fall back to topic with empty scope
            r["tier"] = "topic"
            r["topic_scope"] = list(r.get("topic_scope") or [])
            n_normalized += 1
    if n_normalized:
        (INT / "post_label_5b.json").write_text(
            json.dumps(labels_5b, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[5cd] normalized {n_normalized} 5b posts (invalid tier → topic + scope inferred)")

    clusters = json.loads((INT / "05_clusters.json").read_text(encoding="utf-8"))
    posts: dict[str, dict] = {}
    with (INT / "03_classified.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            posts[r["post_id"]] = r

    stats = json.loads((INT / "05_cluster_stats.json").read_text(encoding="utf-8"))
    cohesion_by_leaf = {leaf["id"]: leaf.get("mean_cohesion", 0.0) for leaf in stats.get("leaves", [])}

    from fbpull import taxonomy as taxonomy_mod
    tax = taxonomy_mod.load()
    cats = tax.categories if tax else []
    slug_to_name = {c.slug: c.name for c in cats}

    # Aggregate 5b post labels by leaf
    posts_by_leaf: dict[str, dict[str, dict]] = defaultdict(dict)
    for pid, r in labels_5b.items():
        posts_by_leaf[r["leaf_id"]][pid] = r

    # Compute final per-leaf decision
    leaf_decision: dict[str, str] = {}  # leaf_id → "core" | "topic" | "noise" | "mixed"
    leaf_5b_rates: dict[str, dict] = {}  # leaf_id → {core, topic, noise}
    for lid, r in labels_5a.items():
        if r["tier"] == "core":
            leaf_decision[lid] = "core"
        elif r["tier"] == "noise":
            leaf_decision[lid] = "noise"
        else:  # topic
            ps = posts_by_leaf.get(lid, {})
            if not ps:
                leaf_decision[lid] = "topic"  # treat as uniform topic if no 5b
                continue
            n_total = len(ps)
            n_core = sum(1 for x in ps.values() if x["tier"] == "core")
            n_topic = sum(1 for x in ps.values() if x["tier"] == "topic")
            n_noise = sum(1 for x in ps.values() if x["tier"] == "noise")
            rates = {
                "core": 100 * n_core / n_total,
                "topic": 100 * n_topic / n_total,
                "noise": 100 * n_noise / n_total,
            }
            leaf_5b_rates[lid] = rates
            if rates["noise"] >= T_NOISE:
                leaf_decision[lid] = "noise"
            elif rates["core"] >= T_CORE:
                leaf_decision[lid] = "core"
            else:
                leaf_decision[lid] = "mixed"

    # Compute per-post tier + topic_scope
    post_importance: dict[str, dict] = {}
    for lid, decision in leaf_decision.items():
        members = clusters.get(lid, [])
        leaf_5a = labels_5a[lid]
        leaf_scope = leaf_5a.get("topic_scope", [])

        if decision == "core":
            for pid in members:
                post_importance[pid] = {"tier": "core", "topic_scope": []}
        elif decision == "noise":
            for pid in members:
                post_importance[pid] = {"tier": "noise", "topic_scope": []}
        elif decision == "topic":
            # uniform topic — entire leaf gets the leaf's topic_scope
            for pid in members:
                post_importance[pid] = {"tier": "topic", "topic_scope": list(leaf_scope)}
        else:  # mixed — 5b post-level wins; topic posts inherit leaf scope unless 5b provides their own
            for pid in members:
                pl = posts_by_leaf.get(lid, {}).get(pid)
                if pl:
                    scope = pl.get("topic_scope") or (leaf_scope if pl["tier"] == "topic" else [])
                    post_importance[pid] = {"tier": pl["tier"], "topic_scope": list(scope)}
                else:
                    # fallback: use leaf 5a tier (shouldn't happen for mixed, but be safe)
                    post_importance[pid] = {"tier": "topic", "topic_scope": list(leaf_scope)}

    # Save threshold record + alternatives
    threshold_path = INT / "leaf_label_threshold.json"
    threshold_path.write_text(
        json.dumps(
            {
                "T_noise": T_NOISE,
                "T_core": T_CORE,
                "leaf_decision": leaf_decision,
                "leaf_5b_rates": {k: {kk: round(vv, 1) for kk, vv in v.items()}
                                   for k, v in leaf_5b_rates.items()},
                "alternatives": _alternative_thresholds(labels_5a, posts_by_leaf),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Update _classify_overrides.json
    ovr = json.loads(OVR_PATH.read_text(encoding="utf-8"))
    ovr["post_importance"] = post_importance
    leaf_dist = Counter(leaf_decision.values())
    post_tier_dist = Counter(v["tier"] for v in post_importance.values())
    ovr.setdefault("_rules", []).append(
        {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "action": "Step 5c — post_importance 작성 (3-tier: core/topic/noise + multi-scope)",
            "detail": (
                f"T_noise={T_NOISE}% / T_core={T_CORE}%. "
                f"{len(post_importance)} posts labeled. "
                f"Leaf decisions: {dict(leaf_dist)}. Post tiers: {dict(post_tier_dist)}."
            ),
        }
    )
    OVR_PATH.write_text(json.dumps(ovr, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    today = datetime.now().strftime("%Y-%m-%d")
    out_dir = VAULT / "_reports" / today / "labeling"
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_stats_md(out_dir / "stats.md", labels_5a, labels_5b, leaf_decision,
                     leaf_5b_rates, clusters, post_importance, cohesion_by_leaf,
                     slug_to_name, posts)
    _write_audit_md(out_dir / "audit.md", labels_5a, labels_5b, leaf_decision,
                     leaf_5b_rates, clusters, cohesion_by_leaf, slug_to_name, posts)

    print(f"[5cd] T_noise={T_NOISE}% / T_core={T_CORE}%")
    print(f"[5cd] leaf decisions: {dict(leaf_dist)}")
    print(f"[5cd] post tiers: {dict(post_tier_dist)}")
    print(f"[5cd] _classify_overrides.json updated (post_importance: {len(post_importance)} entries)")
    print(f"[5cd] reports → {out_dir}")


def _alternative_thresholds(labels_5a, posts_by_leaf) -> dict:
    out = {}
    for T_n, T_c in [(60, 60), (70, 70), (80, 80), (90, 90)]:
        n_pure_core = sum(1 for r in labels_5a.values() if r["tier"] == "core")
        n_pure_noise = sum(1 for r in labels_5a.values() if r["tier"] == "noise")
        n_collapse_core = 0
        n_collapse_noise = 0
        n_topic_uniform = 0
        n_mixed = 0
        for lid, r in labels_5a.items():
            if r["tier"] != "topic":
                continue
            ps = posts_by_leaf.get(lid, {})
            if not ps:
                n_topic_uniform += 1
                continue
            n_total = len(ps)
            n_c = sum(1 for x in ps.values() if x["tier"] == "core")
            n_n = sum(1 for x in ps.values() if x["tier"] == "noise")
            r_c = 100 * n_c / n_total
            r_n = 100 * n_n / n_total
            if r_n >= T_n:
                n_collapse_noise += 1
            elif r_c >= T_c:
                n_collapse_core += 1
            else:
                n_mixed += 1
        out[f"T_noise={T_n}%, T_core={T_c}%"] = {
            "core": n_pure_core + n_collapse_core,
            "noise": n_pure_noise + n_collapse_noise,
            "topic_uniform": n_topic_uniform,
            "mixed": n_mixed,
        }
    return out


def _write_stats_md(path, labels_5a, labels_5b, leaf_decision, leaf_5b_rates,
                    clusters, post_importance, cohesion_by_leaf, slug_to_name, posts):
    today = datetime.now().strftime("%Y-%m-%d")
    md = []
    md.append(f"# Importance Labeling 통계 — {today}\n")
    md.append("3-tier (core / topic / noise) + multi-scope (personal-family · personal-life · society-politics · society-issues · industry-tech · industry-academic · industry-management) labeling. 자기계발서가 1순위 — universal lesson 우선, 개인성 ≠ 가치.\n")

    md.append("## 1. 5a per-leaf tier 분포\n")
    ctr_5a = Counter(r["tier"] for r in labels_5a.values())
    total_5a = sum(ctr_5a.values())
    md.append("| Tier | Leaves | % |")
    md.append("|---|---:|---:|")
    for t in ["core", "topic", "noise"]:
        n = ctr_5a.get(t, 0)
        md.append(f"| {t} | {n} | {100 * n / total_5a:.1f}% |")
    md.append(f"| **총** | **{total_5a}** | 100% |\n")

    # 5a topic_scope
    md.append("## 2. 5a topic_scope 분포 (multi-label, tier=topic 일 때만)\n")
    scope_ctr = Counter()
    for r in labels_5a.values():
        for s in r.get("topic_scope", []):
            scope_ctr[s] += 1
    md.append("| Scope | Leaves |")
    md.append("|---|---:|")
    for s, n in scope_ctr.most_common():
        md.append(f"| {s} | {n} |")
    md.append("")

    # 5b distribution
    md.append("## 3. 5b per-post tier 분포 (topic leaves 안 글들)\n")
    ctr_5b = Counter(r["tier"] for r in labels_5b.values())
    total_5b = sum(ctr_5b.values())
    md.append("| Tier | Posts | % |")
    md.append("|---|---:|---:|")
    for t in ["core", "topic", "noise"]:
        n = ctr_5b.get(t, 0)
        md.append(f"| {t} | {n} | {100 * n / total_5b:.1f}% |")
    md.append(f"| **총** | **{total_5b}** | 100% |\n")

    # 5c leaf decision
    md.append("## 4. 5c Threshold 채택 결과\n")
    md.append(f"채택: **T_noise = {T_NOISE}%, T_core = {T_CORE}%**.\n")
    md.append("- 5b noise rate ≥ T_noise → leaf 통째 noise")
    md.append("- 5b core rate ≥ T_core → leaf 통째 core")
    md.append("- topic leaves 중 5b 안 돈 것 (n=0) → topic uniform")
    md.append("- 그 외 → mixed (post-level 라벨 보존)\n")
    leaf_dist = Counter(leaf_decision.values())
    md.append("| Leaf decision | Count | Posts |")
    md.append("|---|---:|---:|")
    for d in ["core", "topic", "mixed", "noise"]:
        n = leaf_dist.get(d, 0)
        n_p = sum(len(clusters.get(lid, [])) for lid, dd in leaf_decision.items() if dd == d)
        md.append(f"| {d} | {n} | {n_p} |")
    md.append(f"| **총** | **{sum(leaf_dist.values())}** | **{sum(len(v) for v in clusters.values())}** |\n")

    # Final per-post tier
    md.append("## 5. 최종 post-level tier 분포\n")
    post_tier_dist = Counter(v["tier"] for v in post_importance.values())
    total_p = sum(post_tier_dist.values())
    md.append("| Tier | Posts | % |")
    md.append("|---|---:|---:|")
    for t in ["core", "topic", "noise"]:
        n = post_tier_dist.get(t, 0)
        md.append(f"| {t} | {n} | {100 * n / total_p:.1f}% |")
    md.append(f"| **총** | **{total_p}** | 100% |\n")

    # Final post-level topic_scope (multi-label)
    md.append("## 6. 최종 post-level topic_scope 분포 (multi-label, tier=topic post 만)\n")
    scope_p_ctr = Counter()
    for v in post_importance.values():
        if v["tier"] == "topic":
            for s in v.get("topic_scope", []):
                scope_p_ctr[s] += 1
    md.append("| Scope | Posts |")
    md.append("|---|---:|")
    for s, n in scope_p_ctr.most_common():
        md.append(f"| {s} | {n} |")
    md.append("")

    # Per-category breakdown
    md.append("## 7. 카테고리별 post tier 분포\n")
    md.append("| 카테고리 | core | topic | noise | total |")
    md.append("|---|---:|---:|---:|---:|")
    by_cat: dict[str, Counter] = defaultdict(Counter)
    for lid in clusters:
        cat = lid.split("/", 1)[0]
        for pid in clusters[lid]:
            v = post_importance.get(pid)
            if v:
                by_cat[cat][v["tier"]] += 1
    for cat in sorted(by_cat.keys(), key=lambda c: -sum(by_cat[c].values())):
        d = by_cat[cat]
        cname = slug_to_name.get(cat, cat)
        total = sum(d.values())
        md.append(f"| {cname} | {d.get('core', 0)} | {d.get('topic', 0)} | {d.get('noise', 0)} | {total} |")
    md.append("")

    # Cohesion vs core rate
    md.append("## 8. Cohesion vs core rate\n")
    bins = [(0.20, 0.30), (0.30, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.01)]
    md.append("| Cohesion bin | Leaves | core% | topic% | noise% |")
    md.append("|---|---:|---:|---:|---:|")
    for lo, hi in bins:
        leaves_in_bin = [lid for lid in leaf_decision if lo <= cohesion_by_leaf.get(lid, 0) < hi]
        if not leaves_in_bin:
            continue
        all_pids = [pid for lid in leaves_in_bin for pid in clusters.get(lid, [])]
        if not all_pids:
            continue
        n_c = sum(1 for pid in all_pids if post_importance.get(pid, {}).get("tier") == "core")
        n_t = sum(1 for pid in all_pids if post_importance.get(pid, {}).get("tier") == "topic")
        n_n = sum(1 for pid in all_pids if post_importance.get(pid, {}).get("tier") == "noise")
        N = len(all_pids)
        md.append(f"| {lo:.2f}–{hi:.2f} | {len(leaves_in_bin)} | {100*n_c/N:.1f}% | {100*n_t/N:.1f}% | {100*n_n/N:.1f}% |")
    md.append("")

    # Synth scope candidates
    md.append("## 9. Synthesize scope 후보 (Step 6 결정용)\n")
    leaf_min = 8
    n_pure_core = sum(1 for d in leaf_decision.values() if d == "core")
    n_pure_topic = sum(1 for d in leaf_decision.values() if d == "topic")
    n_mixed = sum(1 for d in leaf_decision.values() if d == "mixed")
    n_pure_noise = sum(1 for d in leaf_decision.values() if d == "noise")
    md.append(f"- Pure core leaves: **{n_pure_core}** (1순위 자기계발서 직접 source)")
    md.append(f"- Pure topic leaves: **{n_pure_topic}** (특정 챕터/책에서만 source)")
    md.append(f"- Mixed leaves: **{n_mixed}** (post-level 분리 — core posts 만 자기계발서 / topic posts 는 그 scope 챕터)")
    md.append(f"- Pure noise leaves: {n_pure_noise} (synth 안 함)\n")

    md.append("**합성 옵션**:")
    md.append("- (a) **모두 합성**: pure core + pure topic + mixed (member 들의 majority tier 별로 multi-output) → 책 작성 시 tier·scope 로 필터")
    md.append("- (b) **core 만 합성**: pure core leaves 만 → 자기계발서 source 정수")
    md.append("- (c) **core + topic 분리 합성**: 한 leaf 가 mixed 이면 core 부분만 한 노트, topic 부분만 다른 노트")
    md.append("\n비용: (a)=가장 많음, (b)=가장 적음.\n")

    # cohesion < 0.30
    cohesion_low = [lid for lid in leaf_decision if cohesion_by_leaf.get(lid, 0) < 0.30 and leaf_decision[lid] != "noise"]
    md.append(f"low cohesion + non-noise: **{len(cohesion_low)} leaves** (cohesion < 0.30 — 합성해도 'glob 글 나열' 위험)\n")

    md.append("## 10. Threshold sensitivity (T_noise / T_core 대안)\n")
    md.append("| Threshold | core | mixed | topic_uniform | noise |")
    md.append("|---|---:|---:|---:|---:|")
    alt_path = INT / "leaf_label_threshold.json"
    alt = json.loads(alt_path.read_text(encoding="utf-8"))["alternatives"]
    for k, v in alt.items():
        md.append(f"| {k} | {v['core']} | {v['mixed']} | {v['topic_uniform']} | {v['noise']} |")
    md.append("")

    md.append("## 11. 점검 포인트\n")
    md.append("- [ ] mixed leaves (post-level 분리) 처리 옵션 (a/b/c 중 선택)")
    md.append("- [ ] cohesion < 0.30 + non-noise leaves: synth 추가 제외 여부")
    md.append("- [ ] Threshold T_noise/T_core 변경 (alternatives 표)")
    md.append("- [ ] audit.md 답변 후 post_importance 보강 (Step 5e)")
    md.append("")

    path.write_text("\n".join(md), encoding="utf-8")
    print(f"[5d] stats.md → {path}")


def _write_audit_md(path, labels_5a, labels_5b, leaf_decision, leaf_5b_rates,
                     clusters, cohesion_by_leaf, slug_to_name, posts):
    today = datetime.now().strftime("%Y-%m-%d")
    md = []
    md.append(f"# Importance Audit batch — {today}\n")
    md.append("LLM 이 가장 헷갈려하는 leaf / post. 각 항목 옆 체크박스에 `LEAF=<leaf_id> <tier>` 또는 `POST=<post_id> <tier>` 표시. tier 는 `core` / `topic` / `noise`. 미표시 = LLM 판정 그대로. Step 5e 가 이 파일을 읽어 post_importance 갱신.\n")

    # Leaf-level audit: 5b noise rate 40-70% (mixed-y)
    md.append(f"## A. Leaf-level audit (5b noise rate {AUDIT_LEAF_NOISE_LO}%–{AUDIT_LEAF_NOISE_HI}%)\n")
    md.append("Mixed leaves 중 노이즈 비율이 가장 모호한 구간. 통째 처리 원하면 표시.\n")
    audit_leaves = [
        lid for lid in leaf_5b_rates
        if AUDIT_LEAF_NOISE_LO <= leaf_5b_rates[lid].get("noise", 0) <= AUDIT_LEAF_NOISE_HI
    ]
    audit_leaves.sort(key=lambda l: leaf_5b_rates[l].get("noise", 0))
    for lid in audit_leaves:
        rates = leaf_5b_rates[lid]
        cohesion = cohesion_by_leaf.get(lid, 0)
        n = len(clusters.get(lid, []))
        cat = slug_to_name.get(lid.split("/", 1)[0], lid.split("/", 1)[0])
        sample_pids = clusters.get(lid, [])[:3]
        sample_lines = []
        for pid in sample_pids:
            p = posts.get(pid, {})
            t = (p.get("text") or "").replace("\n", " ").strip()[:140]
            sample_lines.append(f"  - [{p.get('date','')[:10]}] {t}")
        md.append(f"### `{lid}` — {cat} (n={n}, cohesion={cohesion:.2f})")
        md.append(f"  rates: core={rates.get('core',0):.0f}% topic={rates.get('topic',0):.0f}% noise={rates.get('noise',0):.0f}%")
        md.append(f"  5a tier={labels_5a[lid]['tier']} scope={labels_5a[lid].get('topic_scope', [])}")
        md.append(f"  5a reason: {labels_5a[lid]['reason']}")
        md.append("\n".join(sample_lines))
        md.append(f"- [ ] LEAF={lid} core")
        md.append(f"- [ ] LEAF={lid} topic")
        md.append(f"- [ ] LEAF={lid} noise")
        md.append("")

    # Post-level audit: confidence < AUDIT_POST_CONF
    md.append(f"## B. Post-level audit (5b confidence < {AUDIT_POST_CONF})\n")
    md.append("Haiku 도 모호하다고 판정한 글. 표시한 tier 로 override.\n")
    audit_posts = [(pid, r) for pid, r in labels_5b.items() if r["confidence"] < AUDIT_POST_CONF]
    audit_posts.sort(key=lambda x: x[1]["confidence"])
    for pid, r in audit_posts[:200]:
        p = posts.get(pid, {})
        text = (p.get("text") or "").replace("\n", " ").strip()
        snippet = text[:300]
        date = p.get("date", "")[:10]
        md.append(f"### {pid} — leaf=`{r['leaf_id']}` (date={date}, 5b={r['tier']}, conf={r['confidence']:.2f})")
        md.append(f"  reason: {r['reason']}")
        md.append(f"\n  > {snippet}")
        md.append(f"\n- [ ] POST={pid} core")
        md.append(f"- [ ] POST={pid} topic")
        md.append(f"- [ ] POST={pid} noise")
        md.append("")

    if len(audit_posts) > 200:
        md.append(f"\n_(+{len(audit_posts) - 200} more posts with low confidence — 처음 200 개만 audit.)_\n")

    path.write_text("\n".join(md), encoding="utf-8")
    print(f"[5d] audit.md → {path}  ({len(audit_leaves)} leaves + {min(200, len(audit_posts))} posts)")


if __name__ == "__main__":
    main()
