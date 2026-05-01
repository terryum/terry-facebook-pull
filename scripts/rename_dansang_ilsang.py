"""Rename 2 top-level categories (one-shot, 2026-05-01).

  단상         → 일상 감정    (slug: dansang           → ilsang-gamjeong)
  일상·생활    → 일상 사건    (slug: ilsang-saenghwal  → ilsang-sageon)

In-place rewrite of taxonomy + overrides + intermediate stage outputs.
Backups go to <file>.bak-rename-2026-05-01-ilsang. Re-run cluster after to
regenerate any state we don't touch directly. Archive/Synthesized/_index/
_coverage are stale from earlier taxonomies and rebuild on next export/report.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

LABEL_RENAMES = {
    "단상": "일상 감정",
    "일상·생활": "일상 사건",
}
SLUG_RENAMES = {
    "dansang": "ilsang-gamjeong",
    "ilsang-saenghwal": "ilsang-sageon",
}
BAK_SUFFIX = ".bak-rename-2026-05-01-ilsang"


def fb_root() -> Path:
    p = os.environ.get("OBSIDIAN_VAULT")
    if not p:
        sys.exit("OBSIDIAN_VAULT not set")
    return Path(p).expanduser().resolve() / "vault" / "Private" / "Facebook"


def backup(p: Path) -> None:
    bak = p.with_suffix(p.suffix + BAK_SUFFIX)
    if bak.exists():
        return  # don't clobber an earlier run's backup
    shutil.copy2(p, bak)


def rewrite_slug(s: str) -> str:
    """Replace slug-prefix matches (e.g. `dansang/0/1` → `ilsang-gamjeong/0/1`)."""
    if not isinstance(s, str):
        return s
    for old, new in SLUG_RENAMES.items():
        if s == old or s.startswith(old + "/"):
            return new + s[len(old):]
    return s


def rewrite_label(s: str) -> str:
    if isinstance(s, str) and s in LABEL_RENAMES:
        return LABEL_RENAMES[s]
    return s


def walk_json(node, key_remap=None, label_keys=(), slug_keys=()):
    """Recursive transform.
    - key_remap(k): optional, applied to every dict key as a slug-style remap
    - label_keys: dict keys whose value is a Korean label (rewrite via LABEL_RENAMES)
    - slug_keys: dict keys whose value is a slug or slug-path (rewrite via rewrite_slug)
    """
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            new_k = key_remap(k) if key_remap else k
            if k in label_keys and isinstance(v, str):
                v = rewrite_label(v)
            elif k in slug_keys and isinstance(v, str):
                v = rewrite_slug(v)
            else:
                v = walk_json(v, key_remap, label_keys, slug_keys)
            out[new_k] = v
        return out
    if isinstance(node, list):
        return [walk_json(x, key_remap, label_keys, slug_keys) for x in node]
    return node


def patch_text(p: Path, replacements: dict[str, str]) -> int:
    """Plain text substitution; returns total replacement count."""
    txt = p.read_text(encoding="utf-8")
    n = 0
    for old, new in replacements.items():
        cnt = txt.count(old)
        if cnt:
            txt = txt.replace(old, new)
            n += cnt
    p.write_text(txt, encoding="utf-8")
    return n


# ── 1. taxonomy.md ───────────────────────────────────────────────────────
def step_taxonomy(root: Path) -> None:
    p = root / "_taxonomy.md"
    backup(p)
    txt = p.read_text(encoding="utf-8")

    # Headings
    txt = txt.replace("## 단상\n", "## 일상 감정\n")
    txt = txt.replace("## 일상·생활\n", "## 일상 사건\n")

    # Cross-references in body. Replace standalone label tokens. Order matters:
    # do longer/more-specific tokens first so we don't half-substitute.
    # `일상·생활` is a 4-char token; `단상` is 2-char. Both are unambiguous in
    # this taxonomy doc (verified — no 단상 used as a generic noun).
    txt = txt.replace("일상·생활", "일상 사건")
    txt = txt.replace("단상", "일상 감정")

    p.write_text(txt, encoding="utf-8")
    print(f"[1] _taxonomy.md rewritten")


# ── 2. _classify_overrides.json ──────────────────────────────────────────
def step_overrides(root: Path) -> None:
    p = root / "_classify_overrides.json"
    backup(p)
    d = json.loads(p.read_text(encoding="utf-8"))

    # 2a. overrides: post_id → label
    if isinstance(d.get("overrides"), dict):
        for pid, cat in list(d["overrides"].items()):
            d["overrides"][pid] = rewrite_label(cat)

    # 2b. leaf_groups: keyed by slug
    if isinstance(d.get("leaf_groups"), dict):
        d["leaf_groups"] = {rewrite_slug(k): v for k, v in d["leaf_groups"].items()}

    # 2c. leaf_merges: from/to are slug paths
    if isinstance(d.get("leaf_merges"), list):
        for m in d["leaf_merges"]:
            if isinstance(m, dict):
                if "from" in m:
                    m["from"] = rewrite_slug(m["from"])
                if "to" in m:
                    m["to"] = rewrite_slug(m["to"])

    # 2d. _rules: prefix is slug-path, category is label, note is free text
    if isinstance(d.get("_rules"), list):
        for r in d["_rules"]:
            if isinstance(r, dict):
                if "prefix" in r:
                    r["prefix"] = rewrite_slug(r["prefix"])
                if "category" in r:
                    r["category"] = rewrite_label(r["category"])
                if "note" in r and isinstance(r["note"], str):
                    for old, new in LABEL_RENAMES.items():
                        r["note"] = r["note"].replace(old, new)
                    for old, new in SLUG_RENAMES.items():
                        r["note"] = r["note"].replace(old, new)

    # 2e. _comment: free text
    if isinstance(d.get("_comment"), str):
        for old, new in LABEL_RENAMES.items():
            d["_comment"] = d["_comment"].replace(old, new)
        for old, new in SLUG_RENAMES.items():
            d["_comment"] = d["_comment"].replace(old, new)

    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[2] _classify_overrides.json rewritten")


# ── 3. 03_classified.jsonl ───────────────────────────────────────────────
def step_classified(root: Path) -> None:
    p = root / "_intermediate" / "03_classified.jsonl"
    backup(p)
    n_changed = 0
    out_lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            out_lines.append(line)
            continue
        rec = json.loads(line)
        cat = rec.get("category")
        if cat in LABEL_RENAMES:
            rec["category"] = LABEL_RENAMES[cat]
            n_changed += 1
        out_lines.append(json.dumps(rec, ensure_ascii=False))
    p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[3] 03_classified.jsonl: {n_changed} records relabeled")


# ── 4. 05_hierarchy.json ─────────────────────────────────────────────────
def step_hierarchy(root: Path) -> None:
    p = root / "_intermediate" / "05_hierarchy.json"
    backup(p)
    d = json.loads(p.read_text(encoding="utf-8"))
    new = {}
    for k, v in d.items():
        nk = rewrite_slug(k)
        if isinstance(v, dict) and isinstance(v.get("children"), list):
            v = {**v, "children": [rewrite_slug(c) for c in v["children"]]}
        if isinstance(v, dict) and isinstance(v.get("category_name"), str):
            v["category_name"] = rewrite_label(v["category_name"])
        new[nk] = v
    p.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[4] 05_hierarchy.json: {len(new)} nodes rewritten")


# ── 5. 05_clusters.json ──────────────────────────────────────────────────
def step_clusters(root: Path) -> None:
    p = root / "_intermediate" / "05_clusters.json"
    backup(p)
    d = json.loads(p.read_text(encoding="utf-8"))
    new = {rewrite_slug(k): v for k, v in d.items()}
    p.write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[5] 05_clusters.json: {len(new)} cluster ids")


# ── 6. 05_cluster_stats.json ─────────────────────────────────────────────
def step_cluster_stats(root: Path) -> None:
    p = root / "_intermediate" / "05_cluster_stats.json"
    backup(p)
    d = json.loads(p.read_text(encoding="utf-8"))

    # 'categories' is a dict keyed by slug; each value has a 'name' field
    cats = d.get("categories")
    if isinstance(cats, dict):
        new_cats = {}
        for slug, info in cats.items():
            new_slug = rewrite_slug(slug)
            if isinstance(info, dict) and isinstance(info.get("name"), str):
                info["name"] = rewrite_label(info["name"])
            new_cats[new_slug] = info
        d["categories"] = new_cats

    leaves = d.get("leaves")
    if isinstance(leaves, list):
        for leaf in leaves:
            if not isinstance(leaf, dict):
                continue
            if isinstance(leaf.get("id"), str):
                leaf["id"] = rewrite_slug(leaf["id"])
            if isinstance(leaf.get("category_slug"), str):
                leaf["category_slug"] = rewrite_slug(leaf["category_slug"])

    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[6] 05_cluster_stats.json rewritten")


# ── 7. 05_time_series.json ───────────────────────────────────────────────
def step_time_series(root: Path) -> None:
    p = root / "_intermediate" / "05_time_series.json"
    backup(p)
    d = json.loads(p.read_text(encoding="utf-8"))

    if isinstance(d.get("by_category"), dict):
        d["by_category"] = {rewrite_label(k): v for k, v in d["by_category"].items()}
    if isinstance(d.get("by_leaf"), dict):
        d["by_leaf"] = {rewrite_slug(k): v for k, v in d["by_leaf"].items()}

    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[7] 05_time_series.json rewritten")


# ── 8. 06_synthesized.jsonl ──────────────────────────────────────────────
def step_synthesized(root: Path) -> None:
    p = root / "_intermediate" / "06_synthesized.jsonl"
    if not p.exists():
        print("[8] 06_synthesized.jsonl: skipped (missing)")
        return
    backup(p)
    n_changed = 0
    out_lines = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            out_lines.append(line)
            continue
        rec = json.loads(line)
        before = json.dumps(rec, ensure_ascii=False)
        if isinstance(rec.get("cluster_id"), str):
            rec["cluster_id"] = rewrite_slug(rec["cluster_id"])
        if isinstance(rec.get("category_slug"), str):
            rec["category_slug"] = rewrite_slug(rec["category_slug"])
        if isinstance(rec.get("category"), str):
            rec["category"] = rewrite_label(rec["category"])
        after = json.dumps(rec, ensure_ascii=False)
        if before != after:
            n_changed += 1
        out_lines.append(after)
    p.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    print(f"[8] 06_synthesized.jsonl: {n_changed} records updated")


# ── 9. fine-tune scripts (cosmetic) ──────────────────────────────────────
def step_scripts() -> None:
    proj = Path(__file__).resolve().parent
    repls = {**LABEL_RENAMES, **SLUG_RENAMES}
    for name in ("sahoe_finetune.py", "haggye_split.py"):
        p = proj / name
        if not p.exists():
            continue
        backup(p)
        n = patch_text(p, repls)
        print(f"[9] scripts/{name}: {n} replacements")


def main() -> None:
    root = fb_root()
    print(f"vault root: {root}")
    print(f"label renames: {LABEL_RENAMES}")
    print(f"slug renames: {SLUG_RENAMES}")
    print()
    step_taxonomy(root)
    step_overrides(root)
    step_classified(root)
    step_hierarchy(root)
    step_clusters(root)
    step_cluster_stats(root)
    step_time_series(root)
    step_synthesized(root)
    step_scripts()
    print()
    print("done. Re-run `fbpull cluster` next to confirm pipeline still produces")
    print("the same hierarchy with the new slugs.")


if __name__ == "__main__":
    main()
