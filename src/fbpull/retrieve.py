"""Stage 8: query → top-K leaves + top-N posts.

Hybrid 2-stage retrieval over the existing cluster output:
1. Embed the query with the same provider used in stage 4.
2. Score each leaf by cosine(query, leaf_centroid); keep top-K.
3. Score every post inside those top-K leaves by cosine(query, post); keep top-N.
4. Decorate posts with era / category / light-deep / mid-node flags from
   `03_classified.jsonl` and the leaf id path, then write a markdown report
   and a JSON sidecar under `_intermediate/retrieval/<query-slug>.{md,json}`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

from . import embed as embed_mod
from . import llm
from . import taxonomy as taxonomy_mod
from .paths import fb_root, intermediate_dir


_ILSANG_MIDS = {"diary", "exercise", "canada", "sns", "english"}
_VALID_TIERS = {"core", "topic", "noise"}
_VALID_SCOPES = {
    "personal-family", "personal-life",
    "society-politics", "society-issues",
    "industry-tech", "industry-academic", "industry-management",
}


def _slug_to_name() -> dict[str, str]:
    """Map current category slug → 한글 name from `_taxonomy.md`. Used to label
    post records by their *current* category (leaf_id prefix), not the stale
    classify-time label stored in `03_classified.jsonl`."""
    tax = taxonomy_mod.load()
    if not tax:
        return {}
    return {c.slug: c.name for c in tax.categories}


def _slugify_query(query: str) -> str:
    s = query.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^ㄱ-ㆎ가-힣a-z0-9-]", "", s)
    return s[:60] or "query"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _embed_query(query: str) -> tuple[np.ndarray, str]:
    """Embed a single query string. Caches under the same `embed_cache/<model>/`
    directory used by stage 4 so repeated queries are free."""
    model, provider = embed_mod._select_provider(no_llm=False)
    cache_dir = intermediate_dir() / "embed_cache" / model
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = llm.text_hash(query)
    cf = cache_dir / f"{h}.json"
    if cf.exists():
        v = json.loads(cf.read_text(encoding="utf-8"))["v"]
    else:
        v = provider([query])[0]
        cf.write_text(json.dumps({"v": v, "model": model}, ensure_ascii=False), encoding="utf-8")
    arr = np.array(v, dtype=np.float32)
    return _normalize(arr), model


def _ilsang_mid_for(leaf_id: str) -> str | None:
    parts = leaf_id.split("/")
    if len(parts) >= 2 and parts[0] == "ilsang-sageon" and parts[1] in _ILSANG_MIDS:
        return parts[1]
    return None


def _load_post_meta() -> dict[str, dict]:
    out: dict[str, dict] = {}
    p = intermediate_dir() / "03_classified.jsonl"
    with p.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["post_id"]] = r
    return out


def _load_clusters() -> dict[str, list[str]]:
    p = intermediate_dir() / "05_clusters.json"
    return json.loads(p.read_text(encoding="utf-8"))


def _load_embeddings() -> tuple[np.ndarray, dict[str, int]]:
    arr = np.load(intermediate_dir() / "04_embeddings.npy")
    ids = json.loads((intermediate_dir() / "04_post_ids.json").read_text(encoding="utf-8"))
    pid_to_row = {pid: i for i, pid in enumerate(ids)}
    return arr, pid_to_row


def _load_post_importance() -> dict[str, dict]:
    p = fb_root() / "_classify_overrides.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return data.get("post_importance", {})


def _load_leaf_themes_index() -> dict[str, list[dict]]:
    """Returns {leaf_id: [{theme_id, name, score}]}."""
    p = intermediate_dir() / "leaf_themes.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    theme_meta = data.get("theme_meta", {})
    out: dict[str, list[dict]] = {}
    for lid, tids in data.get("leaf_themes", {}).items():
        out[lid] = [
            {
                "theme_id": tid,
                "name": theme_meta.get(tid, {}).get("name", tid),
                "score": float(score),
            }
            for tid, score in tids
        ]
    return out


def _parse_set(spec: str | None, valid: set[str]) -> set[str] | None:
    if not spec:
        return None
    items = {s.strip() for s in spec.split(",") if s.strip()}
    bad = items - valid
    if bad:
        raise ValueError(f"unknown values: {bad}. valid: {valid}")
    return items


def run(
    query: str,
    top_leaves: int = 8,
    top_posts: int = 30,
    tier_filter: set[str] | None = None,
    scope_filter: set[str] | None = None,
) -> dict:
    arr, pid_to_row = _load_embeddings()
    clusters = _load_clusters()
    posts_meta = _load_post_meta()
    slug_to_name = _slug_to_name()
    post_importance = _load_post_importance()
    leaf_themes_index = _load_leaf_themes_index()

    qvec, model = _embed_query(query)
    if qvec.shape[0] != arr.shape[1]:
        raise RuntimeError(
            f"Query embedding dim {qvec.shape[0]} != corpus dim {arr.shape[1]}. "
            f"Re-run `fbpull embed` with the same FBPULL_EMBED_MODEL or unset it."
        )

    def _post_passes(pid: str) -> bool:
        if not (tier_filter or scope_filter):
            return True
        imp = post_importance.get(pid, {})
        if tier_filter is not None and imp.get("tier") not in tier_filter:
            return False
        if scope_filter is not None:
            scopes = set(imp.get("topic_scope") or [])
            if not (scopes & scope_filter):
                return False
        return True

    # Stage 1: leaf centroids — score every leaf, but compute centroid from
    # filtered posts when a filter is active (so leaves dominated by filtered-out
    # tier/scope rank lower for the query).
    leaf_records: list[tuple[str, float, int]] = []
    for leaf_id, leaf_pids in clusters.items():
        rows = [pid_to_row[p] for p in leaf_pids if p in pid_to_row and _post_passes(p)]
        if not rows:
            continue
        centroid = _normalize(arr[rows].mean(axis=0))
        score = float(np.dot(centroid, qvec))
        leaf_records.append((leaf_id, score, len(rows)))
    leaf_records.sort(key=lambda t: -t[1])
    top_leaf_list = leaf_records[:top_leaves]

    # Stage 2: post scores within selected leaves
    candidates: list[tuple[str, str]] = []  # (post_id, leaf_id)
    for leaf_id, _, _ in top_leaf_list:
        for pid in clusters.get(leaf_id, []):
            if pid in pid_to_row and _post_passes(pid):
                candidates.append((pid, leaf_id))

    if not candidates:
        return {
            "query": query,
            "qslug": _slugify_query(query),
            "model": model,
            "top_leaves": [],
            "top_posts": [],
            "top_posts_chrono": [],
            "era_cat_distribution": {},
        }

    rows = np.array([pid_to_row[pid] for pid, _ in candidates])
    scores = (arr[rows] @ qvec).tolist()
    pairs = sorted(zip(candidates, scores), key=lambda t: -t[1])[:top_posts]

    post_records: list[dict] = []
    for (pid, leaf_id), score in pairs:
        m = posts_meta.get(pid, {})
        text = m.get("text", "") or ""
        snippet = text.replace("\n", " ").strip()[:240]
        cat_slug = leaf_id.split("/")[0]
        cat_name = slug_to_name.get(cat_slug, m.get("category", cat_slug))
        imp = post_importance.get(pid, {})
        post_records.append(
            {
                "post_id": pid,
                "leaf_id": leaf_id,
                "score": round(float(score), 4),
                "date": m.get("date", ""),
                "era": m.get("era", "unknown"),
                "category": cat_name,
                "ilsang_mid": _ilsang_mid_for(leaf_id),
                "tier": imp.get("tier"),
                "topic_scope": imp.get("topic_scope") or [],
                "snippet": snippet,
            }
        )

    era_cat: Counter = Counter()
    for r in post_records:
        era_cat[(r["era"], r["category"])] += 1

    chrono = sorted(post_records, key=lambda r: r["date"])

    return {
        "query": query,
        "qslug": _slugify_query(query),
        "model": model,
        "top_leaves": [
            {
                "leaf_id": lid,
                "score": round(float(s), 4),
                "size": n,
                "category_slug": lid.split("/")[0],
                "category": slug_to_name.get(lid.split("/")[0], lid.split("/")[0]),
                "ilsang_mid": _ilsang_mid_for(lid),
                "themes": leaf_themes_index.get(lid, []),
            }
            for lid, s, n in top_leaf_list
        ],
        "filters": {
            "tier": sorted(tier_filter) if tier_filter else None,
            "scope": sorted(scope_filter) if scope_filter else None,
        },
        "top_posts": post_records,
        "top_posts_chrono": chrono,
        "era_cat_distribution": dict(era_cat),
    }


def write_outputs(result: dict, out_dir: Path | None = None) -> tuple[Path, Path]:
    out_dir = out_dir or (intermediate_dir() / "retrieval")
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = result["qslug"]
    json_path = out_dir / f"{slug}.json"
    md_path = out_dir / f"{slug}.md"

    json_payload = {
        "query": result["query"],
        "model": result["model"],
        "top_leaves": result["top_leaves"],
        "top_posts": result["top_posts"],
        "era_cat_distribution": {
            f"{era} | {cat}": n for (era, cat), n in result["era_cat_distribution"].items()
        },
    }
    json_path.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines: list[str] = []
    lines.append(f'# Retrieval — "{result["query"]}"')
    lines.append("")
    lines.append(
        f"_top_leaves={len(result['top_leaves'])}, "
        f"top_posts={len(result['top_posts'])}, model={result['model']}_"
    )
    lines.append("")

    filters = result.get("filters") or {}
    if filters.get("tier") or filters.get("scope"):
        lines.append("**Filters**: " + ", ".join(
            f"{k}={v}" for k, v in filters.items() if v
        ))
        lines.append("")

    lines.append("## Top leaves")
    lines.append("")
    for l in result["top_leaves"]:
        flag = f" · {l['ilsang_mid']}" if l.get("ilsang_mid") else ""
        themes = l.get("themes") or []
        theme_str = ""
        if themes:
            theme_str = " · themes: " + ", ".join(
                f"{t['name']}({t['score']:.2f})" for t in themes[:3]
            )
        lines.append(
            f"- **`{l['leaf_id']}`** · n={l['size']} · score={l['score']:.3f} · {l['category']}{flag}{theme_str}"
        )
    lines.append("")

    lines.append("## Era × Category 분포 (top-N posts)")
    lines.append("")
    rows = sorted(
        result["era_cat_distribution"].items(), key=lambda kv: (-kv[1], kv[0])
    )
    if not rows:
        lines.append("_(empty)_")
    else:
        for (era, cat), n in rows:
            lines.append(f"- **{era}** | {cat} — {n}")
    lines.append("")

    lines.append("## Top posts (시간순)")
    lines.append("")
    for r in result["top_posts_chrono"]:
        flag_parts = []
        if r.get("tier"):
            flag_parts.append(r["tier"])
        if r.get("topic_scope"):
            flag_parts.extend(r["topic_scope"])
        if r.get("ilsang_mid"):
            flag_parts.append(r["ilsang_mid"])
        flag_str = f" · {','.join(flag_parts)}" if flag_parts else ""
        lines.append(
            f"### [{r['date']}] {r['era']} · {r['category']}{flag_str}"
        )
        lines.append(f"_score={r['score']:.3f} · leaf=`{r['leaf_id']}` · pid=`{r['post_id']}`_")
        lines.append("")
        lines.append(r["snippet"] + ("…" if len(r["snippet"]) >= 240 else ""))
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def cli_run(
    query: str,
    top_leaves: int = 8,
    top_posts: int = 30,
    tier: str | None = None,
    scope: str | None = None,
) -> None:
    tier_set = _parse_set(tier, _VALID_TIERS)
    scope_set = _parse_set(scope, _VALID_SCOPES)
    result = run(
        query, top_leaves=top_leaves, top_posts=top_posts,
        tier_filter=tier_set, scope_filter=scope_set,
    )
    json_path, md_path = write_outputs(result)
    print(
        f"[retrieve] {len(result['top_leaves'])} leaves, "
        f"{len(result['top_posts'])} posts → model={result['model']}"
    )
    if tier_set or scope_set:
        print(f"[retrieve] filters: tier={tier_set}, scope={scope_set}")
    print(f"[retrieve] {md_path}")
    print(f"[retrieve] {json_path}")
