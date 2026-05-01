"""사회 카테고리 fine-tune (D + A + B step).

D: /leftover, /35, /2 — multi-category resort via Haiku
A: /8 통째 → 일상 사건, /33 통째 → 취미·문화, /34 → 사회 vs 일상 사건 LLM split
B: /16, /15 → deep reflection check (deep → 인생·일상 감정)

Updates _classify_overrides.json["overrides"]. Caches LLM in
_intermediate/llm_cache/{category_resort,depth_check}/.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fbpull import llm  # noqa: E402
from fbpull.cli import _bootstrap  # noqa: E402

_bootstrap(no_llm=False)

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
INT = VAULT / "_intermediate"
OVERRIDE_PATH = VAULT / "_classify_overrides.json"
MODEL = "claude-haiku-4-5"
WORKERS = 6

CATEGORIES = [
    "연구·학술", "학습·학계", "인생·일상 감정", "창업·경영", "조직·리더십",
    "테크 커뮤니티·산업 동향", "사회", "정치", "가족", "일상 사건",
    "취미·문화", "기타·미분류",
]

RESORT_SYSTEM = """당신은 한 사용자의 페이스북 글을 분류합니다. submit 도구로 결과를 반환하세요.

# 사용자
엄태웅 (Terry, 1983년생). 서울대 기계항공 → KIST/LIG넥스원 연구원 → Waterloo 박사 (딥러닝)
→ ART Lab 창업·대표 → 코스맥스 AI혁신본부장.

# 카테고리 정의
- 연구·학술: 딥러닝·로봇·AI·알고리즘 본업 콘텐츠 (논문 아이디어·기법·분야 동향)
- 학습·학계: 박사과정·학회·논문 워크플로·발표·진로 / 공부 routine·학부 대학원생 일상
- 인생·일상 감정: 삶·시간·죽음·존재·관계 깊은 사색. 길고 reflective. 짧은 감정 토로 X
- 창업·경영: 스타트업 운영·투자·자본·팀 빌딩
- 조직·리더십: 사람·팀·문화·협업·멘토십
- 테크 커뮤니티·산업 동향: 테크가 사회에 미친 영향, AI·블록체인이 일/문화 바꾸는 일상 감정,
  알파고·4차혁명·코딩교육·미디어 변화. 또는 텐서플로우 코리아 같은 커뮤니티 활동.
- 사회: 사회 이슈 비평. 자본·임대·교육·미디어·생명·도시·젠더·법·디아스포라.
  (정치인·정당 비판 X — 그건 정치)
- 정치: 정치인·정당·정부·사법-재벌·이념 비평. 한국 시사 + 개인적 정치 발언.
- 가족: 부모·형제·자녀·결혼 가족 관계
- 일상 사건: 본인의 자전적 일상 — 캐나다 정착·이민, 운동·다이어트, 음식, 학교/회사 일과,
  SNS 활동, 사고·소소한 사건. "내가 오늘 무엇을 했는가" daily narrative.
- 취미·문화: 음악·스포츠·게임·영화·미술관·여행. "내가 좋아하는 것" hobby 기록.
- 기타·미분류: 위 어디에도 명확히 안 맞는 글

기준:
- 사회 vs 정치: 특정 정치인/정당/정부 비판이 핵심이면 정치, 사회구조 비평이면 사회
- 사회 vs 일상 사건: 본인 경험 일상 narrative 면 일상 사건, 사회 비평이면 사회
- 사회 vs 인생·일상 감정: 삶/존재의 깊은 성찰이면 인생·일상 감정, 사회 이슈 비평이면 사회
- 사회 vs 테크 커뮤니티·산업 동향: 테크가 사회에 미친 영향 톤이면 테크 커뮤니티
"""

DEPTH_SYSTEM = """당신은 페이스북 글이 "인생·일상 감정" (깊은 성찰) 후보인지 판단합니다.

deep 기준:
- 삶·시간·죽음·존재·관계·자아 등에 대한 깊은 사색
- reflective 한 길고 정제된 글, 아포리즘, 시적 일상 감정, 일기적 회상
- 자신의 가치관/태도/방향성에 대한 고찰

deep 아님:
- 짧은 감정 토로 (ㅠㅠ, 재밌다, 화가 난다)
- 사회 이슈 비평 (사회구조·시스템·정책 비판)
- 단순 사건 보고
- 정보 공유/링크
"""

DEPTH_SCHEMA = {
    "type": "object",
    "properties": {
        "is_deep_reflection": {"type": "boolean"},
        "reason": {"type": "string", "description": "한 줄 사유"},
    },
    "required": ["is_deep_reflection", "reason"],
}

RESORT_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": CATEGORIES},
        "reason": {"type": "string"},
    },
    "required": ["category", "reason"],
}

SAHOE_VS_ILSANG_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": ["사회", "일상 사건"]},
        "reason": {"type": "string"},
    },
    "required": ["category", "reason"],
}


def _collect_leaf(clusters, hierarchy, leaf_id):
    n = hierarchy.get(leaf_id, {})
    if n.get("is_leaf"):
        return list(clusters.get(leaf_id, []))
    out = []
    for c in n.get("children", []):
        out.extend(_collect_leaf(clusters, hierarchy, c))
    return out


def _call_with_cache(cache_dir, post_id, text, system, schema):
    key = f"{post_id}_{llm.text_hash(text)}"
    cached = llm.cache_get(cache_dir, key)
    if cached:
        return cached
    res = llm.call_json(MODEL, system, text[:6000], max_tokens=300, schema=schema)
    llm.cache_put(cache_dir, key, res)
    return res


def resort(post_id, text):
    cd = INT / "llm_cache" / "category_resort"
    return _call_with_cache(cd, post_id, text, RESORT_SYSTEM, RESORT_SCHEMA)


def depth_check(post_id, text):
    cd = INT / "llm_cache" / "depth_check"
    return _call_with_cache(cd, post_id, text, DEPTH_SYSTEM, DEPTH_SCHEMA)


def sahoe_vs_ilsang(post_id, text):
    cd = INT / "llm_cache" / "sahoe_vs_ilsang"
    sys = (
        "당신은 페이스북 글이 '사회' (사회 이슈 비평) 인지 '일상 사건' (본인의 자전적 일상) 인지 판단합니다.\n"
        "사회: 사회구조·제도·시스템에 대한 비평·관찰 (예: 장애인 권리 비평, 의료 시스템 비교 비평)\n"
        "일상 사건: 본인이 경험한 캐나다 일상, 한국 일상 narrative (예: '한달 살아본 서울', '강남 씽씽이')"
    )
    return _call_with_cache(cd, post_id, text, sys, SAHOE_VS_ILSANG_SCHEMA)


def main():
    clusters = json.loads((INT / "05_clusters.json").read_text())
    hierarchy = json.loads((INT / "05_hierarchy.json").read_text())
    posts = {}
    with (INT / "03_classified.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            posts[r["post_id"]] = r

    overrides_doc = json.loads(OVERRIDE_PATH.read_text())
    overrides = overrides_doc.setdefault("overrides", {})

    # ====== Step D: resort /leftover, /35, /2 ======
    resort_pids = []
    for leaf in ["sahoe/leftover", "sahoe/35", "sahoe/2"]:
        resort_pids.extend(_collect_leaf(clusters, hierarchy, leaf))
    resort_pids = [p for p in resort_pids if p in posts and posts[p].get("text")]
    print(f"[D] resort: {len(resort_pids)} posts (leftover+35+2)")

    # ====== Step A: hard moves + /34 split ======
    sahoe_8_pids = [p for p in _collect_leaf(clusters, hierarchy, "sahoe/8") if p in posts]
    sahoe_33_pids = [p for p in _collect_leaf(clusters, hierarchy, "sahoe/33") if p in posts]
    sahoe_34_pids = [p for p in _collect_leaf(clusters, hierarchy, "sahoe/34") if p in posts and posts[p].get("text")]
    print(f"[A] hard /8 → 일상 사건: {len(sahoe_8_pids)} posts")
    print(f"[A] hard /33 → 취미·문화: {len(sahoe_33_pids)} posts")
    print(f"[A] split /34 (사회 vs 일상 사건): {len(sahoe_34_pids)} posts")

    # ====== Step B: deep check /16, /15 ======
    depth_pids = []
    for leaf in ["sahoe/16", "sahoe/15"]:
        depth_pids.extend(_collect_leaf(clusters, hierarchy, leaf))
    depth_pids = [p for p in depth_pids if p in posts and posts[p].get("text")]
    print(f"[B] depth-check: {len(depth_pids)} posts (/16+/15)")

    # ===== Apply hard moves =====
    for p in sahoe_8_pids:
        overrides[p] = "일상 사건"
    for p in sahoe_33_pids:
        overrides[p] = "취미·문화"

    # ===== Run LLM in parallel =====
    def task(kind, pid):
        text = posts[pid]["text"]
        if kind == "resort":
            return kind, pid, resort(pid, text)
        if kind == "split34":
            return kind, pid, sahoe_vs_ilsang(pid, text)
        if kind == "depth":
            return kind, pid, depth_check(pid, text)

    jobs = []
    for p in resort_pids:
        jobs.append(("resort", p))
    for p in sahoe_34_pids:
        jobs.append(("split34", p))
    for p in depth_pids:
        jobs.append(("depth", p))
    print(f"[run] {len(jobs)} LLM calls (cached if previously seen)")

    results = {"resort": [], "split34": [], "depth": []}
    n_done = 0
    n_err = 0
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(task, k, p) for k, p in jobs]
        for fut in as_completed(futs):
            try:
                kind, pid, res = fut.result()
                results[kind].append((pid, res))
            except Exception as e:
                n_err += 1
                print(f"[err] {type(e).__name__}: {e}", flush=True)
                continue
            n_done += 1
            if n_done % 25 == 0:
                print(f"  {n_done}/{len(jobs)}", flush=True)

    # ===== Apply LLM results =====
    from collections import Counter
    rs_counts = Counter()
    for pid, res in results["resort"]:
        cat = res.get("category", "")
        if cat in CATEGORIES:
            overrides[pid] = cat
            rs_counts[cat] += 1
    print(f"[D] resort distribution:")
    for c, n in rs_counts.most_common():
        print(f"  {c}: {n}")

    sp_counts = Counter()
    for pid, res in results["split34"]:
        cat = res.get("category", "")
        if cat in {"사회", "일상 사건"}:
            overrides[pid] = cat
            sp_counts[cat] += 1
    print(f"[A] /34 split: {dict(sp_counts)}")

    dp_counts = Counter()
    for pid, res in results["depth"]:
        deep = bool(res.get("is_deep_reflection"))
        if deep:
            overrides[pid] = "인생·일상 감정"
            dp_counts["deep→인생·일상 감정"] += 1
        else:
            dp_counts["light→사회 유지"] += 1
    print(f"[B] depth: {dict(dp_counts)}")

    # ===== Save =====
    rules_log = overrides_doc.setdefault("_rules", [])
    rules_log.append({
        "_note": (
            "2026-05-01 사회 fine-tune Pass 1: D (leftover/35/2 resort) + "
            "A (/8→일상 사건, /33→취미·문화, /34 split) + B (/16,/15 deep check)"
        )
    })
    OVERRIDE_PATH.write_text(json.dumps(overrides_doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] errors={n_err}, overrides total={len(overrides)}")
    print(f"[saved] {OVERRIDE_PATH}")


if __name__ == "__main__":
    main()
