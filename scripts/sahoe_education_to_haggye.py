"""사회/education 안의 학계 메타비평 26개 → 학계 카테고리로 이동.

배경: leaf_groups 적용 시 옛 sahoe/2 (이두희+학계권력 mixed 25p) 가
재클러스터링 후 sahoe/education/{0,1} 로 흡수됐는데, 그 중 다수가
실은 사회 가 아니라 학계 (학계 시스템 메타비평·송유근·교수자리·표절·논문대필·
김박사넷·감동근 인물 메모 등) 카테고리에 속함.

이동 대상 26개 — 모두 학계 시스템 메타비평·학자 인물 메모 류:
  /0 (20p): 박사·교수자리, 송유근 표절, 전희경 논문 표절, 김인중 교수 인터뷰,
            연대 사제폭탄, 교수 레슨, 교수 가십, 배명진 PD수첩 (×2),
            논문 대필, 김박사넷, 송유근 방송, 썩은교수, 감동근, 어뷰징 학자,
            송유근 기자, 감동근×신영준 갈등, "대학원생 때 알았더라면", 전문연구요원,
            캐나다 워털루 vs 서울대
  /1 (6p): 언론 노출 학자들, 논문과외, 르쿤 학자 싸움, 의사 강연 집중,
           WASET 사이비 학회, 조선영 교수
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path

VAULT = Path("/Users/terrytaewoongum/Codes/personal/terry-obsidian/vault/Private/Facebook")
OVERRIDE = VAULT / "_classify_overrides.json"

# 26 posts → 학계
TO_HAGGYE: list[str] = [
    # /0 — 학계 메타비평 / 인물 / 송유근 / 표절 / 교수권력 (20p)
    "1430532567-59a97535da",  # 박사 "교수자리 다 자기들끼리..."
    "1448681305-09ba0a934f",  # 송유근 표절
    "1453376284-a9dc0b251e",  # 캐나다 워털루 다양성 vs 서울대
    "1460880250-b61df35801",  # 전희경 의원 석사논문 표절 79%
    "1488608213-318a88e019",  # 김인중 교수 / arXiv·NIPS·ICML 출판 사이클
    "1497356761-8cf389d0a7",  # 연대 대학원생 사제폭탄
    "1498097697-fe9e425d31",  # 유명 교수 얼굴도장·레슨
    "1525998511-e9d3b268ee",  # 교수 가십 문화
    "1527014443-6eef5d9cc2",  # 배명진 "소리박사" 진실
    "1527021069-1109263fd9",  # 배명진 PD수첩 요약
    "1528057375-d08ee8ea9b",  # 논문컨설팅·석박사 대필
    "1538187884-2f8a8f2ac6",  # 김박사넷
    "1540196725-d2b95d9e24",  # 송유근 또 방송
    "1541546728-35647098fe",  # 썩은 교수들 / 학계 망할 것
    "1543711727-e23c1f81b9",  # 감동근 교수 / 글쓰기 태도
    "1544283248-5d49b42e4e",  # 어뷰징 학자들 / 동료 침묵
    "1544448216-eeabd59998",  # 송유근 "교수 10번 떨어진" 기자
    "1560097761-1a65c77120",  # 감동근 × 신영준 갈등
    "1561817093-c9b1322da6",  # "대학원생 때 알았더라면" 4쇄
    "1566723989-1fc2e8f4e7",  # 전문연구요원제 폐지·박사급 의무복무

    # /1 — 학계 메타비평 (6p)
    "1433434542-cf68cee7bc",  # 언론에 자신을 노출하려는 학자들의 위험성
    "1489973470-f58d3a3817",  # 논문과외 1:1 멘토링
    "1512768933-b9b617e857",  # 르쿤 / 학자들 싸움
    "1521923952-1dae4ae8d0",  # 의사들이 강연에 집중 못함 / 과학과 사회
    "1532049194-c307ded9a1",  # WASET 사이비 학회
    "1543679513-31b1a36571",  # 조선영 교수 / 평범한 교수
]


def main() -> None:
    doc = json.loads(OVERRIDE.read_text(encoding="utf-8"))

    bak = OVERRIDE.with_name(
        OVERRIDE.name + ".bak-2026-05-01-sahoe-education-to-haggye"
    )
    shutil.copy(OVERRIDE, bak)
    print(f"[backup] {bak.name}")

    overrides = doc.setdefault("overrides", {})
    n_changed = 0
    for pid in TO_HAGGYE:
        prev = overrides.get(pid)
        if prev != "학계":
            overrides[pid] = "학계"
            n_changed += 1
    print(f"[move] {n_changed}/{len(TO_HAGGYE)} posts → 학계 (rest were already)")

    # leaf_groups[sahoe][education] 안에서도 빼주기 (정리 차원)
    edu = doc.get("leaf_groups", {}).get("sahoe", {}).get("education")
    if isinstance(edu, list):
        before = len(edu)
        new_edu = [p for p in edu if p not in set(TO_HAGGYE)]
        doc["leaf_groups"]["sahoe"]["education"] = new_edu
        print(f"[clean] leaf_groups[sahoe][education]: {before} → {len(new_edu)}")

    doc.setdefault("_rules", []).append({
        "date": "2026-05-01",
        "action": "사회/education → 학계 이동",
        "detail": (
            f"sahoe/education 47p 중 {len(TO_HAGGYE)}p 가 학계 메타비평 "
            f"(송유근·교수자리·표절·논문대필·김박사넷·인물 메모) 으로 분류되어 "
            f"학계 카테고리로 이동. 잔여 21p (수능·입시·이대사태·사교육 등) 는 "
            f"사회 유지."
        ),
    })
    doc["_generated"] = datetime.utcnow().isoformat() + "Z"

    OVERRIDE.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[saved] overrides total={len(overrides)}")


if __name__ == "__main__":
    main()
