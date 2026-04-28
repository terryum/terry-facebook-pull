# Facebook Thought Integration PRD

## 1. Objective

Facebook에 축적된 과거 글들을 "외장 기억 시스템"으로 통합하여,
AI가 개인의 사고 패턴, 철학, 의사결정 기준을 기반으로 고품질 응답을 생성하도록 한다.

핵심 목표:
- 과거 사고 → 현재 글쓰기/의사결정에 연결
- 반복된 생각을 "개념 단위"로 압축
- AI가 빠르게 retrieve 가능한 구조 구축

---

## 2. Core Principle

- Raw ≠ Knowledge
- Knowledge = 반복된 생각의 압축

따라서 구조는 다음과 같다:

Facebook Raw → Filtering → Clustering → Synthesis → Obsidian Integration

---

## 3. System Overview

[Facebook JSON Export]
    ↓
[Heuristic Filter]
    ↓
[LLM Classification]
    ↓
[Embedding + Clustering]
    ↓
[Synthesized Thought Generation]
    ↓
[Obsidian Vault Integration]

---

## 4. Data Layers

### 4.1 Archive (Raw Layer)
- Facebook 원문 저장
- 날짜 기반
- 절대 수정하지 않음

### 4.2 Synthesized (Concept Layer)
- 여러 글을 묶은 "사고 단위"
- AI가 주로 사용하는 레이어

### 4.3 Index (Navigation Layer)
- 전체 사고 맵
- 빠른 retrieval entry point

---

## 5. Obsidian Integration

Private/Facebook/
├── _raw/                  # 원본 export (사용자가 떨굼)
├── _intermediate/         # 파이프라인 중간 산출물 + 임베딩 + 캐시
├── Archive/               # post 1편당 1파일. 본문은 원문 그대로 + "비슷한 글" wiki link
├── Synthesized/           # 합성된 컨셉 노트 (HDBSCAN 클러스터당 1파일)
└── _index.md              # 통계 + Synthesized 목록

Public/Essays/
→ Synthesized 기반 재작성 (이 repo 의 책임 아님; 향후 별도 작업)

---

## 6. Retrieval Strategy (이 repo 의 scope 아님)

Synthesized + Archive 마크다운이 **source of truth**. 옵시디언/AI 도구 측에서:

- Graph view + tag 필터 (`tag:#facebook`) 로 인생 그래프 시각화
- 향후 RAG/벡터 검색 도입 시 `_intermediate/04_embeddings.npy` + `04_post_ids.json` 재활용 가능
- 자체 LLM 로컬 학습으로 임베딩을 교체할 경우 `embed.py` provider 만 바꾸면 동일 파이프라인 동작

---

## 7. Non-Goals

- Graph DB 구축
- 완벽한 분류 체계 설계
- 모든 글의 완벽한 요약

---

## 8. Success Criteria

- "내가 과거에 어떻게 생각했지?" 질문에 일관된 답 가능
- 동일 주제에서 시간에 따른 사고 변화 추적 가능
- Synthesized 노트 수: 20~100개 수준 유지

---

## 9. Constraints

- 비용 최소화 (Haiku/Sonnet 혼용)
- Over-engineering 금지
- Markdown을 source of truth로 유지

---

## 10. Future Extension

- Comments 데이터 추가
- Blog / Notes 통합
- Cross-source synthesis