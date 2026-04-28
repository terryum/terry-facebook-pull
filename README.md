# fbpull

Facebook DYI export → Obsidian "외장 기억" 파이프라인.

오래된 Facebook 글 archive 를 입력으로 받아, 의미 있는 글만 골라 분류하고, 임베딩 기반으로 비슷한 글들을 묶어 옵시디언 vault 에 마크다운으로 떨어뜨립니다.

이 레포는 **공개 가능한 코드만** 담습니다. 실제 Facebook 데이터·임베딩·생성된 노트는 모두 별도의 비공개 옵시디언 vault 에 저장됩니다.

## What it produces

옵시디언 vault 의 `Private/Facebook/` 아래에:

- **Archive/`YYMMDD-<slug>.md`** — Facebook 원문 1편당 1파일 (수정/요약 없음)
- **Synthesized/`<concept-slug>.md`** — 반복되는 사고를 합성한 컨셉 노트 (cluster 1개당 1파일)
- 각 Archive 노트에는 "비슷한 글" wiki link 자동 삽입 → 옵시디언 그래프 뷰에서 인생/에세이 그래프 형성

## Pipeline

```
parse → filter → classify → embed → cluster → synthesize → export
```

| Stage | 역할 |
|-------|------|
| parse | FB JSON → 정규화된 record (jsonl) |
| filter | 짧은 글/링크/공유 등 노이즈 제거 |
| classify | Haiku 가 thought/lesson/event/quote/announcement 로 라벨링, 합성 후보 선별 |
| embed | OpenAI 또는 Voyage 임베딩 |
| cluster | HDBSCAN 으로 진짜 반복 사고만 묶기 + 코사인 top-N 이웃 그래프 |
| synthesize | Sonnet 이 cluster → 컨셉 노트 작성 |
| export | Archive/Synthesized 마크다운 작성 |

자세한 데이터 스키마는 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) 참조.

## Setup

### 1. Get the Facebook archive

API/토큰으로는 못 받습니다. Graph API 의 `user_posts` 권한이 막혀 있고 본인 timeline 의 과거 글은 토큰으로도 못 받음. **Facebook DYI (Download Your Information)** 만 동작합니다:

1. https://www.facebook.com/dyi 접속 (로그인 필요)
2. "Request a copy" → **Format: JSON** (HTML 아님), Media quality: Low
3. 카테고리: **Posts** 선택
4. Date range: All time
5. Request → Facebook 이 처리 후 이메일로 알림 (수 분 ~ 몇 시간)
6. zip 다운로드 → `your_posts_*.json` 추출
7. 파일을 옵시디언 vault 의 `vault/Private/Facebook/_raw/posts_<YYMMDD>.json` 에 복사

### 2. Install

```bash
# uv (recommended)
uv sync

# 또는 pip
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Configure

```bash
cp .env.example .env
# .env 편집: OBSIDIAN_VAULT, ANTHROPIC_API_KEY, OPENAI_API_KEY 채우기
```

### 4. Run

```bash
fbpull all                        # 전체 파이프라인
fbpull parse                      # 단계별 디버깅도 가능
fbpull all --no-llm               # 결정적 stub 으로 (오프라인/CI)
```

## Costs

Anthropic + OpenAI 임베딩 호출은 모두 캐시됩니다 (`vault/Private/Facebook/_intermediate/{embed,llm}_cache/`). 같은 글에 대해 다시 안 부릅니다. 한 번 처리한 archive 를 재처리해도 비용 0.

대략적인 1회 비용 (10년치 ≈ 5,000 글 가정):
- Haiku classify: ~$0.50
- OpenAI embed (3-small): ~$0.05
- Sonnet synthesize (50 cluster): ~$3
- 합계: $5 미만

## Privacy

- 이 레포 (`terry-facebook-pull`): **공개**. 코드/문서만. `.gitignore` 가 데이터/시크릿 차단
- 옵시디언 vault (`terry-obsidian`): **비공개**. raw/intermediate/Archive/Synthesized 모두 거기 보관

테스트 fixture (`tests/fixtures/`) 는 가짜 글로만 구성됩니다. 실제 사용자 글은 절대 이 레포에 들어가지 않습니다.

## See also

- [`docs/PRD.md`](docs/PRD.md) — 왜 만드는가, Non-Goals, 성공 기준
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 데이터 스키마, 캐시 키, 멱등성 규칙
- [`CLAUDE.md`](CLAUDE.md) — 이 레포에서 코드 작업 시 따라야 할 원칙 (Claude/사람 모두)
