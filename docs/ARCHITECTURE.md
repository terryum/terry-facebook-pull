# Architecture

이 문서는 PRD 의 의도를 실제 코드로 어떻게 구현하는지 — 데이터 스키마, stage 별 입출력, 캐시 키, 멱등성 규칙, 공개 repo 안전성 — 을 정리한다.

## 1. Two-repo separation

| Repo | 가시성 | 내용 |
|------|--------|------|
| `terry-facebook-pull` | **공개** | Python 코드, 문서, 가짜 fixture 만 |
| `terry-obsidian` | **비공개** | 모든 실제 데이터: raw, intermediate, Archive, Synthesized |

이 분리는 `.gitignore` 와 디렉터리 배치로 강제한다. 코드는 `OBSIDIAN_VAULT` 환경변수로 vault 경로를 받아 거기에만 쓴다 — 코드 레포 내부에는 어떤 데이터도 떨어뜨리지 않는다.

## 2. Directory layout

### `terry-facebook-pull/` (코드 레포)

```
src/fbpull/
├── __init__.py
├── cli.py            # typer 엔트리포인트
├── paths.py          # OBSIDIAN_VAULT/vault/Private/Facebook 경로 resolver
├── parse.py          # stage 1
├── filter.py         # stage 2
├── classify.py       # stage 3
├── embed.py          # stage 4
├── cluster.py        # stage 5 (HDBSCAN + 코사인 이웃)
├── synthesize.py     # stage 6
├── export.py         # stage 7
├── frontmatter.py    # YAML frontmatter 직렬화
└── llm.py            # Anthropic 호출 + 캐시
```

### `terry-obsidian/vault/Private/Facebook/` (데이터)

```
_raw/                       # 사용자가 FB DYI export 를 여기에 떨굼
  posts_<YYMMDD>.json

_intermediate/              # 파이프라인 중간 산출물
  01_parsed.jsonl
  02_filtered.jsonl
  03_classified.jsonl
  04_embeddings.npy
  04_post_ids.json
  05_clusters.json
  05_neighbors.json
  06_synthesized.jsonl
  embed_cache/<model>/<text_hash>.json
  llm_cache/<model>/<post_id>_<text_hash>.json

Archive/<YYMMDD>-<slug>.md  # post 1편당 1파일
Synthesized/<slug>.md       # cluster 1개당 1파일
_index.md                   # 통계 + 합성 노트 목록
```

## 3. Stage pipeline

각 stage 는 **하나의 일만** 한다 (CLAUDE.md §2). stage 는 단독 실행 가능하고, 입력 파일이 outdated 면 경고만 출력한다 (auto-rerun 없음 — 명시적 호출만).

### Stage 1 — parse

**입력**: `_raw/*.json` (FB DYI export)
**출력**: `_intermediate/01_parsed.jsonl`
**Record schema**:
```json
{
  "post_id": "string",          // FB timestamp 또는 hash 기반 안정적 식별자
  "date": "YYYY-MM-DD",         // 게시일 (FB timestamp → 날짜)
  "timestamp": 1234567890,      // 원본 unix epoch
  "text": "string",             // 본문 (mojibake 복원: latin-1 → utf-8)
  "links": ["url1", "url2"],    // attachments 의 외부 URL
  "source_path": "string"       // 어느 _raw 파일에서 왔는지
}
```

**규칙**:
- FB 의 UTF-8 mojibake 알려진 이슈 — `text.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')` 로 복원
- 같은 post_id 는 dedupe (여러 export 를 합쳐도 안전)
- post_id 는 (1) 원본에 있는 ID, 없으면 (2) `f"{timestamp}-{hashlib.sha256(text[:200].encode()).hexdigest()[:10]}"`

### Stage 2 — filter

**입력**: `01_parsed.jsonl`
**출력**: `02_filtered.jsonl` (모든 record 유지하되 `kept`, `drop_reason` 필드 추가)

**Heuristic**:
- `len(text) < 60` → `kept=false`, `drop_reason="too_short"`
- `text` 가 URL 만 (regex 매치) → `drop_reason="link_only"`
- `text` 가 "Likes a post by ..." / "Shared a memory" 같은 시스템 메시지 → `drop_reason="system"`
- `links` 만 있고 `text` 가 비어 있음 → `drop_reason="link_only"`

이 단계는 결정적 — LLM/외부 호출 없음.

### Stage 3 — classify

**입력**: `02_filtered.jsonl` 의 `kept=true` record
**출력**: `03_classified.jsonl` (kept 만, classify 결과 추가)
**모델**: `claude-haiku-4-5` (env override 가능)
**Cache key**: `llm_cache/{model}/{post_id}_{sha256(text)[:16]}.json`

**Per-post 호출 (1 콜/post)**:
- 입력: post text
- 출력 (JSON): `{"type": "thought|lesson|event|quote|announcement", "primary_topic": "string (단어 1–3개)", "keep_for_synthesis": bool}`
- `keep_for_synthesis=false` 면 Archive 에는 들어가지만 cluster/synthesize 대상에서 제외 (예: 이벤트, 공지)

### Stage 4 — embed

**입력**: `03_classified.jsonl` 의 `keep_for_synthesis=true` record
**출력**:
- `04_embeddings.npy`: shape `(N, D)` float32, **L2-normalized** (코사인 거리 = 유클리드 거리 × 0.5)
- `04_post_ids.json`: `[post_id, ...]` 길이 N — 행 순서와 1:1 매칭
**모델**: `text-embedding-3-small` (OpenAI, 기본) 또는 `voyage-3` (Voyage)
**Cache key**: `embed_cache/{model}/{sha256(text)[:16]}.json`

배치 100, retry 3회.

### Stage 5 — cluster

**입력**: `04_embeddings.npy`, `04_post_ids.json`
**출력**:
- `05_clusters.json`: `{"<cluster_id>": ["post_id", ...]}` (cluster_id `-1` = HDBSCAN 노이즈)
- `05_neighbors.json`: `{"<post_id>": [{"id": "...", "score": 0.83}, ...]}` (top-5)

**알고리즘**:
- **클러스터링**: `sklearn.cluster.HDBSCAN(min_cluster_size=4, metric='euclidean')`
  - L2-normalized 벡터 위에서 유클리드 = √(2−2·cos) → 코사인 클러스터링과 동일
  - `min_cluster_size=4` — 최소 4개의 비슷한 글이 모여야 합성. 1–2개는 노이즈로 취급
- **이웃**: `sklearn.metrics.pairwise.cosine_similarity` 로 N×N → 각 행에서 자기 제외 top-5, score > 0.55

외부 호출 없음. 결정적.

### Stage 6 — synthesize

**입력**: `05_clusters.json` (cluster_id != -1) + `02_filtered.jsonl` 멤버 텍스트
**출력**: `06_synthesized.jsonl`
**모델**: `claude-sonnet-4-6` (env override 가능)
**Cache key**: `llm_cache/{model}/cluster_{cluster_id}_{sha256(member_post_ids_sorted)[:16]}.json`

**Per-cluster 호출 (1 콜/cluster)**:
- 입력: cluster 멤버들의 텍스트 (concat, 각 글에 날짜 prefix)
- 출력 (JSON):
  ```json
  {
    "title": "string (10–30자)",
    "slug": "string (kebab-case, 영문)",
    "body": "string (300–600자, 마크다운)",
    "primary_tag": "string (단어 1개)"
  }
  ```
- 본문 은 "여러 글에서 반복되는 핵심 사고를 한 문단으로 압축" 지시

### Stage 7 — export

**입력**: `02_filtered.jsonl` (kept) + `05_clusters.json` + `05_neighbors.json` + `06_synthesized.jsonl`
**출력**: `Archive/*.md`, `Synthesized/*.md`, `_index.md`

**규칙 (CLAUDE.md §4: 절대 raw 데이터 수정 안 함)**:
- 기존 `Archive/` 파일이 있으면 frontmatter 만 갱신 (`cluster_id`, `tags`), 본문은 안 건드림
- 새 cluster 멤버가 들어오면 "비슷한 글" 섹션만 다시 씀
- Synthesized 노트는 cluster_id 가 동일하면 슬러그 유지 — 사용자가 옮겨도 안 깨지게

**Frontmatter:**

Archive (`Archive/YYMMDD-<slug>.md`):
```yaml
---
type: archive
source: facebook
visibility: private
created_at: 2014-08-12          # 원문 작성일
fb_post_id: "10152..."
cluster_id: 7                   # null 가능
tags: [facebook, archive, <primary_topic>]
---

(원문 그대로)

## 비슷한 글
- [[150103-<slug>]]
- ...
```

Synthesized (`Synthesized/<slug>.md`):
```yaml
---
type: synthesized
source: facebook
visibility: private
created_at: 2026-04-28
cluster_id: 7
member_count: 12
tags: [facebook, synthesized, <primary_tag>]
---

# <Title>

(Sonnet 합성 본문)

## 멤버 글
- [[140812-<slug>]]
- ...
```

## 4. Idempotency

- **결정적 정렬**: 모든 jsonl 은 `post_id` 오름차순으로 정렬해서 출력 → diff 가 안정적
- **캐시 키**는 `(model, post_id, text_hash)` 또는 `(model, text_hash)` — 같은 입력 → 같은 캐시 hit
- **HDBSCAN 결정성**: random_state 고정 (sklearn HDBSCAN 은 결정적)
- **Slug 충돌**: 같은 날짜·같은 시작 단어 두 글 → `YYMMDD-<slug>-2.md` 로 suffix
- **부분 재실행**: stage N 만 다시 돌려도 N+1 이후가 자동으로 outdated 표시되며 사용자가 명시적으로 재실행

## 5. Public-repo safety

코드 레포가 GitHub 에 공개될 때 보장:

- `.gitignore` 가 `*.jsonl`, `*.npy`, `data/`, `raw/`, `.env`, `posts_*.json`, `your_posts*.json` 차단
- `tests/fixtures/*.json` 만 화이트리스트 — 가짜 데이터, 인공 텍스트 ("오늘 카페에서 책을 읽었다" 같은 일반 문장)
- 코드 안 어디에도 OBSIDIAN_VAULT, ANTHROPIC_API_KEY 의 디폴트 값을 하드코딩하지 않음 (모두 env)
- 에러 메시지에 사용자 텍스트를 넣지 않음 (예: "Failed to parse: <text>" → "Failed to parse post_id=...")

배포 전 검증 명령:
```bash
git -C terry-facebook-pull ls-files | grep -E '\.(jsonl|npy|env)$'  # 결과 비어야 함
```

## 6. Future hooks

- 임베딩 provider 교체 (자체 로컬 LLM): `embed.py` 의 `embed_texts()` 함수만 교체. 출력 shape 만 동일하면 됨
- Comments / Notes 추가 입력: `parse.py` 가 `_raw/comments_*.json`, `_raw/your_notes*.json` 도 받도록 확장. 다른 stage 는 schema 동일이라 변경 불필요
- RAG 도입: `04_embeddings.npy` + `04_post_ids.json` 그대로 vector store 에 로드 가능
