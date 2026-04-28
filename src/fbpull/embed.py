import hashlib
import json
import os
from typing import Callable

import httpx
import numpy as np

from . import llm
from .paths import intermediate_dir


def _embed_openai(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    r = httpx.post(
        "https://api.openai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"input": texts, "model": model},
        timeout=60.0,
    )
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


def _embed_voyage(texts: list[str], model: str, api_key: str) -> list[list[float]]:
    r = httpx.post(
        "https://api.voyageai.com/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"input": texts, "model": model, "input_type": "document"},
        timeout=60.0,
    )
    r.raise_for_status()
    return [d["embedding"] for d in r.json()["data"]]


def _embed_stub(texts: list[str]) -> list[list[float]]:
    out: list[list[float]] = []
    for t in texts:
        seed = int(hashlib.sha256(t.encode("utf-8")).hexdigest()[:8], 16) % (2**32)
        rng = np.random.RandomState(seed)
        out.append(rng.randn(64).astype(np.float32).tolist())
    return out


def _select_provider(no_llm: bool) -> tuple[str, Callable[[list[str]], list[list[float]]]]:
    if no_llm:
        return "stub-64", _embed_stub
    if os.environ.get("OPENAI_API_KEY"):
        model = os.environ.get("FBPULL_EMBED_MODEL", "text-embedding-3-small")
        key = os.environ["OPENAI_API_KEY"]
        return model, lambda b: _embed_openai(b, model, key)
    if os.environ.get("VOYAGE_API_KEY"):
        model = os.environ.get("FBPULL_EMBED_MODEL", "voyage-3")
        key = os.environ["VOYAGE_API_KEY"]
        return model, lambda b: _embed_voyage(b, model, key)
    raise RuntimeError("Set OPENAI_API_KEY or VOYAGE_API_KEY (or use --no-llm).")


def _normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def run(no_llm: bool = False, batch: int = 100) -> int:
    in_path = intermediate_dir() / "03_classified.jsonl"
    if not in_path.exists():
        raise FileNotFoundError(f"Run `fbpull classify` first; missing {in_path}")

    posts: list[dict] = []
    with in_path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("keep_for_synthesis"):
                posts.append(rec)

    if not posts:
        print("[embed] 0 posts to embed")
        # Still write empty arrays so downstream can detect "no data"
        np.save(intermediate_dir() / "04_embeddings.npy", np.zeros((0, 64), dtype=np.float32))
        (intermediate_dir() / "04_post_ids.json").write_text("[]", encoding="utf-8")
        return 0

    model, provider = _select_provider(no_llm)
    cache_dir = intermediate_dir() / "embed_cache" / model
    cache_dir.mkdir(parents=True, exist_ok=True)

    vectors_by_id: dict[str, list[float]] = {}
    missing: list[tuple[str, str]] = []  # (post_id, text)

    for p in posts:
        h = llm.text_hash(p["text"])
        cf = cache_dir / f"{h}.json"
        if cf.exists():
            vectors_by_id[p["post_id"]] = json.loads(cf.read_text(encoding="utf-8"))["v"]
        else:
            missing.append((p["post_id"], p["text"]))

    for i in range(0, len(missing), batch):
        chunk = missing[i : i + batch]
        chunk_texts = [t for _, t in chunk]
        new_vecs = provider(chunk_texts)
        for (pid, text), v in zip(chunk, new_vecs):
            h = llm.text_hash(text)
            (cache_dir / f"{h}.json").write_text(
                json.dumps({"v": v, "model": model}, ensure_ascii=False),
                encoding="utf-8",
            )
            vectors_by_id[pid] = v

    arr = np.array([vectors_by_id[p["post_id"]] for p in posts], dtype=np.float32)
    arr = _normalize(arr)

    np.save(intermediate_dir() / "04_embeddings.npy", arr)
    post_ids = [p["post_id"] for p in posts]
    (intermediate_dir() / "04_post_ids.json").write_text(
        json.dumps(post_ids, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[embed] {len(posts)} posts × {arr.shape[1]} dim ({model})")
    return len(posts)
