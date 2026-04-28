import json
import shutil
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "sample_fb_export.json"


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("OBSIDIAN_VAULT", str(tmp_path))
    from fbpull.paths import ensure_dirs, raw_dir

    ensure_dirs()
    shutil.copy(FIXTURE, raw_dir() / "posts_2014.json")
    return tmp_path


def test_normalize_one_basic():
    from fbpull.parse import normalize_one

    raw = {
        "timestamp": 1388534400,
        "data": [{"post": "Hello world this is a long enough post."}],
        "title": None,
    }
    rec = normalize_one(raw, "posts.json")
    assert rec is not None
    assert rec["timestamp"] == 1388534400
    assert rec["date"] == "2014-01-01"
    assert "Hello world" in rec["text"]
    assert rec["post_id"].startswith("1388534400-")


def test_normalize_one_no_timestamp():
    from fbpull.parse import normalize_one

    assert normalize_one({"data": [{"post": "x"}]}, "f.json") is None


def test_fix_mojibake_passes_normal_text():
    from fbpull.parse import fix_mojibake

    assert fix_mojibake("hello") == "hello"
    assert fix_mojibake("") == ""


def test_run_writes_jsonl_and_dedupes(vault: Path):
    from fbpull.parse import run
    from fbpull.paths import intermediate_dir, raw_dir

    # Add a duplicate file that should not double-count
    shutil.copy(FIXTURE, raw_dir() / "posts_2014_dup.json")

    n = run()
    assert n == 15  # fixture has 15 entries

    out = intermediate_dir() / "01_parsed.jsonl"
    assert out.exists()
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 15

    rec = json.loads(lines[0])
    assert {"post_id", "date", "timestamp", "text", "links", "source_path"} <= rec.keys()
