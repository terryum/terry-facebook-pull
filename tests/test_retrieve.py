"""Unit tests for `fbpull retrieve` helpers.

Heavy IO paths (loading embeddings, calling the embedding provider) are not
covered here — those are exercised by running `fbpull retrieve` against the
real vault. These tests cover the pure helpers."""

from fbpull import retrieve


def test_obsidian_open_uri_encodes_absolute_path(tmp_path):
    note = tmp_path / "vault" / "Private" / "Facebook" / "Archive" / "hello world.md"

    uri = retrieve._obsidian_open_uri(note)

    assert uri.startswith("obsidian://open?path=")
    assert "hello+world.md" in uri
    assert "%2F" in uri


def test_load_archive_note_index_matches_export_names(tmp_path, monkeypatch):
    int_dir = tmp_path / "_intermediate"
    archive_dir = tmp_path / "vault" / "Private" / "Facebook" / "Archive"
    int_dir.mkdir()

    (int_dir / "02_filtered.jsonl").write_text(
        "\n".join(
            [
                '{"post_id":"p2","date":"2024-01-01","text":"Hello world","kept":true}',
                '{"post_id":"p1","date":"2024-01-01","text":"Hello world","kept":true}',
                '{"post_id":"skip","date":"2024-01-01","text":"Skip","kept":false}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(retrieve, "intermediate_dir", lambda: int_dir)
    monkeypatch.setattr(retrieve, "archive_dir", lambda: archive_dir)

    index = retrieve._load_archive_note_index()

    assert index["p1"]["archive_note"] == "240101-hello-world"
    assert index["p2"]["archive_note"] == "240101-hello-world-2"
    assert index["p1"]["archive_relpath"] == "Private/Facebook/Archive/240101-hello-world.md"
    assert index["p1"]["obsidian_uri"].startswith("obsidian://open?path=")
    assert "skip" not in index


def test_ilsang_mid_for():
    assert retrieve._ilsang_mid_for("ilsang-sageon/diary/3") == "diary"
    assert retrieve._ilsang_mid_for("ilsang-sageon/exercise/12") == "exercise"
    assert retrieve._ilsang_mid_for("ilsang-sageon/canada") == "canada"
    assert retrieve._ilsang_mid_for("ilsang-sageon/leftover") is None
    assert retrieve._ilsang_mid_for("ilsang-sageon/9") is None
    assert retrieve._ilsang_mid_for("gajog/3") is None
    # old slug should no longer match
    assert retrieve._ilsang_mid_for("ilsang-saenghwal/diary/3") is None


def test_slugify_query():
    assert retrieve._slugify_query("대학원 진학 고민") == "대학원-진학-고민"
    assert retrieve._slugify_query("  multiple   spaces  ") == "multiple-spaces"
    # punctuation stripped
    assert retrieve._slugify_query("hello, world!") == "hello-world"
    # empty input falls back
    assert retrieve._slugify_query("   ") == "query"
    # length cap
    assert len(retrieve._slugify_query("가" * 200)) <= 60
