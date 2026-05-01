"""Unit tests for `fbpull retrieve` helpers.

Heavy IO paths (loading embeddings, calling the embedding provider) are not
covered here — those are exercised by running `fbpull retrieve` against the
real vault. These tests cover the pure helpers."""

from fbpull import retrieve


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
