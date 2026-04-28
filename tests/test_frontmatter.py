def test_dump_basic():
    from fbpull.frontmatter import dump

    out = dump({"type": "archive", "tags": ["a", "b"]}, "Hello\n")
    assert out.startswith("---\n")
    assert "type: archive" in out
    assert "tags:" in out
    assert "Hello" in out
    assert out.endswith("\n")


def test_dump_unicode():
    from fbpull.frontmatter import dump

    out = dump({"title": "한국어"}, "본문\n")
    assert "한국어" in out
    assert "본문" in out


def test_write_note(tmp_path):
    from fbpull.frontmatter import write_note

    p = tmp_path / "sub" / "note.md"
    write_note(p, {"type": "memo"}, "body")
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "type: memo" in txt
    assert "body" in txt
