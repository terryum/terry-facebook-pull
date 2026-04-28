def test_reason_for_too_short():
    from fbpull.filter import reason_for

    assert reason_for({"text": "ㅋㅋ"}) == "too_short"


def test_reason_for_empty():
    from fbpull.filter import reason_for

    assert reason_for({"text": ""}) == "empty"
    assert reason_for({"text": None}) == "empty"


def test_reason_for_link_only():
    from fbpull.filter import reason_for

    assert reason_for({"text": "https://example.com"}) == "link_only"
    assert reason_for({"text": "  https://a.com https://b.com  "}) == "link_only"


def test_reason_for_keeps_long():
    from fbpull.filter import reason_for

    long = "이것은 충분히 긴 글입니다. " * 5
    assert reason_for({"text": long}) is None
