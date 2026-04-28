SAMPLE = """---
type: taxonomy
---

# Bio

A short bio.

# Eras

- 2010–2014: Phase A
- 2014–2018: Phase B
- 2024–:     Phase C

# Coverage gradient

## 2010–2018: rich
Lots of writing.

# Categories

## Cat One
First category.

## Cat Two [SENSITIVE]
Sensitive category.

## Cat Three [STRICT]
Strict category.
"""


def test_parse_basic():
    from fbpull.taxonomy import parse

    tax = parse(SAMPLE)
    assert tax.bio == "A short bio."
    assert "rich" in tax.coverage_gradient


def test_eras_adjusted():
    from fbpull.taxonomy import parse

    tax = parse(SAMPLE)
    assert len(tax.eras) == 3
    # 2014 belongs to Phase B (the later era), so Phase A ends 2013
    assert tax.eras[0].end_year == 2013
    assert tax.era_for_year(2013) == "Phase A"
    assert tax.era_for_year(2014) == "Phase B"
    # Open-ended era extends to "current"
    assert tax.era_for_year(2030) == "Phase C"


def test_category_flags():
    from fbpull.taxonomy import parse

    tax = parse(SAMPLE)
    assert len(tax.categories) == 3
    assert tax.categories[0].name == "Cat One"
    assert not tax.categories[0].sensitive
    assert not tax.categories[0].strict
    assert tax.categories[1].name == "Cat Two"
    assert tax.categories[1].sensitive
    assert not tax.categories[1].strict
    assert tax.categories[2].strict


def test_hash_stable():
    from fbpull.taxonomy import parse

    a = parse(SAMPLE).hash
    b = parse(SAMPLE).hash
    assert a == b
    assert len(a) == 8


def test_fallback_category():
    from fbpull.taxonomy import parse

    fallback_only = """# Categories

## 기타·미분류
catch-all
"""
    tax = parse(fallback_only)
    assert tax.fallback_category().name == "기타·미분류"
