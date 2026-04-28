"""Parse the user-editable _taxonomy.md in the vault.

The file controls how classify/cluster/synthesize behave:
- Bio    : context string passed to LLMs during synthesis
- Eras   : year ranges → era labels (deterministic mapping at classify time)
- Coverage gradient : free-form notes about which (era × category) cells are
  rich vs sparse; surfaced verbatim in Synthesized prompts so the model knows
  why a cluster might be small
- Categories : closed list that Haiku assigns each post to. [SENSITIVE] /
  [STRICT] flags in the heading propagate through the pipeline.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

from slugify import slugify

from .paths import fb_root


@dataclass
class Category:
    name: str          # "연구·학술"
    slug: str          # "yeongu-hagsul"
    description: str
    sensitive: bool = False
    strict: bool = False


@dataclass
class Era:
    label: str         # "박사과정 (Waterloo, 딥러닝)"
    start_year: int
    end_year: int      # inclusive


@dataclass
class Taxonomy:
    bio: str
    eras: list[Era]
    coverage_gradient: str
    categories: list[Category]
    raw_text: str

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()[:8]

    def category_by_name(self, name: str) -> Category | None:
        for c in self.categories:
            if c.name == name:
                return c
        return None

    def fallback_category(self) -> Category:
        for c in self.categories:
            if "기타" in c.name or "미분류" in c.name:
                return c
        return self.categories[-1] if self.categories else Category("기타", "etc", "")

    def era_for_year(self, year: int) -> str:
        for era in self.eras:
            if era.start_year <= year <= era.end_year:
                return era.label
        return "unknown"

    def category_names_for_prompt(self) -> str:
        """Compact list of category names + flags for inclusion in Haiku system prompt."""
        out = []
        for c in self.categories:
            flag = ""
            if c.strict:
                flag = " [STRICT]"
            elif c.sensitive:
                flag = " [SENSITIVE]"
            short = c.description.split("\n")[0].strip()
            out.append(f"- {c.name}{flag}: {short}")
        return "\n".join(out)


def taxonomy_path() -> Path:
    return fb_root() / "_taxonomy.md"


def load() -> Taxonomy | None:
    p = taxonomy_path()
    if not p.exists():
        return None
    return parse(p.read_text(encoding="utf-8"))


def parse(text: str) -> Taxonomy:
    body = _strip_frontmatter(text)
    sections = _split_h1(body)
    eras = _parse_eras(sections.get("Eras", ""))
    categories = _parse_categories(sections.get("Categories", ""))
    return Taxonomy(
        bio=sections.get("Bio", "").strip(),
        eras=eras,
        coverage_gradient=sections.get("Coverage gradient", "").strip(),
        categories=categories,
        raw_text=body,
    )


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    end = text.find("---", 3)
    if end < 0:
        return text
    return text[end + 3 :].lstrip()


def _split_h1(text: str) -> dict[str, str]:
    """Return dict of {h1 title: body up to next h1}."""
    matches = list(re.finditer(r"^# (.+?)$", text, re.MULTILINE))
    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        out[title] = text[start:end].strip()
    return out


_ERA_LINE = re.compile(
    r"^\s*[-*]\s*(\d{4})\s*[–\-~]\s*(\d{4})?\s*:?\s*(.+?)\s*$"
)


def _parse_eras(text: str) -> list[Era]:
    found: list[tuple[int, int | None, str]] = []
    for line in text.splitlines():
        m = _ERA_LINE.match(line)
        if m:
            start = int(m.group(1))
            end_str = m.group(2)
            end = int(end_str) if end_str else None
            label = m.group(3).strip()
            found.append((start, end, label))
    found.sort(key=lambda t: t[0])

    eras: list[Era] = []
    for i, (start, end, label) in enumerate(found):
        if end is None:
            end = (found[i + 1][0] - 1) if i + 1 < len(found) else 9999
        # If consecutive eras share a boundary year, the earlier era ends one
        # year before the next starts (FB exports record discrete years and
        # the user writes overlapping ranges like "2010-2014" / "2014-2018").
        if i + 1 < len(found) and end >= found[i + 1][0]:
            end = found[i + 1][0] - 1
        eras.append(Era(label=label, start_year=start, end_year=end))
    return eras


_FLAG = re.compile(r"\s*\[(SENSITIVE|STRICT)\]\s*")


def _parse_categories(text: str) -> list[Category]:
    matches = list(re.finditer(r"^## (.+?)$", text, re.MULTILINE))
    out: list[Category] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()

        sensitive = "[SENSITIVE]" in heading
        strict = "[STRICT]" in heading
        clean = _FLAG.sub("", heading).strip()
        slug = slugify(clean, allow_unicode=False, max_length=40) or clean
        out.append(
            Category(
                name=clean,
                slug=slug,
                description=body,
                sensitive=sensitive,
                strict=strict,
            )
        )
    return out
