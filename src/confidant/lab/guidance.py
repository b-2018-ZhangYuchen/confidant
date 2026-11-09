"""A relationship-guidance corpus for retrieval: what healthy and unhealthy dating looks like.

The first RAG design retrieved from 11 short pattern cards written for this project. That
is barely retrieval, and the cards say little about what *ordinary* dating looks like,
which is what a model that over-flags terse replies most needs. This corpus is relationship
guidance written by others, for people who are dating:

- **US government health pages** (womenshealth.gov, youth.gov, CDC, HHS OPA): public domain.
- **Wikipedia** articles on relationships, communication, boundaries, and consent: CC BY-SA.

wikiHow was planned too and dropped: it blocks Anthropic's crawler, and that is taken as
the site owner's answer for any AI-driven collection, whatever the user agent.

:func:`build_guidance` fetches each page, keeps the body text (paragraphs, list items,
headings), and cuts it two ways, so chunking can be compared: by **paragraph** and into
**windows** of about 200 words. Passages that use a diagnostic label
(:data:`confidant.evals.DIAGNOSTIC_TERMS`) are dropped, because Confidant's principles rule
those labels out and retrieved text is the easiest way for one to reach a report. Every
passage keeps its source, URL, and license. The corpus is written to ``lab-cache/`` and
rebuilt from this list rather than committed.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

__all__ = ["GUIDANCE_DIR", "SOURCES", "Passage", "build_guidance", "load_passages"]

GUIDANCE_DIR = Path("lab-cache/guidance")
USER_AGENT = "confidant-lab/0.1 (https://github.com/b-2018-ZhangYuchen/confidant)"

PUBLIC = "public domain (US government)"
WIKI = "CC BY-SA 4.0"

# (source, URL or Wikipedia title, license)
SOURCES: tuple[tuple[str, str, str], ...] = (
    ("womenshealth.gov", "https://womenshealth.gov/relationships-and-safety/signs-abuse", PUBLIC),
    (
        "womenshealth.gov",
        "https://womenshealth.gov/relationships-and-safety/other-types/dating-violence-and-abuse",
        PUBLIC,
    ),
    (
        "womenshealth.gov",
        "https://womenshealth.gov/relationships-and-safety/other-types/emotional-and-verbal-abuse",
        PUBLIC,
    ),
    (
        "womenshealth.gov",
        "https://womenshealth.gov/relationships-and-safety/domestic-violence/signs-domestic-violence",
        PUBLIC,
    ),
    (
        "womenshealth.gov",
        "https://womenshealth.gov/relationships-and-safety/get-help/how-help-friend",
        PUBLIC,
    ),
    ("youth.gov", "https://youth.gov/youth-topics/teen-dating-violence/characteristics", PUBLIC),
    ("youth.gov", "https://youth.gov/youth-topics/teen-dating-violence", PUBLIC),
    (
        "cdc.gov",
        "https://www.cdc.gov/intimate-partner-violence/about/about-teen-dating-violence.html",
        PUBLIC,
    ),
    ("cdc.gov", "https://www.cdc.gov/intimate-partner-violence/about/index.html", PUBLIC),
    (
        "opa.hhs.gov",
        "https://opa.hhs.gov/adolescent-health/healthy-relationships-adolescence",
        PUBLIC,
    ),
    (
        "womenshealth.gov",
        "https://womenshealth.gov/relationships-and-safety/get-help/help-abused-friend",
        PUBLIC,
    ),
    (
        "womenshealth.gov",
        "https://womenshealth.gov/relationships-and-safety/domestic-violence/leaving-abusive-relationship",
        PUBLIC,
    ),
    (
        "womenshealth.gov",
        "https://womenshealth.gov/relationships-and-safety/effects-violence-against-women",
        PUBLIC,
    ),
    (
        "womenshealth.gov",
        "https://womenshealth.gov/relationships-and-safety/sexual-assault-and-rape",
        PUBLIC,
    ),
    ("youth.gov", "https://youth.gov/youth-topics/teen-dating-violence/consequences", PUBLIC),
    (
        "youth.gov",
        "https://youth.gov/feature-article/dating-matters-understanding-teen-dating-violence-prevention",
        PUBLIC,
    ),
    ("wikipedia", "Intimate relationship", WIKI),
    ("wikipedia", "Interpersonal relationship", WIKI),
    ("wikipedia", "Courtship", WIKI),
    ("wikipedia", "Relationship maintenance", WIKI),
    ("wikipedia", "Conflict resolution", WIKI),
    ("wikipedia", "Trust (social science)", WIKI),
    ("wikipedia", "Breakup", WIKI),
    ("wikipedia", "Social rejection", WIKI),
    ("wikipedia", "Text messaging", WIKI),
    ("wikipedia", "Infatuation", WIKI),
    ("wikipedia", "Emotional abuse", WIKI),
    ("wikipedia", "Stalking", WIKI),
    ("wikipedia", "Dating violence", WIKI),
    ("wikipedia", "Dating", WIKI),
    ("wikipedia", "Online dating", WIKI),
    ("wikipedia", "Personal boundaries", WIKI),
    ("wikipedia", "Sexual consent", WIKI),
    ("wikipedia", "Ghosting (behavior)", WIKI),
    ("wikipedia", "Jealousy", WIKI),
    ("wikipedia", "Active listening", WIKI),
    ("wikipedia", "Interpersonal communication", WIKI),
    ("wikipedia", "Love bombing", WIKI),
    ("wikipedia", "Romance scam", WIKI),
    ("wikipedia", "Coercive control", WIKI),
)


@dataclass(frozen=True)
class Passage:
    id: str
    text: str
    title: str
    source: str
    url: str
    license: str
    chunking: str


class _Blocks(HTMLParser):
    """Text of paragraphs, list items, and headings, skipping navigation and scripts."""

    BLOCK = {"p", "li", "h2", "h3", "h4"}
    SKIP = {"script", "style", "nav", "header", "footer", "aside", "form", "noscript", "button"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[str] = []
        self._skip = 0
        self._current: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.SKIP:
            self._skip += 1
        elif tag in self.BLOCK and not self._skip:
            self._flush()
            self._current = []

    def handle_endtag(self, tag: str) -> None:
        if tag in self.SKIP and self._skip:
            self._skip -= 1
        elif tag in self.BLOCK:
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._current is not None and not self._skip:
            self._current.append(data)

    def _flush(self) -> None:
        if self._current is not None:
            text = re.sub(r"\s+", " ", "".join(self._current)).strip()
            if text:
                self.blocks.append(text)
        self._current = None


_CITATION = re.compile(r"^\d+\s|doi\.org|https?://|back to top", re.I)


def html_paragraphs(html: str) -> list[str]:
    """The body text of a page as paragraphs; menus, captions, and citations dropped."""
    parser = _Blocks()
    parser.feed(html)
    parser.close()
    return [b for b in parser.blocks if len(b.split()) >= 8 and not _CITATION.search(b)]


def _get(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            time.sleep(5 * 2**attempt)
    raise RuntimeError("unreachable")


def _wikipedia(title: str) -> tuple[str, list[str]]:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
        }
    )
    pages = json.loads(_get(f"https://en.wikipedia.org/w/api.php?{query}"))["query"]["pages"]
    text = next(iter(pages.values())).get("extract") or ""
    text = re.split(r"\n==+ (See also|References|Further reading|External links|Notes) ==+", text)[
        0
    ]
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n+", text)]
    url = "https://en.wikipedia.org/wiki/" + urllib.parse.quote(title.replace(" ", "_"))
    return url, [p for p in paragraphs if len(p.split()) >= 8 and not p.startswith("=")]


def _windows(paragraphs: list[str], size: int = 200) -> list[str]:
    """Paragraphs packed into chunks of about ``size`` words, never splitting a paragraph."""
    chunks, current, count = [], [], 0
    for p in paragraphs:
        words = len(p.split())
        if current and count + words > size:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(p)
        count += words
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_guidance(out: Path = GUIDANCE_DIR, *, pause: float = 1.5) -> dict[str, Any]:
    """Fetch every source, chunk it both ways, filter labels; returns the counts."""
    from confidant.evals import DIAGNOSTIC_TERMS

    label = re.compile(r"\b(" + "|".join(re.escape(t) for t in DIAGNOSTIC_TERMS) + ")", re.I)
    out.mkdir(parents=True, exist_ok=True)
    passages: dict[str, list[Passage]] = {"paragraph": [], "window": []}
    counts: Counter[str] = Counter()
    failed = []
    for source, where, license in SOURCES:
        time.sleep(pause)
        try:
            if source == "wikipedia":
                url, paragraphs = _wikipedia(where)
                title = where
            else:
                url = where
                paragraphs = html_paragraphs(_get(url))
                title = url.rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
        except (urllib.error.URLError, TimeoutError) as exc:
            failed.append(f"{where}: {exc}")
            continue
        if not paragraphs:
            failed.append(f"{where}: no body text found")
            continue
        counts[f"pages:{source}"] += 1
        for chunking, chunks in (("paragraph", paragraphs), ("window", _windows(paragraphs))):
            for i, text in enumerate(chunks):
                if label.search(text):
                    counts[f"dropped_label:{chunking}"] += 1
                    continue
                pid = f"{source}:{title}:{chunking}:{i}"
                passages[chunking].append(Passage(pid, text, title, source, url, license, chunking))
    for chunking, items in passages.items():
        with (out / f"{chunking}.jsonl").open("w", encoding="utf-8") as f:
            for p in items:
                f.write(json.dumps(asdict(p), ensure_ascii=False) + "\n")
        counts[f"passages:{chunking}"] = len(items)
        counts[f"words:{chunking}"] = sum(len(p.text.split()) for p in items)
    return {"counts": dict(sorted(counts.items())), "failed": failed}


# Pages whose subject is abuse, violence, or scams. Every government page in SOURCES is
# one; so are these Wikipedia articles. The rest is about ordinary relationships.
ABUSE_TITLES = frozenset(
    {
        "Coercive control",
        "Emotional abuse",
        "Stalking",
        "Dating violence",
        "Romance scam",
        "Love bombing",
        "Sexual consent",
    }
)


def is_abuse_focused(passage: Passage) -> bool:
    return passage.source != "wikipedia" or passage.title in ABUSE_TITLES


def load_passages(
    chunking: str = "window", directory: Path = GUIDANCE_DIR, pool: str = "all"
) -> list[Passage]:
    """The corpus, or one pool of it: ``ordinary`` (relationships) or ``abuse``."""
    path = directory / f"{chunking}.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    passages = [Passage(**json.loads(line)) for line in lines]
    if pool == "ordinary":
        return [p for p in passages if not is_abuse_focused(p)]
    if pool == "abuse":
        return [p for p in passages if is_abuse_focused(p)]
    return passages
