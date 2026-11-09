"""The guidance corpus and its retrieval, offline: no network, no model."""

from __future__ import annotations

from confidant.lab.guidance import SOURCES, Passage, _windows, html_paragraphs
from confidant.lab.rag import PassageIndex

PAGE = """
<html><head><script>var x = "ignore me entirely please now";</script></head>
<body>
<nav><p>Home About Contact Donate Newsletter Sign up for updates today</p></nav>
<main>
<h2>Signs of a healthy relationship you can look for</h2>
<p>In a healthy relationship, both partners respect each other's boundaries and time.</p>
<p>Short.</p>
<ul><li>Your partner checks your phone and demands to know where you are at all times.</li></ul>
<p>14 Branje, S. (2018). Development of relationships. https://doi.org/10.1111/x back to top</p>
</main>
<footer><p>Copyright notice and all the links that nobody reads at the bottom</p></footer>
</body></html>
"""


def test_body_text_is_kept_and_menus_scripts_and_citations_are_not():
    paragraphs = html_paragraphs(PAGE)
    assert paragraphs == [
        "Signs of a healthy relationship you can look for",
        "In a healthy relationship, both partners respect each other's boundaries and time.",
        "Your partner checks your phone and demands to know where you are at all times.",
    ]


def test_windows_pack_paragraphs_without_splitting_them():
    paragraphs = ["one two three four five"] * 5
    chunks = _windows(paragraphs, size=12)
    assert chunks == [" ".join(["one two three four five"] * 2)] * 2 + ["one two three four five"]


def test_no_source_is_from_a_site_that_opted_out():
    assert all("wikihow" not in where for _, where, _ in SOURCES)
    assert all(license for _, _, license in SOURCES)


def _passage(i: int, text: str, url: str) -> Passage:
    return Passage(f"p{i}", text, f"title {i}", "test", url, "public domain", "window")


def test_passages_come_from_different_pages_and_short_lines_are_ignored():
    passages = [
        _passage(0, "location sharing tracking phone password monitoring", "https://a"),
        _passage(1, "location sharing tracking phone again monitoring", "https://a"),
        _passage(2, "healthy relationships respect boundaries", "https://b"),
        _passage(3, "texting slowly and briefly is ordinary", "https://c"),
    ]
    index = PassageIndex(passages, retriever="bm25max")
    found = index.search(["ok", "turn on location sharing so I can see your phone"], k=2)
    assert found[0].url == "https://a"
    assert len({p.url for p in found}) == 2
