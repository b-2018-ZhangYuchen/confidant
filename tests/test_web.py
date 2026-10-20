"""Tests for the web UI that ``confidant serve`` puts at ``/``.

The page holds a private transcript, so most of these are about what it can reach: it
loads only its own files, the browser is told to refuse anything else, and its script
never turns text into markup. The rest keep the page and the API from drifting apart:
every element the script looks up exists, and every route it calls is one the API has.

None of this runs a browser. The script is checked for syntax with Node when Node is
installed, and otherwise read as text.
"""

from __future__ import annotations

import functools
import re
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from confidant.api import UI_ASSETS, create_app
from confidant.recording import ReplayClient, recording_set
from confidant.store import Store

RECORDINGS = Path(__file__).parent / "fixtures" / "recorded"
EXAMPLES = Path(__file__).parent.parent / "examples"
WEB = files("confidant.web")


def _read(name: str) -> str:
    return WEB.joinpath(name).read_text(encoding="utf-8")


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CONFIDANT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))


@pytest.fixture
def api(tmp_path) -> TestClient:
    app = create_app(store=functools.partial(Store, tmp_path / "confidant.db"))
    return TestClient(app, base_url="http://localhost")


def _answer_with(monkeypatch, name: str) -> None:
    client = ReplayClient(*recording_set(RECORDINGS / f"{name}.json"))
    monkeypatch.setattr("confidant.client.build_client", lambda settings=None: client)


def _transcript(name: str) -> dict:
    return {"text": (EXAMPLES / name).read_text(encoding="utf-8"), "filename": name}


# -- what is served -----------------------------------------------------------------


def test_the_page_is_served_at_the_root(api):
    response = api.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<title>Confidant</title>" in response.text


@pytest.mark.parametrize("path", ["/", "/static/app.js", "/static/style.css"])
def test_the_browser_is_told_to_load_nothing_from_anywhere_else(api, path):
    headers = api.get(path).headers
    policy = headers["content-security-policy"]
    for directive in (
        "default-src 'none'",
        "script-src 'self'",
        "connect-src 'self'",
        "frame-ancestors 'none'",
    ):
        assert directive in policy
    assert headers["x-content-type-options"] == "nosniff"
    assert headers["referrer-policy"] == "no-referrer"


def test_the_page_is_refused_under_another_host_name(api):
    # A rebinding page would otherwise get the UI, and with it same-origin access.
    assert api.get("/", headers={"host": "evil.example"}).status_code == 403


@pytest.mark.parametrize("name", sorted(UI_ASSETS))
def test_each_asset_is_served_with_its_type(api, name):
    response = api.get(f"/static/{name}")
    assert response.status_code == 200
    assert response.headers["content-type"] == UI_ASSETS[name]
    assert response.text == _read(name)


@pytest.mark.parametrize("name", ["__init__.py", "index.html", "missing.js"])
def test_only_the_listed_assets_are_served(api, name):
    response = api.get(f"/static/{name}")
    assert response.status_code == 404
    assert response.json()["error"]["kind"] == "not-found"


def test_the_ui_is_not_part_of_the_api_schema(api):
    paths = api.get("/openapi.json").json()["paths"]
    assert "/" not in paths
    assert not any(path.startswith("/static") for path in paths)


# -- what the page can reach --------------------------------------------------------


def test_the_page_links_exactly_the_assets_that_are_served():
    linked = set(re.findall(r'(?:src|href)="/static/([^"]+)"', _read("index.html")))
    assert linked == set(UI_ASSETS)


@pytest.mark.parametrize("name", ["index.html", *sorted(UI_ASSETS)])
def test_no_file_names_another_site(name):
    assert not re.search(r"https?://|//cdn|@import", _read(name))


@pytest.mark.parametrize(
    "sink", ["innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "Function("]
)
def test_the_script_never_treats_text_as_markup(sink):
    # Transcripts and the model's quotes of them are arbitrary text. textContent is the
    # only way any of it reaches the page.
    assert sink not in _read("app.js")


def test_every_route_the_script_calls_is_one_the_api_has(api):
    called = set(re.findall(r'"(/api/[a-z/]+)"', _read("app.js")))
    assert called == {"/api/stats", "/api/redact", "/api/analyze", "/api/flags"}
    posts = {route.path for route in api.app.routes if "POST" in getattr(route, "methods", set())}
    assert called <= posts


def test_every_element_the_script_looks_up_is_on_the_page():
    ids = set(re.findall(r'id="([^"]+)"', _read("index.html")))
    looked_up = set(re.findall(r'\$\("([^"]+)"\)', _read("app.js")))
    assert looked_up
    assert looked_up <= ids


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_the_script_parses(tmp_path):
    script = tmp_path / "app.js"
    script.write_text(_read("app.js"), encoding="utf-8")
    result = subprocess.run(
        ["node", "--check", str(script)], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr


# -- what the page is given ---------------------------------------------------------


def test_a_danger_finding_comes_with_its_notice_ready_to_show_first(api, monkeypatch):
    _answer_with(monkeypatch, "flags_pressure")
    data = api.post(
        "/api/flags", json=_transcript("pressure_chat.txt"), params={"region": "GB"}
    ).json()
    notice = data["escalation"]
    assert notice["headline"].startswith("Something in Casey's messages is serious")
    assert data["text"].startswith(notice["text"])
    # One line per number, unwrapped, so the page never breaks one mid-sentence.
    assert notice["routing_lines"][0] == "In the United Kingdom:"
    assert any("999" in line for line in notice["routing_lines"])
    assert all("\n" not in line for line in notice["routing_lines"])
    # The recording has a flag whose quote is not in the transcript; the page says so in
    # the same words the terminal ends with.
    assert data["footnotes"] == [line for line in data["text"].splitlines() if "left out" in line]
    assert len(data["footnotes"]) == 1


def test_without_a_region_the_notice_says_how_to_get_numbers(api, monkeypatch):
    monkeypatch.delenv("CONFIDANT_REGION", raising=False)
    _answer_with(monkeypatch, "flags_pressure")
    data = api.post("/api/flags", json=_transcript("pressure_chat.txt")).json()
    [line] = data["escalation"]["routing_lines"]
    assert "--region" in line


def test_a_personality_read_has_no_notice(api, monkeypatch):
    _answer_with(monkeypatch, "analyze_sample")
    data = api.post("/api/analyze", json=_transcript("sample_chat.txt")).json()
    assert data["escalation"] is None
    assert "footnotes" not in data
