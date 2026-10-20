"""Tests for the web UI that ``confidant serve`` puts at ``/``.

The page holds a private transcript, so most of these are about what it can reach: it
loads only its own files, the browser is told to refuse anything else, and its script
never turns text into markup. The rest keep the page and the API from drifting apart:
every element the script looks up exists, every route it calls is one the API has, and
the people list, profile, and nudge digest render from what the API really returns.

None of this runs a browser. When Node is installed the script is checked for syntax, and
its renderers are run against a small stand-in for the DOM; otherwise it is read as text.
"""

from __future__ import annotations

import functools
import json
import re
import shutil
import subprocess
from importlib.resources import files
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from confidant.api import UI_ASSETS, create_app
from confidant.config import Settings
from confidant.profile import read_conversation
from confidant.recording import ReplayClient, recording_set
from confidant.store import ReadingKind, Store

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
def db(tmp_path) -> Path:
    return tmp_path / "confidant.db"


@pytest.fixture
def api(db) -> TestClient:
    app = create_app(store=functools.partial(Store, db))
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


# request("GET", "/api/people") or request("POST", person(name, "/profile/update"), ...).
_CALL = re.compile(
    r'request\(\s*"([A-Z]+)",\s*(?:"(/api/[a-z/]*)"|person\([\w.]+(?:,\s*"([a-z/]*)")?\))'
)


def _calls() -> set[tuple[str, str]]:
    script = _read("app.js")
    found = _CALL.findall(script)
    # Every request( call is one the pattern understood, so none can slip past this test.
    assert len(found) == script.count("request(") - script.count("function request(")
    return {
        (method, literal or f"/api/people/{{name}}{suffix}") for method, literal, suffix in found
    }


def test_every_route_the_script_calls_is_one_the_api_has(api):
    called = _calls()
    assert called == {
        ("POST", "/api/stats"),
        ("POST", "/api/redact"),
        ("POST", "/api/analyze"),
        ("POST", "/api/flags"),
        ("GET", "/api/people"),
        ("DELETE", "/api/people/{name}"),
        ("POST", "/api/people/{name}/conversations"),
        ("GET", "/api/people/{name}/profile"),
        ("POST", "/api/people/{name}/profile/update"),
        ("GET", "/api/nudge"),
    }
    routes = {
        (method, route.path)
        for route in api.app.routes
        for method in getattr(route, "methods", set())
    }
    assert called <= routes


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


def test_the_digest_at_field_is_a_moment_the_api_reads(api):
    _save(api, "Robin", "sample_chat.txt")
    # A datetime-local input gives minutes and no seconds.
    response = api.get("/api/nudge", params={"at": "2026-03-10T09:00"})
    assert response.status_code == 200
    [robin] = response.json()["contacts"]
    assert robin["last_message"] == "2026-03-05T21:36:00"


# -- what the page shows ------------------------------------------------------------

# A stand-in for the few DOM calls the renderers make, enough to run them under Node and
# read back what they put on the page, in order. Each line is one node's own text.
_HARNESS = r"""
const fs = require("fs");
const vm = require("vm");
const [script, call] = process.argv.slice(2).map((p) => fs.readFileSync(p, "utf8"));

class Node {
  constructor(tag) {
    Object.assign(this, { tagName: tag, children: [], own: "", className: "", hidden: false });
    this.attributes = {};
  }
  append(...children) {
    for (const c of children) this.children.push(typeof c === "string" ? new Text(c) : c);
  }
  replaceChildren(...children) { this.children = []; this.append(...children); }
  set textContent(value) { this.children = []; this.own = String(value); }
  get textContent() { return this.own + this.children.map((c) => c.textContent).join(""); }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  addEventListener() {}
  scrollIntoView() {}
}
class Text {
  constructor(t) { this.own = t; this.children = []; }
  get textContent() { return this.own; }
}

const byId = {};
const document = {
  createElement: (tag) => new Node(tag),
  createTextNode: (t) => new Text(t),
  getElementById: (id) => (byId[id] ??= new Node("div")),
  querySelectorAll: () => [],
  addEventListener() {},
};
function lines(node, out = []) {
  if (node.own.trim()) out.push(node.own.trim());
  for (const c of node.children) lines(c, out);
  return out;
}
const context = vm.createContext({ document, window: {}, console, URLSearchParams });
vm.runInContext(script, context);
const { fn, args } = JSON.parse(call);
const result = vm.runInContext(fn, context)(...args);
const nodes = result === undefined ? [] : [result].flat();
const ids = Object.fromEntries(
  Object.entries(byId).map(([id, n]) => [id, { hidden: n.hidden, lines: lines(n) }]),
);
console.log(JSON.stringify({ lines: nodes.flatMap((n) => lines(n)), ids }));
"""


def _render(tmp_path, fn: str, *args) -> dict:
    if shutil.which("node") is None:
        pytest.skip("Node is not installed")
    harness = tmp_path / "harness.js"
    harness.write_text(_HARNESS, encoding="utf-8")
    script = tmp_path / "app.js"
    script.write_text(_read("app.js"), encoding="utf-8")
    call = tmp_path / "call.json"
    call.write_text(json.dumps({"fn": fn, "args": list(args)}), encoding="utf-8")
    result = subprocess.run(
        ["node", str(harness), str(script), str(call)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def _save(api, name: str, example: str) -> None:
    response = api.post(
        f"/api/people/{name}/conversations", json=_transcript(example), params={"create": "true"}
    )
    assert response.status_code in (200, 201), response.text


def _check_for_flags(db: Path, name: str) -> None:
    with Store(db) as store:
        [stored] = store.conversations(name)
        read_conversation(
            store,
            stored,
            ReadingKind.FLAGS,
            settings=Settings(),
            client=ReplayClient(RECORDINGS / "flags_pressure.json"),
        )


def test_the_digest_puts_the_safety_notice_before_any_suggestion(api, db, tmp_path):
    _save(api, "Casey", "pressure_chat.txt")
    _save(api, "Robin", "sample_chat.txt")
    _save(api, "Robin", "sample_chat_later.txt")
    _check_for_flags(db, "Casey")
    data = api.get("/api/nudge", params={"at": "2026-04-14T09:00"}).json()

    lines = _render(tmp_path, "digest", data)["lines"]
    casey = data["contacts"][[c["name"] for c in data["contacts"]].index("Casey")]
    order = [
        lines.index(casey["headline"]),
        lines.index("Read this first"),
        lines.index(casey["escalation"]["headline"]),
        lines.index("Worth a message today"),
    ]
    assert order == sorted(order)
    # The suggestion comes with its reasons, and the caveat closes the digest.
    robin = next(c for c in data["contacts"] if c["name"] == "Robin")
    assert robin["suggested"]
    assert all(reason in lines for reason in robin["reasons"])
    assert lines[-1] == data["caveat"]


def test_the_digest_with_no_one_saved_says_how_to_start(api, tmp_path):
    lines = _render(tmp_path, "digest", api.get("/api/nudge").json())["lines"]
    assert lines == ["No one saved yet. Save a conversation above to start."]


def test_a_profile_shows_its_notice_first_and_every_flag_and_footnote(api, db, tmp_path):
    _save(api, "Casey", "pressure_chat.txt")
    _check_for_flags(db, "Casey")
    data = api.get("/api/people/Casey/profile", params={"region": "GB"}).json()

    shown = _render(tmp_path, "showProfile", data)["ids"]
    # The notice is the page's own block above the report, as for a single check.
    assert shown["escalation"]["hidden"] is False
    assert data["escalation"]["headline"] in shown["escalation"]["lines"]
    assert "In the United Kingdom:" in shown["escalation"]["lines"]
    assert shown["result-title"]["lines"] == ["Casey"]

    report = shown["report"]["lines"]
    assert report[0] == "1 conversation, 10 messages"
    assert "Not read yet." in report
    for flag in data["flags"]:
        assert flag["behavior"] in report
    assert all(note in report for note in data["footnotes"])
    # The personality read was never made, so the page offers to make it, and says
    # what that sends.
    assert "Bring 1 read up to date" in report
    assert any("sends the redacted conversations" in line for line in report)


def test_a_profile_with_nothing_saved_says_so(api, tmp_path):
    api.post("/api/people", json={"name": "Robin"})
    data = api.get("/api/people/Robin/profile").json()
    lines = _render(tmp_path, "profile", data)["lines"]
    assert lines == [
        "0 conversations, 0 messages",
        "Nothing saved yet. Save a conversation above under Robin.",
    ]


def test_the_people_list_shows_each_person_with_their_counts(api, tmp_path):
    _save(api, "Robin", "sample_chat.txt")
    _save(api, "Casey", "pressure_chat.txt")
    lines = _render(tmp_path, "peopleTable", api.get("/api/people").json())["lines"]
    assert lines[:4] == ["Name", "Conversations", "Messages", "Last message"]
    assert lines[4:9] == ["Casey", "1", "10", "2026-04-12 09:20", "Profile"]
    assert lines[10:15] == ["Robin", "1", "17", "2026-03-05 21:36", "Profile"]
