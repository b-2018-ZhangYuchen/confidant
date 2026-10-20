"""Tests for the HTTP API.

The app runs in-process through FastAPI's TestClient, which speaks ASGI directly and
never opens a socket, so the suite's no-network guard stays on. Model calls answer from
the same recordings the CLI tests use: the API has to give the same reads, with the same
grounding and the same safety notices, or it is a second product rather than a backend.
"""

from __future__ import annotations

import functools
import sys
from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from confidant.api import create_app, is_local_host
from confidant.cli import main
from confidant.config import Settings
from confidant.profile import read_conversation
from confidant.recording import ReplayClient, recording_set
from confidant.store import ReadingKind, Store

RECORDINGS = Path(__file__).parent / "fixtures" / "recorded"
EXAMPLES = Path(__file__).parent.parent / "examples"
NOW = datetime(2026, 10, 16, 9, 0)


def _text(name: str) -> str:
    return (EXAMPLES / name).read_text(encoding="utf-8")


def _transcript(name: str, **options) -> dict:
    return {"text": _text(name), "filename": f"examples/{name}", **options}


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CONFIDANT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))


@pytest.fixture
def db(tmp_path) -> Path:
    return tmp_path / "store" / "confidant.db"


@pytest.fixture
def api(db) -> TestClient:
    app = create_app(store=functools.partial(Store, db, clock=lambda: NOW))
    return TestClient(app, base_url="http://localhost")


@pytest.fixture
def answer_with(monkeypatch):
    """Make the API answer from recordings, in order, for the rest of the test."""

    def install(*names: str) -> ReplayClient:
        recordings = [r for name in names for r in recording_set(RECORDINGS / f"{name}.json")]
        client = ReplayClient(*recordings)
        monkeypatch.setattr("confidant.client.build_client", lambda settings=None: client)
        return client

    return install


def _check_casey_for_flags(db: Path) -> None:
    with Store(db, clock=lambda: NOW) as store:
        [stored] = store.conversations("Casey")
        read_conversation(
            store,
            stored,
            ReadingKind.FLAGS,
            settings=Settings(),
            client=ReplayClient(RECORDINGS / "flags_pressure.json"),
        )


def _save(api: TestClient, name: str, example: str) -> None:
    if api.get(f"/api/people/{name}").status_code == 404:
        assert api.post("/api/people", json={"name": name}).status_code == 201
    response = api.post(f"/api/people/{name}/conversations", json=_transcript(example))
    assert response.status_code in (200, 201), response.json()


# -- only this machine ----------------------------------------------------------------


def test_health(api):
    response = api.get("/api/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]", "localhost:8000"])
def test_local_host_headers_are_answered(api, host):
    assert api.get("/api/health", headers={"host": host}).status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "192.168.1.20:8000", "localhost.evil.example"])
def test_any_other_host_is_refused(api, host):
    # A DNS-rebinding page reaches 127.0.0.1 under its own domain name; the Host header
    # is what gives it away.
    response = api.get("/api/people", headers={"host": host})
    assert response.status_code == 403
    assert response.json()["error"]["kind"] == "forbidden"


@pytest.mark.parametrize("origin", ["https://evil.example", "null"])
def test_requests_from_other_websites_are_refused(api, origin):
    response = api.post("/api/people", json={"name": "Robin"}, headers={"origin": origin})
    assert response.status_code == 403
    # Refused before the route: nothing was saved.
    assert api.get("/api/people").json() == []


def test_a_local_origin_is_allowed(api):
    response = api.post(
        "/api/people", json={"name": "Robin"}, headers={"origin": "http://localhost:8000"}
    )
    assert response.status_code == 201


def test_the_schema_is_served_here_and_no_page_pulls_from_a_cdn(api):
    assert api.get("/openapi.json").json()["info"]["title"] == "Confidant"
    assert api.get("/docs").status_code == 404
    assert api.get("/redoc").status_code == 404


def test_is_local_host():
    assert is_local_host("localhost") and is_local_host("::1") and is_local_host("[::1]")
    assert not is_local_host("0.0.0.0")
    assert not is_local_host(None)


# -- local reads --------------------------------------------------------------------


def test_stats_match_the_cli(api, capsys):
    response = api.post("/api/stats", json=_transcript("sample_chat.txt"))
    assert response.status_code == 200
    data = response.json()
    assert (data["owner"], data["match"], data["messages"]) == ("Sam", "Robin", 17)
    assert data["note"] is None

    assert main(["stats", str(EXAMPLES / "sample_chat.txt")]) == 0
    assert data["text"] == capsys.readouterr().out.rstrip("\n")


def test_an_export_says_how_it_was_read(api):
    body = _transcript("whatsapp_chat.txt", owner="Noor")
    data = api.post("/api/stats", json=body).json()
    assert data["match"] == "Ari"
    assert data["note"].startswith("read as a WhatsApp export, 10 messages; left out")


def test_a_transcript_that_cannot_be_read_is_a_422_with_the_reason(api):
    response = api.post("/api/stats", json={"text": "nothing anyone said\n"})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["kind"] == "invalid-transcript"
    assert error["message"]


def test_a_missing_field_has_the_same_error_shape(api):
    response = api.post("/api/stats", json={})
    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "invalid"
    assert "text" in response.json()["error"]["message"]


def test_redact_shows_what_would_be_sent_and_the_key(api):
    data = api.post("/api/redact", json=_transcript("details_chat.txt")).json()
    assert "[PHONE_1]" in data["transcript"]
    assert "(555) 010-4477" not in data["transcript"]
    assert data["key"]["[PHONE_1]"] == "(555) 010-4477"
    assert data["key"]["[MATCH]"] == "Theo"


# -- people -------------------------------------------------------------------------


def test_people_can_be_added_saved_shown_and_removed(api):
    assert api.get("/api/people").json() == []
    assert api.post("/api/people", json={"name": "Robin"}).status_code == 201
    assert api.post("/api/people", json={"name": "robin"}).status_code == 409

    first = api.post("/api/people/Robin/conversations", json=_transcript("sample_chat.txt"))
    assert first.status_code == 201
    assert first.json()["outcome"] == "new"
    again = api.post("/api/people/Robin/conversations", json=_transcript("sample_chat.txt"))
    assert again.status_code == 200
    assert again.json()["outcome"] == "unchanged"
    later = api.post(
        "/api/people/Robin/conversations", json=_transcript("sample_chat_later.txt")
    ).json()
    assert later["outcome"] == "extended"
    assert later["conversation"]["messages"] == later["added"] + 17

    [summary] = api.get("/api/people").json()
    assert summary["name"] == "Robin"
    assert summary["conversations"] == 1

    shown = api.get("/api/people/robin").json()
    assert shown["conversations"][0]["stats"]["match"] == "Robin"

    removed = api.delete("/api/people/Robin")
    assert removed.json() == {"removed": "Robin", "conversations": 1}
    assert api.get("/api/people/Robin").status_code == 404


def test_saving_someone_elses_chat_under_a_name_is_refused(api):
    api.post("/api/people", json={"name": "Robin"})
    response = api.post("/api/people/Robin/conversations", json=_transcript("pressure_chat.txt"))
    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "wrong-person"
    assert api.get("/api/people/Robin").json()["conversations"] == []


def test_an_unknown_person_is_a_404(api):
    for path in ("/api/people/Nobody", "/api/people/Nobody/profile", "/api/people/Nobody/nudge"):
        response = api.get(path)
        assert response.status_code == 404, path
        assert response.json()["error"]["kind"] == "unknown-person"


def test_profile_timeline_and_nudge_carry_the_saved_safety_notice(api, db):
    _save(api, "Casey", "pressure_chat.txt")
    _save(api, "Robin", "sample_chat.txt")
    _check_casey_for_flags(db)

    profile = api.get("/api/people/Casey/profile", params={"region": "GB"}).json()
    assert profile["escalation"]["routing"]["region"] == "GB"
    assert "\n\n!! Something in Casey's messages is serious" in profile["text"]
    assert profile["flags"] and all(f["conversation"] == 1 for f in profile["flags"])
    assert profile["pending"] == 1  # the personality read was never made

    timeline = api.get("/api/people/Casey/timeline").json()
    assert timeline["period"] == "week"
    assert timeline["escalation"] is not None
    assert timeline["buckets"][0]["match"]["messages"] > 0

    nudge = api.get("/api/nudge", params={"at": "2026-04-14T09:00:00"}).json()
    by_name = {c["name"]: c for c in nudge["contacts"]}
    assert by_name["Casey"]["state"] == "safety"
    assert by_name["Casey"]["suggested"] is False
    assert by_name["Casey"]["escalation"] is not None
    assert "priority" not in by_name["Robin"]
    assert nudge["text"].startswith("Casey: not suggested.")


# -- model calls ----------------------------------------------------------------------


def test_analyze_returns_the_report_its_text_and_what_it_used(api, answer_with):
    answer_with("analyze_sample")
    response = api.post("/api/analyze", json=_transcript("sample_chat.txt"))
    assert response.status_code == 200
    data = response.json()
    assert data["report"]["confidence"] == "medium"
    assert data["text"].startswith("Robin comes across as")
    assert data["usage"]["calls"] == 1
    assert data["usage"]["summary"].startswith("used ")


def test_flags_with_a_region_route_the_notice(api, answer_with):
    answer_with("flags_pressure")
    data = api.post(
        "/api/flags", json=_transcript("pressure_chat.txt"), params={"region": "uk"}
    ).json()
    assert data["report"]["escalation"]["routing"]["region"] == "GB"
    assert data["text"].startswith("!! Something in Casey's messages is serious")


def test_a_refusal_is_a_422_and_still_reports_what_it_cost(api, answer_with):
    answer_with("analyze_refusal")
    response = api.post("/api/analyze", json=_transcript("sample_chat.txt"))
    assert response.status_code == 422
    assert response.json()["error"]["kind"] == "refused"
    assert response.json()["usage"]["calls"] == 1


def test_a_cut_off_answer_is_a_502(api, answer_with):
    answer_with("analyze_truncated")
    response = api.post("/api/analyze", json=_transcript("sample_chat.txt"))
    assert response.status_code == 502
    assert response.json()["error"]["kind"] == "incomplete"


def test_without_credentials_nothing_is_sent_and_it_says_why(api):
    response = api.post("/api/analyze", json=_transcript("sample_chat.txt"))
    assert response.status_code == 503
    error = response.json()["error"]
    assert error["kind"] == "configuration"
    assert "credentials" in error["message"]
    assert "usage" not in response.json()


def test_a_thread_needs_exactly_one_source(api):
    both = {"transcript": _transcript("sample_chat.txt"), "person": "Robin"}
    assert api.post("/api/draft", json=both).status_code == 422
    assert api.post("/api/draft", json={}).status_code == 422


def test_draft_for_someone_saved_opens_with_their_safety_notice(api, db, answer_with):
    _save(api, "Casey", "pressure_chat.txt")
    _check_casey_for_flags(db)
    answer_with("draft_pressure_firm")
    data = api.post("/api/draft", json={"person": "casey", "tone": "firm"}).json()
    assert data["report"]["tone"] == "firm"
    assert data["report"]["escalation"] is not None
    assert data["text"].index("!!") < data["text"].index("DRAFTS (firm)")


def test_an_unknown_tone_is_refused_before_anything_is_sent(api, answer_with):
    client = answer_with()
    body = {"transcript": _transcript("sample_chat.txt"), "tone": "smouldering"}
    assert api.post("/api/draft", json=body).status_code == 422
    assert client.requests == []


def test_comfort_from_a_transcript(api, answer_with):
    answer_with("comfort_faded")
    body = {
        "transcript": _transcript("faded_chat.txt"),
        "what": "two good dates, then Jamie went quiet",
    }
    data = api.post("/api/comfort", json=body).json()
    assert data["report"]["unanswered"] == 2
    assert "WHAT YOU DID WELL" in data["text"]


def test_ask_runs_the_agent_with_every_check(api, answer_with):
    answer_with("ask_pressure")
    body = {
        "transcript": _transcript("pressure_chat.txt"),
        "question": "Casey says everyone shares their location. Is that normal?",
    }
    response = api.post("/api/ask", json=body)
    assert response.status_code == 200, response.json()
    data = response.json()
    assert data["report"]["escalation"] is not None
    assert data["usage"]["calls"] == 4


def test_profile_update_reads_what_is_pending_and_keeps_it(api, answer_with):
    _save(api, "Robin", "sample_chat.txt")
    answer_with("analyze_sample", "flags_sample")
    data = api.post("/api/people/Robin/profile/update").json()
    assert data["updated"] == 2
    assert data["usage"]["calls"] == 2
    assert data["profile"]["pending"] == 0

    # Kept, so reading the profile again is local and free.
    answer_with()
    again = api.post("/api/people/Robin/profile/update").json()
    assert again["updated"] == 0
    assert again["usage"]["calls"] == 0
    [conversation] = again["profile"]["conversations"]
    assert conversation["personality"]["report"]["confidence"] == "medium"


# -- confidant serve ------------------------------------------------------------------


def test_serve_refuses_an_address_other_machines_can_reach(capsys, monkeypatch):
    started = []
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: started.append(k))
    assert main(["serve", "--host", "0.0.0.0"]) == 2
    assert "not this machine" in capsys.readouterr().err
    assert started == []


def test_serve_runs_on_localhost_without_an_access_log(capsys, monkeypatch):
    started = []
    monkeypatch.setattr("uvicorn.run", lambda app, **k: started.append(k))
    assert main(["serve", "--port", "8123"]) == 0
    [options] = started
    assert (options["host"], options["port"], options["access_log"]) == ("127.0.0.1", 8123, False)
    assert "http://127.0.0.1:8123/ for the web UI" in capsys.readouterr().err


def test_serve_without_the_api_extra_says_how_to_get_it(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "uvicorn", None)
    assert main(["serve"]) == 2
    assert "pip install 'confidant[api]'" in capsys.readouterr().err
