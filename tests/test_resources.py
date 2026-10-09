"""Tests for crisis-resource routing. Offline: the table is data, and every scan is canned."""

from __future__ import annotations

import importlib
import json
import re

import pytest

from confidant.analysis.comfort import ComfortScan
from confidant.analysis.flags import DANGER_CATEGORIES, Flag, FlagScan
from confidant.analysis.personality import Evidence
from confidant.cli import main
from confidant.resources import (
    DIRECTORY,
    REGION_ENV,
    REGIONS,
    current_region,
    normalize_region,
    route,
    using_region,
)
from confidant.safety import escalate, support_notice

FADED = "examples/faded_chat.txt"
PRESSURE = "examples/pressure_chat.txt"


def flag(category: str, severity: str = "danger") -> Flag:
    return Flag(
        category=category,
        severity=severity,
        behavior="Something the match did.",
        evidence=[Evidence(quote="I like knowing where you are", why_it_matters="It matters.")],
        innocent_reading=None if severity == "danger" else "They may be busy.",
    )


def kinds(routing) -> list[str]:
    return [r.kind for r in routing.resources]


# -- the table ----------------------------------------------------------------------


@pytest.mark.parametrize("code", sorted(REGIONS))
def test_every_region_leads_with_its_emergency_number(code):
    region = REGIONS[code]
    assert region.code == code
    assert region.resources[0].kind == "emergency"
    assert sum(r.kind == "emergency" for r in region.resources) == 1
    # Every region the owner can pick has a crisis line: the owner-at-risk notice is the
    # one that most needs a number, and a region without one should not be on the list.
    assert any(r.kind == "crisis" for r in region.resources)


@pytest.mark.parametrize("code", sorted(REGIONS))
def test_every_entry_says_how_to_reach_it_with_a_number(code):
    for resource in REGIONS[code].resources:
        assert resource.how.startswith(("call ", "text ", "call or text ")), resource
        assert re.search(r"\d", resource.how), resource
        assert not resource.how.endswith("."), "to_text adds the full stop"


def test_the_numbers_are_the_ones_reviewed():
    # Pinned on purpose, so a changed number is a visible diff in a test as well as in
    # the table. Check against the service's own site when updating.
    def how(code: str, name: str) -> str:
        return next(r.how for r in REGIONS[code].resources if r.name.startswith(name))

    assert how("US", "Emergency") == "call 911"
    assert how("US", "988") == "call or text 988"
    assert how("GB", "Emergency") == "call 999"
    assert how("GB", "Samaritans") == "call 116 123"
    assert how("IE", "Samaritans") == "call 116 123"
    assert how("AU", "Emergency") == "call 000"
    assert how("AU", "Lifeline") == "call 13 11 14"
    assert how("NZ", "Emergency") == "call 111"
    assert how("NZ", "Need to Talk") == "call or text 1737"
    assert how("FR", "Numéro") == "call 3114"


def test_the_table_names_services_not_verdicts():
    text = " ".join(r.name + " " + r.how for region in REGIONS.values() for r in region.resources)
    for word in ("abuser", "victim", "toxic", "narcissist"):
        assert word not in text.casefold()


# -- choosing the region ------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, code",
    [
        ("gb", "GB"),
        ("UK", "GB"),
        (" united  kingdom ", "GB"),
        ("Scotland", "GB"),
        ("U.S.", "US"),
        ("usa", "US"),
        ("new_zealand", "NZ"),
        ("Aotearoa", "NZ"),
        ("Deutschland", "DE"),
        ("the netherlands", "NL"),
        ("xx", "XX"),
    ],
)
def test_what_people_type_is_normalized(raw, code):
    assert normalize_region(raw) == code


@pytest.mark.parametrize("raw", [None, "", "   "])
def test_blank_means_unset(raw):
    assert normalize_region(raw) is None


def test_the_region_comes_from_the_environment(monkeypatch):
    assert current_region() is None
    monkeypatch.setenv(REGION_ENV, "au")
    assert current_region() == "AU"


def test_an_explicit_region_beats_the_environment_and_is_undone_after(monkeypatch):
    monkeypatch.setenv(REGION_ENV, "AU")
    with using_region("nz"):
        assert current_region() == "NZ"
        with using_region(None):
            assert current_region() == "AU"
    assert current_region() == "AU"


def test_nothing_is_inferred_from_the_locale(monkeypatch):
    monkeypatch.setenv("LANG", "en_GB.UTF-8")
    monkeypatch.setenv("LC_ALL", "en_GB.UTF-8")
    monkeypatch.setenv("TZ", "Europe/London")
    assert current_region() is None


# -- routing ------------------------------------------------------------------------


def test_no_kinds_means_no_routing():
    assert route([], "GB") is None


def test_a_crisis_routing_lists_emergency_then_crisis_lines():
    routing = route({"crisis"}, "GB")
    assert kinds(routing) == ["emergency", "crisis", "crisis"]
    assert routing.region == "GB"
    assert routing.note is None


def test_an_abuse_routing_leaves_out_the_crisis_lines():
    routing = route({"abuse"}, "US")
    assert kinds(routing) == ["emergency", "abuse"]


def test_a_gap_in_the_table_is_said_rather_than_hidden():
    # Canada has no national domestic-violence line; the emergency number alone must not
    # read as all there is.
    routing = route({"abuse"}, "CA")
    assert kinds(routing) == ["emergency"]
    assert routing.note == DIRECTORY


def test_unset_says_how_to_set_it():
    routing = route({"crisis"})
    assert routing.resources == []
    assert routing.region is None
    assert "--region" in routing.note and REGION_ENV in routing.note
    assert DIRECTORY in routing.note


def test_an_unknown_region_is_not_an_error():
    routing = route({"crisis"}, "Narnia")
    assert routing.resources == []
    assert routing.note.startswith("Confidant has no numbers on file for NARNIA.")


def test_route_defaults_to_the_current_region():
    with using_region("IE"):
        assert route({"crisis"}).region == "IE"


# -- what each notice is routed to ---------------------------------------------------


def test_the_owner_at_risk_is_pointed_at_a_crisis_line():
    with using_region("US"):
        notice = support_notice()
    assert kinds(notice.routing) == ["emergency", "crisis"]


@pytest.mark.parametrize("category", sorted(DANGER_CATEGORIES - {"threat"}))
def test_physical_danger_is_pointed_at_an_abuse_line(category):
    with using_region("AU"):
        notice = escalate([flag(category)], "Casey")
    assert kinds(notice.routing) == ["emergency", "abuse"]


def test_a_threat_also_gets_the_crisis_line():
    # The threat may be to self-harm, and the notice already says the owner can point
    # them to help; this is where.
    with using_region("GB"):
        notice = escalate([flag("threat")], "Casey")
    assert kinds(notice.routing) == ["emergency", "crisis", "crisis", "abuse"]


def test_money_pressure_alone_routes_nowhere():
    with using_region("US"):
        notice = escalate([flag("financial_pressure")], "Casey")
    assert notice.routing is None
    assert "In the United States" not in notice.to_text()


def test_below_danger_there_is_still_no_notice():
    with using_region("US"):
        assert escalate([flag("monitoring", "concern")], "Casey") is None


# -- how it renders -----------------------------------------------------------------


def test_numbers_come_after_the_steps_and_stay_narrow():
    with using_region("US"):
        text = escalate([flag("threat"), flag("monitoring")], "Casey").to_text()
    steps_end = text.index("call your local emergency number.")
    assert steps_end < text.index("In the United States:") < text.index("call 911.")
    assert "     National Domestic Violence Hotline: call 1-800-799-7233" in text
    assert all(len(line) <= 88 for line in text.splitlines())


def test_the_unset_note_renders_under_the_steps():
    text = support_notice().to_text()
    assert text.endswith("most\n   countries.")
    assert "set CONFIDANT_REGION" in text


def test_the_routing_is_in_the_json():
    with using_region("NZ"):
        data = json.loads(support_notice().model_dump_json())
    assert data["routing"]["region"] == "NZ"
    assert data["routing"]["resources"][1] == {
        "kind": "crisis",
        "name": "Need to Talk?",
        "how": "call or text 1737",
    }


# -- end to end through the CLI -----------------------------------------------------


@pytest.fixture
def offline(monkeypatch):
    monkeypatch.setattr("confidant.cli.Settings.from_env", lambda: None)


def test_flags_with_region_lists_the_numbers(capsys, monkeypatch, offline):
    canned = FlagScan(flags=[flag("monitoring")], summary="A summary.", confidence="high")
    monkeypatch.setattr("confidant.analysis.flags.structured_call", lambda **_: canned)

    assert main(["flags", PRESSURE, "--region", "uk"]) == 0
    out = capsys.readouterr().out
    head = out.split("\n\nA summary.")[0]
    assert "In the United Kingdom:\n     Emergency services: call 999." in head
    assert "National Domestic Abuse Helpline (England): call 0808 2000 247." in head
    assert "Samaritans" not in head


def test_the_environment_sets_the_region_when_the_flag_is_absent(capsys, monkeypatch, offline):
    canned = FlagScan(flags=[flag("monitoring")], summary="A summary.", confidence="high")
    monkeypatch.setattr("confidant.analysis.flags.structured_call", lambda **_: canned)
    monkeypatch.setenv(REGION_ENV, "DE")

    assert main(["flags", PRESSURE, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["escalation"]["routing"]["region"] == "DE"
    assert current_region() == "DE"  # the flag's context does not leak past main()


def test_comfort_with_region_routes_the_support_notice(capsys, monkeypatch, offline):
    canned = ComfortScan(
        what_happened="Your last messages had no reply.",
        what_it_says="The messages do not say why.",
        you_did_well=[],
        worth_knowing=None,
        next_steps=["Mute the thread for tonight."],
        owner_at_risk=True,
    )
    # The package re-exports a function named comfort, which shadows the module path.
    module = importlib.import_module("confidant.analysis.comfort")
    monkeypatch.setattr(module, "structured_call", lambda **_: canned)

    assert main(["comfort", FADED, "--region", "Australia"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("!! Some of what you wrote")
    notice = out.split("\n\nYour last messages")[0]
    assert "In Australia:" in notice
    assert "Lifeline: call 13 11 14." in notice
    assert "1800RESPECT" not in notice


def test_a_mistyped_region_still_prints_the_notice(capsys, monkeypatch, offline):
    canned = FlagScan(flags=[flag("monitoring")], summary="A summary.", confidence="high")
    monkeypatch.setattr("confidant.analysis.flags.structured_call", lambda **_: canned)

    assert main(["flags", PRESSURE, "--region", "Untied Kingdom"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("!! Something in Casey's messages is serious")
    assert "Confidant has no numbers on file for UNTIED KINGDOM." in out


@pytest.mark.parametrize("command", ["flags", "draft", "comfort", "profile", "timeline", "nudge"])
def test_every_command_that_can_show_a_notice_takes_region(capsys, command):
    with pytest.raises(SystemExit):
        main([command, "--help"])
    out = capsys.readouterr().out
    assert "--region COUNTRY" in out


def test_help_documents_the_variable(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert REGION_ENV in out
    assert "--region GB" in out
