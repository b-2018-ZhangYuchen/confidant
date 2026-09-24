"""Tests for the red-flag check. Offline: the model is replaced by a canned scan."""

from __future__ import annotations

import json

import pytest

from confidant.analysis.flags import (
    DANGER_CATEGORIES,
    Flag,
    FlagReport,
    FlagScan,
    analyze_flags,
    ground_flags,
    quote_appears_in,
)
from confidant.analysis.personality import Evidence
from confidant.cli import main
from confidant.ingest.transcript import parse_transcript, read_transcript
from confidant.prompts.flags import FLAGS_SYSTEM, build_flags_request


@pytest.fixture
def pressure():
    return read_transcript("examples/pressure_chat.txt")


def flag(
    *quotes: str,
    category: str = "boundary_pushing",
    severity: str = "concern",
    innocent_reading: str | None = "They may just be eager.",
) -> Flag:
    return Flag(
        category=category,
        severity=severity,
        behavior="Pushed after being told no.",
        evidence=[Evidence(quote=q, why_it_matters="It matters.") for q in quotes],
        innocent_reading=innocent_reading,
    )


def scan(*flags: Flag) -> FlagScan:
    return FlagScan(flags=list(flags), summary="A summary.", confidence="medium")


# -- quote grounding ------------------------------------------------------


@pytest.mark.parametrize(
    "quote",
    [
        "come on, it's not a big deal",
        "Come on, it's not a big deal.",  # case and trailing punctuation
        "come on, it’s not a big deal",  # curly apostrophe
        '"come on, it\'s not a big deal"',  # wrapped in quote marks
        "come on... everyone does it",  # ellipsis for omitted text
        "come on…everyone does it",
    ],
)
def test_faithful_quotes_are_found(quote):
    assert quote_appears_in(quote, ["come on, it's not a big deal. everyone does it"])


def test_multiline_messages_match_across_the_line_break():
    message = "your friends again? they don't even like me\nyou should be spending that time"
    assert quote_appears_in("they don't even like me you should be spending", [message])


@pytest.mark.parametrize(
    "quote",
    [
        "you are not allowed to see them",  # invented
        "everyone does it... come on",  # right words, wrong order
        "",
        "...",
    ],
)
def test_unfaithful_quotes_are_rejected(quote):
    assert not quote_appears_in(quote, ["come on, it's not a big deal. everyone does it"])


def test_a_quote_cannot_stitch_two_messages_together():
    messages = ["I like knowing", "where you are"]
    assert not quote_appears_in("I like knowing where you are", messages)
    assert not quote_appears_in("I like knowing ... where you are", messages)


# -- grounding a scan -----------------------------------------------------


def test_a_flag_with_an_invented_quote_is_discarded(pressure):
    report = ground_flags(scan(flag("I will find out where you live")), pressure)
    assert report.flags == []
    assert report.discarded == 1


def test_quotes_from_the_owner_do_not_count(pressure):
    # Jordan said this, not Casey. A flag about the match cannot rest on it.
    report = ground_flags(scan(flag("I'm not really comfortable sharing that yet")), pressure)
    assert report.flags == []
    assert report.discarded == 1


def test_bad_quotes_are_pruned_and_good_ones_kept(pressure):
    report = ground_flags(
        scan(flag("come on, it's not a big deal", "sharing is caring, babe")), pressure
    )
    assert report.discarded == 0
    [kept] = report.flags
    assert [e.quote for e in kept.evidence] == ["come on, it's not a big deal"]


@pytest.mark.parametrize("category", sorted(DANGER_CATEGORIES))
def test_danger_categories_cannot_be_softened(pressure, category):
    report = ground_flags(
        scan(flag("I like knowing where you are", category=category, severity="watch")),
        pressure,
    )
    assert report.flags[0].severity == "danger"
    assert report.has_danger


def test_other_categories_keep_the_models_tier(pressure):
    report = ground_flags(
        scan(flag("I guess I just care more than you do", category="guilt_or_blame")),
        pressure,
    )
    assert report.flags[0].severity == "concern"
    assert report.highest_tier == 2


def test_flags_are_sorted_most_serious_first(pressure):
    watch = flag("whatever", category="other", severity="watch")
    concern = flag("I guess I just care more", category="guilt_or_blame")
    danger = flag("who are you with right now", category="monitoring", severity="danger")
    report = ground_flags(scan(watch, concern, danger), pressure)
    assert [f.severity for f in report.flags] == ["danger", "concern", "watch"]


def test_an_empty_scan_is_a_valid_answer(pressure):
    report = ground_flags(scan(), pressure)
    assert report.flags == []
    assert report.highest_tier == 0
    assert not report.has_danger


# -- rendering ------------------------------------------------------------


def test_no_flags_says_so_without_alarm():
    text = FlagReport(flags=[], summary="A warm exchange.", confidence="medium").to_text()
    assert "Nothing in this transcript rises to a red flag." in text
    assert "serious" not in text


def test_innocent_reading_is_shown_below_danger_and_withheld_at_danger():
    concern = flag("whatever", innocent_reading="They were tired.")
    danger = flag(
        "who are you with",
        category="monitoring",
        severity="danger",
        innocent_reading="They were just curious.",
    )
    text = FlagReport(flags=[danger, concern], summary="s", confidence="high").to_text()
    assert "Could also be: They were tired." in text
    assert "They were just curious." not in text
    assert "[DANGER] monitoring" in text


def test_discarded_flags_are_reported():
    text = FlagReport(flags=[], summary="s", confidence="low", discarded=2).to_text()
    assert "2 flags left out" in text


# -- the prompt -----------------------------------------------------------


def test_prompt_carries_the_principles():
    lowered = FLAGS_SYSTEM.casefold()
    for phrase in ("do not diagnose", "verbatim", "boring explanation", "empty list"):
        assert phrase in lowered
    for tier in ("watch", "concern", "danger"):
        assert f"- {tier}:" in FLAGS_SYSTEM


def test_prompt_names_the_same_danger_categories_the_code_enforces():
    for category in DANGER_CATEGORIES:
        assert category in FLAGS_SYSTEM


def test_request_tags_both_sides(pressure):
    request = build_flags_request(pressure)
    assert "MATCH (Casey): come on, it's not a big deal" in request
    assert "OWNER (Jordan):" in request


# -- the analyzer, with the model faked -----------------------------------


def test_analyze_flags_grounds_what_the_model_returns(monkeypatch, pressure):
    calls = []

    def fake_structured_call(**kwargs):
        calls.append(kwargs)
        return scan(
            flag("who are you with right now", category="monitoring", severity="concern"),
            flag("a line Casey never wrote"),
        )

    monkeypatch.setattr("confidant.analysis.flags.structured_call", fake_structured_call)
    report = analyze_flags(pressure)

    [call] = calls
    assert call["schema"] is FlagScan
    assert call["system"] is FLAGS_SYSTEM
    assert "--- TRANSCRIPT ---" in call["user_content"]
    assert [f.severity for f in report.flags] == ["danger"]
    assert report.discarded == 1


def test_analyze_flags_refuses_a_one_sided_transcript(monkeypatch):
    monkeypatch.setattr(
        "confidant.analysis.flags.structured_call",
        lambda **_: pytest.fail("should not reach the model"),
    )
    conversation = parse_transcript("# owner: Sam\n# match: Robin\nSam: hello?\n")
    with pytest.raises(ValueError, match="nothing to check"):
        analyze_flags(conversation)


# -- the CLI --------------------------------------------------------------


@pytest.fixture
def fake_flags(monkeypatch, pressure):
    report = ground_flags(
        FlagScan(
            flags=[flag("who are you with right now", category="monitoring", severity="danger")],
            summary="Casey pushes for access to Jordan's whereabouts.",
            confidence="medium",
        ),
        pressure,
    )
    monkeypatch.setattr("confidant.cli.Settings.from_env", lambda: None)
    monkeypatch.setattr("confidant.cli.analyze_flags", lambda conversation, settings: report)


def test_cli_flags_prints_the_report(capsys, fake_flags):
    assert main(["flags", "examples/pressure_chat.txt"]) == 0
    out = capsys.readouterr().out
    assert "[DANGER] monitoring" in out
    assert '"who are you with right now"' in out
    assert out.startswith("!! Something in Casey's messages is serious: tracking where you are.")


def test_cli_flags_json(capsys, fake_flags):
    assert main(["flags", "examples/pressure_chat.txt", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["flags"][0]["severity"] == "danger"
    assert payload["discarded"] == 0
    assert payload["escalation"]["categories"] == ["monitoring"]
    assert payload["escalation"]["physical_risk"] is True


def test_cli_flags_without_credentials_exits_2(capsys, monkeypatch, tmp_path):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))
    assert main(["flags", "examples/sample_chat.txt"]) == 2
    assert "No Anthropic credentials found" in capsys.readouterr().err
