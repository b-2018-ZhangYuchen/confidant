"""Tests for the tier-3 escalation path. Offline: every scan is canned."""

from __future__ import annotations

import json
import typing

import pytest

from confidant.analysis.flags import DANGER_CATEGORIES, Category, Flag, FlagScan, ground_flags
from confidant.analysis.personality import Evidence
from confidant.cli import main
from confidant.ingest.transcript import read_transcript
from confidant.safety import CATEGORY_STEPS, ESCALATION_ORDER, STEPS, escalate


@pytest.fixture
def pressure():
    return read_transcript("examples/pressure_chat.txt")


def flag(quote: str, category: str, severity: str = "danger") -> Flag:
    return Flag(
        category=category,
        severity=severity,
        behavior="Something the match did.",
        evidence=[Evidence(quote=quote, why_it_matters="It matters.")],
        innocent_reading=None if severity == "danger" else "They may be tired.",
    )


def scan(*flags: Flag, summary: str = "A summary.", confidence: str = "medium") -> FlagScan:
    return FlagScan(flags=list(flags), summary=summary, confidence=confidence)


# -- when it fires --------------------------------------------------------


def test_nothing_below_danger_escalates():
    below = [flag("x", "guilt_or_blame", "concern"), flag("y", "other", "watch")]
    assert escalate(below, "Casey") is None
    assert escalate([], "Casey") is None
    assert escalate([flag("x", "evasion", "watch")], "Casey") is None


def test_the_severity_floor_feeds_the_escalation(pressure):
    # Filed as "watch" by the model, floored to danger by grounding, and so escalated.
    report = ground_flags(
        scan(flag("I like knowing where you are", "monitoring", "watch")), pressure
    )
    assert report.escalation is not None
    assert report.escalation.categories == ["monitoring"]


def test_a_danger_flag_that_fails_grounding_does_not_escalate(pressure):
    report = ground_flags(scan(flag("I know where you live", "threat")), pressure)
    assert report.escalation is None
    assert report.discarded == 1
    assert report.discarded_danger == 1


@pytest.mark.parametrize("confidence", ["low", "medium", "high"])
def test_confidence_does_not_gate_the_notice(pressure, confidence):
    report = ground_flags(
        scan(flag("who are you with right now", "monitoring"), confidence=confidence), pressure
    )
    assert report.escalation is not None


def test_financial_pressure_escalates_only_when_filed_as_danger(pressure):
    # financial_pressure has no floor, so the model's tier decides.
    mild = ground_flags(scan(flag("whatever", "financial_pressure", "concern")), pressure)
    assert mild.escalation is None
    serious = ground_flags(scan(flag("whatever", "financial_pressure", "danger")), pressure)
    assert serious.escalation is not None
    assert STEPS["no_money"] in serious.escalation.steps


# -- what it says ---------------------------------------------------------


def test_headline_names_the_behavior_most_immediate_first():
    notice = escalate(
        [flag("a", "isolation"), flag("b", "monitoring"), flag("c", "threat")], "Casey"
    )
    assert notice.headline == (
        "Something in Casey's messages is serious: a threat, tracking where you are, "
        "and pressure to pull away from friends or family."
    )
    assert notice.categories == ["threat", "monitoring", "isolation"]


def test_only_danger_flags_reach_the_headline():
    notice = escalate([flag("a", "monitoring"), flag("b", "guilt_or_blame", "concern")], "Casey")
    assert notice.categories == ["monitoring"]
    assert "guilt" not in notice.headline


def test_steps_are_deduplicated_and_ordered_by_the_most_serious_category():
    notice = escalate([flag("a", "sexual_pressure"), flag("b", "threat")], "Casey")
    assert notice.steps.count(STEPS["keep_records"]) == 1
    # Threat comes first, so its steps lead; the closing steps come last.
    assert notice.steps[0] == STEPS["threat_counts"]
    assert notice.steps[-2:] == [STEPS["no_reply_owed"], STEPS["emergency"]]


def test_every_notice_suggests_someone_they_trust():
    for category in ESCALATION_ORDER:
        notice = escalate([flag("a", category)], "Casey")
        assert any("someone you trust" in step for step in notice.steps), category


def test_emergency_line_is_for_physical_risk_only():
    scam = escalate([flag("a", "financial_pressure")], "Casey")
    assert not scam.physical_risk
    assert STEPS["emergency"] not in scam.steps
    for category in sorted(DANGER_CATEGORIES):
        notice = escalate([flag("a", category)], "Casey")
        assert notice.physical_risk, category
        assert STEPS["emergency"] in notice.steps


def test_every_category_is_ordered_and_named():
    # A category the model can return but the notice cannot name would crash at the
    # worst possible moment, so this is checked against the schema itself.
    assert set(ESCALATION_ORDER) == set(typing.get_args(Category))
    for category in ESCALATION_ORDER:
        escalate([flag("a", category)], "Casey")


def test_every_referenced_step_exists():
    for keys in CATEGORY_STEPS.values():
        for key in keys:
            assert key in STEPS


def test_the_copy_follows_the_principles():
    text = " ".join(STEPS.values()).casefold()
    # No diagnosis and no verdicts on the person, not even in the safety notice.
    for word in ("narcissist", "abuser", "toxic", "manipulative", "psycho", "controlling"):
        assert word not in text
    # No specific hotline yet: routing to the right one depends on where the owner is.
    assert not any(ch.isdigit() for ch in text)


# -- how it renders -------------------------------------------------------


def test_the_notice_comes_before_a_milder_summary(pressure):
    report = ground_flags(
        scan(flag("who are you with right now", "monitoring"), summary="Mostly friendly."),
        pressure,
    )
    text = report.to_text()
    assert text.startswith("!! Something in Casey's messages is serious")
    assert text.index("!!") < text.index("Mostly friendly.") < text.index("[DANGER]")


def test_no_notice_without_danger(pressure):
    report = ground_flags(scan(flag("whatever", "evasion", "watch")), pressure)
    assert "!!" not in report.to_text()


def test_a_discarded_danger_flag_is_mentioned_without_repeating_it(pressure):
    report = ground_flags(scan(flag("I know where you live", "threat")), pressure)
    text = report.to_text()
    assert "One of them was filed as serious." in text
    assert "felt unsafe to you" in text
    assert "I know where you live" not in text


def test_a_discarded_concern_does_not_mention_danger(pressure):
    report = ground_flags(scan(flag("a line nobody wrote", "evasion", "concern")), pressure)
    text = report.to_text()
    assert "1 flag left out" in text
    assert "filed as serious" not in text


# -- end to end through the CLI -------------------------------------------


def test_cli_leads_with_the_notice_for_the_pressure_example(capsys, monkeypatch):
    canned = scan(
        flag("I like knowing where you are", "monitoring"),
        flag("you should be spending that time with me", "isolation"),
        flag("I guess I just care more than you do", "guilt_or_blame", "concern"),
    )
    monkeypatch.setattr("confidant.cli.Settings.from_env", lambda: None)
    monkeypatch.setattr("confidant.analysis.flags.structured_call", lambda **_: canned)

    assert main(["flags", "examples/pressure_chat.txt"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("!! Something in Casey's messages is serious: tracking where you are")
    assert all(len(line) <= 88 for line in out.split("\n\nA summary.")[0].splitlines())

    assert main(["flags", "examples/pressure_chat.txt", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["escalation"]["headline"] == (
        "Something in Casey's messages is serious: tracking where you are "
        "and pressure to pull away from friends or family."
    )
    assert payload["escalation"]["categories"] == ["monitoring", "isolation"]
    assert payload["discarded_danger"] == 0
