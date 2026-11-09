"""The local-model backend, with the model replaced by a function: no mlx-lm needed."""

from __future__ import annotations

import json
import sys

import pytest

from confidant.analysis.flags import FlagBackend, FlagScan, analyze_flags
from confidant.ingest.transcript import parse_transcript, read_transcript
from confidant.lab import LocalOutputError, MlxBackend, build_local_messages, parse_scan
from confidant.lab.backend import split_stitched

GOOD = {
    "flags": [
        {
            "category": "monitoring",
            "severity": "watch",
            "behavior": "Asks where the owner is and who with.",
            "evidence": [
                {"quote": "who are you with right now", "why_it_matters": "Checks up on them."}
            ],
            "innocent_reading": None,
        }
    ],
    "summary": "One thing worth watching.",
    "confidence": "medium",
}


@pytest.fixture
def pressure():
    return read_transcript("examples/pressure_chat.txt")


def answering(text: str) -> MlxBackend:
    return MlxBackend("no-model", generate=lambda messages: text)


# -- reading the answer ---------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        json.dumps(GOOD),
        "Here is my answer:\n" + json.dumps(GOOD, indent=2) + "\nHope that helps.",
        "```json\n" + json.dumps(GOOD) + "\n```",
    ],
)
def test_a_scan_is_found_inside_whatever_came_back(raw):
    assert parse_scan(raw).flags[0].category == "monitoring"


def test_braces_inside_strings_do_not_end_the_object():
    answer = dict(GOOD, summary="They wrote {this} and }that{.")
    assert parse_scan(json.dumps(answer)).summary == "They wrote {this} and }that{."


def test_a_stray_brace_before_the_json_is_skipped():
    assert parse_scan("I'd say { roughly:\n" + json.dumps(GOOD)).confidence == "medium"


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("Nothing to flag here.", "no complete JSON"),
        (json.dumps(GOOD)[:-5], "no complete JSON"),  # cut off by max_tokens
        (json.dumps(dict(GOOD, confidence="certain")), "confidence"),
        (json.dumps({"summary": "fine"}), "flags"),
    ],
)
def test_an_unusable_answer_is_an_error_not_an_empty_scan(raw, message):
    with pytest.raises(LocalOutputError, match=message) as caught:
        parse_scan(raw)
    assert caught.value.raw == raw


# -- the backend ----------------------------------------------------------


def test_the_local_backend_meets_the_protocol():
    assert isinstance(MlxBackend("no-model"), FlagBackend)


def test_building_a_backend_loads_nothing():
    MlxBackend("no-model")
    assert "mlx_lm" not in sys.modules


def test_the_prompt_is_claudes_plus_the_schema(pressure):
    system, user = build_local_messages(pressure)
    assert system["role"] == "system" and user["role"] == "user"
    assert "JSON Schema" in system["content"]
    assert '"innocent_reading"' in system["content"]
    assert "--- TRANSCRIPT ---" in user["content"]


def test_a_local_scan_goes_through_the_same_checks_as_claudes(pressure):
    backend = answering(json.dumps(GOOD))
    report = analyze_flags(pressure, backend=backend)

    # Filed as "watch", but monitoring has a danger floor whoever filed it.
    assert [f.severity for f in report.flags] == ["danger"]
    assert report.escalation is not None
    assert backend.last_raw == json.dumps(GOOD)


def test_the_local_model_sees_only_redacted_names(pressure):
    shown = []

    def generate(messages):
        shown.append(messages)
        return json.dumps(GOOD)

    analyze_flags(pressure, backend=MlxBackend("no-model", generate=generate))
    text = json.dumps(shown)
    assert "Casey" not in text and "Jordan" not in text


def test_an_unusable_answer_reaches_the_caller(pressure):
    backend = answering("I can't help with that.")
    with pytest.raises(LocalOutputError):
        analyze_flags(pressure, backend=backend)
    assert backend.last_raw == "I can't help with that."


def test_without_mlx_lm_the_error_says_how_to_get_it(pressure, monkeypatch):
    monkeypatch.setitem(sys.modules, "mlx_lm", None)
    with pytest.raises(RuntimeError, match=r"confidant-agent\[lab\]"):
        MlxBackend("no-model").scan(pressure)


# -- stitched quotes ------------------------------------------------------


def _stitched(*quotes: str) -> FlagScan:
    flag = dict(GOOD["flags"][0], evidence=[{"quote": q, "why_it_matters": "x"} for q in quotes])
    return FlagScan.model_validate(dict(GOOD, flags=[flag]))


def test_a_quote_that_runs_two_whole_messages_together_is_split():
    conversation = parse_transcript(
        "# owner: Sam\n# match: Robin\nSam: hi\nRobin: you will regret it\nRobin: I mean it\n"
    )
    repaired = split_stitched(_stitched("you will regret it, I mean it"), conversation)
    assert [e.quote for e in repaired.flags[0].evidence] == ["you will regret it", "I mean it"]


def test_a_partial_or_invented_stitch_is_left_for_grounding_to_discard():
    conversation = parse_transcript(
        "# owner: Sam\n# match: Robin\nSam: hi\nRobin: you will regret it\nRobin: I mean it\n"
    )
    for quote in ("you will regret it, and I mean it", "regret it, I mean it", "hello"):
        repaired = split_stitched(_stitched(quote), conversation)
        assert [e.quote for e in repaired.flags[0].evidence] == [quote]


def test_the_owners_messages_are_never_used_to_repair_a_quote():
    conversation = parse_transcript(
        "# owner: Sam\n# match: Robin\nSam: stop\nRobin: you will regret it\n"
    )
    repaired = split_stitched(_stitched("you will regret it, stop"), conversation)
    assert [e.quote for e in repaired.flags[0].evidence] == ["you will regret it, stop"]
