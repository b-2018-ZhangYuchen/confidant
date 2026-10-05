"""Tests for the redaction layer: what is stripped, what is kept, and what comes back.

The end-to-end tests at the bottom replay recordings through the real analyzers and look
at the request the SDK would have received, which is the only place that matters.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from confidant.analysis.flags import FlagScan, ground_flags
from confidant.analysis.personality import Evidence
from confidant.cli import main
from confidant.ingest.transcript import parse_transcript, read_transcript
from confidant.models import Conversation, Message, Role
from confidant.recording import run_analysis
from confidant.redaction import redact


def convo(*match_lines: str, owner: str = "Sam", match: str = "Robin", **kwargs) -> Conversation:
    return Conversation(
        owner_name=owner,
        match_name=match,
        messages=[Message(role=Role.MATCH, text=t, sender=match) for t in match_lines],
        **kwargs,
    )


def redacted(text: str, **kwargs) -> str:
    [message] = redact(convo(text, **kwargs)).conversation.messages
    return message.text


# -- what is stripped ---------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("call me on (555) 010-4477", "call me on [PHONE_1]"),
        ("my cell is +1 555 010 4477.", "my cell is [PHONE_1]."),
        ("it's 555.010.4477", "it's [PHONE_1]"),
        ("card ends up as 4111 1111 1111 1111", "card ends up as [NUMBER_1]"),
        ("email me: robin.l+dates@example.com", "email me: [EMAIL_1]"),
        ("I'm @robin_bikes on there", "I'm [HANDLE_1] on there"),
        ("see https://example.com/robin?x=1, it's me", "see [LINK_1], it's me"),
        ("I live at 214 Linden Street, apt 3B", "I live at [ADDRESS_1]"),
        ("meet at 9 Old Mill Rd.", "meet at [ADDRESS_1]"),
        ("pick me up from 42 elm st", "pick me up from [ADDRESS_1]"),
    ],
)
def test_identifying_details_are_replaced(text, expected):
    assert redacted(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "can you send $1500 by friday",  # the amount is what a money flag is about
        "it was like 2,000,000 dollars lol",
        "see you at 7:30, or 19:45 at the latest",
        "we went on 2026-03-02 and again 03.05.2026",
        "I was there 2019-2023",
        "I ran 5 miles down the road",
        "10 more minutes this way",
        "I have 2 dogs and 3 cats",
        "that's 100% true",
    ],
)
def test_ordinary_details_are_kept(text):
    assert redacted(text) == text


def test_both_names_go_whatever_the_case():
    text = redacted("robin here. ROBIN is my name and Sam's my favorite person", match="Robin")
    assert text == "[MATCH] here. [MATCH] is my name and [OWNER]'s my favorite person"


def test_a_full_name_and_its_parts_share_a_placeholder():
    assert redacted("Robin Lee, or just Lee", match="Robin Lee") == "[MATCH], or just [MATCH]"


def test_names_inside_words_are_left_alone():
    assert redacted("Samantha and I went to Samarkand", owner="Sam") == (
        "Samantha and I went to Samarkand"
    )


def test_a_speaker_labelled_me_does_not_eat_the_pronoun():
    result = redact(convo("tell me more", owner="Me"))
    assert result.conversation.messages[0].text == "tell me more"
    assert result.conversation.owner_name == "[OWNER]"


def test_other_names_come_from_the_transcript_and_the_caller():
    conversation = convo("Maya and Dev say hi", private_names=["Maya"])
    result = redact(conversation, names=["Dev"])
    assert result.conversation.messages[0].text == "[NAME_1] and [NAME_2] say hi"
    assert result.originals["[NAME_2]"] == "Dev"


def test_the_same_detail_keeps_its_placeholder_and_a_new_one_gets_the_next():
    result = redact(convo("555 010 4477 is mine", "again: (555) 010-4477", "work: 555-010-9900"))
    texts = [m.text for m in result.conversation.messages]
    assert texts == ["[PHONE_1] is mine", "again: [PHONE_1]", "work: [PHONE_2]"]


def test_an_email_is_not_also_a_handle_or_a_name():
    assert redacted("robin@example.com") == "[EMAIL_1]"


def test_senders_and_header_names_are_replaced_and_the_source_dropped():
    result = redact(read_transcript("examples/pressure_chat.txt"))
    conversation = result.conversation
    assert (conversation.owner_name, conversation.match_name) == ("[OWNER]", "[MATCH]")
    assert {m.sender for m in conversation} == {"[OWNER]", "[MATCH]"}
    assert conversation.source is None
    # Nothing else about the conversation changes.
    original = read_transcript("examples/pressure_chat.txt")
    assert [m.timestamp for m in conversation] == [m.timestamp for m in original]
    assert [m.role for m in conversation] == [m.role for m in original]


def test_the_original_is_not_modified():
    conversation = convo("call (555) 010-4477")
    redact(conversation)
    assert conversation.messages[0].text == "call (555) 010-4477"
    assert conversation.match_name == "Robin"


# -- what comes back ----------------------------------------------------------


def test_restore_puts_every_issued_placeholder_back():
    result = redact(convo("I'm at 214 Linden Street", "text (555) 010-4477"))
    text = "[MATCH] gave [OWNER] [ADDRESS_1] and [PHONE_1]"
    assert result.restore_text(text) == "Robin gave Sam 214 Linden Street and (555) 010-4477"


def test_restore_leaves_placeholders_it_never_issued():
    result = redact(convo("hello"))
    assert result.restore_text("[PHONE_9] and [MATCH]") == "[PHONE_9] and Robin"


def test_restore_walks_nested_models():
    result = redact(convo("call (555) 010-4477"))
    scan = FlagScan(
        flags=[],
        summary="[MATCH] shared [PHONE_1] with [OWNER].",
        confidence="low",
    )
    assert result.restore(scan).summary == "Robin shared (555) 010-4477 with Sam."


def test_grounding_in_placeholders_then_restoring_keeps_a_true_quote():
    conversation = convo("robin@example.com, send me yours", match="Robin")
    result = redact(conversation)
    quote = "[EMAIL_1], send me yours"
    scan = FlagScan(
        flags=[
            {
                "category": "other",
                "severity": "watch",
                "behavior": "[MATCH] asked for contact details early.",
                "evidence": [Evidence(quote=quote, why_it_matters="Early.")],
                "innocent_reading": "Wants to move off the app.",
            }
        ],
        summary="Nothing serious.",
        confidence="low",
    )
    report = result.restore(ground_flags(scan, result.conversation))
    assert report.discarded == 0
    assert report.flags[0].evidence[0].quote == "robin@example.com, send me yours"
    assert report.flags[0].behavior == "Robin asked for contact details early."


def test_summary_counts_each_kind():
    result = redact(read_transcript("examples/details_chat.txt"))
    assert result.summary() == (
        "3 names, 1 phone number, 1 email address, 1 street address, 1 handle"
    )


# -- the transcript directive -------------------------------------------------


def test_redact_directives_accumulate():
    conversation = parse_transcript(
        "# owner: Sam\n# redact: Maya, Dev\nRobin: hi\n# redact: Ash\nSam: hey\n"
    )
    assert conversation.private_names == ["Maya", "Dev", "Ash"]


# -- end to end: what the SDK is actually handed ------------------------------


@pytest.mark.parametrize(
    ("analysis", "recording_name", "transcript", "names"),
    [
        ("flags", "flags_pressure", "examples/pressure_chat.txt", ("Casey", "Jordan")),
        ("analyze", "analyze_sample", "examples/sample_chat.txt", ("Robin", "Sam")),
    ],
)
def test_no_name_reaches_the_request(replay_client, analysis, recording_name, transcript, names):
    client = replay_client(recording_name)
    run_analysis(analysis, transcript, client=client)
    [request] = client.requests
    sent = json.dumps(request["system"]) + json.dumps(request["messages"])
    for name in names:
        assert name not in sent
    assert "[MATCH]" in sent


def test_the_report_comes_back_with_names(replay_client):
    report = run_analysis(
        "flags", "examples/pressure_chat.txt", client=replay_client("flags_pressure")
    )
    assert report.summary.startswith("Casey asks for Jordan's location")
    assert report.escalation.headline.startswith("Something in Casey's messages")
    assert "[MATCH]" not in report.model_dump_json()


# -- the CLI ------------------------------------------------------------------


def test_cli_redact_prints_what_would_be_sent(capsys):
    assert main(["redact", "examples/details_chat.txt", "--redact", "Theo M"]) == 0
    out = capsys.readouterr().out
    for detail in ("Priya", "Maya", "(555) 010-4477", "theo.m@example.com", "@theo.cooks"):
        transcript_part = out.split("--- END TRANSCRIPT ---")[0]
        assert detail not in transcript_part
    assert "MATCH: or instagram, [HANDLE_1]. I'm slow on email ([EMAIL_1])" in out
    assert "  [PHONE_1]    (555) 010-4477" in out
    assert "$40" in out


def test_cli_redact_never_needs_credentials(capsys, monkeypatch):
    monkeypatch.setattr(
        "confidant.cli.Settings.from_env", lambda: pytest.fail("redact is local only")
    )
    assert main(["redact", "examples/pressure_chat.txt"]) == 0


def test_readme_redact_example_matches_the_cli(capsys):
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    marker = "confidant redact examples/details_chat.txt\n```\n\nprints\n\n```\n"
    documented = readme[readme.index(marker) + len(marker) :].split("\n```", 1)[0]
    assert main(["redact", "examples/details_chat.txt"]) == 0
    assert capsys.readouterr().out == documented + "\n"
