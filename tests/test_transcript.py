"""Tests for the transcript parser. No network, no API key needed."""

from __future__ import annotations

from datetime import datetime

import pytest

from confidant.ingest.transcript import TranscriptError, parse_transcript, read_transcript
from confidant.models import Role

BASIC = """\
# owner: Sam
# match: Robin

Robin: hey there
Sam: hi!
"""


def test_directives_identify_both_speakers():
    conversation = parse_transcript(BASIC)
    assert conversation.owner_name == "Sam"
    assert conversation.match_name == "Robin"
    assert [m.role for m in conversation] == [Role.MATCH, Role.OWNER]


def test_arguments_override_directives():
    conversation = parse_transcript(BASIC, owner="Robin")
    assert conversation.owner_name == "Robin"
    assert conversation.match_name == "Sam"
    assert [m.role for m in conversation] == [Role.OWNER, Role.MATCH]


def test_match_is_inferred_when_only_the_owner_is_named():
    conversation = parse_transcript("Robin: hey\nSam: hi", owner="Sam")
    assert conversation.match_name == "Robin"


def test_comments_that_are_not_directives_are_ignored():
    conversation = parse_transcript("# owner: Sam\n# note to self: ask about the cat\nRobin: hey")
    assert len(conversation) == 1


def test_timestamps_are_optional_and_mixable():
    conversation = parse_transcript("[2026-03-02 19:04] Robin: hey\nSam: hi", owner="Sam")
    assert conversation.messages[0].timestamp == datetime(2026, 3, 2, 19, 4)
    assert conversation.messages[1].timestamp is None


@pytest.mark.parametrize(
    ("stamp", "expected"),
    [
        ("2026-03-02 19:04:30", datetime(2026, 3, 2, 19, 4, 30)),
        ("2026-03-02 19:04", datetime(2026, 3, 2, 19, 4)),
        ("2026-03-02T19:04:00", datetime(2026, 3, 2, 19, 4)),
        ("2026-03-02", datetime(2026, 3, 2)),
        ("03/02/2026 19:04", datetime(2026, 3, 2, 19, 4)),
    ],
)
def test_common_timestamp_formats(stamp, expected):
    conversation = parse_transcript(f"[{stamp}] Robin: hey", owner="Sam", match="Robin")
    assert conversation.messages[0].timestamp == expected


def test_unrecognized_timestamp_is_an_error():
    with pytest.raises(TranscriptError, match="Unrecognized timestamp"):
        parse_transcript("[last tuesday] Robin: hey", owner="Sam", match="Robin")


def test_indented_lines_continue_the_previous_message():
    conversation = parse_transcript(
        "Robin: first line\n    second line\n\t third line\nSam: ok", owner="Sam"
    )
    assert conversation.messages[0].text == "first line\nsecond line\nthird line"
    assert len(conversation) == 2


def test_a_colon_inside_a_message_does_not_start_a_new_one():
    """Once both speakers are known, prose that looks like 'name: text' stays as text."""
    conversation = parse_transcript(
        "Robin: hey\nSam: one thing: I had fun\nRobin: same", owner="Sam"
    )
    assert len(conversation) == 3
    assert conversation.messages[1].text == "one thing: I had fun"


def test_a_leading_colon_line_folds_into_the_message_above_it():
    conversation = parse_transcript(
        "Robin: hey\nSam: hi\nreminder: bring the book\nRobin: ok", owner="Sam"
    )
    assert len(conversation) == 3
    assert conversation.messages[1].text == "hi\nreminder: bring the book"


def test_urls_do_not_get_mistaken_for_speakers():
    conversation = parse_transcript(
        "Robin: look at this\nhttps://example.com/thing\nSam: nice", owner="Sam"
    )
    assert len(conversation) == 2
    assert "https://example.com/thing" in conversation.messages[0].text


def test_missing_owner_is_a_clear_error():
    with pytest.raises(TranscriptError, match="which side of this conversation is yours"):
        parse_transcript("Robin: hey\nAlex: hi")


def test_group_chats_are_rejected():
    """A third name that keeps opening lines is a person, not stray punctuation."""
    with pytest.raises(TranscriptError, match="one-on-one"):
        parse_transcript(
            "Robin: hey\nSam: hi\nAlex: hello\nRobin: sup\nAlex: what did I miss", owner="Sam"
        )


def test_a_single_odd_colon_line_is_not_treated_as_a_third_speaker():
    """The flip side of the rule above: one-off prose still folds into the message."""
    conversation = parse_transcript(
        "Robin: hey\nSam: hi\nreminder: bring the book\nRobin: ok", owner="Sam"
    )
    assert len(conversation) == 3


def test_a_monologue_is_rejected():
    with pytest.raises(TranscriptError, match="Only 'Sam' speaks"):
        parse_transcript("Sam: hello?\nSam: anyone there", owner="Sam")


def test_empty_transcript_is_rejected():
    with pytest.raises(TranscriptError, match="no messages"):
        parse_transcript("# owner: Sam\n\n\n")


def test_leading_junk_with_nothing_to_attach_to_is_rejected():
    with pytest.raises(TranscriptError, match="nothing before it"):
        parse_transcript("just some loose text\nRobin: hey", owner="Sam")


def test_read_transcript_reads_a_file(tmp_path):
    path = tmp_path / "chat.txt"
    path.write_text(BASIC, encoding="utf-8")
    conversation = read_transcript(path)
    assert len(conversation) == 2
    assert conversation.source == str(path)


def test_read_transcript_reports_a_missing_file(tmp_path):
    with pytest.raises(TranscriptError, match="No transcript at"):
        read_transcript(tmp_path / "nope.txt")


def test_the_bundled_example_parses():
    conversation = read_transcript("examples/sample_chat.txt")
    assert conversation.owner_name == "Sam"
    assert conversation.match_name == "Robin"
    assert len(conversation) == 17
    assert conversation.messages[0].timestamp == datetime(2026, 3, 2, 19, 4)
