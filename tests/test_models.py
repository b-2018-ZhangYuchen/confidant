"""Tests for the core data types. No network, no API key needed."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from confidant.models import Conversation, Message, Role


def msg(role: Role, text: str, *, sender: str | None = None, ts: datetime | None = None) -> Message:
    return Message(
        role=role,
        text=text,
        sender=sender or ("Sam" if role is Role.OWNER else "Robin"),
        timestamp=ts,
    )


def test_message_rejects_empty_text():
    with pytest.raises(ValueError, match="text cannot be empty"):
        Message(role=Role.OWNER, text="   ", sender="Sam")


def test_message_rejects_empty_sender():
    with pytest.raises(ValueError, match="sender cannot be empty"):
        Message(role=Role.OWNER, text="hi", sender=" ")


def test_word_count_counts_words_not_characters():
    assert msg(Role.MATCH, "three whole words").word_count == 3


def test_word_count_handles_multiline_messages():
    assert msg(Role.MATCH, "first line\nsecond line here").word_count == 5


@pytest.fixture
def conversation() -> Conversation:
    start = datetime(2026, 3, 2, 19, 0)
    return Conversation(
        match_name="Robin",
        owner_name="Sam",
        messages=[
            msg(Role.MATCH, "hey which route do you bike", ts=start),
            msg(Role.OWNER, "river path", ts=start + timedelta(hours=1)),
            msg(
                Role.MATCH,
                "the one past the old mill I love that one",
                ts=start + timedelta(days=2),
            ),
        ],
    )


def test_len_and_iteration(conversation):
    assert len(conversation) == 3
    assert [m.role for m in conversation] == [Role.MATCH, Role.OWNER, Role.MATCH]


def test_by_role_splits_the_two_sides(conversation):
    assert len(conversation.match_messages) == 2
    assert len(conversation.owner_messages) == 1


def test_mean_words_averages_per_side(conversation):
    assert conversation.mean_words(Role.OWNER) == 2.0
    assert conversation.mean_words(Role.MATCH) == pytest.approx(8.0)


def test_mean_words_is_zero_for_a_silent_side():
    solo = Conversation(
        match_name="Robin",
        owner_name="Sam",
        messages=[msg(Role.MATCH, "hello there")],
    )
    assert solo.mean_words(Role.OWNER) == 0.0


def test_effort_ratio_compares_the_two_sides(conversation):
    assert conversation.effort_ratio == pytest.approx(8.0 / 2.0)


def test_effort_ratio_is_none_when_the_owner_never_spoke():
    """Dividing by an absent side would be a nonsense number, so we decline to give one."""
    solo = Conversation(
        match_name="Robin",
        owner_name="Sam",
        messages=[msg(Role.MATCH, "hello there")],
    )
    assert solo.effort_ratio is None


def test_timespan_measures_first_to_last(conversation):
    assert conversation.timespan == timedelta(days=2)


def test_timespan_is_none_without_two_timestamps():
    undated = Conversation(
        match_name="Robin",
        owner_name="Sam",
        messages=[msg(Role.MATCH, "hey"), msg(Role.OWNER, "hi")],
    )
    assert undated.timespan is None


def test_last_activity_picks_the_latest_timestamp(conversation):
    assert conversation.last_activity == datetime(2026, 3, 4, 19, 0)


def test_transcript_round_trips_through_the_parser(conversation):
    from confidant.ingest.transcript import parse_transcript

    reparsed = parse_transcript(conversation.transcript(), owner="Sam", match="Robin")
    assert [m.text for m in reparsed] == [m.text for m in conversation]
    assert [m.role for m in reparsed] == [m.role for m in conversation]
