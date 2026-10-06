"""Tests for the keep-in-touch model: decay over last contact, weighted by reciprocity.

Offline. The danger case is read from the recorded flags response, saved through the same
path ``profile --update`` uses.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from confidant.config import Settings
from confidant.contact import (
    DEFAULT_RHYTHM,
    HALF_LIFE,
    MAX_RHYTHM,
    MIN_RHYTHM,
    ContactState,
    Reciprocity,
    assess,
    assess_everyone,
    assess_person,
)
from confidant.ingest.transcript import parse_transcript, read_transcript
from confidant.profile import read_conversation
from confidant.recording import ReplayClient
from confidant.store import ReadingKind, Store

RECORDINGS = Path(__file__).parent / "fixtures" / "recorded"
HEAD = "# owner: Sam\n# match: Robin\n"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CONFIDANT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)


def chat(*lines: str, head: str = HEAD):
    return parse_transcript(head + "\n".join(lines) + "\n")


def at(text: str) -> datetime:
    return datetime.fromisoformat(text)


# Three evenings, a few days apart, each a real exchange: the usual gap is about three
# days, both sides start things, and the last stretch has both of them in it.
EVEN = chat(
    "[2026-03-02 19:00] Robin: how was the ride?",
    "[2026-03-02 19:10] Sam: windy. you?",
    "[2026-03-02 19:12] Robin: stayed in",
    "[2026-03-05 20:00] Sam: did you try the cart?",
    "[2026-03-05 20:30] Robin: yes, you were right",
    "[2026-03-08 18:00] Robin: free sunday?",
    "[2026-03-08 18:05] Sam: yes!",
)
LAST = at("2026-03-08 18:05")


# -- state ------------------------------------------------------------------------


def test_inside_the_usual_gap_nothing_is_due():
    contact = assess("Robin", [EVEN], now=LAST + timedelta(days=1))
    assert contact.state is ContactState.IN_RHYTHM
    assert contact.priority == 0
    assert not contact.suggested


def test_past_the_usual_gap_a_warm_thread_is_due():
    contact = assess("Robin", [EVEN], now=LAST + timedelta(days=5))
    assert contact.rhythm_measured
    assert timedelta(days=2) < contact.rhythm < timedelta(days=4)
    assert contact.state is ContactState.DUE
    assert contact.suggested
    assert contact.headline() == "Robin: worth a message. You last talked 5 days ago."


def test_after_two_half_lives_the_thread_has_faded():
    contact = assess("Robin", [EVEN], now=LAST + 2 * HALF_LIFE + timedelta(days=1))
    assert contact.state is ContactState.FADED
    assert not contact.suggested
    assert "a choice to make, not a reply that is owed" in contact.to_text()


def test_their_unanswered_message_is_your_turn_however_recent():
    later = chat(*EVEN.transcript().splitlines(), "[2026-03-10 09:00] Robin: morning!")
    contact = assess("Robin", [later], now=at("2026-03-10 09:20"))
    assert contact.state is ContactState.YOUR_TURN
    assert contact.headline() == (
        "Robin: your turn. They wrote last, 20 minutes ago, and have not had an answer."
    )
    assert contact.priority > 0.9


def test_your_unanswered_message_is_never_suggested():
    later = chat(*EVEN.transcript().splitlines(), "[2026-03-10 09:00] Sam: morning!")
    contact = assess("Robin", [later], now=at("2026-03-20 09:00"))
    assert contact.state is ContactState.WAITING
    assert contact.priority == 0
    text = contact.to_text()
    assert "Another message now would be your second in a row." in text
    assert "A slow reply is usually a busy week" in text


def test_a_reply_inside_the_last_stretch_makes_it_an_exchange_whoever_spoke_last():
    # Sam wrote last, but Robin answered within the same stretch: the evening ended, and
    # nothing is waiting on anyone.
    contact = assess("Robin", [EVEN], now=LAST + timedelta(days=5))
    assert contact.last_message.role.value == "owner"
    assert contact.state is ContactState.DUE


# -- decay ------------------------------------------------------------------------


def test_warmth_halves_every_half_life():
    fresh = assess("Robin", [EVEN], now=LAST)
    one = assess("Robin", [EVEN], now=LAST + HALF_LIFE)
    two = assess("Robin", [EVEN], now=LAST + 2 * HALF_LIFE)
    assert fresh.warmth == 1.0
    assert one.warmth == pytest.approx(0.5)
    assert two.warmth == pytest.approx(0.25)


def test_priority_rises_past_the_usual_gap_then_decays():
    days = [4, 6, 10, 20, 40]
    priorities = [assess("Robin", [EVEN], now=LAST + timedelta(days=d)).priority for d in days]
    assert priorities[0] < priorities[1]
    assert priorities[1] > priorities[2] > priorities[3] > priorities[4]
    assert all(0 <= p <= 1 for p in priorities)


def test_a_clock_behind_the_export_reads_as_just_now():
    contact = assess("Robin", [EVEN], now=LAST - timedelta(minutes=5))
    assert contact.silence == timedelta(0)
    assert contact.warmth == 1.0


# -- rhythm -----------------------------------------------------------------------


def test_too_few_quiet_gaps_assume_a_rhythm_and_say_so():
    short = chat("[2026-03-02 19:00] Robin: hey", "[2026-03-02 19:05] Sam: hi")
    contact = assess("Robin", [short], now=at("2026-03-04 19:00"))
    assert not contact.rhythm_measured
    assert contact.rhythm == DEFAULT_RHYTHM
    assert "Too few quiet stretches to tell your usual rhythm" in contact.to_text()
    assert contact.state is ContactState.IN_RHYTHM


def test_rhythm_is_clamped_to_a_day_and_two_weeks():
    hourly = chat(
        "[2026-03-02 08:00] Robin: a",
        "[2026-03-02 08:01] Sam: b",
        "[2026-03-02 17:00] Robin: c",
        "[2026-03-02 17:01] Sam: d",
        "[2026-03-03 02:00] Robin: e",
        "[2026-03-03 02:01] Sam: f",
    )
    monthly = chat(
        "[2026-01-01 10:00] Robin: a",
        "[2026-01-01 10:01] Sam: b",
        "[2026-02-01 10:00] Robin: c",
        "[2026-02-01 10:01] Sam: d",
        "[2026-03-04 10:00] Robin: e",
        "[2026-03-04 10:01] Sam: f",
    )
    assert assess("Robin", [hourly], now=at("2026-03-03 03:00")).rhythm == MIN_RHYTHM
    assert assess("Robin", [monthly], now=at("2026-03-05 10:00")).rhythm == MAX_RHYTHM


def test_gaps_are_measured_within_a_conversation_never_across_two():
    first = chat("[2026-01-01 10:00] Robin: a", "[2026-01-01 10:05] Sam: b")
    second = chat("[2026-03-01 10:00] Robin: c", "[2026-03-01 10:05] Sam: d")
    contact = assess("Robin", [first, second], now=at("2026-03-02 10:00"))
    # The two months between the chats is not a gap in either of them.
    assert not contact.rhythm_measured
    assert contact.last_message.text == "d"


def test_an_undated_message_breaks_the_chain_but_not_the_last_contact():
    mixed = chat(
        "[2026-03-02 19:00] Robin: hey",
        "[2026-03-02 19:05] Sam: hi",
        "Robin: (pasted without a time)",
    )
    contact = assess("Robin", [mixed], now=at("2026-03-03 19:00"))
    assert contact.last_message.text == "hi"
    assert contact.undated == 1
    assert "(1 message without a timestamp left out.)" in contact.to_text()


def test_no_timestamps_at_all_cannot_say_how_long():
    undated = chat("Robin: hey", "Sam: hi")
    contact = assess("Robin", [undated], now=at("2026-03-03 19:00"))
    assert contact.state is ContactState.NO_DATES
    assert contact.priority == 0
    assert "no telling how long" in contact.headline()


# -- reciprocity ------------------------------------------------------------------


def test_an_even_share_or_better_counts_in_full():
    assert Reciprocity(3, 3, 10, 10).weight == 1.0
    assert Reciprocity(1, 6, 5, 20).weight == 1.0


def test_a_one_sided_share_counts_for_less_but_a_tiny_sample_cannot_swing_it():
    carried = Reciprocity(owner_starts=8, match_starts=0, owner_messages=30, match_messages=3)
    assert carried.weight < 0.3
    assert carried.owner_carrying
    tiny = Reciprocity(owner_starts=1, match_starts=0, owner_messages=1, match_messages=0)
    assert tiny.weight == pytest.approx(2 / 3)
    assert not tiny.owner_carrying


def one_sided():
    lines = []
    for day in (2, 5, 8):
        lines += [
            f"[2026-03-{day:02d} 19:00] Sam: hey, how was today? anything good happen",
            f"[2026-03-{day:02d} 19:20] Sam: also found a new coffee place you'd like",
            f"[2026-03-{day:02d} 21:00] Robin: ok",
        ]
    return chat(*lines)


def test_carrying_the_thread_lowers_priority_and_is_said_plainly():
    # Both quiet for five days since their last message.
    balanced = assess("Robin", [EVEN], now=LAST + timedelta(days=5))
    carried = assess("Robin", [one_sided()], now=at("2026-03-13 21:00"))
    assert carried.state is ContactState.DUE
    assert carried.priority < balanced.priority
    text = carried.to_text()
    assert "Robin started 0 of 3 stretches and sent 3 of 9 messages." in text
    assert "You have been doing most of the reaching out." in text


def test_reciprocity_looks_back_four_weeks_from_the_last_message():
    old = [f"[2026-01-{day:02d} 19:00] Robin: thinking of you {day}" for day in (2, 6, 10, 14)] + [
        "[2026-01-14 19:05] Sam: same"
    ]
    recent = [
        "[2026-03-02 19:00] Sam: hey",
        "[2026-03-02 19:05] Robin: hi",
    ]
    contact = assess("Robin", [chat(*old, *recent)], now=at("2026-03-03 19:00"))
    assert contact.reciprocity == Reciprocity(1, 0, 1, 1)


# -- the store, and safety --------------------------------------------------------


def stored(tmp_path: Path):
    s = Store(tmp_path / "confidant.db", clock=lambda: datetime(2026, 10, 5, 9, 0))
    s.add_person("Robin")
    s.save_conversation("Robin", read_transcript("examples/sample_chat.txt"))
    s.add_person("Casey")
    saved = s.save_conversation("Casey", read_transcript("examples/pressure_chat.txt"))
    s.add_person("Ada")
    return s, saved


def test_a_danger_flag_overrides_the_timing(tmp_path):
    s, casey = stored(tmp_path)
    with s:
        before = assess_person(s, "Casey", now=at("2026-04-20 09:00"))
        read_conversation(
            s,
            casey.conversation,
            ReadingKind.FLAGS,
            settings=Settings(api_key="test"),
            client=ReplayClient(RECORDINGS / "flags_pressure.json"),
        )
        after = assess_person(s, "Casey", now=at("2026-04-20 09:00"))
    assert before.state is not ContactState.SAFETY
    assert after.state is ContactState.SAFETY
    assert after.priority == 0
    assert not after.suggested
    text = after.to_text()
    assert text.startswith("Casey: not suggested. A danger-tier red flag was found with them.")
    assert after.escalation.to_text() in text


def test_everyone_is_ordered_most_worth_a_message_first(tmp_path):
    s, _ = stored(tmp_path)
    with s:
        later = parse_transcript(
            HEAD.replace("Robin", "Ada") + "[2026-03-10 09:00] Ada: still on for friday?\n"
        )
        s.save_conversation("Ada", later)
        s.add_person("Zed")
        contacts = assess_everyone(s, now=at("2026-03-10 12:00"))
    names = [c.name for c in contacts]
    states = {c.name: c.state for c in contacts}
    assert names[0] == "Ada"
    assert states["Ada"] is ContactState.YOUR_TURN
    assert states["Zed"] is ContactState.NOTHING_SAVED
    assert names.index("Robin") < names.index("Zed")
    priorities = [c.priority for c in contacts]
    assert priorities == sorted(priorities, reverse=True)


def test_assessing_never_writes_to_the_store(tmp_path):
    s, _ = stored(tmp_path)
    with s:
        before = (tmp_path / "confidant.db").read_bytes()
        assess_everyone(s, now=at("2026-04-20 09:00"))
        assert (tmp_path / "confidant.db").read_bytes() == before
