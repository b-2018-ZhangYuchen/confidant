"""Tests for the timeline: `confidant timeline`, and the counting behind it.

Offline. Reads on the timeline come from the recorded responses in
``tests/fixtures/recorded``, saved through the same path ``profile --update`` uses.
"""

from __future__ import annotations

import functools
from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from confidant.cli import main
from confidant.config import Settings
from confidant.ingest.transcript import parse_transcript, read_transcript
from confidant.profile import read_conversation
from confidant.recording import ReplayClient
from confidant.store import ReadingKind, Store
from confidant.timeline import Period, _duration, build_timeline

README = Path(__file__).parents[1] / "README.md"
RECORDINGS = Path(__file__).parent / "fixtures" / "recorded"
READ_AT = datetime(2026, 10, 3, 9, 0)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path) -> Path:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CONFIDANT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))
    path = tmp_path / "store" / "confidant.db"
    monkeypatch.setenv("CONFIDANT_DB", str(path))
    monkeypatch.setattr("confidant.cli.Store", functools.partial(Store, clock=lambda: READ_AT))
    return path


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def store(path: Path) -> Store:
    return Store(path, clock=lambda: READ_AT)


def saved(path: Path, *transcripts: str, name: str = "Robin") -> Store:
    s = store(path)
    s.add_person(name)
    for text in transcripts:
        s.save_conversation(name, parse_transcript(text))
    return s


HEAD = "# owner: Sam\n# match: Robin\n"


# -- counting ---------------------------------------------------------------------


def test_messages_fall_into_weeks_starting_on_monday(isolated):
    chat = (
        HEAD + "[2026-03-01 10:00] Robin: sunday\n"  # a Sunday: the week of Feb 23
        "[2026-03-02 10:00] Sam: monday\n"
        "[2026-03-08 23:59] Robin: still the same week\n"
    )
    with saved(isolated, chat) as s:
        timeline = build_timeline(s, "Robin")
    assert [b.start for b in timeline.buckets] == [date(2026, 2, 23), date(2026, 3, 2)]
    assert [b.messages for b in timeline.buckets] == [1, 2]


def test_replies_are_timed_from_the_other_sides_last_message(isolated):
    chat = HEAD + (
        "[2026-03-02 10:00] Sam: one\n"
        "[2026-03-02 10:05] Sam: two\n"
        "[2026-03-02 10:20] Robin: answered fifteen minutes after the last one\n"
        "[2026-03-02 10:21] Robin: and a follow-up, which is not a reply\n"
        "[2026-03-02 11:21] Sam: an hour\n"
        "[2026-03-02 14:21] Sam: a double text is not a reply either\n"
    )
    with saved(isolated, chat) as s:
        [bucket] = build_timeline(s, "Robin").buckets
    assert bucket.match.replies == [timedelta(minutes=15)]
    assert bucket.owner.replies == [timedelta(hours=1)]


def test_writing_after_a_quiet_stretch_counts_as_starting(isolated):
    chat = HEAD + (
        "[2026-03-02 10:00] Sam: opened the chat\n"
        "[2026-03-02 10:30] Robin: same stretch\n"
        "[2026-03-02 18:29] Robin: under eight hours later, same stretch\n"
        "[2026-03-03 09:00] Robin: next morning, a new one\n"
        "[2026-03-04 09:00] Sam: a slow reply starts things up again too\n"
    )
    with saved(isolated, chat) as s:
        [bucket] = build_timeline(s, "Robin").buckets
    assert (bucket.owner.starts, bucket.match.starts) == (2, 1)
    assert bucket.owner.replies == [timedelta(days=1)]


def test_separate_conversations_are_never_read_as_answering_each_other(isolated):
    first = HEAD + "[2026-03-02 10:00] Sam: in one app\n"
    second = HEAD + "[2026-03-02 10:01] Robin: in another\n"
    with saved(isolated, first, second) as s:
        [bucket] = build_timeline(s, "Robin").buckets
    assert bucket.match.replies == []
    assert (bucket.owner.starts, bucket.match.starts) == (1, 1)


def test_an_undated_message_is_left_out_and_breaks_the_chain(isolated):
    chat = HEAD + (
        "[2026-03-02 10:00] Sam: dated\nRobin: undated\n[2026-03-02 10:30] Robin: dated again\n"
    )
    with saved(isolated, chat) as s:
        timeline = build_timeline(s, "Robin")
    [bucket] = timeline.buckets
    assert timeline.undated == 1
    assert bucket.messages == 2
    assert bucket.match.replies == []
    assert "(1 message without a timestamp left out.)" in timeline.to_text()


def test_days_and_the_quiet_between_them(isolated):
    chat = HEAD + "[2026-03-02 10:00] Sam: hi\n[2026-03-05 10:00] Robin: hi\n"
    with saved(isolated, chat) as s:
        text = build_timeline(s, "Robin", Period.DAY).to_text()
    assert text.startswith("Robin: 2 messages over 4 days\n")
    assert "            (no messages for 2 days)\n2026-03-05" in text
    assert "FIRST DAY TO LATEST" in text


@pytest.mark.parametrize(
    ("value", "shown"),
    [
        (timedelta(seconds=20), "<1m"),
        (timedelta(minutes=14, seconds=40), "15m"),
        (timedelta(minutes=90), "2h"),
        (timedelta(hours=47), "47h"),
        (timedelta(hours=60), "3d"),
        (None, "-"),
    ],
)
def test_reply_times_are_rounded_to_one_unit(value, shown):
    assert _duration(value) == shown


# -- reads on the timeline --------------------------------------------------------


def test_a_read_sits_at_the_last_message_it_read(isolated):
    with store(isolated) as s:
        s.add_person("Robin")
        first = s.save_conversation("Robin", read_transcript("examples/sample_chat.txt"))
        read_conversation(
            s,
            first.conversation,
            ReadingKind.PERSONALITY,
            settings=Settings(),
            client=ReplayClient(RECORDINGS / "analyze_sample.json"),
        )
        s.save_conversation("Robin", read_transcript("examples/sample_chat_later.txt"))
        timeline = build_timeline(s, "Robin")

    [mark] = [m for b in timeline.buckets for m in b.marks]
    assert mark.messages_read == 17
    assert mark.at == datetime(2026, 3, 5, 21, 36)
    assert timeline.buckets[0].marks == [mark]
    assert mark.to_text().startswith("#1 read at 17 messages: Robin comes across as curious")


def test_every_read_is_shown_not_just_the_latest(isolated):
    empty = '{"flags": [], "summary": "Nothing.", "confidence": "low"}'
    with store(isolated) as s:
        s.add_person("Robin")
        s.save_conversation("Robin", read_transcript("examples/sample_chat.txt"))
        s.save_reading(1, "flags", empty, messages_read=5)
        s.save_reading(1, "flags", empty, messages_read=17)
        s.save_reading(1, "personality", '{"vibe": "an older shape"}', messages_read=17)
        timeline = build_timeline(s, "Robin")
    marks = [m.to_text() for b in timeline.buckets for m in b.marks]
    assert marks == [
        "#1 checked at 5 messages: no red flags",
        "#1 checked at 17 messages: no red flags",
    ]


def test_a_read_with_nothing_dated_is_counted_not_placed(isolated):
    chat = HEAD + "Sam: one\nRobin: two\n[2026-03-02 10:00] Robin: three\n"
    with saved(isolated, chat) as s:
        s.save_reading(
            1, "flags", '{"flags": [], "summary": "", "confidence": "low"}', messages_read=2
        )
        timeline = build_timeline(s, "Robin")
    assert timeline.unplaced_reads == 1
    assert "(1 read left out: the messages read have no timestamps.)" in timeline.to_text()


def test_a_danger_flag_puts_the_safety_notice_first(isolated):
    with store(isolated) as s:
        s.add_person("Casey")
        result = s.save_conversation("Casey", read_transcript("examples/pressure_chat.txt"))
        read_conversation(
            s,
            result.conversation,
            ReadingKind.FLAGS,
            settings=Settings(),
            client=ReplayClient(RECORDINGS / "flags_pressure.json"),
        )
        text = build_timeline(s, "Casey").to_text()
    lines = text.splitlines()
    assert lines[2].startswith("!! Something in Casey's messages is serious")
    assert text.index("!! ") < text.index("MESSAGES")
    assert "the most serious danger" in text


# -- the command ------------------------------------------------------------------


def test_nothing_saved_says_how_to_add_something(capsys):
    run(capsys, "add", "Robin")
    code, out, _ = run(capsys, "timeline", "Robin")
    assert code == 0
    assert out == (
        "Robin: 0 conversations\n\nNothing saved yet. Add a transcript with: "
        "confidant add Robin FILE\n"
    )


def test_a_chat_without_timestamps_says_why_there_is_no_timeline(capsys, tmp_path):
    chat = tmp_path / "undated.txt"
    chat.write_text(HEAD + "Sam: hi\nRobin: hello\n", encoding="utf-8")
    run(capsys, "add", "Robin", str(chat))
    code, out, _ = run(capsys, "timeline", "Robin")
    assert code == 0
    assert "None of the 2 saved messages has a timestamp" in out


def test_one_week_has_nothing_to_compare(capsys):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    code, out, _ = run(capsys, "timeline", "Robin")
    assert code == 0
    assert out.startswith("Robin: 17 messages over 1 week\n")
    assert "TO LATEST" not in out


def test_an_unknown_name_exits_2_and_points_at_list(capsys):
    code, _, err = run(capsys, "timeline", "Robin")
    assert code == 2
    assert "No one called 'Robin'" in err
    assert "confidant list" in err


def test_the_timeline_never_needs_credentials_or_the_network(capsys):
    # No key in the environment, and conftest refuses every connection.
    run(capsys, "add", "Robin", "examples/sample_chat.txt", "examples/sample_chat_later.txt")
    code, out, err = run(capsys, "timeline", "Robin", "--by", "day")
    assert (code, err) == (0, "")
    assert "2026-03-21" in out


def test_readme_timeline_example_matches_what_the_cli_prints(capsys, monkeypatch):
    readme = README.read_text(encoding="utf-8")
    marker = "```bash\nconfidant timeline Robin\n```\n\nprints\n\n```\n"
    documented = readme[readme.index(marker) + len(marker) :].split("\n```", 1)[0]

    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    client = ReplayClient(RECORDINGS / "analyze_sample.json", RECORDINGS / "flags_sample.json")
    monkeypatch.setattr("confidant.client.build_client", lambda settings=None: client)
    run(capsys, "profile", "Robin", "--update")
    run(capsys, "add", "Robin", "examples/sample_chat_later.txt")

    code, out, _ = run(capsys, "timeline", "Robin")
    assert code == 0
    assert out == documented + "\n"


def test_help_names_timeline_as_local(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "confidant timeline Robin" in out
    assert "remove, profile, and timeline, which keep" in out
