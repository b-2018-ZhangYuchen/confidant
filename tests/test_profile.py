"""Tests for per-person profiles: stored reads, and `confidant profile`.

Offline, like the rest of the suite. ``--update`` is exercised against the recorded
responses in ``tests/fixtures/recorded``: the store hands back conversations exactly as
they were saved, so reading a saved copy of an example transcript makes the same request
the recording answers.
"""

from __future__ import annotations

import functools
import sqlite3
from datetime import datetime
from pathlib import Path

import pytest

from confidant.cli import main
from confidant.client import ModelRefusal
from confidant.config import Settings
from confidant.ingest.transcript import parse_transcript, read_transcript
from confidant.profile import build_profile, read_conversation
from confidant.recording import ReplayClient
from confidant.store import ReadingKind, Store, StoreError

README = Path(__file__).parents[1] / "README.md"
RECORDINGS = Path(__file__).parent / "fixtures" / "recorded"
READ_AT = datetime(2026, 10, 2, 9, 0)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path) -> Path:
    """A private store, no real credentials, and a clock that does not move."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CONFIDANT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))
    path = tmp_path / "store" / "confidant.db"
    monkeypatch.setenv("CONFIDANT_DB", str(path))
    monkeypatch.setattr("confidant.cli.Store", functools.partial(Store, clock=lambda: READ_AT))
    return path


@pytest.fixture
def answer_with(monkeypatch):
    """Make the API answer from recordings, in order, for the rest of the test."""

    def install(*names: str) -> ReplayClient:
        client = ReplayClient(*(RECORDINGS / f"{name}.json" for name in names))
        monkeypatch.setattr("confidant.client.build_client", lambda settings=None: client)
        return client

    return install


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def store(path: Path) -> Store:
    return Store(path, clock=lambda: READ_AT)


# -- the store keeps reads --------------------------------------------------------


def test_reads_are_kept_in_order_and_the_latest_wins(isolated):
    with store(isolated) as s:
        s.add_person("Robin")
        stored = s.save_conversation("Robin", read_transcript("examples/sample_chat.txt"))
        first = s.save_reading(stored.conversation.id, "flags", '{"a": 1}', messages_read=10)
        second = s.save_reading(stored.conversation.id, "flags", '{"a": 2}', messages_read=17)
        assert [r.id for r in s.readings(stored.conversation.id)] == [first.id, second.id]
        assert s.latest_reading(stored.conversation.id, ReadingKind.FLAGS) == second
        assert s.latest_reading(stored.conversation.id, ReadingKind.PERSONALITY) is None


def test_a_read_needs_a_conversation_to_belong_to(isolated):
    with store(isolated) as s, pytest.raises(StoreError, match="No conversation with id 9"):
        s.save_reading(9, "flags", "{}", messages_read=1)


def test_forgetting_someone_forgets_the_reads_too(isolated):
    with store(isolated) as s:
        s.add_person("Robin")
        saved = s.save_conversation("Robin", read_transcript("examples/sample_chat.txt"))
        s.save_reading(saved.conversation.id, "flags", "{}", messages_read=17)
        s.remove_person("Robin")
        assert s._conn.execute("SELECT COUNT(*) FROM readings").fetchone()[0] == 0


def test_a_store_from_before_readings_is_upgraded_in_place(isolated):
    from confidant.store import _MIGRATIONS, SCHEMA_VERSION

    with store(isolated) as s:
        s.add_person("Robin")
    # Wind the file back to schema 2, as yesterday's Confidant left it.
    conn = sqlite3.connect(isolated)
    conn.executescript("DROP TABLE readings; PRAGMA user_version = 2;")
    conn.close()
    assert SCHEMA_VERSION == len(_MIGRATIONS) == 3

    with store(isolated) as s:
        assert s.find_person("robin") is not None
        assert s._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        saved = s.save_conversation("Robin", read_transcript("examples/sample_chat.txt"))
        s.save_reading(saved.conversation.id, "personality", "{}", messages_read=17)


# -- building a profile -----------------------------------------------------------


def test_statistics_cover_every_conversation(isolated):
    with store(isolated) as s:
        s.add_person("Robin")
        s.save_conversation("Robin", read_transcript("examples/sample_chat.txt"))
        s.save_conversation(
            "Robin",
            parse_transcript("# owner: Sam\nRobin: one two three four\nSam: five six\n"),
        )
        profile = build_profile(s, "robin")

    assert len(profile.combined) == 19
    assert len(profile.combined.owner_messages) == 9
    text = profile.to_text()
    assert text.startswith("Robin: 2 conversations, 19 messages\n")
    assert "  messages         19 (9 you / 10 them)" in text
    assert "4 reads missing or out of date" in text
    assert "(sends 2 redacted conversations to the Anthropic API)" in text


def test_a_person_with_nothing_saved_says_how_to_add_something(capsys):
    run(capsys, "add", "Robin")
    code, out, _ = run(capsys, "profile", "Robin")
    assert code == 0
    assert out == (
        "Robin: 0 conversations, 0 messages\n\n"
        "Nothing saved yet. Add a transcript with: confidant add Robin FILE\n"
    )


def test_a_chat_they_never_wrote_in_is_not_sent_to_be_read(isolated):
    with store(isolated) as s:
        s.add_person("Robin")
        s.save_conversation(
            "Robin", parse_transcript("# owner: Sam\n# match: Robin\nSam: hello?\n")
        )
        profile = build_profile(s, "Robin")
    assert profile.pending == []
    assert "Nothing from Robin to read." in profile.to_text()
    assert "missing or out of date" not in profile.to_text()


def test_an_unknown_name_exits_2_and_points_at_list(capsys):
    code, _, err = run(capsys, "profile", "Robin")
    assert code == 2
    assert "No one called 'Robin'" in err
    assert "confidant list" in err


def test_showing_a_profile_never_needs_credentials(capsys):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    code, out, _ = run(capsys, "profile", "Robin")
    assert code == 0
    assert "     Not read yet." in out
    assert "No conversation has been checked yet." in out


# -- --update ---------------------------------------------------------------------


def test_update_reads_each_conversation_once(capsys, answer_with, isolated):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    client = answer_with("analyze_sample", "flags_sample")

    code, out, err = run(capsys, "profile", "Robin", "--update")
    assert (code, err) == (0, "")
    assert client.remaining == 0
    assert out.startswith("Updated 2 reads.\n\nRobin: 1 conversation, 17 messages\n")
    # Placeholders are swapped back before a read is saved, so the store holds what the
    # owner sees and nothing in it needs the redaction key to make sense.
    assert "Robin comes across as curious and playful" in out
    assert "[MATCH]" not in out
    assert "Nothing in 1 checked conversation rises to a red flag." in out

    # Saved, so asking again costs nothing.
    answer_with()
    code, out, _ = run(capsys, "profile", "Robin", "--update")
    assert code == 0
    assert out.startswith("Nothing to update: every conversation is read up to its last message.")
    with store(isolated) as s:
        assert len(s.readings(1)) == 2


def test_new_messages_make_a_read_out_of_date(capsys, answer_with, tmp_path):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    answer_with("analyze_sample", "flags_sample")
    run(capsys, "profile", "Robin", "--update")

    later = tmp_path / "later.txt"
    later.write_text(
        "# owner: Sam\n# match: Robin\n"
        "[2026-03-05 21:35] Sam: distracted please\n"
        "[2026-03-05 21:36] Robin: excellent. I have a strongly held opinion about airport "
        "carpet that I have been saving\n"
        "[2026-03-07 10:00] Sam: go on then\n",
        encoding="utf-8",
    )
    run(capsys, "add", "Robin", str(later))

    _, out, _ = run(capsys, "profile", "Robin")
    assert "     read 2026-10-02, at 17 of 18 messages, medium confidence" in out
    assert "     checked 2026-10-02, at 17 of 18 messages, 0 flags, medium confidence" in out
    assert "2 reads missing or out of date." in out
    # The out-of-date read is still shown: it is what is known, and it says how old it is.
    assert "Robin comes across as curious and playful" in out


def test_update_without_credentials_exits_2_and_saves_nothing(capsys, isolated):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    code, out, err = run(capsys, "profile", "Robin", "--update")
    assert (code, out) == (2, "")
    assert "No Anthropic credentials found" in err
    with store(isolated) as s:
        assert s.readings(1) == []


def test_a_refusal_exits_3(capsys, answer_with, isolated):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    answer_with("analyze_refusal")
    code, _, err = run(capsys, "profile", "Robin", "--update")
    assert code == 3
    assert "declined" in err
    assert "kept the" not in err


def test_a_failure_part_way_keeps_the_reads_already_done(
    capsys, answer_with, monkeypatch, isolated
):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    answer_with("analyze_sample")

    def refuse(*args, **kwargs):
        raise ModelRefusal("cyber", None)

    monkeypatch.setattr("confidant.profile.analyze_flags", refuse)
    code, _, err = run(capsys, "profile", "Robin", "--update")
    assert code == 3
    assert "kept the 1 read that finished before this" in err
    with store(isolated) as s:
        assert [r.kind for r in s.readings(1)] == [ReadingKind.PERSONALITY]

    # Only the missing read is still pending.
    with store(isolated) as s:
        assert [kind for _, kind in build_profile(s, "Robin").pending] == [ReadingKind.FLAGS]


# -- red flags across conversations -----------------------------------------------


def _check_casey(path: Path) -> Store:
    s = store(path)
    s.add_person("Casey")
    saved = s.save_conversation("Casey", read_transcript("examples/pressure_chat.txt"))
    read_conversation(
        s,
        saved.conversation,
        ReadingKind.FLAGS,
        settings=Settings(),
        client=ReplayClient(RECORDINGS / "flags_pressure.json"),
    )
    return s


def test_a_danger_flag_puts_the_safety_notice_first(isolated):
    with _check_casey(isolated) as s:
        text = build_profile(s, "Casey").to_text()

    lines = text.splitlines()
    assert lines[0] == "Casey: 1 conversation, 10 messages"
    assert lines[2].startswith("!! Something in Casey's messages is serious: tracking where you")
    assert text.index("!! ") < text.index("ACROSS ALL CONVERSATIONS")
    assert "  [DANGER] monitoring  [#1]" in text
    assert text.index("[DANGER] isolation") < text.index("[CONCERN] guilt or blame")
    assert "  (1 flag left out: the quotes behind them could not be found" in text


def test_an_out_of_date_check_still_raises_the_notice(isolated):
    with _check_casey(isolated) as s:
        s.save_conversation(
            "Casey",
            parse_transcript(
                read_transcript("examples/pressure_chat.txt").transcript()
                + "\nJordan: I need some space this week\nCasey: ok\n",
                owner="Jordan",
                match="Casey",
            ),
        )
        profile = build_profile(s, "Casey")
    [entry] = profile.entries
    assert entry.flags.stale
    assert profile.escalation is not None
    assert profile.to_text().splitlines()[2].startswith("!! ")


def test_danger_is_never_given_a_charitable_reading(isolated):
    from confidant.analysis.flags import Flag, FlagReport
    from confidant.analysis.personality import Evidence

    # The prompt asks for null here; a model that supplies one anyway must not be shown.
    gloss = "Maybe they just worry a lot."
    report = FlagReport(
        flags=[
            Flag(
                category="monitoring",
                severity="danger",
                behavior="Asked for a location twice.",
                evidence=[Evidence(quote="where are you", why_it_matters="Tracking.")],
                innocent_reading=gloss,
            )
        ],
        summary="Serious.",
        confidence="medium",
    )
    with store(isolated) as s:
        s.add_person("Casey")
        saved = s.save_conversation("Casey", read_transcript("examples/pressure_chat.txt"))
        s.save_reading(saved.conversation.id, "flags", report.model_dump_json(), messages_read=10)
        text = build_profile(s, "Casey").to_text()
    assert "[DANGER] monitoring" in text
    assert gloss not in text


def test_a_read_in_an_older_shape_counts_as_missing(isolated):
    with store(isolated) as s:
        s.add_person("Robin")
        saved = s.save_conversation("Robin", read_transcript("examples/sample_chat.txt"))
        s.save_reading(saved.conversation.id, "personality", '{"vibe": "great"}', messages_read=17)
        profile = build_profile(s, "Robin")
    assert profile.entries[0].personality is None
    assert ReadingKind.PERSONALITY in profile.entries[0].pending


# -- documentation ----------------------------------------------------------------


def test_readme_profile_example_matches_what_the_cli_prints(capsys, answer_with):
    readme = README.read_text(encoding="utf-8")
    marker = "```bash\nconfidant profile Robin --update\n```\n\nprints\n\n```\n"
    documented = readme[readme.index(marker) + len(marker) :].split("\n```", 1)[0]

    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    answer_with("analyze_sample", "flags_sample")
    code, out, _ = run(capsys, "profile", "Robin", "--update")
    assert code == 0
    assert out == documented + "\n"


def test_help_names_profile_and_its_network_use(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "confidant profile Robin --update" in out
    assert "profile --update send the redacted transcript" in out
