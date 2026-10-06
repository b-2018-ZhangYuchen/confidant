"""Tests for `confidant nudge`: who is worth a message today, and why.

Offline. Every test points CONFIDANT_DB at a temporary file, and passes ``--at`` so the
answer does not depend on the day the suite runs. The danger case is read from the
recorded flags response, saved through the same path ``profile --update`` uses.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from confidant.cli import main
from confidant.config import Settings
from confidant.contact import ContactState, assess
from confidant.ingest.transcript import parse_transcript
from confidant.nudge import CAVEAT
from confidant.profile import read_conversation
from confidant.recording import ReplayClient
from confidant.store import ReadingKind, Store

README = Path(__file__).parents[1] / "README.md"
RECORDINGS = Path(__file__).parent / "fixtures" / "recorded"


@pytest.fixture(autouse=True)
def db(monkeypatch, tmp_path) -> Path:
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CONFIDANT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    path = tmp_path / "store" / "confidant.db"
    monkeypatch.setenv("CONFIDANT_DB", str(path))
    return path


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def save(capsys, tmp_path: Path, name: str, *lines: str) -> None:
    path = tmp_path / f"{name.lower()}.txt"
    path.write_text(f"# owner: Sam\n# match: {name}\n" + "\n".join(lines) + "\n")
    assert run(capsys, "add", name, str(path))[0] == 0


def check_flags(db: Path, name: str) -> None:
    with Store(db) as store:
        [stored] = store.conversations(name)
        read_conversation(
            store,
            stored,
            ReadingKind.FLAGS,
            settings=Settings(api_key="test"),
            client=ReplayClient(RECORDINGS / "flags_pressure.json"),
        )


# -- the digest -----------------------------------------------------------------


def test_an_empty_store_says_how_to_add_someone(capsys):
    assert run(capsys, "nudge") == (
        0,
        "No one saved yet. Add someone with: confidant add NAME [TRANSCRIPT ...]\n",
        "",
    )


def test_suggestions_come_with_reasons_and_everyone_else_gets_one_line(capsys, tmp_path):
    save(capsys, tmp_path, "Ada", "[2026-03-10 09:00] Ada: still on for friday?")
    save(capsys, tmp_path, "Bo", "[2026-03-10 08:00] Sam: good luck today!")
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    run(capsys, "add", "Zed")
    code, out, _ = run(capsys, "nudge", "--at", "2026-03-10 12:00")
    assert code == 0
    suggested, rest = out.split("\nNot today\n")
    assert suggested.startswith("Worth a message today\n\nAda: your turn.")
    assert "  - " in suggested
    assert "  Bo: waiting on them. You wrote last, 4 hours ago" in rest
    assert "  Zed: nothing saved yet." in rest
    # Only the headline for those not suggested: the reasons are one command away.
    assert "second in a row" not in rest
    assert "'confidant nudge NAME' shows the reasons for anyone." in rest
    assert out.endswith(CAVEAT + "\n")


def test_suggestions_are_in_priority_order(capsys, tmp_path):
    # Both due, but Cam's thread has been quiet longer against the same rhythm.
    exchange = ("Sam: hi", "{name}: hi!")
    for name, day in (("Ada", "2026-03-08"), ("Cam", "2026-03-05")):
        lines = [f"[{day} 19:0{i}] {line.format(name=name)}" for i, line in enumerate(exchange)]
        save(capsys, tmp_path, name, *lines)
    out = run(capsys, "nudge", "--at", "2026-03-11 20:00")[1]
    assert out.index("Cam: worth a message") < out.index("Ada: worth a message")


def test_when_no_one_is_due_it_says_so(capsys, tmp_path):
    save(capsys, tmp_path, "Ada", "[2026-03-10 09:00] Sam: hi", "[2026-03-10 09:05] Ada: hey")
    out = run(capsys, "nudge", "--at", "2026-03-10 12:00")[1]
    assert out.startswith("No one is due a message today.\n\nNot today\n  Ada: nothing due.")


def test_no_caveat_when_there_was_no_timing_to_go_on(capsys):
    run(capsys, "add", "Zed")
    out = run(capsys, "nudge")[1]
    assert CAVEAT not in out
    assert "Zed: nothing saved yet." in out


def test_a_danger_flag_is_printed_first_and_never_suggested(capsys, db):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    run(capsys, "add", "Casey", "examples/pressure_chat.txt")
    before = run(capsys, "nudge", "--at", "2026-04-14 09:00")[1]
    assert "Casey: worth a message" in before

    check_flags(db, "Casey")
    after = run(capsys, "nudge", "--at", "2026-04-14 09:00")[1]
    assert after.startswith("Casey: not suggested. A danger-tier red flag was found with them.")
    assert "!! Something in Casey's messages is serious" in after
    assert after.index("!!") < after.index("Robin:")
    assert "Casey: worth a message" not in after


# -- one person -------------------------------------------------------------------


def test_one_person_shows_every_reason_whatever_the_state(capsys, tmp_path):
    save(capsys, tmp_path, "Bo", "[2026-03-10 08:00] Sam: good luck today!")
    code, out, _ = run(capsys, "nudge", "bo", "--at", "2026-03-10 12:00")
    assert code == 0
    assert out.startswith("Bo: waiting on them.")
    assert "Another message now would be your second in a row." in out
    assert "Not today" not in out
    assert out.endswith(CAVEAT + "\n")


def test_someone_not_saved_is_an_error_that_says_where_to_look(capsys):
    code, out, err = run(capsys, "nudge", "Zed")
    assert (code, out) == (2, "")
    assert "'confidant list' shows who is" in err


# -- looking back -------------------------------------------------------------------


def test_at_leaves_out_messages_sent_after_it(capsys, tmp_path):
    save(
        capsys,
        tmp_path,
        "Ada",
        "[2026-03-10 09:00] Ada: still on for friday?",
        "[2026-03-10 13:00] Sam: yes!",
    )
    assert "Ada: your turn." in run(capsys, "nudge", "Ada", "--at", "2026-03-10 12:00")[1]
    # Four hours on, the reply is in the same stretch: an exchange, not a turn owed.
    assert "Ada: nothing due." in run(capsys, "nudge", "Ada", "--at", "2026-03-10 14:00")[1]


def test_at_before_everything_saved_says_so(capsys, tmp_path):
    save(capsys, tmp_path, "Ada", "[2026-03-10 09:00] Ada: hi")
    out = run(capsys, "nudge", "Ada", "--at", "2026-03-01")[1]
    assert out.startswith("Ada: nothing saved from before then.")


def test_a_bad_at_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["nudge", "--at", "friday"])
    assert exit_info.value.code == 2
    assert "'friday' is not a date" in capsys.readouterr().err


def test_without_until_a_clock_running_ahead_still_counts():
    # The real clock is never a cutoff: an export a few hours ahead of this machine is a
    # timezone, not a message from the future.
    chat = parse_transcript("# owner: Sam\n# match: Ada\n[2026-03-10 15:00] Ada: hi\n")
    contact = assess("Ada", [chat], now=datetime(2026, 3, 10, 12, 0))
    assert contact.state is ContactState.YOUR_TURN
    assert contact.silence.total_seconds() == 0


def test_nudge_never_writes_to_the_store(capsys, db):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    before = db.read_bytes()
    run(capsys, "nudge", "--at", "2026-04-14 09:00")
    assert db.read_bytes() == before


# -- docs -------------------------------------------------------------------------


def test_help_names_nudge_as_local(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "confidant nudge" in out
    assert "remove, profile, timeline, and nudge, which keep" in out


def test_readme_nudge_example_matches_what_the_cli_prints(capsys, db):
    readme = README.read_text(encoding="utf-8")
    marker = 'confidant nudge --at "2026-04-14 09:00"\n```\n\nprints\n\n```\n'
    documented = readme[readme.index(marker) + len(marker) :].split("\n```", 1)[0]

    run(capsys, "add", "Robin", "examples/sample_chat.txt", "examples/sample_chat_later.txt")
    run(capsys, "add", "Casey", "examples/pressure_chat.txt")
    check_flags(db, "Casey")
    assert run(capsys, "nudge", "--at", "2026-04-14 09:00")[1] == documented + "\n"
