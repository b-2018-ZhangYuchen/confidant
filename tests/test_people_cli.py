"""Tests for `confidant add`, `list`, `show`, and `remove`.

Every test points CONFIDANT_DB at a temporary file, so none of them can touch a real
store in the developer's home directory.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from confidant.cli import main
from confidant.store import Store

README = Path(__file__).parents[1] / "README.md"


@pytest.fixture(autouse=True)
def db(monkeypatch, tmp_path) -> Path:
    path = tmp_path / "store" / "confidant.db"
    monkeypatch.setenv("CONFIDANT_DB", str(path))
    return path


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# -- add ------------------------------------------------------------------------


def test_add_a_person_with_a_conversation(capsys, db):
    code, out, _ = run(capsys, "add", "Robin", "examples/sample_chat.txt")
    assert code == 0
    assert out == "Added Robin.\nSaved examples/sample_chat.txt as #1 (17 messages).\n"
    with Store(db) as store:
        [summary] = store.people()
    assert (summary.person.name, summary.conversations, summary.messages) == ("Robin", 1, 17)


def test_add_just_a_name(capsys):
    assert run(capsys, "add", "Robin") == (0, "Added Robin.\n", "")


def test_adding_someone_already_saved_without_a_transcript_says_how_to_add_one(capsys):
    run(capsys, "add", "Robin")
    code, _, err = run(capsys, "add", "robin")
    assert code == 2
    assert "'Robin' is already saved" in err
    assert "confidant add Robin FILE" in err


def test_add_attaches_more_conversations_to_someone_already_saved(capsys, db, tmp_path):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    later = tmp_path / "later.txt"
    later.write_text("# owner: Sam\nRobin: coffee cart on saturday?\nSam: yes!\n")
    code, out, _ = run(capsys, "add", "ROBIN", str(later))
    assert code == 0
    assert out == f"Saved {later} as #2 (2 messages).\n"
    with Store(db) as store:
        assert [c.messages for c in store.conversations("Robin")] == [17, 2]


def test_adding_the_same_conversation_twice_keeps_one_copy(capsys, db):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    code, out, _ = run(capsys, "add", "Robin", "examples/sample_chat.txt")
    assert code == 0
    assert out == "examples/sample_chat.txt is already saved, as #1.\n"
    with Store(db) as store:
        assert len(store.conversations("Robin")) == 1


def test_a_conversation_with_someone_else_is_not_filed_under_this_person(capsys, db):
    code, _, err = run(capsys, "add", "Robin", "examples/pressure_chat.txt")
    assert code == 2
    assert "a conversation with 'Casey', not 'Robin'" in err
    assert "--match 'Casey'" in err
    # Refused before anything was written, including the new person.
    with Store(db) as store:
        assert store.people() == []


def test_match_accepts_an_export_that_spells_the_name_differently(capsys, db, tmp_path):
    chat = tmp_path / "export.txt"
    chat.write_text("# owner: Sam\nRob: hey\nSam: hi\n")
    code, out, _ = run(capsys, "add", "Robin", str(chat), "--match", "Rob")
    assert code == 0
    assert "Saved" in out
    with Store(db) as store:
        [stored] = store.conversations("Robin")
        assert store.load_conversation(stored.id).match_name == "Rob"


def test_one_bad_transcript_saves_nothing(capsys, db):
    code, _, err = run(capsys, "add", "Robin", "examples/sample_chat.txt", "missing.txt")
    assert code == 2
    assert "No transcript at missing.txt" in err
    with Store(db) as store:
        assert store.people() == []


def test_owner_can_be_given_for_transcripts_without_a_directive(capsys, tmp_path):
    chat = tmp_path / "chat.txt"
    chat.write_text("Robin: hey\nAlex: hi\n")
    assert run(capsys, "add", "Robin", str(chat))[0] == 2
    assert run(capsys, "add", "Robin", str(chat), "--owner", "Alex")[0] == 0


def test_an_unusable_store_exits_2(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIDANT_DB", str(tmp_path))
    code, _, err = run(capsys, "list")
    assert code == 2
    assert "is a folder" in err


# -- list -----------------------------------------------------------------------


def test_list_with_no_one_saved_says_how_to_start(capsys):
    code, out, _ = run(capsys, "list")
    assert code == 0
    assert "No one saved yet" in out
    assert "confidant add NAME" in out


def test_list_shows_everyone_alphabetically(capsys):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    run(capsys, "add", "Casey", "examples/pressure_chat.txt")
    run(capsys, "add", "Ari")
    code, out, _ = run(capsys, "list")
    assert code == 0
    assert out.splitlines() == [
        "NAME   CONVERSATIONS  MESSAGES  LAST MESSAGE",
        "Ari    0              0         -",
        "Casey  1              10        2026-04-12 09:20",
        "Robin  1              17        2026-03-05 21:36",
    ]


def test_readme_list_example_matches_what_the_cli_prints(capsys):
    readme = README.read_text(encoding="utf-8")
    marker = "`list` shows everyone at a glance:\n\n```\n"
    documented = readme[readme.index(marker) + len(marker) :].split("\n```", 1)[0]

    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    run(capsys, "add", "Casey", "examples/pressure_chat.txt")
    assert run(capsys, "list")[1] == documented + "\n"


# -- show -----------------------------------------------------------------------


def test_show_gives_the_stats_for_each_conversation(capsys):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    code, out, _ = run(capsys, "show", "robin")
    assert code == 0
    stats = run(capsys, "stats", "examples/sample_chat.txt")[1]
    header, _, first = out.partition("\n\n")
    assert re.fullmatch(r"Robin: added \d{4}-\d\d-\d\d, 1 conversation", header)
    heading, _, body = first.partition("\n")
    assert re.fullmatch(r"#1  examples/sample_chat\.txt, saved \d{4}-\d\d-\d\d \d\d:\d\d", heading)
    # Loaded back from the store, the conversation describes itself exactly as the file does.
    assert body == stats


def test_show_someone_with_nothing_saved(capsys):
    run(capsys, "add", "Ari")
    out = run(capsys, "show", "Ari")[1]
    assert "0 conversations" in out
    assert "confidant add Ari FILE" in out


def test_show_someone_unknown_points_at_list(capsys):
    code, _, err = run(capsys, "show", "Nobody")
    assert code == 2
    assert "No one called 'Nobody'" in err
    assert "confidant list" in err


# -- remove ---------------------------------------------------------------------


def test_remove_with_yes_forgets_them_and_their_conversations(capsys, db):
    run(capsys, "add", "Casey", "examples/pressure_chat.txt")
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    code, out, _ = run(capsys, "remove", "casey", "--yes")
    assert code == 0
    assert out == "Forgot Casey and 1 conversation.\n"
    with Store(db) as store:
        assert [p.person.name for p in store.people()] == ["Robin"]


def test_remove_without_a_terminal_needs_yes(capsys, monkeypatch, db):
    run(capsys, "add", "Casey", "examples/pressure_chat.txt")
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    code, _, err = run(capsys, "remove", "Casey")
    assert code == 2
    assert "pass --yes" in err
    with Store(db) as store:
        assert len(store.people()) == 1


@pytest.mark.parametrize(("answer", "kept"), [("y", 0), ("", 1), ("no", 1)])
def test_remove_asks_first_at_a_terminal(capsys, monkeypatch, db, answer, kept):
    run(capsys, "add", "Casey", "examples/pressure_chat.txt")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    prompts = []
    monkeypatch.setattr("builtins.input", lambda prompt: prompts.append(prompt) or answer)
    code, out, _ = run(capsys, "remove", "Casey")
    assert code == 0
    assert prompts == ["Forget Casey and 1 conversation? This cannot be undone. [y/N] "]
    assert out == ("Nothing removed.\n" if kept else "Forgot Casey and 1 conversation.\n")
    with Store(db) as store:
        assert len(store.people()) == kept


def test_remove_someone_unknown_exits_2(capsys):
    assert run(capsys, "remove", "Nobody", "--yes")[0] == 2


# -- help -----------------------------------------------------------------------


def test_help_mentions_the_store(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for expected in ("confidant add Robin", "CONFIDANT_DB", "list", "show", "remove"):
        assert expected in out


# -- incremental add --------------------------------------------------------------


def _export(tmp_path, name: str, lines: list[str]) -> Path:
    path = tmp_path / name
    path.write_text("# owner: Sam\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return path


def _messages(path: str) -> list[str]:
    """A transcript's messages as written, each with its indented continuation lines."""
    messages: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("["):
            messages.append(line)
        elif line[:1].isspace() and messages:
            messages[-1] += "\n" + line
    return messages


SAMPLE_LINES = _messages("examples/sample_chat.txt")


def test_a_newer_export_adds_only_the_new_messages(capsys, db, tmp_path):
    early = _export(tmp_path, "early.txt", SAMPLE_LINES[:10])
    run(capsys, "add", "Robin", str(early))
    code, out, _ = run(capsys, "add", "Robin", "examples/sample_chat.txt")
    assert code == 0
    new = len(SAMPLE_LINES) - 10
    assert out == f"Added {new} new messages from examples/sample_chat.txt to #1 (17 in all).\n"
    with Store(db) as store:
        assert [c.messages for c in store.conversations("Robin")] == [17]


def test_an_older_export_after_a_newer_one_adds_nothing(capsys, db, tmp_path):
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    early = _export(tmp_path, "early.txt", SAMPLE_LINES[:10])
    code, out, _ = run(capsys, "add", "Robin", str(early))
    assert code == 0
    assert out == f"{early} is already saved, as #1.\n"


def test_one_new_message_is_singular(capsys, tmp_path):
    run(capsys, "add", "Robin", str(_export(tmp_path, "a.txt", SAMPLE_LINES[:16])))
    out = run(capsys, "add", "Robin", "examples/sample_chat.txt")[1]
    assert "Added 1 new message from" in out


def test_show_says_when_a_conversation_was_last_added_to(capsys, tmp_path):
    run(capsys, "add", "Robin", str(_export(tmp_path, "a.txt", SAMPLE_LINES[:10])))
    run(capsys, "add", "Robin", "examples/sample_chat.txt")
    out = run(capsys, "show", "Robin")[1]
    heading = out.split("\n\n")[1].splitlines()[0]
    assert re.fullmatch(
        r"#1  .*a\.txt, saved \d{4}-\d\d-\d\d \d\d:\d\d, updated \d{4}-\d\d-\d\d \d\d:\d\d",
        heading,
    )
    assert "messages         17" in out


def test_readme_incremental_example_matches_what_the_cli_prints(capsys, monkeypatch, tmp_path):
    readme = README.read_text(encoding="utf-8")
    marker = "to the same conversation rather than a second\ncopy of it:\n\n```\n"
    documented = readme[readme.index(marker) + len(marker) :].split("\n```", 1)[0]

    full = Path("examples/sample_chat.txt").read_text(encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    _export(tmp_path, "robin_week1.txt", SAMPLE_LINES[:10])
    Path("robin_week2.txt").write_text(full, encoding="utf-8")
    run(capsys, "add", "Robin", "robin_week1.txt")
    assert run(capsys, "add", "Robin", "robin_week2.txt")[1] == documented + "\n"
