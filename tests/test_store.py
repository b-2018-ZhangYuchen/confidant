"""Tests for the SQLite store. Every store lives in a temporary directory."""

from __future__ import annotations

import sqlite3
import stat
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from confidant.ingest.transcript import parse_transcript, read_transcript
from confidant.models import Message
from confidant.store import (
    SCHEMA_VERSION,
    DuplicatePerson,
    SaveOutcome,
    Store,
    StoreError,
    UnknownPerson,
    default_db_path,
)

EXAMPLES = Path(__file__).parent.parent / "examples"


class Clock:
    """A clock that moves forward a minute every time it is read."""

    def __init__(self) -> None:
        self.now = datetime(2026, 9, 29, 9, 0)

    def __call__(self) -> datetime:
        self.now += timedelta(minutes=1)
        return self.now


@pytest.fixture
def db(tmp_path) -> Path:
    return tmp_path / "confidant" / "confidant.db"


@pytest.fixture
def store(db):
    with Store(db, clock=Clock()) as s:
        yield s


@pytest.fixture
def sample():
    return read_transcript(EXAMPLES / "sample_chat.txt")


# -- where it lives and who can read it ---------------------------------------


def test_default_path_is_in_the_home_directory_not_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.delenv("CONFIDANT_DB", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert default_db_path() == tmp_path / ".confidant" / "confidant.db"


def test_confidant_db_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIDANT_DB", str(tmp_path / "elsewhere.db"))
    assert default_db_path() == tmp_path / "elsewhere.db"


def test_store_file_and_folder_are_private_to_the_owner(store, db):
    assert stat.S_IMODE(db.stat().st_mode) == 0o600
    assert stat.S_IMODE(db.parent.stat().st_mode) == 0o700


def test_secure_delete_is_on(store):
    assert store._conn.execute("PRAGMA secure_delete").fetchone()[0] == 1


def test_removed_messages_are_not_left_in_the_file(db, sample):
    with Store(db) as s:
        s.add_person("Robin")
        s.save_conversation("Robin", sample)
        s.remove_person("Robin")
    assert b"coffee cart" not in db.read_bytes()


# -- opening -------------------------------------------------------------------


def test_new_store_is_at_the_current_schema_version(store):
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_reopening_keeps_what_was_saved(db, sample):
    with Store(db) as s:
        s.add_person("Robin")
        s.save_conversation("Robin", sample)
    with Store(db) as s:
        assert [p.person.name for p in s.people()] == ["Robin"]
        assert len(s.conversations("Robin")) == 1


def test_refuses_a_store_from_a_newer_version(db):
    Store(db).close()
    conn = sqlite3.connect(db)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(StoreError, match="newer version of Confidant"):
        Store(db)


def test_refuses_a_file_that_is_not_a_database(tmp_path):
    path = tmp_path / "notes.db"
    path.write_text("these are my notes, not a database\n" * 50)
    with pytest.raises(StoreError, match="not a Confidant store"):
        Store(path)


def test_refuses_a_folder(tmp_path):
    with pytest.raises(StoreError, match="is a folder"):
        Store(tmp_path)


# -- people --------------------------------------------------------------------


def test_add_and_find_a_person(store):
    added = store.add_person("Robin")
    assert added.name == "Robin"
    assert added.added_at == datetime(2026, 9, 29, 9, 1)
    assert store.person("Robin") == added


def test_names_match_regardless_of_case_and_spacing(store):
    store.add_person("Zoë  Park")
    assert store.person("ZOË park").name == "Zoë Park"


def test_adding_the_same_person_twice_is_refused(store):
    store.add_person("Robin")
    with pytest.raises(DuplicatePerson, match="'Robin' is already"):
        store.add_person("robin")


def test_a_blank_name_is_refused(store):
    with pytest.raises(StoreError, match="needs a name"):
        store.add_person("   ")


def test_unknown_person(store):
    assert store.find_person("Casey") is None
    with pytest.raises(UnknownPerson, match="No one called 'Casey'"):
        store.person("Casey")


def test_people_are_listed_alphabetically_with_their_counts(store, sample):
    store.add_person("robin")
    store.add_person("Casey")
    store.save_conversation("robin", sample)

    casey, robin = store.people()
    assert casey.person.name == "Casey"
    assert (casey.conversations, casey.messages, casey.last_message) == (0, 0, None)
    assert robin.person.name == "robin"
    assert robin.conversations == 1
    assert robin.messages == len(sample)
    assert robin.last_message == sample.last_activity


def test_last_message_spans_every_conversation(store, sample):
    later = parse_transcript("# owner: Sam\n[2026-04-01 10:00] Robin: still on for friday?\n")
    store.add_person("Robin")
    store.save_conversation("Robin", sample)
    store.save_conversation("Robin", later)
    (robin,) = store.people()
    assert robin.conversations == 2
    assert robin.messages == len(sample) + 1
    assert robin.last_message == datetime(2026, 4, 1, 10, 0)


def test_removing_a_person_removes_their_conversations(store, sample):
    store.add_person("Robin")
    stored = store.save_conversation("Robin", sample).conversation
    assert store.remove_person("Robin") == 1
    assert store.people() == []
    with pytest.raises(StoreError):
        store.load_conversation(stored.id)
    assert store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 0


def test_removing_someone_unknown_says_so(store):
    with pytest.raises(UnknownPerson):
        store.remove_person("Casey")


# -- conversations -------------------------------------------------------------


def test_a_saved_conversation_loads_back_exactly(store):
    original = read_transcript(EXAMPLES / "details_chat.txt")
    store.add_person("Theo")
    result = store.save_conversation("Theo", original)
    assert (result.outcome, result.added) == (SaveOutcome.NEW, len(original))
    stored = result.conversation

    loaded = store.load_conversation(stored.id)
    assert loaded.owner_name == original.owner_name
    assert loaded.match_name == original.match_name
    assert loaded.source == original.source
    assert loaded.private_names == original.private_names
    assert loaded.messages == original.messages


def test_unredacted_text_is_what_gets_stored(store):
    """Redaction is for the trip to the model; the owner's own copy stays whole."""
    original = read_transcript(EXAMPLES / "details_chat.txt")
    store.add_person("Theo")
    stored = store.save_conversation("Theo", original).conversation
    assert "214 Linden Street" in store.load_conversation(stored.id).transcript()


def test_multiline_messages_and_missing_timestamps_survive(store):
    original = parse_transcript(
        "# owner: Sam\nRobin: first line\n    second line\nSam: no timestamp here\n"
    )
    store.add_person("Robin")
    stored = store.save_conversation("Robin", original).conversation
    loaded = store.load_conversation(stored.id)
    assert loaded.messages == original.messages
    assert loaded.messages[0].text == "first line\nsecond line"


def test_stored_conversation_summarises_without_loading_messages(store, sample):
    store.add_person("Robin")
    stored = store.save_conversation("Robin", sample).conversation
    assert stored.messages == len(sample)
    assert stored.first_message == datetime(2026, 3, 2, 19, 4)
    assert stored.last_message == sample.last_activity
    assert stored.source == str(EXAMPLES / "sample_chat.txt")
    assert stored.imported_at == datetime(2026, 9, 29, 9, 2)


def test_saving_the_same_conversation_twice_keeps_one_copy(store, sample):
    store.add_person("Robin")
    first = store.save_conversation("Robin", sample)
    again = read_transcript(EXAMPLES / "sample_chat.txt")
    again.source = "renamed-export.txt"
    second = store.save_conversation("Robin", again)
    assert (first.outcome, second.outcome) == (SaveOutcome.NEW, SaveOutcome.UNCHANGED)
    assert second.added == 0
    assert second.conversation == first.conversation
    assert len(store.conversations("Robin")) == 1


def test_the_same_conversation_can_belong_to_two_people(store, sample):
    """Deduplication is per person: filing a chat under the wrong name is fixable."""
    store.add_person("Robin")
    store.add_person("Rob")
    assert store.save_conversation("Robin", sample).outcome is SaveOutcome.NEW
    assert store.save_conversation("Rob", sample).outcome is SaveOutcome.NEW


def test_a_conversation_with_an_edited_message_is_saved_as_new(store, sample):
    store.add_person("Robin")
    store.save_conversation("Robin", sample)
    edited = read_transcript(EXAMPLES / "sample_chat.txt")
    first = edited.messages[0]
    edited.messages[0] = Message(first.role, first.text + "!", first.sender, first.timestamp)
    assert store.save_conversation("Robin", edited).outcome is SaveOutcome.NEW
    assert len(store.conversations("Robin")) == 2


def test_conversations_are_listed_oldest_import_first(store, sample):
    later = parse_transcript("# owner: Sam\nRobin: still on for friday?\n")
    store.add_person("Robin")
    first = store.save_conversation("Robin", sample).conversation
    second = store.save_conversation("Robin", later).conversation
    assert [c.id for c in store.conversations("Robin")] == [first.id, second.id]
    assert second.first_message is None and second.last_message is None


def test_saving_for_an_unknown_person_is_refused(store, sample):
    with pytest.raises(UnknownPerson):
        store.save_conversation("Robin", sample)
    assert store._conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0] == 0


def test_an_empty_conversation_is_refused(store, sample):
    store.add_person("Robin")
    sample.messages.clear()
    with pytest.raises(StoreError, match="no messages"):
        store.save_conversation("Robin", sample)


def test_removing_one_conversation_keeps_the_rest(store, sample):
    later = parse_transcript("# owner: Sam\nRobin: still on for friday?\n")
    store.add_person("Robin")
    first = store.save_conversation("Robin", sample).conversation
    second = store.save_conversation("Robin", later).conversation
    store.remove_conversation(first.id)
    assert [c.id for c in store.conversations("Robin")] == [second.id]
    with pytest.raises(StoreError, match=f"No conversation with id {first.id}"):
        store.remove_conversation(first.id)


def test_a_failed_save_leaves_nothing_behind(store, sample, monkeypatch):
    """The conversation row and its messages land together or not at all."""
    store.add_person("Robin")
    real_conn = store._conn

    class FailingMessages:
        def __getattr__(self, name):
            return getattr(real_conn, name)

        def executemany(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        def __enter__(self):
            return real_conn.__enter__()

        def __exit__(self, *exc):
            return real_conn.__exit__(*exc)

    monkeypatch.setattr(store, "_conn", FailingMessages())
    with pytest.raises(sqlite3.OperationalError):
        store.save_conversation("Robin", sample)
    monkeypatch.setattr(store, "_conn", real_conn)
    assert store.conversations("Robin") == []


# -- incremental saves ---------------------------------------------------------


def _slice(conversation, start, stop=None):
    """The same chat, exported over a different stretch of it."""
    part = read_transcript(EXAMPLES / "sample_chat.txt")
    part.messages = conversation.messages[start:stop]
    part.source = f"export-{start}-{stop}.txt"
    return part


def test_a_longer_export_appends_only_the_new_messages(store, sample):
    store.add_person("Robin")
    first = store.save_conversation("Robin", _slice(sample, 0, 10)).conversation

    result = store.save_conversation("Robin", sample)
    assert (result.outcome, result.added) == (SaveOutcome.EXTENDED, 7)
    assert result.conversation.id == first.id
    assert result.conversation.messages == 17
    assert result.conversation.updated_at is not None
    assert result.conversation.imported_at == first.imported_at
    assert store.load_conversation(first.id).messages == sample.messages
    (robin,) = store.people()
    assert (robin.conversations, robin.messages) == (1, 17)


def test_an_export_that_starts_partway_through_appends_after_the_overlap(store, sample):
    store.add_person("Robin")
    first = store.save_conversation("Robin", _slice(sample, 0, 12)).conversation
    result = store.save_conversation("Robin", _slice(sample, 8))
    assert (result.outcome, result.added) == (SaveOutcome.EXTENDED, 5)
    assert store.load_conversation(first.id).messages == sample.messages


def test_appending_keeps_the_positions_in_order(store, sample):
    store.add_person("Robin")
    first = store.save_conversation("Robin", _slice(sample, 0, 5)).conversation
    store.save_conversation("Robin", _slice(sample, 3, 11))
    store.save_conversation("Robin", _slice(sample, 9))
    positions = [
        row[0]
        for row in store._conn.execute(
            "SELECT position FROM messages WHERE conversation_id = ? ORDER BY position",
            (first.id,),
        )
    ]
    assert positions == list(range(17))
    assert store.load_conversation(first.id).messages == sample.messages


@pytest.mark.parametrize(("start", "stop"), [(0, 10), (4, 9), (10, None)])
def test_an_export_already_inside_a_stored_conversation_adds_nothing(store, sample, start, stop):
    store.add_person("Robin")
    first = store.save_conversation("Robin", sample).conversation
    result = store.save_conversation("Robin", _slice(sample, start, stop))
    assert (result.outcome, result.added) == (SaveOutcome.UNCHANGED, 0)
    assert result.conversation.id == first.id
    assert store._conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0] == 17


def test_saving_the_longer_export_again_after_appending_adds_nothing(store, sample):
    store.add_person("Robin")
    store.save_conversation("Robin", _slice(sample, 0, 10))
    store.save_conversation("Robin", sample)
    assert store.save_conversation("Robin", sample).outcome is SaveOutcome.UNCHANGED
    assert len(store.conversations("Robin")) == 1


def test_names_to_redact_accumulate_across_appends(store, sample):
    store.add_person("Robin")
    early = _slice(sample, 0, 10)
    early.private_names = ["Maya"]
    first = store.save_conversation("Robin", early).conversation
    later = _slice(sample, 5)
    later.private_names = ["Jo", "Maya"]
    store.save_conversation("Robin", later)
    assert store.load_conversation(first.id).private_names == ["Maya", "Jo"]


def test_a_short_undated_overlap_is_treated_as_coincidence(store):
    """Two chats that both open with "hey" / "hi" are not the same chat."""
    store.add_person("Robin")
    store.save_conversation("Robin", parse_transcript("# owner: Sam\nRobin: hey\nSam: hi\n"))
    result = store.save_conversation(
        "Robin", parse_transcript("# owner: Sam\nRobin: hey\nSam: hi\nRobin: new week, who dis\n")
    )
    assert result.outcome is SaveOutcome.NEW
    assert len(store.conversations("Robin")) == 2


def test_a_long_enough_undated_overlap_is_trusted(store):
    opening = "# owner: Sam\nRobin: hey\nSam: hi\nRobin: how was the climb\n"
    store.add_person("Robin")
    store.save_conversation("Robin", parse_transcript(opening))
    result = store.save_conversation("Robin", parse_transcript(opening + "Sam: windy\n"))
    assert (result.outcome, result.added) == (SaveOutcome.EXTENDED, 1)


def test_a_single_timestamped_message_is_enough_overlap(store):
    store.add_person("Robin")
    store.save_conversation(
        "Robin", parse_transcript("# owner: Sam\n[2026-04-01 10:00] Robin: hey\n")
    )
    result = store.save_conversation(
        "Robin",
        parse_transcript(
            "# owner: Sam\n[2026-04-01 10:00] Robin: hey\n[2026-04-01 10:05] Sam: hi\n"
        ),
    )
    assert (result.outcome, result.added) == (SaveOutcome.EXTENDED, 1)


def test_the_longest_overlap_wins_when_two_conversations_could_be_extended(store):
    shared = "Robin: hey\nSam: hi\nRobin: how was the climb\n"
    store.add_person("Robin")
    shorter = parse_transcript("# owner: Sam\nSam: morning\n" + shared)
    longer = parse_transcript("# owner: Sam\nSam: evening\n" + shared + "Sam: windy\n")
    store.save_conversation("Robin", shorter)
    expected = store.save_conversation("Robin", longer).conversation
    # Both stored conversations end in a way this transcript opens with; the one it
    # repeats more of is the one it continues.
    result = store.save_conversation(
        "Robin", parse_transcript("# owner: Sam\n" + shared + "Sam: windy\nRobin: brr\n")
    )
    assert (result.outcome, result.added) == (SaveOutcome.EXTENDED, 1)
    assert result.conversation.id == expected.id


def test_extending_is_per_person(store, sample):
    store.add_person("Robin")
    store.add_person("Rob")
    store.save_conversation("Robin", _slice(sample, 0, 10))
    assert store.save_conversation("Rob", sample).outcome is SaveOutcome.NEW


def test_a_store_from_before_digests_is_upgraded_and_can_be_extended(db, sample):
    """Schema 1 had no message digests; opening it backfills them in place."""
    from confidant.store import _MIGRATIONS

    db.parent.mkdir(parents=True)
    conn = sqlite3.connect(db)
    conn.executescript(f"BEGIN;\n{_MIGRATIONS[0]}\nPRAGMA user_version = 1;\nCOMMIT;")
    conn.execute(
        "INSERT INTO people (name, name_key, added_at) VALUES ('Robin', 'robin', ?)",
        ("2026-09-29T09:00:00",),
    )
    conn.execute(
        "INSERT INTO conversations (person_id, owner_name, match_name, fingerprint, imported_at)"
        " VALUES (1, 'Sam', 'Robin', 'old', '2026-09-29T09:00:00')"
    )
    conn.executemany(
        "INSERT INTO messages (conversation_id, position, role, sender, text, sent_at)"
        " VALUES (1, ?, ?, ?, ?, ?)",
        [
            (i, m.role.value, m.sender, m.text, m.timestamp.isoformat())
            for i, m in enumerate(sample.messages[:10])
        ],
    )
    conn.commit()
    conn.close()

    with Store(db) as s:
        assert s._conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        result = s.save_conversation("Robin", sample)
        assert (result.outcome, result.added, result.conversation.id) == (
            SaveOutcome.EXTENDED,
            7,
            1,
        )
        assert s.load_conversation(1).messages == sample.messages
