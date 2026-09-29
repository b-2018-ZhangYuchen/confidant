"""Tests for the SQLite store. Every store lives in a temporary directory."""

from __future__ import annotations

import sqlite3
import stat
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from confidant.ingest.transcript import parse_transcript, read_transcript
from confidant.store import (
    SCHEMA_VERSION,
    DuplicatePerson,
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
    stored, _ = store.save_conversation("Robin", sample)
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
    stored, created = store.save_conversation("Theo", original)
    assert created

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
    stored, _ = store.save_conversation("Theo", original)
    assert "214 Linden Street" in store.load_conversation(stored.id).transcript()


def test_multiline_messages_and_missing_timestamps_survive(store):
    original = parse_transcript(
        "# owner: Sam\nRobin: first line\n    second line\nSam: no timestamp here\n"
    )
    store.add_person("Robin")
    stored, _ = store.save_conversation("Robin", original)
    loaded = store.load_conversation(stored.id)
    assert loaded.messages == original.messages
    assert loaded.messages[0].text == "first line\nsecond line"


def test_stored_conversation_summarises_without_loading_messages(store, sample):
    store.add_person("Robin")
    stored, _ = store.save_conversation("Robin", sample)
    assert stored.messages == len(sample)
    assert stored.first_message == datetime(2026, 3, 2, 19, 4)
    assert stored.last_message == sample.last_activity
    assert stored.source == str(EXAMPLES / "sample_chat.txt")
    assert stored.imported_at == datetime(2026, 9, 29, 9, 2)


def test_saving_the_same_conversation_twice_keeps_one_copy(store, sample):
    store.add_person("Robin")
    first, created_first = store.save_conversation("Robin", sample)
    again = read_transcript(EXAMPLES / "sample_chat.txt")
    again.source = "renamed-export.txt"
    second, created_second = store.save_conversation("Robin", again)
    assert (created_first, created_second) == (True, False)
    assert second == first
    assert len(store.conversations("Robin")) == 1


def test_the_same_conversation_can_belong_to_two_people(store, sample):
    """Deduplication is per person: filing a chat under the wrong name is fixable."""
    store.add_person("Robin")
    store.add_person("Rob")
    assert store.save_conversation("Robin", sample)[1]
    assert store.save_conversation("Rob", sample)[1]


def test_a_changed_conversation_is_saved_as_new(store, sample):
    store.add_person("Robin")
    store.save_conversation("Robin", sample)
    longer = read_transcript(EXAMPLES / "sample_chat.txt")
    longer.messages.append(longer.messages[-1])
    assert store.save_conversation("Robin", longer)[1]


def test_conversations_are_listed_oldest_import_first(store, sample):
    later = parse_transcript("# owner: Sam\nRobin: still on for friday?\n")
    store.add_person("Robin")
    first, _ = store.save_conversation("Robin", sample)
    second, _ = store.save_conversation("Robin", later)
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
    first, _ = store.save_conversation("Robin", sample)
    second, _ = store.save_conversation("Robin", later)
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
