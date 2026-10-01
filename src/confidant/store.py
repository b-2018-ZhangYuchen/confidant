"""A local SQLite store for the people you are seeing and your conversations with them.

One file, on your machine, holding the same thing a transcript holds: the unredacted
messages. Redaction happens on the way to the model, not on the way to disk, so a stored
conversation can be re-read, re-redacted with a newer redactor, or shown back to you
exactly as it was written. That makes the file as private as the chats themselves, and
the store treats it that way:

* It lives in your home directory (``~/.confidant/confidant.db``), not the working
  directory, so running Confidant inside a git checkout cannot leave chat history where
  ``git add .`` will find it. ``CONFIDANT_DB`` points it somewhere else.
* The folder is created ``0700`` and the file ``0600``: readable by you and nobody else.
* ``secure_delete`` is on, so removing a person overwrites their messages rather than
  leaving them in free pages for anyone who opens the file with a hex editor.

Chats keep going after they are saved, so saving is incremental. Every stored message
carries a digest of its content, indexed, and a new transcript whose opening messages
line up with the end of a stored conversation is appended to it. Lining up is done on
digests alone, through the index, from the first matching message on; only the messages
after the overlap are written; and a transcript already wholly stored writes nothing.

Nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from confidant.models import Conversation, Message, Role

__all__ = [
    "DuplicatePerson",
    "Person",
    "PersonSummary",
    "SaveOutcome",
    "SaveResult",
    "Store",
    "StoreError",
    "StoredConversation",
    "UnknownPerson",
    "default_db_path",
    "name_key",
]


class StoreError(RuntimeError):
    """Raised when the store cannot be opened or asked to do something it cannot do."""


class UnknownPerson(StoreError):
    """Raised when a name does not match anyone in the store."""


class DuplicatePerson(StoreError):
    """Raised when adding a name that is already in the store."""


def default_db_path() -> Path:
    """Where the store lives unless told otherwise."""
    override = os.getenv("CONFIDANT_DB")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".confidant" / "confidant.db"


@dataclass(frozen=True, slots=True)
class Person:
    id: int
    name: str
    added_at: datetime


@dataclass(frozen=True, slots=True)
class PersonSummary:
    """A person plus the counts a list view needs, computed in one query."""

    person: Person
    conversations: int
    messages: int
    last_message: datetime | None
    """The latest timestamped message across all their conversations."""


@dataclass(frozen=True, slots=True)
class StoredConversation:
    """What the store knows about a conversation without loading its messages."""

    id: int
    person_id: int
    source: str | None
    imported_at: datetime
    messages: int
    first_message: datetime | None
    last_message: datetime | None
    updated_at: datetime | None = None
    """When messages were last appended to it, or ``None`` if never."""


class SaveOutcome(StrEnum):
    NEW = "new"
    """Nothing stored lined up with it, so it was saved as a conversation of its own."""

    EXTENDED = "extended"
    """It continued a stored conversation, and the messages after the overlap were added."""

    UNCHANGED = "unchanged"
    """Every message in it was already stored, in order, in one conversation."""


@dataclass(frozen=True, slots=True)
class SaveResult:
    conversation: StoredConversation
    outcome: SaveOutcome
    added: int
    """How many messages this save wrote."""


# Each entry upgrades the schema by one version; PRAGMA user_version records how many
# have run. Append, never edit: a file created by an older Confidant replays only the
# entries it has not seen yet.
_MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE people (
        id        INTEGER PRIMARY KEY,
        name      TEXT NOT NULL,
        -- Python's casefold, not COLLATE NOCASE, which only folds ASCII: "Zoë" and
        -- "ZOË" must be the same person, as they are to the transcript parser.
        name_key  TEXT NOT NULL UNIQUE,
        added_at  TEXT NOT NULL
    );

    CREATE TABLE conversations (
        id             INTEGER PRIMARY KEY,
        person_id      INTEGER NOT NULL REFERENCES people(id) ON DELETE CASCADE,
        owner_name     TEXT NOT NULL,
        match_name     TEXT NOT NULL,
        source         TEXT,
        private_names  TEXT NOT NULL DEFAULT '[]',
        fingerprint    TEXT NOT NULL,
        imported_at    TEXT NOT NULL,
        UNIQUE (person_id, fingerprint)
    );
    CREATE INDEX conversations_by_person ON conversations(person_id);

    CREATE TABLE messages (
        conversation_id  INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
        position         INTEGER NOT NULL,
        role             TEXT NOT NULL CHECK (role IN ('owner', 'match')),
        sender           TEXT NOT NULL,
        text             TEXT NOT NULL,
        sent_at          TEXT,
        PRIMARY KEY (conversation_id, position)
    ) WITHOUT ROWID;
    """,
    # Per-message digests, so a re-exported chat can be lined up against what is already
    # stored without loading it. confidant_digest is registered on the connection before
    # migrating; it is the same function new messages are stamped with.
    """
    ALTER TABLE messages ADD COLUMN digest TEXT NOT NULL DEFAULT '';
    UPDATE messages SET digest = confidant_digest(role, sender, text, sent_at);
    CREATE INDEX messages_by_digest ON messages(digest);
    ALTER TABLE conversations ADD COLUMN updated_at TEXT;
    """,
)

SCHEMA_VERSION = len(_MIGRATIONS)


def name_key(name: str) -> str:
    """How the store compares names: spacing collapsed, case folded."""
    return " ".join(name.split()).casefold()


def _stamp(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _unstamp(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None


def _digest(role: str, sender: str, text: str, sent_at: str | None) -> str:
    payload = json.dumps([role, sender, text, sent_at], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def message_digest(message: Message) -> str:
    """What makes two messages the same message: who, what, and when, exactly."""
    return _digest(message.role.value, message.sender, message.text, _stamp(message.timestamp))


# Below this many messages, an untimestamped overlap is too likely to be coincidence:
# plenty of separate chats open with the same "hey" / "hi!". Timestamped messages carry
# the minute they were sent, so a single one lining up is already convincing.
MIN_UNDATED_OVERLAP = 3


def _overlap_is_trustworthy(overlap: list[Message]) -> bool:
    return len(overlap) >= MIN_UNDATED_OVERLAP or all(m.timestamp is not None for m in overlap)


def fingerprint(conversation: Conversation) -> str:
    """A hash of the messages alone, so importing the same chat twice is noticed.

    The source path is left out on purpose: the same export saved under a new name is
    still the same conversation.
    """
    canonical = [
        [m.role.value, m.sender, m.text, _stamp(m.timestamp)] for m in conversation.messages
    ]
    payload = json.dumps(canonical, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Store:
    """The people and conversations Confidant remembers.

    Use it as a context manager so the connection is closed::

        with Store() as store:
            store.add_person("Robin")
            store.save_conversation("Robin", read_transcript("chat.txt"))
    """

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        clock: Callable[[], datetime] = datetime.now,
    ) -> None:
        self.path = Path(path).expanduser() if path is not None else default_db_path()
        self._clock = clock
        self._conn = self._connect()

    # -- lifecycle -----------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        try:
            self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            # Create the file ourselves so it is private from its first byte; sqlite
            # would create it with the process umask, usually world-readable.
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
            os.close(fd)
        except IsADirectoryError as exc:
            raise StoreError(f"{self.path} is a folder; the store needs a file path") from exc
        except OSError as exc:
            raise StoreError(f"Cannot create the store at {self.path}: {exc.strerror}") from exc

        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        conn.create_function("confidant_digest", 4, _digest, deterministic=True)
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute("PRAGMA secure_delete = ON")
            self._migrate(conn)
        except sqlite3.DatabaseError as exc:
            conn.close()
            raise StoreError(
                f"{self.path} is not a Confidant store ({exc}). Point CONFIDANT_DB somewhere else."
            ) from exc
        except StoreError:
            conn.close()
            raise
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            # Guessing at a newer schema risks writing rows a newer Confidant then
            # misreads. Better to refuse and say why.
            raise StoreError(
                f"{self.path} was written by a newer version of Confidant "
                f"(schema {version}, this version understands up to {SCHEMA_VERSION}). "
                "Upgrade Confidant to open it."
            )
        for number, script in enumerate(_MIGRATIONS[version:], start=version + 1):
            # executescript commits whatever is open first, so each migration and its
            # version bump land together or not at all.
            conn.executescript(f"BEGIN;\n{script}\nPRAGMA user_version = {number};\nCOMMIT;")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- people --------------------------------------------------------------

    def add_person(self, name: str) -> Person:
        name = " ".join(name.split())
        if not name:
            raise StoreError("A person needs a name")
        added_at = self._clock().replace(microsecond=0)
        try:
            with self._conn:
                cursor = self._conn.execute(
                    "INSERT INTO people (name, name_key, added_at) VALUES (?, ?, ?)",
                    (name, name_key(name), _stamp(added_at)),
                )
        except sqlite3.IntegrityError as exc:
            existing = self.find_person(name)
            shown = existing.name if existing else name
            raise DuplicatePerson(f"{shown!r} is already in the store") from exc
        return Person(id=cursor.lastrowid, name=name, added_at=added_at)

    def find_person(self, name: str) -> Person | None:
        row = self._conn.execute(
            "SELECT id, name, added_at FROM people WHERE name_key = ?", (name_key(name),)
        ).fetchone()
        return self._person(row) if row else None

    def person(self, name: str) -> Person:
        found = self.find_person(name)
        if found is None:
            raise UnknownPerson(f"No one called {name!r} is in the store")
        return found

    def people(self) -> list[PersonSummary]:
        """Everyone in the store, alphabetically, with what has been saved for each."""
        rows = self._conn.execute(
            """
            SELECT p.id, p.name, p.added_at,
                   COUNT(DISTINCT c.id)  AS conversations,
                   COUNT(m.position)     AS messages,
                   MAX(m.sent_at)        AS last_message
            FROM people p
            LEFT JOIN conversations c ON c.person_id = p.id
            LEFT JOIN messages m      ON m.conversation_id = c.id
            GROUP BY p.id
            ORDER BY p.name_key
            """
        ).fetchall()
        return [
            PersonSummary(
                person=self._person(row),
                conversations=row["conversations"],
                messages=row["messages"],
                # ISO strings sort chronologically, so MAX in SQL is the latest moment.
                last_message=_unstamp(row["last_message"]),
            )
            for row in rows
        ]

    def remove_person(self, name: str) -> int:
        """Forget someone and every conversation with them. Returns how many went."""
        person = self.person(name)
        with self._conn:
            removed = self._conn.execute(
                "SELECT COUNT(*) FROM conversations WHERE person_id = ?", (person.id,)
            ).fetchone()[0]
            self._conn.execute("DELETE FROM people WHERE id = ?", (person.id,))
        return removed

    # -- conversations -------------------------------------------------------

    def save_conversation(self, person: str | Person, conversation: Conversation) -> SaveResult:
        """Store ``conversation`` under ``person``, adding only what is new.

        Three things can happen, and the result says which:

        * every message is already stored, in order, in one of their conversations, as
          when the same export is saved twice: nothing is written;
        * its opening messages are the last messages of a stored conversation, as when
          the chat is exported again after it carried on: the rest is appended to it;
        * neither: it is saved as a conversation of its own.

        Writing a second copy instead would double every count built on top of it, and
        split one conversation's history in two.
        """
        if isinstance(person, str):
            person = self.person(person)
        if not conversation.messages:
            raise StoreError("Cannot store a conversation with no messages")

        digest = fingerprint(conversation)
        existing = self._conn.execute(
            "SELECT id FROM conversations WHERE person_id = ? AND fingerprint = ?",
            (person.id, digest),
        ).fetchone()
        if existing:
            return SaveResult(self.conversation(existing["id"]), SaveOutcome.UNCHANGED, 0)

        digests = [message_digest(m) for m in conversation.messages]
        contained, extends, overlap = self._line_up(person, digests)
        if contained is not None and _overlap_is_trustworthy(conversation.messages):
            return SaveResult(self.conversation(contained), SaveOutcome.UNCHANGED, 0)
        if extends is not None and _overlap_is_trustworthy(conversation.messages[:overlap]):
            added = self._append(extends, conversation, digests, overlap)
            return SaveResult(self.conversation(extends), SaveOutcome.EXTENDED, added)

        with self._conn:
            cursor = self._conn.execute(
                """
                INSERT INTO conversations
                    (person_id, owner_name, match_name, source, private_names,
                     fingerprint, imported_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    person.id,
                    conversation.owner_name,
                    conversation.match_name,
                    conversation.source,
                    json.dumps(conversation.private_names, ensure_ascii=False),
                    digest,
                    _stamp(self._clock().replace(microsecond=0)),
                ),
            )
            conversation_id = cursor.lastrowid
            self._insert_messages(conversation_id, 0, conversation.messages, digests)
        return SaveResult(
            self.conversation(conversation_id), SaveOutcome.NEW, len(conversation.messages)
        )

    def _line_up(self, person: Person, digests: list[str]) -> tuple[int | None, int | None, int]:
        """Find where a new transcript sits against this person's stored conversations.

        Returns ``(contained, extends, overlap)``: the id of a conversation that already
        holds every message, the id of the conversation whose last ``overlap`` messages
        open the transcript, or ``None`` for either. When several conversations could be
        extended, the longest overlap wins, then the earliest conversation.
        """
        candidates = self._conn.execute(
            """
            SELECT m.conversation_id, m.position
            FROM messages m JOIN conversations c ON c.id = m.conversation_id
            WHERE c.person_id = ? AND m.digest = ?
            ORDER BY m.conversation_id, m.position
            """,
            (person.id, digests[0]),
        ).fetchall()

        extends, overlap = None, 0
        for candidate in candidates:
            # The stored messages from the candidate on, capped at the transcript's length:
            # anything further would only ever show it is contained, which the cap shows.
            tail = [
                row["digest"]
                for row in self._conn.execute(
                    "SELECT digest FROM messages WHERE conversation_id = ? AND position >= ? "
                    "ORDER BY position LIMIT ?",
                    (candidate["conversation_id"], candidate["position"], len(digests)),
                )
            ]
            if tail == digests:
                return candidate["conversation_id"], None, len(digests)
            # A shorter tail means the stored conversation ends inside the transcript.
            if len(tail) < len(digests) and digests[: len(tail)] == tail and len(tail) > overlap:
                extends, overlap = candidate["conversation_id"], len(tail)
        return None, extends, overlap

    def _append(
        self, conversation_id: int, conversation: Conversation, digests: list[str], overlap: int
    ) -> int:
        new_messages = conversation.messages[overlap:]
        header = self._conn.execute(
            "SELECT private_names FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
        # Names to redact only ever accumulate: one given for an earlier export still
        # appears in the messages saved from it.
        private_names = json.loads(header["private_names"])
        private_names += [n for n in conversation.private_names if n not in private_names]
        start = self._conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()[0]

        with self._conn:
            self._insert_messages(conversation_id, start, new_messages, digests[overlap:])
            # The fingerprint follows the content, so saving this longer export again is
            # recognised as nothing new.
            merged = self.load_conversation(conversation_id)
            self._conn.execute(
                "UPDATE conversations SET fingerprint = ?, private_names = ?, updated_at = ? "
                "WHERE id = ?",
                (
                    fingerprint(merged),
                    json.dumps(private_names, ensure_ascii=False),
                    _stamp(self._clock().replace(microsecond=0)),
                    conversation_id,
                ),
            )
        return len(new_messages)

    def _insert_messages(
        self, conversation_id: int, start: int, messages: list[Message], digests: list[str]
    ) -> None:
        self._conn.executemany(
            """
            INSERT INTO messages
                (conversation_id, position, role, sender, text, sent_at, digest)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (conversation_id, start + i, m.role.value, m.sender, m.text, _stamp(m.timestamp), d)
                for i, (m, d) in enumerate(zip(messages, digests, strict=True))
            ],
        )

    def conversation(self, conversation_id: int) -> StoredConversation:
        row = self._conn.execute(
            self._CONVERSATION_SUMMARY + " WHERE c.id = ? GROUP BY c.id", (conversation_id,)
        ).fetchone()
        if row is None:
            raise StoreError(f"No conversation with id {conversation_id}")
        return self._stored(row)

    def conversations(self, person: str | Person) -> list[StoredConversation]:
        """A person's conversations, oldest import first."""
        if isinstance(person, str):
            person = self.person(person)
        rows = self._conn.execute(
            self._CONVERSATION_SUMMARY
            + " WHERE c.person_id = ? GROUP BY c.id ORDER BY c.imported_at, c.id",
            (person.id,),
        ).fetchall()
        return [self._stored(row) for row in rows]

    def load_conversation(self, conversation_id: int) -> Conversation:
        """Rebuild the :class:`Conversation` exactly as it was saved."""
        header = self._conn.execute(
            "SELECT owner_name, match_name, source, private_names FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if header is None:
            raise StoreError(f"No conversation with id {conversation_id}")
        rows = self._conn.execute(
            "SELECT role, sender, text, sent_at FROM messages "
            "WHERE conversation_id = ? ORDER BY position",
            (conversation_id,),
        ).fetchall()
        return Conversation(
            match_name=header["match_name"],
            owner_name=header["owner_name"],
            messages=[
                Message(
                    role=Role(row["role"]),
                    text=row["text"],
                    sender=row["sender"],
                    timestamp=_unstamp(row["sent_at"]),
                )
                for row in rows
            ],
            source=header["source"],
            private_names=json.loads(header["private_names"]),
        )

    def remove_conversation(self, conversation_id: int) -> None:
        with self._conn:
            cursor = self._conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conversation_id,)
            )
        if cursor.rowcount == 0:
            raise StoreError(f"No conversation with id {conversation_id}")

    # -- row mapping ---------------------------------------------------------

    _CONVERSATION_SUMMARY = """
        SELECT c.id, c.person_id, c.source, c.imported_at, c.updated_at,
               COUNT(m.position) AS messages,
               MIN(m.sent_at)    AS first_message,
               MAX(m.sent_at)    AS last_message
        FROM conversations c
        LEFT JOIN messages m ON m.conversation_id = c.id
    """

    @staticmethod
    def _person(row: sqlite3.Row) -> Person:
        return Person(id=row["id"], name=row["name"], added_at=_unstamp(row["added_at"]))

    @staticmethod
    def _stored(row: sqlite3.Row) -> StoredConversation:
        return StoredConversation(
            id=row["id"],
            person_id=row["person_id"],
            source=row["source"],
            imported_at=_unstamp(row["imported_at"]),
            messages=row["messages"],
            first_message=_unstamp(row["first_message"]),
            last_message=_unstamp(row["last_message"]),
            updated_at=_unstamp(row["updated_at"]),
        )
