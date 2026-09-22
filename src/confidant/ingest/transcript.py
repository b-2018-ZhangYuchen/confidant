"""Parser for Confidant's plain-text transcript format.

The format is meant to be something you can paste together by hand in thirty seconds,
because that is what people actually do with their chat history::

    # owner: Yuchen
    # match: Alex

    [2026-03-01 19:04] Alex: hey! how was your week?
    [2026-03-01 19:20] Yuchen: honestly kind of long
        still recovering from midterms
    Alex: oof

Rules:

* ``# key: value`` lines at any point are directives; ``owner`` and ``match`` are read,
  anything else is ignored so you can leave yourself notes.
* ``[timestamp] `` in front of a message is optional.
* Indented lines continue the previous message, so multi-line messages survive a paste.
* Blank lines are ignored.

Ambiguity is the interesting part. A line like ``one thing: I had fun`` looks exactly
like a new message from a speaker named "one thing". So once the parser has learned who
is actually talking, it stops guessing: only known speakers can start a new message.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from confidant.models import Conversation, Message, Role

__all__ = ["TranscriptError", "parse_transcript", "read_transcript"]


class TranscriptError(ValueError):
    """Raised when a transcript cannot be parsed unambiguously."""


_DIRECTIVE = re.compile(r"^#\s*(?P<key>[a-zA-Z_]+)\s*:\s*(?P<value>.+?)\s*$")
_LINE = re.compile(
    r"^(?:\[(?P<ts>[^\]]+)\]\s*)?"  # optional [timestamp]
    r"(?P<sender>[^:]{1,40}?)"  # speaker name
    r":\s(?P<text>.*)$"  # ": " then the message
)

_TIMESTAMP_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M",
    "%m/%d/%y %H:%M",
    "%b %d, %Y %H:%M",
)

# A speaker name is short, and does not look like prose or a URL fragment.
_PLAUSIBLE_NAME = re.compile(r"^[\w .'’\-]{1,40}$", re.UNICODE)
_MAX_NAME_WORDS = 3


def _parse_timestamp(raw: str) -> datetime:
    raw = raw.strip()
    for fmt in _TIMESTAMP_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw)
    except ValueError as exc:
        raise TranscriptError(f"Unrecognized timestamp: {raw!r}") from exc


def _looks_like_a_name(candidate: str) -> bool:
    return bool(_PLAUSIBLE_NAME.match(candidate)) and len(candidate.split()) <= _MAX_NAME_WORDS


def parse_transcript(
    text: str,
    *,
    owner: str | None = None,
    match: str | None = None,
    source: str | None = None,
) -> Conversation:
    """Parse transcript ``text`` into a :class:`~confidant.models.Conversation`.

    ``owner`` and ``match`` override the ``# owner:`` / ``# match:`` directives in the
    file. The owner must be identifiable one way or the other — without it there is no
    way to tell whose side of the conversation is whose.
    """
    directives: dict[str, str] = {}
    raw_messages: list[tuple[str, str, datetime | None]] = []
    known_senders: set[str] = set()
    # Names that looked like speakers but arrived after both slots were taken. A single
    # one is almost always prose ("reminder: bring the book"); the same name opening
    # several lines is a third person in the room.
    turned_away: Counter[str] = Counter()

    if owner:
        known_senders.add(owner.casefold())
    if match:
        known_senders.add(match.casefold())

    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            continue

        if raw_line.lstrip().startswith("#"):
            directive = _DIRECTIVE.match(raw_line.strip())
            if directive:
                directives[directive["key"].casefold()] = directive["value"]
            continue

        # Indented lines continue the message above them.
        is_continuation = raw_line[:1].isspace()
        line = raw_line.strip()

        if is_continuation and raw_messages:
            sender, body, stamp = raw_messages[-1]
            raw_messages[-1] = (sender, f"{body}\n{line}", stamp)
            continue

        parsed = _LINE.match(line)
        if parsed:
            sender = parsed["sender"].strip()
            # Once we know who is in this conversation, only they can open a message.
            # Before then, fall back to "does this look like a name at all".
            recognized = sender.casefold() in known_senders
            room_for_another = len(known_senders) < 2 and _looks_like_a_name(sender)
            if recognized or room_for_another:
                stamp = _parse_timestamp(parsed["ts"]) if parsed["ts"] else None
                raw_messages.append((sender, parsed["text"].strip(), stamp))
                known_senders.add(sender.casefold())
                continue
            if _looks_like_a_name(sender):
                turned_away[sender] += 1

        if raw_messages:
            sender, body, stamp = raw_messages[-1]
            raw_messages[-1] = (sender, f"{body}\n{line}", stamp)
        else:
            raise TranscriptError(
                f"Line {lineno} is not a message and there is nothing before it to "
                f"attach it to: {line!r}"
            )

    owner_name = owner or directives.get("owner")
    match_name = match or directives.get("match")

    if not raw_messages:
        raise TranscriptError("Transcript contains no messages")

    senders_in_order: list[str] = []
    for sender, _, _ in raw_messages:
        if sender not in senders_in_order:
            senders_in_order.append(sender)

    if owner_name is None:
        raise TranscriptError(
            "Cannot tell which side of this conversation is yours. Add a '# owner: <name>' "
            f"line to the transcript or pass owner=... (speakers found: {senders_in_order})"
        )

    others = [s for s in senders_in_order if s.casefold() != owner_name.casefold()]
    repeat_outsiders = sorted(name for name, count in turned_away.items() if count >= 2)
    if len(others) > 1 or repeat_outsiders:
        extra = others[1:] + repeat_outsiders
        raise TranscriptError(
            "Confidant handles one-on-one conversations, but these also appear to be "
            f"speaking: {extra}. If those are not people, rename them; if the owner or "
            "match name is wrong, pass owner=... / match=... explicitly."
        )
    # A stale '# match:' directive naming the owner is worse than no directive at all.
    if match_name is None or match_name.casefold() == owner_name.casefold():
        if not others:
            raise TranscriptError(f"Only {owner_name!r} speaks in this transcript")
        match_name = others[0]

    messages = [
        Message(
            role=Role.OWNER if sender.casefold() == owner_name.casefold() else Role.MATCH,
            text=body,
            sender=sender,
            timestamp=stamp,
        )
        for sender, body, stamp in raw_messages
    ]

    return Conversation(
        match_name=match_name,
        owner_name=owner_name,
        messages=messages,
        source=source,
    )


def read_transcript(
    path: str | Path,
    *,
    owner: str | None = None,
    match: str | None = None,
) -> Conversation:
    """Read and parse a transcript file."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TranscriptError(f"No transcript at {path}") from exc
    return parse_transcript(text, owner=owner, match=match, source=str(path))
