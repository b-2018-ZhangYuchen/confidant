"""Strip names, numbers, and addresses from a conversation before it leaves the machine.

Everything the model needs to read a conversation is in how people write to each other,
not in who they are. So before a request is built, the two names, any other names the
owner asks for, phone and long account-like numbers, email and street addresses, links,
and social handles are swapped for bracketed placeholders: ``[MATCH]``, ``[PHONE_1]``,
``[ADDRESS_1]``. The same detail always gets the same placeholder, so "sent their number
twice" is still visible; different details never share one.

The model answers in placeholders, and everything local happens in placeholders too:
grounding checks quotes against the *redacted* transcript, which is exactly what the
model saw, and only the finished report is swapped back with :meth:`Redaction.restore`.
Restoring first and grounding afterwards would be fragile, because a placeholder stands
for every spelling of a name ("casey", "Casey") and can only be restored to one.

This is pattern matching, not understanding. It catches the shapes that identify people
in a chat; it does not catch "the bakery on the corner by my work". Names other than the
two in the conversation are only known if the owner lists them, with ``# redact:`` in
the transcript or ``--redact`` on the command line. ``confidant redact`` prints what
would be sent, so none of this has to be taken on trust.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, replace
from typing import Any, TypeVar

from pydantic import BaseModel

from confidant.models import Conversation, Role

__all__ = ["KINDS", "Redaction", "redact"]

T = TypeVar("T", bound=BaseModel)

# Plural labels for the summary line, in the order they are listed.
KINDS: dict[str, tuple[str, str]] = {
    "name": ("name", "names"),
    "phone": ("phone number", "phone numbers"),
    "number": ("long number", "long numbers"),
    "email": ("email address", "email addresses"),
    "address": ("street address", "street addresses"),
    "link": ("link", "links"),
    "handle": ("handle", "handles"),
}

_URL = re.compile(r"(?:https?://|www\.)[^\s<>\"]+", re.IGNORECASE)
_EMAIL = re.compile(r"[\w.%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}")
_HANDLE = re.compile(r"(?<![\w.@])@\w(?:[\w.]{0,28}\w)?")

# A house number, one to three words, then a street suffix. The lookahead keeps
# "ran 5 miles down the road" and "10 more minutes this way" from reading as addresses.
_ADDRESS = re.compile(
    r"\b\d{1,5}[A-Za-z]?\s+"
    r"(?:(?!(?:the|a|an|my|your|our|this|that|down|up|to|of|on|in|at|by|and|or|more|"
    r"miles?|km|minutes?|mins?|blocks?|hours?|times?|days?)\b)[A-Za-z][\w'.-]*\s+){1,3}"
    r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|place|pl|"
    r"terrace|way|parkway|pkwy|square|sq|highway|hwy)\b\.?"
    r"(?:,?\s*(?:apt|apartment|unit|suite|ste|#)\.?\s*[\w-]+)?",
    re.IGNORECASE,
)

# Digits with the separators people put in phone numbers. Currency is excluded by the
# lookbehind: "$1500" is exactly the kind of detail a financial-pressure flag needs.
_NUMBER = re.compile(r"(?<![\w$€£¥.,])\+?\(?\d[\d ().-]{5,}\d(?![\w%])")
_CURRENCY_AFTER = re.compile(r"\s*(?:dollars|bucks|usd|eur|euros?|pounds|gbp)\b", re.IGNORECASE)
_DATE_LIKE = re.compile(
    r"\d{4}[-.]\d{1,2}[-.]\d{1,2}|\d{1,2}[-.]\d{1,2}[-.]\d{2,4}|(?:19|20)\d{2} ?- ?(?:19|20)\d{2}"
)
_MIN_PHONE_DIGITS = 7
_MIN_LONG_DIGITS = 13
_MAX_DIGITS = 19

# People label themselves "Me" in hand-made transcripts. The label is replaced where it
# names a speaker, but "me" inside a message is a pronoun, not a name.
_NOT_NAMES = frozenset({"me", "you", "i", "myself", "them", "him", "her", "us", "we"})

_TRAILING = ".,!?;:)'\"—–"


@dataclass(slots=True)
class _Span:
    start: int
    end: int
    kind: str
    key: str


@dataclass(slots=True)
class Redaction:
    """A redacted conversation, and the key to put the details back afterwards."""

    conversation: Conversation
    """Safe to send: every detail found is a placeholder."""

    originals: dict[str, str] = field(default_factory=dict)
    """Placeholder to the detail it replaced, in the order they were first seen."""

    kinds: dict[str, str] = field(default_factory=dict)
    """Placeholder to its kind, a key of :data:`KINDS`."""

    notes: list[str] = field(default_factory=list)
    """The owner's own free text for the request, redacted with the same key."""

    def restore_text(self, text: str) -> str:
        """Swap every placeholder this redaction issued back for what it replaced.

        Placeholders the model invented, like a ``[PHONE_9]`` that was never issued, are
        left alone: there is nothing true to put back.
        """
        if not self.originals:
            return text
        return _PLACEHOLDER.sub(lambda m: self.originals.get(m[0], m[0]), text)

    def unissued(self, text: str) -> list[str]:
        """Placeholders in ``text`` that this redaction never issued.

        Restoring leaves these as they are, so anything the owner might copy and send
        should be checked for them first.
        """
        return [m[0] for m in _PLACEHOLDER.finditer(text) if m[0] not in self.originals]

    def restore(self, value: T) -> T:
        """A copy of ``value`` with placeholders restored in every string field."""
        data = _map_strings(value.model_dump(), self.restore_text)
        return type(value).model_validate(data)

    def summary(self) -> str:
        """``"2 names, 1 phone number"`` — what was replaced, for the terminal."""
        counts = Counter(self.kinds.values())
        parts = []
        for kind, (one, many) in KINDS.items():
            if counts[kind]:
                parts.append(f"{counts[kind]} {one if counts[kind] == 1 else many}")
        return ", ".join(parts) if parts else "nothing"


_PLACEHOLDER = re.compile(r"\[(?:OWNER|MATCH|[A-Z]+_\d+)\]")


def _map_strings(value: Any, fn: Callable[[str], str]) -> Any:
    if isinstance(value, str):
        return fn(value)
    if isinstance(value, dict):
        return {k: _map_strings(v, fn) for k, v in value.items()}
    if isinstance(value, list):
        return [_map_strings(v, fn) for v in value]
    return value


def _strip_trailing(text: str, start: int, end: int) -> int:
    while end > start and text[end - 1] in _TRAILING:
        end -= 1
    return end


def _number_kind(match: re.Match[str], text: str) -> str | None:
    raw = match[0]
    digits = re.sub(r"\D", "", raw)
    if not _MIN_PHONE_DIGITS <= len(digits) <= _MAX_DIGITS:
        return None
    if _DATE_LIKE.fullmatch(raw.strip()):
        return None
    if _CURRENCY_AFTER.match(text, match.end()):
        return None
    return "number" if len(digits) >= _MIN_LONG_DIGITS else "phone"


class _Redactor:
    def __init__(self, conversation: Conversation, extra_names: Iterable[str]) -> None:
        self.placeholders: dict[tuple[str, str], str] = {}
        self.originals: dict[str, str] = {}
        self.kinds: dict[str, str] = {}
        self.counters: Counter[str] = Counter()

        # Every spelling that should be replaced, mapped to whose name it is. Full names
        # go in before their parts so "Robin Lee" wins over "Robin" in the alternation.
        self.name_owner: dict[str, str] = {}
        # Whose name it is, to the spelling it restores to. A placeholder stands for every
        # spelling of a name, so it goes back as the one the owner wrote in the header.
        self.name_original: dict[str, str] = {}
        self._claim(conversation.owner_name, "owner", "[OWNER]", conversation.owner_name)
        self._claim(conversation.match_name, "match", "[MATCH]", conversation.match_name)
        for name in extra_names:
            name = name.strip()
            if name and name.casefold() not in self.name_owner:
                self._claim(name, name.casefold(), None, name)

        spellings = sorted(
            (s for s in self.name_owner if s not in _NOT_NAMES), key=len, reverse=True
        )
        self.name_pattern = (
            re.compile(
                r"(?<!\w)(?:" + "|".join(re.escape(s) for s in spellings) + r")(?!\w)",
                re.IGNORECASE,
            )
            if spellings
            else None
        )

    def _claim(self, name: str, who: str, placeholder: str | None, original: str) -> None:
        self.name_original[who] = original
        for form in (name, *name.split()):
            form = form.casefold()
            if len(form) >= 2 and form not in self.name_owner:
                self.name_owner[form] = who
        # The two people are issued up front, since the header always names them. Anyone
        # else gets a number only if they actually appear.
        if placeholder is not None:
            self._issue("name", who, original, placeholder)

    def _issue(self, kind: str, key: str, original: str, placeholder: str | None = None) -> str:
        existing = self.placeholders.get((kind, key))
        if existing is not None:
            return existing
        if placeholder is None:
            self.counters[kind] += 1
            placeholder = f"[{kind.upper()}_{self.counters[kind]}]"
        self.placeholders[(kind, key)] = placeholder
        self.originals[placeholder] = original
        self.kinds[placeholder] = kind
        return placeholder

    def _spans(self, text: str) -> list[_Span]:
        found: list[_Span] = []

        def take(start: int, end: int, kind: str, key: str) -> None:
            if end <= start or any(s.start < end and start < s.end for s in found):
                return
            found.append(_Span(start, end, kind, key))

        # Most specific first: a link can contain an @, an email contains a name, and an
        # address contains a number. The first pattern to claim a stretch of text keeps it.
        for m in _URL.finditer(text):
            end = _strip_trailing(text, m.start(), m.end())
            take(m.start(), end, "link", text[m.start() : end])
        for m in _EMAIL.finditer(text):
            take(m.start(), m.end(), "email", m[0].casefold())
        for m in _HANDLE.finditer(text):
            end = _strip_trailing(text, m.start(), m.end())
            take(m.start(), end, "handle", text[m.start() : end].casefold())
        for m in _ADDRESS.finditer(text):
            take(m.start(), m.end(), "address", " ".join(m[0].casefold().split()))
        for m in _NUMBER.finditer(text):
            kind = _number_kind(m, text)
            if kind:
                take(m.start(), m.end(), kind, re.sub(r"\D", "", m[0]))
        if self.name_pattern is not None:
            for m in self.name_pattern.finditer(text):
                take(m.start(), m.end(), "name", self.name_owner[m[0].casefold()])
        return sorted(found, key=lambda s: s.start)

    def text(self, text: str) -> str:
        out: list[str] = []
        position = 0
        for span in self._spans(text):
            out.append(text[position : span.start])
            original = text[span.start : span.end]
            if span.kind == "name":
                original = self.name_original[span.key]
            out.append(self._issue(span.kind, span.key, original))
            position = span.end
        out.append(text[position:])
        return "".join(out)


def redact(
    conversation: Conversation, *, names: Iterable[str] = (), notes: Iterable[str] = ()
) -> Redaction:
    """Replace identifying details in ``conversation`` with placeholders.

    ``names`` are redacted in addition to the owner, the match, and anything listed in
    the conversation's ``private_names``. ``notes`` is anything else the owner wrote for
    the request; it shares the transcript's placeholders, so "ask Maya along" becomes
    "ask [NAME_1] along" if the transcript already called her that.
    """
    extra = [*conversation.private_names, *names]
    redactor = _Redactor(conversation, extra)

    tags = {Role.OWNER: "[OWNER]", Role.MATCH: "[MATCH]"}
    messages = [
        replace(message, text=redactor.text(message.text), sender=tags[message.role])
        for message in conversation.messages
    ]
    redacted = Conversation(
        match_name="[MATCH]",
        owner_name="[OWNER]",
        messages=messages,
        source=None,
    )
    # After the messages, so a note never changes how the transcript itself is numbered.
    redacted_notes = [redactor.text(note) for note in notes]
    return Redaction(
        conversation=redacted,
        originals=redactor.originals,
        kinds=redactor.kinds,
        notes=redacted_notes,
    )
