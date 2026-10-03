"""A person's profile: what every saved conversation with them adds up to.

A single read answers "how does this chat come across". Someone you keep seeing turns
into several chats, and the useful question becomes what holds across them. The profile
puts together, for one person:

* statistics over every saved message, computed locally;
* the latest personality read and red-flag check of each conversation, kept in the store
  so they are paid for once rather than every time the profile is shown;
* every red flag from those checks, most serious first, each tagged with the
  conversation it came from, under the fixed safety notice if any of them is danger.

Reads stay per conversation, and the profile shows them side by side instead of asking
the model to merge them. A merged summary would be a claim about the person that no
single quote supports, which ``docs/principles.md`` rules out; side by side, the owner
can see for themselves what repeats.

A read records how many messages its conversation had. When a newer export has added
messages since, the read is shown as out of date, and ``confidant profile NAME
--update`` reads it again. An out-of-date red-flag check still counts towards the safety
notice: a threat does not stop having been made because the chat carried on.

Building a profile never touches the network. Only :func:`read_conversation` does.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from confidant.analysis.flags import Flag, FlagReport, analyze_flags
from confidant.analysis.personality import PersonalityReport, analyze_personality
from confidant.config import Settings
from confidant.models import Conversation, Role
from confidant.progress import Progress
from confidant.safety import Escalation, escalate
from confidant.store import Person, Reading, ReadingKind, Store, StoredConversation

__all__ = [
    "ConversationEntry",
    "CurrentRead",
    "Profile",
    "TaggedFlag",
    "build_profile",
    "parse_reading",
    "read_conversation",
]

R = TypeVar("R", bound=BaseModel)

_SCHEMAS: dict[ReadingKind, type[BaseModel]] = {
    ReadingKind.PERSONALITY: PersonalityReport,
    ReadingKind.FLAGS: FlagReport,
}


@dataclass(frozen=True, slots=True)
class CurrentRead(Generic[R]):
    """The latest read of one kind for one conversation."""

    reading: Reading
    report: R
    messages_now: int

    @property
    def stale(self) -> bool:
        return self.reading.messages_read < self.messages_now


@dataclass(frozen=True, slots=True)
class ConversationEntry:
    stored: StoredConversation
    match_messages: int
    personality: CurrentRead[PersonalityReport] | None = None
    flags: CurrentRead[FlagReport] | None = None

    @property
    def pending(self) -> list[ReadingKind]:
        """The reads this conversation is missing or has outgrown."""
        if self.match_messages == 0:
            # The analyzers refuse a chat the other person never wrote in; there is
            # nothing of theirs to read, and asking would only fail.
            return []
        return [
            kind
            for kind, current in (
                (ReadingKind.PERSONALITY, self.personality),
                (ReadingKind.FLAGS, self.flags),
            )
            if current is None or current.stale
        ]


@dataclass(frozen=True, slots=True)
class TaggedFlag:
    conversation_id: int
    flag: Flag


@dataclass(slots=True)
class Profile:
    person: Person
    entries: list[ConversationEntry] = field(default_factory=list)
    combined: Conversation | None = None
    """Every saved message, from every conversation, for the local statistics."""

    @property
    def pending(self) -> list[tuple[StoredConversation, ReadingKind]]:
        return [(entry.stored, kind) for entry in self.entries for kind in entry.pending]

    @property
    def flags(self) -> list[TaggedFlag]:
        tagged = [
            TaggedFlag(entry.stored.id, flag)
            for entry in self.entries
            if entry.flags is not None
            for flag in entry.flags.report.flags
        ]
        # Stable, so within a tier the flags stay in conversation order.
        return sorted(tagged, key=lambda t: t.flag.tier, reverse=True)

    @property
    def escalation(self) -> Escalation | None:
        return escalate((t.flag for t in self.flags), self.person.name)

    def to_text(self) -> str:
        """Render the profile for a terminal."""
        name = self.person.name
        messages = len(self.combined) if self.combined is not None else 0
        out = [
            f"{name}: {_plural(len(self.entries), 'conversation')}, {_plural(messages, 'message')}"
        ]
        escalation = self.escalation
        if escalation is not None:
            # First, as in a single check: a danger flag from any one conversation is
            # what the owner most needs to see, however mild the rest of the profile is.
            out += ["", escalation.to_text()]

        if self.combined is None:
            out += ["", f"Nothing saved yet. Add a transcript with: confidant add {name} FILE"]
            return "\n".join(out)

        out += ["", "ACROSS ALL CONVERSATIONS", *_statistics(self.combined)]
        out += ["", "CONVERSATIONS"]
        for entry in self.entries:
            out += _entry_lines(entry, name)
        out += self._personality_lists()
        out += ["", "RED FLAGS", *self._flag_lines()]

        pending = self.pending
        if pending:
            count = len({stored.id for stored, _ in pending})
            out += [
                "",
                f"{_plural(len(pending), 'read')} missing or out of date. To bring this up to "
                f"date: confidant profile {name} --update",
                f"(sends {_plural(count, 'redacted conversation')} to the Anthropic API)",
            ]
        return "\n".join(out)

    def _personality_lists(self) -> list[str]:
        out: list[str] = []
        for title, pick in (
            ("GREEN FLAGS", lambda r: r.green_flags),
            ("WORTH WATCHING", lambda r: r.worth_watching),
            ("WORTH ASKING", lambda r: r.questions_to_ask),
        ):
            items = [
                f"  * {item}  [#{entry.stored.id}]"
                for entry in self.entries
                if entry.personality is not None
                for item in pick(entry.personality.report)
            ]
            if items:
                out += ["", title, *items]
        return out

    def _flag_lines(self) -> list[str]:
        checked = [entry for entry in self.entries if entry.flags is not None]
        if not checked:
            return ["  No conversation has been checked yet."]

        out: list[str] = []
        flags = self.flags
        if not flags:
            out.append(
                f"  Nothing in {_plural(len(checked), 'checked conversation')} rises to a red flag."
            )
        for tagged in flags:
            flag = tagged.flag
            out.append(
                f"  [{flag.severity.upper()}] {flag.category.replace('_', ' ')}  "
                f"[#{tagged.conversation_id}]"
            )
            out.append(f"    {flag.behavior}")
            for item in flag.evidence:
                out.append(f'      "{item.quote}"')
                out.append(f"        -> {item.why_it_matters}")
            if flag.innocent_reading and flag.severity != "danger":
                out.append(f"    Could also be: {flag.innocent_reading}")

        discarded = sum(entry.flags.report.discarded for entry in checked)
        discarded_danger = sum(entry.flags.report.discarded_danger for entry in checked)
        if discarded:
            out.append(
                f"  ({_plural(discarded, 'flag')} left out: the quotes behind them could not "
                "be found in the transcript.)"
            )
        if discarded_danger:
            # The same reasoning as in a single check: a dropped danger flag must not
            # leave the profile sounding more reassuring than it has earned.
            out.append(
                f"  ({_plural(discarded_danger, 'of them was', 'of them were')} filed as "
                "serious. If something in these conversations felt unsafe to you, that "
                "counts for more than this check does.)"
            )
        unchecked = [
            f"#{entry.stored.id}"
            for entry in self.entries
            if entry.flags is None and entry.match_messages
        ]
        if unchecked:
            out.append(f"  (Not checked yet: {', '.join(unchecked)}.)")
        return out


def _plural(count: int, word: str, many: str | None = None) -> str:
    if count == 1:
        return f"{count} {word}"
    return f"{count} {many}" if many is not None else f"{count} {word}s"


def _when(entry: CurrentRead) -> str:
    text = f"{entry.reading.read_at:%Y-%m-%d}"
    if entry.stale:
        text += f", at {entry.reading.messages_read} of {entry.messages_now} messages"
    return text


def _statistics(conversation: Conversation) -> list[str]:
    lines = [
        f"  messages         {len(conversation)} "
        f"({len(conversation.owner_messages)} you / {len(conversation.match_messages)} them)",
        f"  avg words (you)  {conversation.mean_words(Role.OWNER):.1f}",
        f"  avg words (them) {conversation.mean_words(Role.MATCH):.1f}",
    ]
    ratio = conversation.effort_ratio
    if ratio is not None:
        lines.append(f"  effort ratio     {ratio:.2f} (their words per word of yours)")
    stamps = sorted(m.timestamp for m in conversation if m.timestamp is not None)
    if stamps:
        lines.append(f"  first message    {stamps[0]:%Y-%m-%d %H:%M}")
        lines.append(f"  last message     {stamps[-1]:%Y-%m-%d %H:%M}")
    return lines


def _entry_lines(entry: ConversationEntry, name: str) -> list[str]:
    stored = entry.stored
    out = [
        f"  #{stored.id} {stored.source or '(no source)'}, {_plural(stored.messages, 'message')}"
    ]
    if entry.match_messages == 0:
        out.append(f"     Nothing from {name} to read.")
        return out

    personality = entry.personality
    if personality is None:
        out.append("     Not read yet.")
    else:
        report = personality.report
        out.append(f"     {report.headline}")
        for trait in report.traits:
            out.append(f"     * {trait.name}  [{trait.confidence} confidence]")
        out.append(f"     read {_when(personality)}, {report.confidence} confidence")
    if entry.flags is not None:
        report = entry.flags.report
        found = _plural(len(report.flags), "flag")
        out.append(f"     checked {_when(entry.flags)}, {found}, {report.confidence} confidence")
    return out


def parse_reading(reading: Reading) -> PersonalityReport | FlagReport | None:
    """The report a stored read holds, or ``None`` if it is not one this version reads."""
    try:
        return _SCHEMAS[reading.kind].model_validate_json(reading.report)
    except ValidationError:
        # Written by a Confidant whose report had a different shape. Treated as missing,
        # so --update reads it again, rather than as an error that hides the whole profile.
        return None


def _current(store: Store, stored: StoredConversation, kind: ReadingKind) -> CurrentRead | None:
    reading = store.latest_reading(stored.id, kind)
    report = parse_reading(reading) if reading is not None else None
    if report is None:
        return None
    return CurrentRead(reading=reading, report=report, messages_now=stored.messages)


def build_profile(store: Store, person: str | Person) -> Profile:
    """Put together everything saved for ``person``. Local only."""
    if isinstance(person, str):
        person = store.person(person)
    profile = Profile(person=person)
    messages = []
    owner_name = None
    for stored in store.conversations(person):
        conversation = store.load_conversation(stored.id)
        owner_name = owner_name or conversation.owner_name
        messages += conversation.messages
        profile.entries.append(
            ConversationEntry(
                stored=stored,
                match_messages=len(conversation.match_messages),
                personality=_current(store, stored, ReadingKind.PERSONALITY),
                flags=_current(store, stored, ReadingKind.FLAGS),
            )
        )
    if messages:
        profile.combined = Conversation(
            match_name=person.name, owner_name=owner_name, messages=messages
        )
    return profile


def read_conversation(
    store: Store,
    stored: StoredConversation,
    kind: ReadingKind,
    *,
    settings: Settings,
    client: anthropic.Anthropic | None = None,
    on_progress: Callable[[Progress], None] | None = None,
) -> Reading:
    """Run one analysis over a stored conversation and keep the result.

    The conversation is analyzed exactly as ``confidant analyze`` or ``confidant flags``
    would analyze the transcript it came from, redaction and grounding included.
    """
    conversation = store.load_conversation(stored.id)
    analyze = analyze_personality if kind is ReadingKind.PERSONALITY else analyze_flags
    report = analyze(conversation, settings=settings, client=client, on_progress=on_progress)
    return store.save_reading(
        stored.id,
        kind,
        report.model_dump_json(),
        messages_read=len(conversation),
        model=settings.model,
    )
