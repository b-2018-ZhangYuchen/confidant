"""Keep-in-touch: where a thread stands today, and whether it is worth a message.

The timeline says how the back-and-forth changed. This says where it has got to, for one
person or for everyone saved, in the terms someone deciding whether to send a message
actually thinks in:

* **Whose turn it is.** The last stretch of messages — everything since the last quiet
  gap of eight hours or more — is one of three things. If only they wrote in it, they
  reached out and are waiting on an answer. If only you did, you are the one waiting,
  and another message would be the second in a row. If both of you did, the exchange
  ended the way exchanges do, and the rest of the model decides.
* **How long it has been, against your usual rhythm.** The rhythm is the median quiet
  gap between stretches in your conversations, so a pair that talks every few days is
  not "overdue" after two, and a pair that talks every evening is after three. With too
  few gaps to measure, it assumes three days, and says so.
* **Warmth, which decays.** Every two weeks of quiet halves it. Past four weeks — two
  half-lives — the thread is treated as gone quiet: reaching out then is a choice the
  owner makes, not something Confidant nudges them towards.
* **Reciprocity, which weights all of it.** Over the four weeks up to the last message,
  how much of the reaching out — stretches started, and messages sent — came from them.
  An even share counts in full, and so does more than even. Less than even counts for
  less, because a thread the owner is carrying alone is one to be honest with them
  about (``docs/principles.md``, principle 6), not one to push them further into. One
  pretend stretch and message is added to each side first, so three messages cannot
  swing it either way.

These combine into a ``priority`` between 0 and 1, which exists only to put threads in
order. It is a property of the conversation's timing, not a score of the person, and it
is not meant to be printed: the reasons are what the owner reads.

Two things override the arithmetic. A danger-tier red flag in any saved conversation
means Confidant never suggests reaching out, whatever the timing says, and carries the
same fixed safety notice the profile leads with. And a thread where the owner is waiting
on an answer is never suggested either: a slow reply is usually a busy week (principle 5),
and the decision to follow up belongs to the person who knows them.

Looking back from an earlier moment (``until``) leaves out every message sent after it,
so the answer is the one Confidant would have given then. Without ``until``, nothing is
left out: an export whose clock runs ahead of this machine's should still count.

Nothing here touches the network.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from enum import StrEnum

from confidant.models import Conversation, Message, Role
from confidant.profile import build_profile
from confidant.safety import Escalation
from confidant.store import Person, Store
from confidant.timeline import QUIET_GAP

__all__ = [
    "DEFAULT_RHYTHM",
    "FADED_WARMTH",
    "HALF_LIFE",
    "RECENT",
    "Contact",
    "ContactState",
    "Reciprocity",
    "assess",
    "assess_everyone",
    "assess_person",
]

HALF_LIFE = timedelta(days=14)
# Two half-lives. Four weeks of nothing is, early in dating, usually an answer.
FADED_WARMTH = 0.25
DEFAULT_RHYTHM = timedelta(days=3)
# A median of a few hours would call a pair overdue by breakfast; one of a month would
# never call anyone overdue before the thread had faded anyway.
MIN_RHYTHM = timedelta(days=1)
MAX_RHYTHM = timedelta(days=14)
MIN_GAPS = 2
"""Fewer quiet gaps than this, and the rhythm is assumed rather than measured."""

RECENT = timedelta(days=28)
"""How far back from the last message reciprocity looks."""


class ContactState(StrEnum):
    NOTHING_SAVED = "nothing-saved"
    NOT_YET = "not-yet"
    """Looking back from ``until``, and every dated message is from after it."""

    NO_DATES = "no-dates"
    """Messages are saved, but none has a timestamp, so there is no "how long"."""

    SAFETY = "safety"
    """A danger-tier flag was found. Never suggested, whatever the timing."""

    YOUR_TURN = "your-turn"
    """They reached out and have not had an answer."""

    WAITING = "waiting"
    """The owner reached out and has not had an answer."""

    IN_RHYTHM = "in-rhythm"
    """Quiet for less than the usual gap between stretches."""

    DUE = "due"
    """Quiet for longer than usual, and the thread is still warm."""

    FADED = "faded"
    """Quiet for long enough that the thread has gone cold."""


@dataclass(frozen=True, slots=True)
class Reciprocity:
    """Who did the reaching out, in the four weeks up to the last message."""

    owner_starts: int
    match_starts: int
    owner_messages: int
    match_messages: int

    @staticmethod
    def _credit(theirs: int, yours: int) -> float:
        # One pretend item each side, so a handful of messages reads as "not enough to
        # say" rather than as all or nothing.
        share = (theirs + 1) / (theirs + yours + 2)
        return min(1.0, 2 * share)

    @property
    def weight(self) -> float:
        """1.0 for an even share or better from them, falling towards 0 as it shrinks."""
        starts = self._credit(self.match_starts, self.owner_starts)
        messages = self._credit(self.match_messages, self.owner_messages)
        return (starts + messages) / 2

    @property
    def owner_carrying(self) -> bool:
        """Whether the owner is doing clearly more of the reaching out."""
        return self.weight < 0.6 and self.owner_starts > self.match_starts


@dataclass(frozen=True, slots=True)
class Contact:
    """Where the thread with one person stands at ``now``."""

    name: str
    state: ContactState
    now: datetime
    last_message: Message | None = None
    rhythm: timedelta = DEFAULT_RHYTHM
    rhythm_measured: bool = False
    reciprocity: Reciprocity | None = None
    escalation: Escalation | None = None
    undated: int = 0
    reasons: list[str] = field(default_factory=list)

    @property
    def silence(self) -> timedelta | None:
        if self.last_message is None or self.last_message.timestamp is None:
            return None
        # A clock a few minutes ahead of the export should read as "just now", not as
        # negative time.
        return max(self.now - self.last_message.timestamp, timedelta(0))

    @property
    def warmth(self) -> float:
        """1.0 at the moment of the last message, halving every :data:`HALF_LIFE`."""
        silence = self.silence
        if silence is None:
            return 0.0
        return 0.5 ** (silence / HALF_LIFE)

    @property
    def priority(self) -> float:
        """For ordering threads only: higher is more worth a message today."""
        if self.state is ContactState.YOUR_TURN:
            readiness = 1.0
        elif self.state in (ContactState.DUE, ContactState.FADED):
            # Half-ready at exactly the usual gap, fully ready at twice it.
            readiness = min(1.0, (self.silence / self.rhythm) / 2)
        else:
            return 0.0
        weight = self.reciprocity.weight if self.reciprocity is not None else 0.5
        return self.warmth * weight * readiness

    @property
    def suggested(self) -> bool:
        return self.state in (ContactState.YOUR_TURN, ContactState.DUE)

    def headline(self) -> str:
        name, ago = self.name, _ago(self.silence)
        if self.state is ContactState.NOTHING_SAVED:
            return f"{name}: nothing saved yet."
        if self.state is ContactState.NOT_YET:
            return f"{name}: nothing saved from before then."
        if self.state is ContactState.NO_DATES:
            return f"{name}: no saved message has a timestamp, so there is no telling how long."
        if self.state is ContactState.SAFETY:
            return f"{name}: not suggested. A danger-tier red flag was found with them."
        if self.state is ContactState.YOUR_TURN:
            return f"{name}: your turn. They wrote last, {ago}, and have not had an answer."
        if self.state is ContactState.WAITING:
            return f"{name}: waiting on them. You wrote last, {ago}, and have not heard back."
        if self.state is ContactState.IN_RHYTHM:
            return f"{name}: nothing due. You last talked {ago}."
        if self.state is ContactState.DUE:
            return f"{name}: worth a message. You last talked {ago}."
        return f"{name}: gone quiet. You last talked {ago}."

    def to_text(self) -> str:
        out = [self.headline()]
        if self.escalation is not None:
            out += ["", self.escalation.to_text()]
        out += [f"  - {reason}" for reason in self.reasons]
        return "\n".join(out)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _span(value: timedelta) -> str:
    """Rounded, in words: these are rough figures and should read like them."""
    hours = value.total_seconds() / 3600
    if hours < 1:
        return _plural(max(1, round(hours * 60)), "minute")
    if hours < 36:
        return _plural(round(hours), "hour")
    days = hours / 24
    if days < 14:
        return _plural(round(days), "day")
    return _plural(round(days / 7), "week")


def _ago(value: timedelta | None) -> str:
    if value is None:
        return "at an unknown time"
    if value < timedelta(minutes=1):
        return "just now"
    return f"{_span(value)} ago"


def _every(rhythm: timedelta) -> str:
    days = round(rhythm / timedelta(days=1))
    return "every day" if days <= 1 else f"every {days} days"


@dataclass(slots=True)
class _Gathered:
    last: Message | None = None
    final_roles: set[Role] = field(default_factory=set)
    gaps: list[timedelta] = field(default_factory=list)
    starts: list[Message] = field(default_factory=list)
    dated: list[Message] = field(default_factory=list)
    undated: int = 0
    later: int = 0


def _gather(conversations: Iterable[Conversation], until: datetime | None = None) -> _Gathered:
    """Walk every conversation once, as the timeline does: within a chat, never across.

    The last message of one chat is not something the first message of another is
    answering, so gaps and starts are measured inside each conversation, and an undated
    message breaks the chain just as it does on the timeline.
    """
    found = _Gathered()
    for conversation in conversations:
        previous: Message | None = None
        stretch: list[Message] = []
        for message in conversation:
            if message.timestamp is None:
                # It breaks the chain, so the next dated message starts a stretch, but the
                # stretch before it is kept: it is still the last contact that can be dated.
                found.undated += 1
                previous = None
                continue
            if until is not None and message.timestamp > until:
                # Messages are in order within a conversation, so the rest are later too.
                found.later += 1
                continue
            found.dated.append(message)
            gap = message.timestamp - previous.timestamp if previous is not None else None
            if gap is None or gap >= QUIET_GAP:
                found.starts.append(message)
                stretch = []
                if gap is not None:
                    found.gaps.append(gap)
            stretch.append(message)
            previous = message
        if stretch and (found.last is None or stretch[-1].timestamp >= found.last.timestamp):
            found.last = stretch[-1]
            found.final_roles = {m.role for m in stretch}
    return found


def _rhythm(gaps: list[timedelta]) -> tuple[timedelta, bool]:
    if len(gaps) < MIN_GAPS:
        return DEFAULT_RHYTHM, False
    return min(max(statistics.median(gaps), MIN_RHYTHM), MAX_RHYTHM), True


def _reciprocity(found: _Gathered) -> Reciprocity:
    since = found.last.timestamp - RECENT

    def count(messages: list[Message], role: Role) -> int:
        return sum(1 for m in messages if m.role is role and m.timestamp >= since)

    return Reciprocity(
        owner_starts=count(found.starts, Role.OWNER),
        match_starts=count(found.starts, Role.MATCH),
        owner_messages=count(found.dated, Role.OWNER),
        match_messages=count(found.dated, Role.MATCH),
    )


def assess(
    name: str,
    conversations: Iterable[Conversation],
    *,
    now: datetime,
    escalation: Escalation | None = None,
    until: datetime | None = None,
) -> Contact:
    """Where the thread with ``name`` stands at ``now``, from these conversations.

    With ``until``, messages sent after it are left out. A danger flag is not: what a
    red-flag check found is known now, whenever the message it found it in was sent.
    """
    conversations = list(conversations)
    if not conversations:
        return Contact(name, ContactState.NOTHING_SAVED, now, escalation=escalation)
    found = _gather(conversations, until)
    if escalation is not None:
        # Checked before the timing on purpose: no rhythm or reciprocity makes reaching
        # out to someone who threatened or tracked the owner a suggestion worth making.
        return Contact(
            name,
            ContactState.SAFETY,
            now,
            last_message=found.last,
            escalation=escalation,
            undated=found.undated,
            reasons=[
                "Confidant does not suggest getting in touch with someone a red-flag check "
                "found danger-tier behavior from. Whether to is yours to decide, ideally "
                "with someone you trust."
            ],
        )
    if found.last is None:
        state = ContactState.NOT_YET if found.later else ContactState.NO_DATES
        return Contact(name, state, now, undated=found.undated)

    rhythm, measured = _rhythm(found.gaps)
    contact = Contact(
        name,
        ContactState.IN_RHYTHM,
        now,
        last_message=found.last,
        rhythm=rhythm,
        rhythm_measured=measured,
        reciprocity=_reciprocity(found),
        undated=found.undated,
    )
    if found.final_roles == {Role.MATCH}:
        state = ContactState.YOUR_TURN
    elif found.final_roles == {Role.OWNER}:
        state = ContactState.WAITING
    elif contact.warmth < FADED_WARMTH:
        state = ContactState.FADED
    elif contact.silence >= rhythm:
        state = ContactState.DUE
    else:
        state = ContactState.IN_RHYTHM
    contact = replace(contact, state=state)
    return replace(contact, reasons=_reasons(contact))


def _reasons(contact: Contact) -> list[str]:
    name, r = contact.name, contact.reciprocity
    reasons = []
    if contact.rhythm_measured:
        reasons.append(f"The two of you usually pick things up {_every(contact.rhythm)}.")
    else:
        reasons.append(
            f"Too few quiet stretches to tell your usual rhythm, so this assumes "
            f"{_every(contact.rhythm)}."
        )
    reasons.append(
        f"In the four weeks to the last message, {name} started "
        f"{r.match_starts} of {r.match_starts + r.owner_starts} stretches and sent "
        f"{r.match_messages} of {r.match_messages + r.owner_messages} messages."
    )
    if r.owner_carrying:
        reasons.append(
            "You have been doing most of the reaching out. That is worth noticing before "
            "doing more of it."
        )
    if contact.state is ContactState.WAITING:
        reasons.append(
            "Another message now would be your second in a row. A slow reply is usually a "
            "busy week; whether to follow up depends on how the two of you text."
        )
    elif contact.state is ContactState.FADED:
        reasons.append(
            "After this long, getting back in touch is a choice to make, not a reply that "
            "is owed either way."
        )
    if contact.undated:
        reasons.append(f"({_plural(contact.undated, 'message')} without a timestamp left out.)")
    return reasons


def assess_person(
    store: Store,
    person: str | Person,
    *,
    now: datetime | None = None,
    until: datetime | None = None,
) -> Contact:
    """Assess one saved person. Local only."""
    if isinstance(person, str):
        person = store.person(person)
    conversations = [store.load_conversation(s.id) for s in store.conversations(person)]
    # The same notice the profile and timeline lead with, from the same reads.
    escalation = build_profile(store, person).escalation
    return assess(
        person.name,
        conversations,
        now=now or datetime.now(),
        escalation=escalation,
        until=until,
    )


def assess_everyone(
    store: Store, *, now: datetime | None = None, until: datetime | None = None
) -> list[Contact]:
    """Everyone saved, most worth a message first. Local only."""
    now = now or datetime.now()
    contacts = [
        assess_person(store, summary.person, now=now, until=until) for summary in store.people()
    ]
    # Stable sort: equal priorities stay in the store's alphabetical order.
    return sorted(contacts, key=lambda contact: -contact.priority)
