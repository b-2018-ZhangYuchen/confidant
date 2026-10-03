"""A person's timeline: how the back-and-forth changed, week by week.

A profile says what every saved conversation adds up to. It cannot say that the replies
used to take minutes and now take a day, because a total has no "used to". The timeline
splits every timestamped message into weeks (or days) and shows, for each side:

* how many messages they sent, and how long they were;
* how often they started things up, meaning wrote first after eight hours or more of
  quiet, or opened a conversation;
* how long they usually took to answer, as a median, so one slow reply in a busy week
  does not stand for the whole week.

Every saved read sits on the timeline at the last message it read, so it is visible
which messages a read was made from and how reads of the same chat changed as it went
on. Superseded reads are shown too: they are a record, not a mistake.

These are counts, and counts say what changed, not why. The rendered timeline says so in
fixed text under the comparison, because a reply time going from minutes to hours is
exactly the kind of number that invites the dramatic reading (``docs/principles.md``,
principle 5), and the ordinary one — a busy week, a trip, a move to calls — cannot be
seen from here.

Building a timeline never touches the network.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import StrEnum

from confidant.analysis.flags import TIERS, FlagReport
from confidant.analysis.personality import PersonalityReport
from confidant.models import Conversation, Message, Role
from confidant.profile import build_profile, parse_reading
from confidant.safety import Escalation
from confidant.store import Person, ReadingKind, Store

__all__ = [
    "QUIET_GAP",
    "Bucket",
    "Mark",
    "Period",
    "Timeline",
    "build_timeline",
]

# Long enough that an evening's exchange is one stretch, short enough that picking it up
# the next morning counts as starting again. Overnight is the boundary most chats have.
QUIET_GAP = timedelta(hours=8)


class Period(StrEnum):
    WEEK = "week"
    DAY = "day"

    def start_of(self, moment: datetime) -> date:
        day = moment.date()
        if self is Period.DAY:
            return day
        # Weeks start on Monday, as ISO weeks do.
        return day - timedelta(days=day.weekday())

    @property
    def step(self) -> timedelta:
        return timedelta(days=7 if self is Period.WEEK else 1)


@dataclass(frozen=True, slots=True)
class Mark:
    """A saved read, placed at the last message it read."""

    conversation_id: int
    kind: ReadingKind
    messages_read: int
    at: datetime
    report: PersonalityReport | FlagReport

    def to_text(self) -> str:
        where = f"#{self.conversation_id} {_verb(self.kind)} at {self.messages_read} messages"
        report = self.report
        if isinstance(report, PersonalityReport):
            return f"{where}: {report.headline} ({report.confidence} confidence)"
        if not report.flags:
            return f"{where}: no red flags"
        worst = max(report.flags, key=lambda f: TIERS[f.severity]).severity
        return f"{where}: {_plural(len(report.flags), 'flag')}, the most serious {worst}"


@dataclass(slots=True)
class _Side:
    messages: int = 0
    words: int = 0
    starts: int = 0
    replies: list[timedelta] = field(default_factory=list)

    @property
    def mean_words(self) -> float | None:
        return self.words / self.messages if self.messages else None

    @property
    def median_reply(self) -> timedelta | None:
        return statistics.median(self.replies) if self.replies else None


@dataclass(slots=True)
class Bucket:
    """One week (or day) of messages, across every conversation with the person."""

    start: date
    owner: _Side = field(default_factory=_Side)
    match: _Side = field(default_factory=_Side)
    marks: list[Mark] = field(default_factory=list)

    def side(self, role: Role) -> _Side:
        return self.owner if role is Role.OWNER else self.match

    @property
    def messages(self) -> int:
        return self.owner.messages + self.match.messages


@dataclass(slots=True)
class Timeline:
    person: Person
    period: Period
    buckets: list[Bucket] = field(default_factory=list)
    """Only the periods that had messages, oldest first."""

    conversations: int = 0
    undated: int = 0
    """Saved messages with no timestamp, which cannot be placed."""

    unplaced_reads: int = 0
    """Reads whose messages have no timestamps, so they have nowhere to go."""

    escalation: Escalation | None = None

    def to_text(self) -> str:
        """Render the timeline for a terminal."""
        name = self.person.name
        dated = sum(bucket.messages for bucket in self.buckets)
        if self.buckets:
            span = (self.buckets[-1].start - self.buckets[0].start) // self.period.step + 1
            out = [f"{name}: {_plural(dated, 'message')} over {_plural(span, self.period)}"]
        else:
            out = [f"{name}: {_plural(self.conversations, 'conversation')}"]
        if self.escalation is not None:
            # As in the profile: whatever the counts look like, this is read first.
            out += ["", self.escalation.to_text()]

        if not self.conversations:
            out += ["", f"Nothing saved yet. Add a transcript with: confidant add {name} FILE"]
            return "\n".join(out)
        if not self.buckets:
            out += [
                "",
                f"None of the {_plural(self.undated, 'saved message')} has a timestamp, so "
                "there is nothing to put on a timeline.",
            ]
            return "\n".join(out)

        out += ["", *self._table()]
        notes = []
        if self.undated:
            notes.append(f"({_plural(self.undated, 'message')} without a timestamp left out.)")
        if self.unplaced_reads:
            notes.append(
                f"({_plural(self.unplaced_reads, 'read')} left out: the messages "
                "read have no timestamps.)"
            )
        if notes:
            out += ["", *notes]
        out += self._change()
        return "\n".join(out)

    def _table(self) -> list[str]:
        label = "WEEK OF" if self.period is Period.WEEK else "DAY"
        out = [
            f"{'':<10}  {'MESSAGES':^9}  {'AVG WORDS':^11}  {'STARTED':^9}  {'MEDIAN REPLY':^11}",
            f"{label:<10}  {'you':>4} {'them':>4}  {'you':>5} {'them':>5}  "
            f"{'you':>4} {'them':>4}  {'you':>5} {'them':>5}",
        ]
        previous: date | None = None
        for bucket in self.buckets:
            if previous is not None:
                quiet = (bucket.start - previous) // self.period.step - 1
                if quiet:
                    out.append(f"{'':<10}  (no messages for {_plural(quiet, self.period)})")
            previous = bucket.start
            o, m = bucket.owner, bucket.match
            out.append(
                f"{bucket.start:%Y-%m-%d}  {o.messages:>4} {m.messages:>4}  "
                f"{_words(o.mean_words):>5} {_words(m.mean_words):>5}  "
                f"{o.starts:>4} {m.starts:>4}  "
                f"{_duration(o.median_reply):>5} {_duration(m.median_reply):>5}"
            )
            out += [f"{'':<10}  ~ {mark.to_text()}" for mark in bucket.marks]
        return out

    def _change(self) -> list[str]:
        if len(self.buckets) < 2:
            return []
        first, last = self.buckets[0], self.buckets[-1]
        rows = [
            (
                "messages (you / them)",
                f"{first.owner.messages} / {first.match.messages}",
                f"{last.owner.messages} / {last.match.messages}",
            ),
            ("started by them", _share(first), _share(last)),
        ]
        for title, role in (("their median reply", Role.MATCH), ("your median reply", Role.OWNER)):
            before, after = first.side(role).median_reply, last.side(role).median_reply
            if before is not None and after is not None:
                rows.append((title, _duration(before), _duration(after)))
        width = max(len(row[1]) for row in rows)
        return [
            "",
            f"FIRST {self.period.upper()} TO LATEST",
            *(f"  {title:<24}{before:>{width}}  ->  {after}" for title, before, after in rows),
            "",
            # Fixed text, not the model's: it is the boring explanation, said every time.
            "These count what changed, not why. A busy week, a trip, or moving to calls or",
            f"another app look the same from here, and one {self.period} is a small sample.",
        ]


def _verb(kind: ReadingKind) -> str:
    return "read" if kind is ReadingKind.PERSONALITY else "checked"


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _words(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "-"


def _share(bucket: Bucket) -> str:
    total = bucket.owner.starts + bucket.match.starts
    return f"{bucket.match.starts} of {total}" if total else "-"


def _duration(value: timedelta | None) -> str:
    """Short and rounded: a median reply time is a rough figure, and should look like one."""
    if value is None:
        return "-"
    minutes = value.total_seconds() / 60
    if minutes < 1:
        return "<1m"
    if minutes < 60:
        return f"{int(minutes + 0.5)}m"
    hours = minutes / 60
    if hours < 48:
        return f"{int(hours + 0.5)}h"
    return f"{int(hours / 24 + 0.5)}d"


def _events(conversation: Conversation, bucket_for) -> None:
    """Count one conversation's messages, starts, and replies into their buckets.

    Starts and replies are worked out per conversation, never across two: the last
    message in one chat is not something the first message of another is answering.
    """
    previous: Message | None = None
    for message in conversation:
        if message.timestamp is None:
            # An undated message still breaks the chain: what comes after it is not
            # known to be answering what came before.
            previous = None
            continue
        side = bucket_for(message.timestamp).side(message.role)
        side.messages += 1
        side.words += message.word_count
        gap = message.timestamp - previous.timestamp if previous is not None else None
        if gap is None or gap >= QUIET_GAP:
            side.starts += 1
        if previous is not None and previous.role is not message.role:
            # Clocks in exports can disagree by a minute; never report a negative delay.
            side.replies.append(max(gap, timedelta(0)))
        previous = message


def build_timeline(store: Store, person: str | Person, period: Period = Period.WEEK) -> Timeline:
    """Lay out everything saved for ``person`` by ``period``. Local only."""
    if isinstance(person, str):
        person = store.person(person)
    timeline = Timeline(person=person, period=Period(period))
    buckets: dict[date, Bucket] = {}

    def bucket_for(moment: datetime) -> Bucket:
        start = timeline.period.start_of(moment)
        if start not in buckets:
            buckets[start] = Bucket(start=start)
        return buckets[start]

    for stored in store.conversations(person):
        timeline.conversations += 1
        conversation = store.load_conversation(stored.id)
        timeline.undated += sum(1 for m in conversation if m.timestamp is None)
        _events(conversation, bucket_for)

        for reading in store.readings(stored.id):
            report = parse_reading(reading)
            if report is None:
                continue
            read = conversation.messages[: reading.messages_read]
            at = max((m.timestamp for m in read if m.timestamp is not None), default=None)
            if at is None:
                timeline.unplaced_reads += 1
                continue
            mark = Mark(stored.id, reading.kind, reading.messages_read, at, report)
            bucket_for(at).marks.append(mark)

    for bucket in buckets.values():
        bucket.marks.sort(key=lambda mark: (mark.at, mark.conversation_id))
    timeline.buckets = sorted(buckets.values(), key=lambda bucket: bucket.start)
    # The same notice the profile leads with, from the same reads, so the two views can
    # never disagree about whether something was serious.
    timeline.escalation = build_profile(store, person).escalation
    return timeline
