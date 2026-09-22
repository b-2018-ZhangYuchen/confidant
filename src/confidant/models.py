"""Core data types shared by every part of Confidant.

These are deliberately plain: parsing, analysis, and storage all speak in terms of
:class:`Message` and :class:`Conversation`, so a new input format or a new analyzer
never has to invent its own shape.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum


class Role(StrEnum):
    """Who sent a message, from the point of view of the person using Confidant."""

    OWNER = "owner"
    """You — the person this tool is working for."""

    MATCH = "match"
    """The person you are talking to."""


@dataclass(frozen=True, slots=True)
class Message:
    """A single message in a conversation."""

    role: Role
    text: str
    sender: str
    """Display name exactly as it appeared in the source transcript."""

    timestamp: datetime | None = None

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError("Message.text cannot be empty or whitespace-only")
        if not self.sender.strip():
            raise ValueError("Message.sender cannot be empty or whitespace-only")

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass(slots=True)
class Conversation:
    """An ordered exchange between the owner and one match."""

    match_name: str
    owner_name: str
    messages: list[Message] = field(default_factory=list)
    source: str | None = None
    """Where this came from — a file path, an export name, or ``None`` for in-memory."""

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)

    # -- slicing by speaker -------------------------------------------------

    def by_role(self, role: Role) -> list[Message]:
        return [m for m in self.messages if m.role is role]

    @property
    def match_messages(self) -> list[Message]:
        return self.by_role(Role.MATCH)

    @property
    def owner_messages(self) -> list[Message]:
        return self.by_role(Role.OWNER)

    # -- cheap signals, computed without calling a model --------------------

    @property
    def timespan(self) -> timedelta | None:
        """Time between the first and last timestamped message, if any are dated."""
        stamps = sorted(m.timestamp for m in self.messages if m.timestamp is not None)
        if len(stamps) < 2:
            return None
        return stamps[-1] - stamps[0]

    @property
    def last_activity(self) -> datetime | None:
        stamps = [m.timestamp for m in self.messages if m.timestamp is not None]
        return max(stamps) if stamps else None

    def mean_words(self, role: Role) -> float:
        """Average message length for one side. ``0.0`` when that side said nothing."""
        counts = [m.word_count for m in self.by_role(role)]
        return statistics.fmean(counts) if counts else 0.0

    @property
    def effort_ratio(self) -> float | None:
        """Match words per owner word.

        Roughly ``1.0`` means both sides are putting in comparable effort; well under
        ``1.0`` means you are carrying the conversation. It is a blunt signal — someone
        can be warm and terse — so treat it as a question, not an answer.
        """
        owner = self.mean_words(Role.OWNER)
        if owner == 0:
            return None
        return self.mean_words(Role.MATCH) / owner

    def transcript(self, *, include_timestamps: bool = True) -> str:
        """Render back to the canonical ``[timestamp] Name: text`` text format."""
        lines = []
        for m in self.messages:
            if include_timestamps and m.timestamp is not None:
                lines.append(f"[{m.timestamp:%Y-%m-%d %H:%M}] {m.sender}: {m.text}")
            else:
                lines.append(f"{m.sender}: {m.text}")
        return "\n".join(lines)
