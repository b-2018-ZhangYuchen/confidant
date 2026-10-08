"""Comfort mode: what Confidant says when it has gone badly.

The owner arrives here at a low moment, primed to read every message as a verdict on
them. So most of what this module adds is about holding the model to what the thread
can show:

* **Praise is grounded too.** Each thing the owner did well quotes one of the owner's
  own messages, and is dropped if the quote is not there. Comfort built on an invented
  compliment falls apart the moment the owner rereads the chat.
* **The silence is counted, not inferred.** How many of the owner's messages at the end
  of the thread have had no reply is worked out here and printed in fixed words, so the
  model never has to guess at it and the report never exaggerates it.
* **The owner's own safety comes first.** If the model thinks the owner may be at risk,
  or what they typed uses one of the plain phrasings of wanting to hurt themselves, a
  fixed notice from :mod:`confidant.safety` opens the report.

Nothing here is a plan to win anyone back, and the prompt says so.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

import anthropic
from pydantic import BaseModel, Field

from confidant.analysis.flags import quote_appears_in
from confidant.client import structured_call
from confidant.config import Settings
from confidant.models import Conversation, Role
from confidant.progress import Progress
from confidant.prompts.comfort import COMFORT_SYSTEM, build_comfort_request
from confidant.redaction import redact
from confidant.safety import Escalation, mentions_self_harm, support_notice
from confidant.usage import Usage

__all__ = [
    "FOOTER",
    "ComfortReport",
    "ComfortScan",
    "Moment",
    "comfort",
    "ground_comfort",
    "silence_note",
    "unanswered",
]

FOOTER = (
    "This is read from the messages alone. Someone who knows you will be better company "
    "for the rest of it."
)


class Moment(BaseModel):
    """Something the owner did well, tied to their own words."""

    point: str = Field(description="What the owner did, specifically, in one short sentence.")
    quote: str = Field(description="A short verbatim quote from one of the OWNER's messages.")


class ComfortScan(BaseModel):
    """What the model is asked to return."""

    what_happened: str = Field(
        description="One or two plain sentences on what the messages show, as behavior."
    )
    what_it_says: str = Field(
        description=(
            "What the thread can and cannot say about why, boring explanations first. "
            "No verdict on anyone's character or worth."
        )
    )
    you_did_well: list[Moment] = Field(
        description="Up to three specific things the owner did well. Empty if none."
    )
    worth_knowing: str | None = Field(
        description="One kind, grounded observation worth carrying forward, or null."
    )
    next_steps: list[str] = Field(
        description="Two to four small, practical things for the next day or two."
    )
    owner_at_risk: bool = Field(
        description="True if anything the owner wrote suggests they might hurt themselves."
    )


class ComfortReport(BaseModel):
    """The comfort read after local checks, with any fixed notices that go above it."""

    what_happened: str
    what_it_says: str
    you_did_well: list[Moment]
    worth_knowing: str | None = None
    next_steps: list[str]
    unanswered: int = Field(
        default=0,
        description="How many of the owner's messages at the end of the thread have no reply.",
    )
    match_name: str = ""
    match_last_wrote: datetime | None = Field(
        default=None, description="When the match last wrote, if that message is dated."
    )
    discarded: int = Field(
        default=0,
        description="Things the owner did well that were dropped because the quote was not theirs.",
    )
    support: Escalation | None = Field(
        default=None,
        description="The fixed notice for when the owner may be at risk themselves.",
    )
    escalation: Escalation | None = Field(
        default=None,
        description="The safety notice, when a saved red-flag check found danger-tier behavior.",
    )

    def to_text(self) -> str:
        """Render the comfort read for a terminal."""
        out: list[str] = []
        # The owner's own safety, then theirs from the match, then everything else.
        for notice in (self.support, self.escalation):
            if notice is not None:
                out += [notice.to_text(), ""]

        out += [self.what_happened, ""]
        note = silence_note(self.unanswered, self.match_name, self.match_last_wrote)
        if note:
            out += [note, ""]
        out += [self.what_it_says, ""]

        if self.you_did_well:
            out.append("WHAT YOU DID WELL")
            for moment in self.you_did_well:
                out.append(f"  * {moment.point}")
                out.append(f'      "{moment.quote}"')
            out.append("")
        if self.worth_knowing:
            out += [f"Worth knowing: {self.worth_knowing}", ""]
        if self.next_steps:
            out.append("FOR THE NEXT DAY OR TWO")
            out += [f"  - {step}" for step in self.next_steps]
            out.append("")
        if self.discarded:
            noun = "thing" if self.discarded == 1 else "things"
            out += [
                f"({self.discarded} {noun} left out of what you did well: quoting lines "
                "you did not write.)",
                "",
            ]
        out.append(FOOTER)
        return "\n".join(out)


def unanswered(conversation: Conversation) -> int:
    """How many of the owner's messages at the end of the thread came after the match's last."""
    count = 0
    for message in reversed(conversation.messages):
        if message.role is not Role.OWNER:
            break
        count += 1
    return count


def silence_note(count: int, match_name: str, last_wrote: datetime | None) -> str | None:
    """The fixed sentence about an unanswered stretch, or ``None`` when there is none."""
    if not count:
        return None
    what = "message has" if count == 1 else f"{count} messages have"
    note = f"Your last {what} had no reply."
    if last_wrote is not None:
        note += f" {match_name} last wrote on {last_wrote:%Y-%m-%d}."
    return note


def ground_comfort(scan: ComfortScan, conversation: Conversation) -> ComfortReport:
    """Check a scan against the (redacted) conversation it was written from.

    Only the owner's messages count for ``you_did_well``: praise for something the match
    said would be a misquote, however kind.
    """
    owner_texts = [m.text for m in conversation.owner_messages]
    kept = [m for m in scan.you_did_well if quote_appears_in(m.quote, owner_texts)]
    count = unanswered(conversation)
    last_wrote = next(
        (m.timestamp for m in reversed(conversation.messages) if m.role is Role.MATCH),
        None,
    )
    return ComfortReport(
        what_happened=scan.what_happened,
        what_it_says=scan.what_it_says,
        you_did_well=kept,
        worth_knowing=scan.worth_knowing,
        next_steps=scan.next_steps,
        unanswered=count,
        match_name=conversation.match_name,
        match_last_wrote=last_wrote,
        discarded=len(scan.you_did_well) - len(kept),
        support=support_notice() if scan.owner_at_risk else None,
    )


def comfort(
    conversation: Conversation,
    *,
    what: str | None = None,
    settings: Settings | None = None,
    client: anthropic.Anthropic | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    on_usage: Callable[[Usage], None] | None = None,
) -> ComfortReport:
    """Read ``conversation`` gently, for an owner it has gone badly for.

    ``what`` is the owner's account of what happened, which often took place outside the
    chat. It is redacted with the transcript before anything is sent.
    """
    what = what.strip() if what else None
    redaction = redact(conversation, notes=[what] if what else [])
    scan = structured_call(
        schema=ComfortScan,
        system=COMFORT_SYSTEM,
        user_content=build_comfort_request(
            redaction.conversation,
            what=redaction.notes[0] if what else None,
            unanswered=unanswered(conversation),
        ),
        settings=settings,
        client=client,
        on_progress=on_progress,
        on_usage=on_usage,
    )
    report = redaction.restore(ground_comfort(scan, redaction.conversation))
    # Checked on the owner's own words, unredacted, and whatever the model said: the
    # notice should not depend on one sampled boolean.
    if report.support is None and mentions_self_harm(what):
        report = report.model_copy(update={"support": support_notice()})
    return report
