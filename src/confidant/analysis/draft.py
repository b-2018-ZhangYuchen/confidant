"""Draft the owner's next message, in a tone they choose, grounded in the thread so far.

The other analyses describe the match. This one writes words the owner might send, so
its checks are about what could reach the match rather than what the owner reads:

* **Grounding.** Every draft quotes the line it picks up on, and the quote is checked
  against the transcript, from either side. A draft that answers nothing anyone said is
  a generic line wearing the thread's clothes, and it is dropped.
* **Placeholders.** A draft is restored before the owner sees it, but only placeholders
  this request issued can be put back. One the model made up, like ``[PHONE_4]``, would
  be copied into a real message as it stands, so a draft carrying one is dropped too.
* **Whose turn it is.** Whether the owner sent the last message is worked out here, not
  left for the model to notice, and the report says so in fixed words. ``confidant
  nudge`` never suggests writing to someone the owner is waiting on; this command is
  asked for directly, so it drafts a follow-up, but it does not pretend one is owed.

Nothing here sends anything. A draft is printed and the owner does the rest.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import anthropic
from pydantic import BaseModel, Field

from confidant.analysis.flags import quote_appears_in
from confidant.client import structured_call
from confidant.config import Settings
from confidant.models import Conversation, Role
from confidant.progress import Progress
from confidant.prompts.draft import DEFAULT_TONE, DRAFT_SYSTEM, TONES, build_draft_request
from confidant.redaction import Redaction, redact
from confidant.safety import Escalation
from confidant.usage import Usage

__all__ = [
    "DEFAULT_TONE",
    "WAITING_NOTE",
    "DraftReport",
    "DraftScan",
    "Reply",
    "Tone",
    "draft_reply",
    "ground_drafts",
]

# The descriptions the model sees are in prompts.draft.TONES; the tests hold the two
# lists together, since a tone in one and not the other fails only when someone picks it.
Tone = Literal["natural", "warm", "playful", "direct", "brief", "firm"]

WAITING_NOTE = (
    "Your message is the last one in this thread, so nothing here is waiting on you. "
    "These are follow-ups, if you want one; a slow reply is usually a busy week."
)

FOOTER = "Nothing has been sent. These are starting points; what goes out should sound like you."


class Reply(BaseModel):
    """One possible next message from the owner."""

    text: str = Field(
        description=(
            "The message exactly as the owner would send it, in their voice. Curly-brace "
            "blanks like {day} for details only the owner knows."
        )
    )
    picks_up: list[str] = Field(
        description=(
            "One to three short verbatim quotes from the thread that this message answers "
            "or builds on. From either person. Never empty."
        )
    )
    why: str = Field(description="One sentence on why this fits the thread and the tone.")


class DraftScan(BaseModel):
    """What the model is asked to return."""

    where_it_stands: str = Field(
        description=(
            "One or two sentences on where the thread is now: what was said last, and "
            "what, if anything, is waiting for an answer. Behavior, not labels."
        )
    )
    replies: list[Reply] = Field(
        description="Two or three drafts in the requested tone, each doing something different."
    )
    before_sending: str | None = Field(
        description=(
            "Anything the owner should know before sending one of these, grounded in the "
            "thread. Null if there is nothing."
        )
    )


class DraftReport(BaseModel):
    """Drafts after local checks, and what the owner needs to read alongside them."""

    tone: Tone
    where_it_stands: str
    replies: list[Reply]
    before_sending: str | None = None
    waiting: bool = Field(
        default=False, description="Whether the owner sent the last message in the thread."
    )
    discarded: int = Field(
        default=0,
        description=(
            "Drafts dropped because their quotes could not be found or they carried a "
            "placeholder that cannot be put back."
        ),
    )
    escalation: Escalation | None = Field(
        default=None,
        description="The safety notice, when a saved red-flag check found danger-tier behavior.",
    )

    def to_text(self) -> str:
        """Render the drafts for a terminal."""
        out: list[str] = []
        # First, as everywhere else: a draft is not the most important thing on a page
        # that also says the person has threatened you.
        if self.escalation is not None:
            out += [self.escalation.to_text(), ""]
        out += [self.where_it_stands, ""]
        if self.waiting:
            out += [WAITING_NOTE, ""]

        if self.replies:
            out.append(f"DRAFTS ({self.tone})")
            for number, reply in enumerate(self.replies, 1):
                out.append(f"  {number}. {reply.text}")
                for quote in reply.picks_up:
                    out.append(f'       re: "{quote}"')
                out.append(f"       {reply.why}")
                out.append("")
        else:
            out += ["No draft held up against the thread, so none is shown.", ""]

        if self.before_sending:
            out += [f"Before sending: {self.before_sending}", ""]
        if self.discarded:
            noun = "draft" if self.discarded == 1 else "drafts"
            out.append(
                f"({self.discarded} {noun} left out: quoting lines the thread does not have, "
                "or holding details that could not be filled back in.)"
            )
        out.append(FOOTER)
        return "\n".join(out)


def ground_drafts(
    scan: DraftScan,
    redaction: Redaction,
    *,
    tone: Tone,
    waiting: bool,
) -> DraftReport:
    """Check a model scan against the redacted transcript it was written from.

    Quotes that cannot be found are removed, and a draft left with none is dropped. So is
    a draft whose text holds a placeholder the redaction did not issue. Both count in
    ``discarded``. The report is still in placeholders; restore it afterwards.
    """
    texts = [m.text for m in redaction.conversation]
    kept: list[Reply] = []
    discarded = 0
    for reply in scan.replies:
        quotes = [q for q in reply.picks_up if quote_appears_in(q, texts)]
        if not quotes or redaction.unissued(reply.text):
            discarded += 1
            continue
        kept.append(reply.model_copy(update={"picks_up": quotes}))

    return DraftReport(
        tone=tone,
        where_it_stands=scan.where_it_stands,
        replies=kept,
        before_sending=scan.before_sending,
        waiting=waiting,
        discarded=discarded,
    )


def draft_reply(
    conversation: Conversation,
    *,
    tone: Tone = DEFAULT_TONE,
    say: str | None = None,
    settings: Settings | None = None,
    client: anthropic.Anthropic | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    on_usage: Callable[[Usage], None] | None = None,
) -> DraftReport:
    """Draft the owner's next message to the match in ``conversation``.

    ``say`` is what the owner wants the message to do, in their own words. It is redacted
    with the transcript before anything is sent.
    """
    if tone not in TONES:
        raise ValueError(f"Unknown tone {tone!r}; choose one of {', '.join(TONES)}.")
    if not conversation.match_messages:
        raise ValueError(
            f"{conversation.match_name} has no messages in this transcript — nothing to reply to."
        )

    waiting = conversation.messages[-1].role is Role.OWNER
    say = say.strip() if say else None
    redaction = redact(conversation, notes=[say] if say else [])
    scan = structured_call(
        schema=DraftScan,
        system=DRAFT_SYSTEM,
        user_content=build_draft_request(
            redaction.conversation,
            tone=tone,
            say=redaction.notes[0] if say else None,
            waiting=waiting,
        ),
        settings=settings,
        client=client,
        on_progress=on_progress,
        on_usage=on_usage,
    )
    return redaction.restore(ground_drafts(scan, redaction, tone=tone, waiting=waiting))
