"""Check a conversation for red flags, with a severity tier on each one.

This is deliberately a separate pass from the personality read. A personality prompt is
asked to be even-handed and warm; a safety check is asked to name real danger plainly.
Putting both in one request means one of them gets diluted, and in practice it is the
safety finding that ends up softened into a "worth watching" bullet.

Two things happen locally, after the model answers, because they are too important to
leave to the prompt alone:

* **Grounding.** Every quote is checked against what the match actually wrote. A flag
  whose quotes cannot be found is dropped. A fabricated quote in a personality read is
  embarrassing; a fabricated quote in a red flag is an accusation.
* **Severity floors.** Threats, coercion, sexual pressure, isolation, and monitoring are
  always tier 3. If the model filed one of those as "watch", that is exactly the
  softening ``docs/principles.md`` rules out, so it is corrected rather than trusted.

When a tier-3 flag survives both, the report carries an :class:`~confidant.safety.Escalation`
and leads with it. What that notice says lives in ``confidant.safety``.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Literal

from pydantic import BaseModel, Field

from confidant.analysis.personality import Confidence, Evidence
from confidant.client import structured_call
from confidant.config import Settings
from confidant.models import Conversation
from confidant.prompts.flags import FLAGS_SYSTEM, build_flags_request
from confidant.safety import Escalation, escalate

__all__ = [
    "DANGER_CATEGORIES",
    "TIERS",
    "Category",
    "Flag",
    "FlagReport",
    "FlagScan",
    "Severity",
    "Escalation",
    "analyze_flags",
    "ground_flags",
    "quote_appears_in",
]

Severity = Literal["watch", "concern", "danger"]
TIERS: dict[str, int] = {"watch": 1, "concern": 2, "danger": 3}

Category = Literal[
    "threat",
    "coercion",
    "sexual_pressure",
    "financial_pressure",
    "isolation",
    "monitoring",
    "boundary_pushing",
    "demeaning",
    "guilt_or_blame",
    "inconsistency",
    "evasion",
    "other",
]

# financial_pressure is not here on purpose: "can you spot me for the concert ticket" and
# "move your savings into this platform" share a category and should not share a tier.
DANGER_CATEGORIES: frozenset[str] = frozenset(
    {"threat", "coercion", "sexual_pressure", "isolation", "monitoring"}
)


class Flag(BaseModel):
    """One piece of behavior worth flagging, and how serious it is."""

    category: Category = Field(description="The closest category for this behavior.")
    severity: Severity = Field(description="watch, concern, or danger. See the tier definitions.")
    behavior: str = Field(
        description="One or two sentences describing what the match did. Behavior, not labels."
    )
    evidence: list[Evidence] = Field(
        description="One to three verbatim quotes from the match's messages. Never empty."
    )
    innocent_reading: str | None = Field(
        description=(
            "The most plausible mundane explanation, in one sentence. Required for watch "
            "and concern. Null for danger."
        )
    )

    @property
    def tier(self) -> int:
        return TIERS[self.severity]


class FlagScan(BaseModel):
    """What the model is asked to return."""

    flags: list[Flag] = Field(description="Red flags found, most serious first. Often empty.")
    summary: str = Field(
        description=(
            "One or two sentences on the overall picture. If there are no flags, say so "
            "plainly without adding reassurance the transcript does not support."
        )
    )
    confidence: Confidence = Field(description="How well the transcript supports this check.")


class FlagReport(BaseModel):
    """A grounded red-flag check: the model's scan after local verification."""

    flags: list[Flag]
    summary: str
    confidence: Confidence
    discarded: int = Field(
        default=0,
        description="Flags dropped because none of their quotes appear in the transcript.",
    )
    discarded_danger: int = Field(
        default=0,
        description="How many of the discarded flags had been filed as danger.",
    )
    escalation: Escalation | None = Field(
        default=None,
        description="The safety notice, present whenever a danger flag survived grounding.",
    )

    @property
    def highest_tier(self) -> int:
        """``0`` when nothing was flagged, otherwise the tier of the most serious flag."""
        return max((flag.tier for flag in self.flags), default=0)

    @property
    def has_danger(self) -> bool:
        return self.highest_tier == TIERS["danger"]

    def to_text(self) -> str:
        """Render the check for a terminal."""
        out: list[str] = []
        # First, above the model's own summary, which may be milder than its flags.
        if self.escalation is not None:
            out.extend([self.escalation.to_text(), ""])
        out.extend([self.summary, ""])

        if not self.flags:
            out.append("Nothing in this transcript rises to a red flag.")
            out.append("")
        for flag in self.flags:
            label = flag.category.replace("_", " ")
            out.append(f"[{flag.severity.upper()}] {label}")
            out.append(f"  {flag.behavior}")
            for item in flag.evidence:
                out.append(f'    "{item.quote}"')
                out.append(f"      -> {item.why_it_matters}")
            # Printed only below danger: the prompt asks for null there, but a charitable
            # gloss on a threat should not reach the screen even if the model supplies one.
            if flag.innocent_reading and flag.tier < TIERS["danger"]:
                out.append(f"  Could also be: {flag.innocent_reading}")
            out.append("")

        out.append(f"Confidence: {self.confidence}")
        if self.discarded:
            noun = "flag" if self.discarded == 1 else "flags"
            out.append(
                f"({self.discarded} {noun} left out: the quotes behind them could not be "
                "found in the transcript.)"
            )
        if self.discarded_danger:
            # Grounding protects the match from an invented accusation, but the owner was
            # there and this check was not. Dropping a danger flag silently would let the
            # report sound more reassuring than it has earned.
            which = (
                "One of them was"
                if self.discarded_danger == 1
                else f"{self.discarded_danger} of them were"
            )
            out.append(
                f"({which} filed as serious. If something in this conversation felt unsafe "
                "to you, that counts for more than this check does.)"
            )
        return "\n".join(out)


# -- grounding --------------------------------------------------------------

_TRANSLATE = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
    }
)
_ELLIPSIS = re.compile(r"\.\.\.|…")
_EDGE_PUNCTUATION = " \t\n\"'.,!?;:-"


def _normalize(text: str) -> str:
    """Fold away differences a faithful quote can still have from the source.

    Models straighten curly quotes, change case at the start of a quote, and collapse
    line breaks. None of those change what was said, so none should fail grounding.
    """
    text = unicodedata.normalize("NFKC", text).translate(_TRANSLATE).casefold()
    return " ".join(text.split())


def quote_appears_in(quote: str, messages: list[str]) -> bool:
    """Whether ``quote`` is a faithful excerpt of one of ``messages``.

    An ellipsis in the quote may stand for omitted text, so each fragment must appear in
    the same message, in order. Quotes are not allowed to stitch two messages together:
    that is how a sentence gets a meaning its author never gave it.
    """
    fragments = [
        _normalize(part).strip(_EDGE_PUNCTUATION) for part in _ELLIPSIS.split(_normalize(quote))
    ]
    fragments = [f for f in fragments if f]
    if not fragments:
        return False

    for message in messages:
        haystack = _normalize(message)
        position = 0
        for fragment in fragments:
            found = haystack.find(fragment, position)
            if found < 0:
                break
            position = found + len(fragment)
        else:
            return True
    return False


def ground_flags(scan: FlagScan, conversation: Conversation) -> FlagReport:
    """Verify a model scan against the transcript and apply severity floors.

    Quotes that cannot be found are removed; a flag left with no quotes is dropped and
    counted in ``discarded``. The result is sorted most serious first, and carries an
    escalation if any danger flag remains.
    """
    match_texts = [m.text for m in conversation.match_messages]
    kept: list[Flag] = []
    discarded = 0
    discarded_danger = 0

    for flag in scan.flags:
        evidence = [e for e in flag.evidence if quote_appears_in(e.quote, match_texts)]
        if not evidence:
            discarded += 1
            if flag.severity == "danger" or flag.category in DANGER_CATEGORIES:
                discarded_danger += 1
            continue

        severity = flag.severity
        if flag.category in DANGER_CATEGORIES:
            severity = "danger"
        kept.append(flag.model_copy(update={"evidence": evidence, "severity": severity}))

    # sorted() is stable, so flags within a tier keep the model's ordering.
    kept = sorted(kept, key=lambda f: f.tier, reverse=True)
    return FlagReport(
        flags=kept,
        summary=scan.summary,
        confidence=scan.confidence,
        discarded=discarded,
        discarded_danger=discarded_danger,
        escalation=escalate(kept, conversation.match_name),
    )


def analyze_flags(
    conversation: Conversation,
    *,
    settings: Settings | None = None,
) -> FlagReport:
    """Check the match in ``conversation`` for red flags, grounded against the transcript."""
    if not conversation.match_messages:
        raise ValueError(
            f"{conversation.match_name} has no messages in this transcript — nothing to check."
        )

    scan = structured_call(
        schema=FlagScan,
        system=FLAGS_SYSTEM,
        user_content=build_flags_request(conversation),
        settings=settings,
    )
    return ground_flags(scan, conversation)
