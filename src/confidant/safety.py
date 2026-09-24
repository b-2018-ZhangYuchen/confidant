"""What Confidant does when a red-flag check finds something at tier 3.

The model decides *whether* something is dangerous, and grounding checks that it can
point at the line. What the owner is told to *do* about it is not left to the model at
all. It is fixed text, written in advance and reviewed here as prose, for three reasons:

* It has to be the same every time. Advice that varies with sampling is advice nobody
  can review, and this is the one part of the output where a bad phrasing does harm.
* The model's summary can undercut it. A scan can return a danger flag alongside a
  summary that calls the exchange "mostly friendly"; the notice is printed first so the
  summary cannot talk the owner out of it.
* It does not depend on confidence. A thin transcript with a threat in it is still a
  transcript with a threat in it, so a low-confidence scan escalates exactly like a
  high-confidence one.

No phone numbers or named services yet. Those depend on where the owner is, and
routing to the right one is its own piece of work with its own tests (see ROADMAP.md).
Until then the notice points at the one resource that is right everywhere: someone the
owner already trusts, and the local emergency number when there is a physical risk.
"""

from __future__ import annotations

import textwrap
from collections.abc import Iterable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from confidant.analysis.flags import Flag

__all__ = [
    "CATEGORY_STEPS",
    "ESCALATION_ORDER",
    "STEPS",
    "Escalation",
    "escalate",
]

# The order the notice names things in, most immediate risk first, so that a threat is
# never the third item in a sentence that starts with "tracking where you are".
ESCALATION_ORDER: tuple[str, ...] = (
    "threat",
    "coercion",
    "sexual_pressure",
    "monitoring",
    "isolation",
    "financial_pressure",
    "boundary_pushing",
    "demeaning",
    "guilt_or_blame",
    "inconsistency",
    "evasion",
    "other",
)

# How each category is named in the headline. Descriptions of behavior, never labels
# for the person.
_NAMES: dict[str, str] = {
    "threat": "a threat",
    "coercion": "coercion",
    "sexual_pressure": "pressure around sex",
    "monitoring": "tracking where you are",
    "isolation": "pressure to pull away from friends or family",
    "financial_pressure": "pressure around money",
    "boundary_pushing": "pushing past a no",
    "demeaning": "belittling you",
    "guilt_or_blame": "guilt and blame",
    "inconsistency": "a story that does not hold together",
    "evasion": "evasion",
    "other": "something serious",
}

STEPS: dict[str, str] = {
    "keep_records": (
        "Keep a copy of these messages, with the dates visible, somewhere they cannot reach."
    ),
    "threat_counts": (
        "A threat counts even in a text, and even when it is followed by a joke or an apology."
    ),
    "self_harm": (
        "If the threat was to hurt themselves, their safety is not yours to carry alone. You "
        "can point them to emergency services and still step back."
    ),
    "can_say_no": (
        "You can say no and you can stop replying. Pressure that rises when you hesitate is "
        "the pattern, not a misunderstanding."
    ),
    "owe_nothing_sexual": (
        "You do not owe anyone photos, sex, or a meeting somewhere private, however it is "
        "asked for."
    ),
    "meet_public": (
        "If you do meet, make it somewhere public, get there on your own, and tell a friend "
        "where you will be."
    ),
    "location": (
        "You do not have to share your location or account for where you are. If you have "
        "already shared it in an app, you can turn that off without explaining."
    ),
    "keep_people": (
        "Keep your plans with friends and family. They are the people to talk this through with."
    ),
    "no_money": (
        "Do not send money, gift cards, or crypto, and do not move savings anywhere they "
        "suggest. From someone you have only met online, this is one of the most common "
        "shapes a scam takes."
    ),
    "no_reply_owed": (
        "You do not owe them a reply, and you do not have to decide anything right away. "
        "Talk it through with someone you trust first."
    ),
    "emergency": "If you ever feel physically unsafe, call your local emergency number.",
}

# Categories with no entry get only the closing steps, which is right for a tier-3 flag
# the model filed under a category that has no specific advice of its own.
CATEGORY_STEPS: dict[str, tuple[str, ...]] = {
    "threat": ("threat_counts", "keep_records", "self_harm"),
    "coercion": ("can_say_no", "keep_records"),
    "sexual_pressure": ("owe_nothing_sexual", "meet_public", "keep_records"),
    "monitoring": ("location",),
    "isolation": ("keep_people",),
    "financial_pressure": ("no_money",),
}

# Where the risk can become physical. Money pressure at tier 3 is serious, but telling
# someone to call emergency services over a crypto pitch reads as alarm, not help.
_PHYSICAL: frozenset[str] = frozenset(
    {"threat", "coercion", "sexual_pressure", "monitoring", "isolation"}
)

_CLOSING: tuple[str, ...] = ("no_reply_owed",)

_WIDTH = 88


class Escalation(BaseModel):
    """The notice shown ahead of everything else when a check finds tier-3 behavior."""

    categories: list[str] = Field(
        description="The categories of the danger flags, most immediate risk first."
    )
    headline: str = Field(description="One sentence naming what was found, plainly.")
    steps: list[str] = Field(description="What the owner can do, in order. Fixed text.")
    physical_risk: bool = Field(
        description="Whether any of it could become a risk to physical safety."
    )

    def to_text(self) -> str:
        # Wrapped, unlike the rest of the report: this is the block that has to be read in
        # full, and a single 200-column line is the easiest thing on a screen to skim past.
        lines = [
            textwrap.fill(self.headline, _WIDTH, initial_indent="!! ", subsequent_indent="   ")
        ]
        lines.append("")
        for step in self.steps:
            lines.append(
                textwrap.fill(step, _WIDTH, initial_indent="   - ", subsequent_indent="     ")
            )
        return "\n".join(lines)


def _join(names: list[str]) -> str:
    if len(names) <= 2:
        return " and ".join(names)
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def escalate(flags: Iterable[Flag], match_name: str) -> Escalation | None:
    """Build the tier-3 notice for ``flags``, or ``None`` if none of them are danger.

    Only danger flags contribute. A concern flag alongside a danger flag is already on the
    page below the notice, and folding it into the headline would blur the one sentence
    that most needs to be sharp.
    """
    fired = {flag.category for flag in flags if flag.severity == "danger"}
    if not fired:
        return None

    categories = [c for c in ESCALATION_ORDER if c in fired]
    names = [_NAMES[c] for c in categories]
    headline = f"Something in {match_name}'s messages is serious: {_join(names)}."

    physical = bool(fired & _PHYSICAL)
    keys: list[str] = []
    for category in categories:
        keys.extend(CATEGORY_STEPS.get(category, ()))
    keys.extend(_CLOSING)
    if physical:
        keys.append("emergency")
    # dict.fromkeys dedupes while keeping first-seen order, so "keep_records" appears
    # once, where the most serious category put it.
    steps = [STEPS[key] for key in dict.fromkeys(keys)]

    return Escalation(
        categories=categories,
        headline=headline,
        steps=steps,
        physical_risk=physical,
    )
