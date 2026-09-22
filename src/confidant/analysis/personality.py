"""Read a conversation and describe how the other person comes across.

The schema below is doing real work. Asking for a quote alongside every trait makes an
unsupported claim structurally awkward to produce, and a required ``confidence`` field
makes hedging a first-class answer rather than something the model has to smuggle into
prose.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from confidant.client import structured_call
from confidant.config import Settings
from confidant.models import Conversation
from confidant.prompts.personality import PERSONALITY_SYSTEM, build_personality_request

__all__ = ["Confidence", "Evidence", "PersonalityReport", "Trait", "analyze_personality"]

Confidence = Literal["low", "medium", "high"]


class Evidence(BaseModel):
    """A line from the transcript, and why it was worth pointing at."""

    quote: str = Field(description="A short verbatim quote from the match's messages.")
    why_it_matters: str = Field(description="One sentence on what this line suggests.")


class Trait(BaseModel):
    """One observed pattern in how the match behaves."""

    name: str = Field(description="A short, plain-language label, e.g. 'asks follow-up questions'.")
    reading: str = Field(description="Two or three sentences describing the pattern.")
    confidence: Confidence = Field(description="How well the transcript actually supports this.")
    evidence: list[Evidence] = Field(
        description="One to three quotes supporting this reading. Never empty."
    )


class PersonalityReport(BaseModel):
    """Confidant's read on one person, based on one conversation."""

    headline: str = Field(description="One sentence capturing the overall impression.")
    communication_style: str = Field(
        description="How they text: pace, length, warmth, humor, who drives the conversation."
    )
    traits: list[Trait] = Field(description="Three to six observed patterns, strongest first.")
    what_they_seem_to_want: str = Field(
        description="What the transcript suggests they are looking for, or why it is unclear."
    )
    green_flags: list[str] = Field(description="Specific things they did well. May be empty.")
    worth_watching: list[str] = Field(
        description=(
            "Patterns that are not alarming yet but would matter if they continue. "
            "Behavior, not labels. May be empty."
        )
    )
    questions_to_ask: list[str] = Field(
        description="Two to four things the owner could ask to resolve the open questions."
    )
    confidence: Confidence = Field(description="Overall confidence given how much was said.")
    caveat: str = Field(
        description="The most important reason this read could be wrong, in one sentence."
    )

    def to_text(self) -> str:
        """Render the report for a terminal."""
        out = [self.headline, "", "HOW THEY COMMUNICATE", self.communication_style, ""]

        out.append("WHAT STANDS OUT")
        for trait in self.traits:
            out.append(f"  * {trait.name}  [{trait.confidence} confidence]")
            out.append(f"    {trait.reading}")
            for item in trait.evidence:
                out.append(f'      "{item.quote}"')
                out.append(f"        -> {item.why_it_matters}")
        out.append("")

        out += ["WHAT THEY SEEM TO WANT", self.what_they_seem_to_want, ""]

        for title, items in (
            ("GREEN FLAGS", self.green_flags),
            ("WORTH WATCHING", self.worth_watching),
            ("WORTH ASKING", self.questions_to_ask),
        ):
            if items:
                out.append(title)
                out += [f"  * {item}" for item in items]
                out.append("")

        out.append(f"Overall confidence: {self.confidence}")
        out.append(f"Caveat: {self.caveat}")
        return "\n".join(out)


def analyze_personality(
    conversation: Conversation,
    *,
    settings: Settings | None = None,
) -> PersonalityReport:
    """Produce a :class:`PersonalityReport` for the match in ``conversation``."""
    if not conversation.match_messages:
        raise ValueError(
            f"{conversation.match_name} has no messages in this transcript — nothing to read."
        )

    return structured_call(
        schema=PersonalityReport,
        system=PERSONALITY_SYSTEM,
        user_content=build_personality_request(conversation),
        settings=settings,
    )
