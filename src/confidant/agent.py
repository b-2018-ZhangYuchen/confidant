"""``confidant ask``: one question about a conversation, with Claude choosing how to answer it.

Every other command decides in advance what the model is shown and what it is asked. This
one hands Claude the question and a set of tools — the transcript readers, the local
statistics, and Confidant's own analyzers — and lets it decide which ones the question
needs. "Did they ever say what they do for work?" wants a search; "should I be worried
about how they talk about my friends?" wants the red-flag check. The loop itself is in
:func:`confidant.client.structured_call`.

What stays fixed, whatever Claude chooses:

* **Everything is redacted, all the way down.** The conversation is redacted once, here,
  and every tool works on that redacted copy, so a tool result can never carry a name the
  first request did not. The analyzers redact again when they run, which changes nothing
  in text that is already placeholders: their requests are byte-for-byte the ones
  ``analyze`` and ``flags`` send. Names are put back only in the finished answer.
* **The answer is grounded.** Each quote in it is checked against the transcript, and one
  that cannot be found is dropped and counted, as a red flag's would be.
* **The safety check always runs.** If Claude answered without calling check_red_flags,
  it is run afterwards anyway. A question about where to go for dinner can still be
  asked about someone who has been tracking the owner's location, and the fixed safety
  notice should not depend on whether the question happened to make the model look.
* **Each analyzer runs at most once per question.** A second call for the same read is
  answered from the first, so a model that asks twice does not bill twice.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

import anthropic
from pydantic import BaseModel, Field

from confidant.analysis.flags import FlagReport, analyze_flags, quote_appears_in
from confidant.analysis.personality import (
    Confidence,
    Evidence,
    PersonalityReport,
    analyze_personality,
)
from confidant.client import (
    DEFAULT_TOOL_ROUNDS,
    IncompleteResponse,
    ModelRefusal,
    Tool,
    ToolError,
    structured_call,
)
from confidant.config import Settings
from confidant.contact import assess
from confidant.models import Conversation, Role
from confidant.progress import Progress
from confidant.prompts.agent import AGENT_SYSTEM, TOOL_DESCRIPTIONS, build_agent_request
from confidant.prompts.common import render_line
from confidant.redaction import redact
from confidant.safety import Escalation
from confidant.timeline import QUIET_GAP
from confidant.usage import Usage

__all__ = ["READ_LIMIT", "SEARCH_LIMIT", "AgentAnswer", "AskReport", "ask"]

SEARCH_LIMIT = 20
"""Most matches a search returns. A term that matches more is too broad to be useful."""

READ_LIMIT = 60
"""Most messages one read returns, so one call cannot pull a whole long history."""

# What each tool is called in the report's "looked at" line.
_LABELS: dict[str, str] = {
    "conversation_stats": "the statistics",
    "search_messages": "a search",
    "read_messages": "the messages",
    "read_personality": "the personality read",
    "check_red_flags": "the red-flag check",
    "keep_in_touch": "the keep-in-touch timing",
}


class AgentAnswer(BaseModel):
    """What the model is asked to return once it is done with the tools."""

    answer: str = Field(
        description="The answer to the question, in a few sentences. Behavior, not labels."
    )
    evidence: list[Evidence] = Field(
        description=(
            "The lines the answer rests on, quoted verbatim from messages a tool showed you. "
            "Empty only when the answer is that the transcript does not say."
        )
    )
    confidence: Confidence = Field(description="How well the transcript supports the answer.")
    caveat: str = Field(description="The most important reason this answer could be wrong.")


class AskReport(BaseModel):
    """An answer to one question, after local checks."""

    question: str
    answer: str
    evidence: list[Evidence]
    confidence: Confidence
    caveat: str
    consulted: list[str] = Field(
        default_factory=list, description="The tools used, in the order first used."
    )
    discarded: int = Field(
        default=0, description="Quotes dropped because they are not in the transcript."
    )
    escalation: Escalation | None = Field(
        default=None, description="The safety notice, present whenever danger was found."
    )
    unchecked: str | None = Field(
        default=None, description="Why the red-flag check could not run, if it could not."
    )

    def to_text(self) -> str:
        """Render the answer for a terminal."""
        out: list[str] = []
        # First, for the same reason as everywhere else: the answer may be milder.
        if self.escalation is not None:
            out += [self.escalation.to_text(), ""]
        out += [self.answer, ""]
        if self.evidence:
            out.append("FROM THE MESSAGES")
            for item in self.evidence:
                out.append(f'  "{item.quote}"')
                out.append(f"    -> {item.why_it_matters}")
            out.append("")
        out.append(f"Confidence: {self.confidence}")
        out.append(f"Caveat: {self.caveat}")
        if self.consulted:
            out.append(f"Looked at: {', '.join(_LABELS.get(n, n) for n in self.consulted)}.")
        if self.discarded:
            noun = "quote" if self.discarded == 1 else "quotes"
            out.append(
                f"({self.discarded} {noun} left out: not found in the transcript as written.)"
            )
        if self.unchecked is not None:
            # Said outright, because silence here would read as "nothing was found".
            out.append(
                "(The red-flag check could not run, so this answer has not been checked for "
                f"anything dangerous. {self.unchecked})"
            )
        return "\n".join(out)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}{'' if count == 1 else 's'}"


def _span(value: timedelta) -> str:
    if value >= timedelta(days=2):
        return f"{value.days} days"
    hours = round(value.total_seconds() / 3600)
    return _plural(hours, "hour") if hours else "under an hour"


class _Session:
    """The tools for one question, over one redacted conversation."""

    def __init__(
        self,
        conversation: Conversation,
        *,
        settings: Settings,
        client: Any,
        now: datetime,
        on_progress: Callable[[Progress], None] | None,
        on_usage: Callable[[Usage], None] | None,
    ) -> None:
        self.conversation = conversation
        self.settings = settings
        self.client = client
        self.now = now
        self.on_progress = on_progress
        self.on_usage = on_usage
        self.personality: PersonalityReport | None = None
        self.flags: FlagReport | None = None
        self.flags_failed: str | None = None
        self.consulted: list[str] = []

    def tools(self) -> list[Tool]:
        def tool(name: str, run: Callable[[dict[str, Any]], str], **schema: Any) -> Tool:
            def tracked(arguments: dict[str, Any]) -> str:
                if name not in self.consulted:
                    self.consulted.append(name)
                return run(arguments)

            extra = {"input_schema": schema} if schema else {}
            return Tool(name, TOOL_DESCRIPTIONS[name], tracked, **extra)

        return [
            tool("conversation_stats", self.stats),
            tool(
                "search_messages",
                self.search,
                type="object",
                properties={
                    "text": {"type": "string", "description": "What to look for."},
                    "who": {
                        "type": "string",
                        "enum": ["owner", "match", "both"],
                        "description": "Whose messages to search. Default both.",
                    },
                },
                required=["text"],
                additionalProperties=False,
            ),
            tool(
                "read_messages",
                self.read,
                type="object",
                properties={
                    "first": {"type": "integer", "description": "First message number."},
                    "last": {"type": "integer", "description": "Last message number."},
                },
                required=["first", "last"],
                additionalProperties=False,
            ),
            tool("read_personality", self.read_personality),
            tool("check_red_flags", self.check_red_flags),
            tool("keep_in_touch", self.keep_in_touch),
        ]

    # -- local tools -----------------------------------------------------------

    def _numbered(self, index: int) -> str:
        return f"#{index + 1} {render_line(self.conversation.messages[index])}"

    def stats(self, _: dict[str, Any]) -> str:
        c = self.conversation
        lines = [
            f"Messages: {len(c)} ({len(c.owner_messages)} owner / {len(c.match_messages)} match)",
            f"Average words per message: owner {c.mean_words(Role.OWNER):.1f}, "
            f"match {c.mean_words(Role.MATCH):.1f}",
        ]
        dated = [m for m in c if m.timestamp is not None]
        if dated:
            starts = {Role.OWNER: 0, Role.MATCH: 0}
            longest = timedelta(0)
            previous = None
            for message in dated:
                gap = message.timestamp - previous.timestamp if previous else None
                if gap is None or gap >= QUIET_GAP:
                    starts[message.role] += 1
                if gap is not None:
                    longest = max(longest, gap)
                previous = message
            hours = int(QUIET_GAP.total_seconds() // 3600)
            lines += [
                f"First message: {dated[0].timestamp:%Y-%m-%d %H:%M}",
                f"Last message: {dated[-1].timestamp:%Y-%m-%d %H:%M}",
                f"Stretches of conversation (split at {hours}+ hours of quiet) started by: "
                f"owner {starts[Role.OWNER]}, match {starts[Role.MATCH]}",
                f"Longest quiet stretch: {_span(longest)}",
            ]
        else:
            lines.append("No message has a timestamp, so there is no timing to report.")
        if c.messages:
            last = "owner" if c.messages[-1].role is Role.OWNER else "match"
            lines.append(f"Wrote last: {last}")
        return "\n".join(lines)

    def search(self, arguments: dict[str, Any]) -> str:
        text = str(arguments.get("text", "")).strip()
        if not text:
            raise ToolError("Give some text to search for.")
        who = arguments.get("who", "both")
        roles = {"owner": {Role.OWNER}, "match": {Role.MATCH}}.get(who, {Role.OWNER, Role.MATCH})
        hits = [
            i
            for i, message in enumerate(self.conversation.messages)
            if message.role in roles and quote_appears_in(text, [message.text])
        ]
        if not hits:
            return f"No message contains {text!r}."
        lines = [self._numbered(i) for i in hits[:SEARCH_LIMIT]]
        if len(hits) > SEARCH_LIMIT:
            lines.append(
                f"({len(hits) - SEARCH_LIMIT} more not shown; search for something narrower.)"
            )
        return "\n".join(lines)

    def read(self, arguments: dict[str, Any]) -> str:
        total = len(self.conversation)
        try:
            first, last = int(arguments["first"]), int(arguments["last"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ToolError("Give first and last as message numbers.") from exc
        if not 1 <= first <= total:
            raise ToolError(f"There are {total} messages, numbered 1 to {total}.")
        last = min(max(last, first), total, first + READ_LIMIT - 1)
        lines = [self._numbered(i) for i in range(first - 1, last)]
        if last < total:
            lines.append(f"(messages {last + 1} to {total} not shown)")
        return "\n".join(lines)

    def keep_in_touch(self, _: dict[str, Any]) -> str:
        escalation = self.flags.escalation if self.flags is not None else None
        contact = assess("[MATCH]", [self.conversation], now=self.now, escalation=escalation)
        lines = [contact.headline(), *(f"- {reason}" for reason in contact.reasons)]
        return "\n".join(lines)

    # -- analyzers -------------------------------------------------------------

    @property
    def _kwargs(self) -> dict[str, Any]:
        return {
            "settings": self.settings,
            "client": self.client,
            "on_progress": self.on_progress,
            "on_usage": self.on_usage,
        }

    def read_personality(self, _: dict[str, Any]) -> str:
        if self.personality is None:
            self.personality = _as_tool(analyze_personality, self.conversation, **self._kwargs)
        return self.personality.model_dump_json()

    def run_flags(self) -> FlagReport:
        # A check that declined once is not tried again: the answer says it did not run.
        if self.flags_failed is not None:
            raise ToolError(self.flags_failed)
        if self.flags is None:
            try:
                self.flags = _as_tool(analyze_flags, self.conversation, **self._kwargs)
            except ToolError as exc:
                self.flags_failed = str(exc)
                raise
        return self.flags

    def check_red_flags(self, _: dict[str, Any]) -> str:
        report = self.run_flags()
        # The notice itself is fixed text for the owner, not something for the model to
        # rephrase; the model only needs to know it will be there.
        data = report.model_dump(mode="json", exclude={"escalation"})
        data["safety_notice_shown_to_owner"] = report.escalation is not None
        return json.dumps(data, ensure_ascii=False)


def _as_tool(analyzer: Callable[..., Any], conversation: Conversation, **kwargs: Any) -> Any:
    """Run an analyzer, turning the failures the model can work around into tool errors."""
    try:
        return analyzer(conversation, **kwargs)
    except ModelRefusal as exc:
        raise ToolError(f"This check declined to run: {exc}") from exc
    except IncompleteResponse as exc:
        raise ToolError(f"This check did not finish: {exc}") from exc
    except ValueError as exc:
        raise ToolError(str(exc)) from exc


def ask(
    conversation: Conversation,
    question: str,
    *,
    settings: Settings | None = None,
    client: anthropic.Anthropic | None = None,
    now: datetime | None = None,
    escalation: Escalation | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    on_usage: Callable[[Usage], None] | None = None,
    max_tool_rounds: int = DEFAULT_TOOL_ROUNDS,
) -> AskReport:
    """Answer ``question`` about the match in ``conversation``, using tools as needed.

    ``escalation`` is a safety notice already known from elsewhere, such as another saved
    conversation with the same person. It leads the answer if this conversation's own
    check finds nothing.
    """
    question = question.strip()
    if not question:
        raise ValueError("Ask a question, e.g. 'have they ever asked about my day?'")
    if not conversation.match_messages:
        raise ValueError(
            f"{conversation.match_name} has no messages in this transcript — nothing to ask about."
        )

    settings = settings or Settings.from_env()
    redaction = redact(conversation, notes=[question])
    session = _Session(
        redaction.conversation,
        settings=settings,
        client=client,
        now=now or datetime.now(),
        on_progress=on_progress,
        on_usage=on_usage,
    )
    answer = structured_call(
        schema=AgentAnswer,
        system=AGENT_SYSTEM,
        user_content=build_agent_request(redaction.conversation, redaction.notes[0]),
        settings=settings,
        client=client,
        on_progress=on_progress,
        on_usage=on_usage,
        tools=session.tools(),
        max_tool_rounds=max_tool_rounds,
    )
    try:
        flags: FlagReport | None = session.run_flags()
    except ToolError as exc:
        flags = None
        unchecked = str(exc)
    else:
        unchecked = None

    texts = [message.text for message in redaction.conversation]
    evidence = [e for e in answer.evidence if quote_appears_in(e.quote, texts)]
    report = AskReport(
        question=question,
        answer=answer.answer,
        evidence=evidence,
        confidence=answer.confidence,
        caveat=answer.caveat,
        consulted=session.consulted,
        discarded=len(answer.evidence) - len(evidence),
        escalation=flags.escalation if flags is not None else None,
        unchecked=unchecked,
    )
    report = redaction.restore(report)
    if report.escalation is None and escalation is not None:
        report = report.model_copy(update={"escalation": escalation})
    return report
