"""Thin wrapper over the Anthropic SDK.

Every model call in Confidant goes through :func:`structured_call`, which keeps six
things consistent across the codebase: adaptive thinking is on, the system prompt is
cached, the response is streamed, it is validated against a Pydantic schema, what it
used is reported (see :mod:`confidant.usage`), and a refusal is surfaced as an exception
instead of quietly becoming an empty report.

Given ``tools``, the same call becomes a loop: while the model asks for a tool, the tool
is run here and its result sent back, and only the answer it finally writes is parsed.
Every round goes through the same streaming, refusal check, and usage report as a single
call, so an agent is not a second, less careful way of talking to the model.

What gets cached, and what does not
-----------------------------------

A cache hit needs a byte-identical prefix, and the API renders the system prompt ahead
of the messages. Only the system prompt is stable: every analysis of a given kind sends
the same one, along with the same output schema, whoever the conversation is about. So
``profile --update`` reading several conversations in a row pays full price for that
prefix once and a tenth of it after.

The transcript is deliberately left uncached. It is the biggest part of the request, but
it is not a stable one: the personality read and the red-flag check put it after
different system prompts, so neither can reuse the other's entry, and the header above it
counts messages, so an appended message changes it from the first line. A breakpoint
there would pay the cache-write premium on nearly every call and almost never be read.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from confidant.config import ConfigError, Settings
from confidant.progress import Progress, ProgressTracker
from confidant.usage import Usage

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "DEFAULT_TOOL_ROUNDS",
    "IncompleteResponse",
    "ModelRefusal",
    "Tool",
    "ToolError",
    "ToolLoopError",
    "block_param",
    "build_client",
    "cached_system",
    "describe_api_error",
    "structured_call",
]

# The default five-minute lifetime, renewed on every hit. The one-hour option costs twice
# as much to write, and the calls that share a prefix (one `profile --update`, or an
# `analyze` and a `flags` run back to back) land minutes apart, not hours.
_CACHE_CONTROL = {"type": "ephemeral"}

# Enough for a careful answer that looks at the transcript, reads it, and checks a quote
# or two; well short of a model that has lost the thread and is calling tools in circles.
DEFAULT_TOOL_ROUNDS = 8


class ModelRefusal(RuntimeError):
    """The model declined to answer.

    Worth handling explicitly here: Confidant's whole job is reading emotionally charged
    conversations, so the caller needs a clear signal rather than a blank report.
    """

    def __init__(self, category: str | None, explanation: str | None) -> None:
        self.category = category
        self.explanation = explanation
        detail = explanation or "no explanation given"
        super().__init__(f"The model declined this request ({category or 'unspecified'}): {detail}")


class IncompleteResponse(RuntimeError):
    """The model answered, but not with a whole report.

    In practice this is almost always the token limit: a long transcript plus adaptive
    thinking can run out before the JSON closes. Saying so, with the setting that fixes
    it, beats a pydantic traceback about an unexpected end of input.
    """

    def __init__(self, stop_reason: str | None, max_tokens: int) -> None:
        self.stop_reason = stop_reason
        if stop_reason in ("max_tokens", None):
            # None: the stream failed to parse before the stop reason arrived, which is
            # how a cut-off answer shows up when streaming.
            detail = (
                f"Claude's answer was cut off before the report was finished "
                f"(CONFIDANT_MAX_TOKENS={max_tokens}). Raise CONFIDANT_MAX_TOKENS, or "
                "lower CONFIDANT_EFFORT, and try again."
            )
        else:
            detail = f"Claude stopped without returning a report (stop_reason={stop_reason!r})."
        super().__init__(detail)


class ToolLoopError(RuntimeError):
    """The model kept asking for tools past the limit, without ever answering."""

    def __init__(self, rounds: int) -> None:
        self.rounds = rounds
        super().__init__(
            f"Claude asked for {rounds} rounds of tools without settling on an answer. "
            "Try asking a narrower question."
        )


class ToolError(Exception):
    """A tool could not do what it was asked; the message goes back to the model.

    Raised by a tool for problems the model can do something about (a line number past
    the end, a check that declined). Anything else a tool raises is a bug, and propagates.
    """


@dataclass(frozen=True, slots=True)
class Tool:
    """Something the model may ask Confidant to do while it works out an answer."""

    name: str
    description: str
    run: Callable[[dict[str, Any]], str]
    input_schema: dict[str, Any] = field(
        default_factory=lambda: {"type": "object", "properties": {}, "additionalProperties": False}
    )

    def param(self) -> dict[str, Any]:
        """The tool as the API wants it described."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


def block_param(block: Any) -> dict[str, Any]:
    """A response content block as it must be sent back in the next request.

    Spelled out field by field rather than dumped: the SDK's parsed blocks carry extra
    attributes (a text block's ``parsed_output``) that the API would reject, and thinking
    blocks have to go back exactly as they came, signature included, or the loop breaks.
    """
    if isinstance(block, dict):
        return block
    kind = block.type
    if kind == "thinking":
        return {"type": "thinking", "thinking": block.thinking, "signature": block.signature}
    if kind == "redacted_thinking":
        return {"type": "redacted_thinking", "data": block.data}
    if kind == "text":
        return {"type": "text", "text": block.text}
    if kind == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    raise ValueError(f"Cannot send a {kind!r} block back to the model")


def _credentials_available(settings: Settings) -> bool:
    """Check for credentials before building a client.

    The SDK does not complain about missing credentials until the first request, and
    then only with a ``TypeError`` from deep inside its header code. Checking here turns
    a forty-line traceback into one sentence.
    """
    if settings.api_key or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    # An `ant auth login` profile, which the SDK picks up on its own.
    profile_dir = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config")) / "anthropic"
    return profile_dir.is_dir() and any(profile_dir.iterdir())


def build_client(settings: Settings | None = None) -> anthropic.Anthropic:
    """Construct an Anthropic client.

    Passing ``api_key=None`` is intentional — the SDK then resolves credentials from the
    environment or an ``ant auth login`` profile on its own. It raises a bare
    ``TypeError`` when it finds nothing, which is not a useful thing to show someone who
    simply has not made a ``.env`` file yet.
    """
    settings = settings or Settings.from_env()
    if not _credentials_available(settings):
        raise ConfigError(
            "No Anthropic credentials found. Copy .env.example to .env and set "
            "ANTHROPIC_API_KEY, or export it in your shell. "
            "(`confidant stats` and `confidant redact` work without a key.)"
        )
    return anthropic.Anthropic(api_key=settings.api_key)


def cached_system(system: str) -> list[dict[str, Any]]:
    """``system`` as a single text block with a cache breakpoint on it.

    The prompt modules hold plain strings so they read as prose; this is the one place
    that knows the API wants a list of blocks to carry ``cache_control``.
    """
    return [{"type": "text", "text": system, "cache_control": dict(_CACHE_CONTROL)}]


def structured_call(
    *,
    schema: type[T],
    system: str,
    user_content: str,
    settings: Settings | None = None,
    client: anthropic.Anthropic | None = None,
    on_progress: Callable[[Progress], None] | None = None,
    on_usage: Callable[[Usage], None] | None = None,
    tools: Sequence[Tool] = (),
    max_tool_rounds: int = DEFAULT_TOOL_ROUNDS,
) -> T:
    """Ask the model one question and get back a validated ``schema`` instance.

    ``client`` is normally left out and built from ``settings``. Tests pass a
    :class:`~confidant.recording.ReplayClient` here, so everything below runs against a
    recorded response instead of being patched away.

    ``on_progress`` is told when the model starts thinking, starts writing, and moves on
    to each part of the report. It never sees the report itself; see
    :mod:`confidant.progress` for why.

    ``on_usage`` is handed the call's token usage as soon as the response is whole, before
    it is checked: a refusal or a cut-off answer is billed too. An answer whose stream
    breaks off mid-JSON never produces a final message, so its usage cannot be reported.
    With tools, it is handed each round's usage in turn.

    ``tools`` lets the model ask for things before it answers. A tool that raises
    :class:`ToolError` has the message sent back as a failed result, so the model can try
    something else; ``max_tool_rounds`` rounds of tool use without an answer raise
    :class:`ToolLoopError`.
    """
    settings = settings or Settings.from_env()
    client = client or build_client(settings)
    by_name = {tool.name: tool for tool in tools}
    messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]

    for _ in range(max_tool_rounds + 1):
        response = _respond(
            client, settings, schema, system, messages, tools, on_progress, on_usage
        )
        if response.stop_reason != "tool_use" or not tools:
            break
        messages.append(
            {"role": "assistant", "content": [block_param(b) for b in response.content]}
        )
        messages.append(
            {
                "role": "user",
                "content": [
                    _run_tool(by_name, block)
                    for block in response.content
                    if getattr(block, "type", None) == "tool_use"
                ],
            }
        )
    else:
        raise ToolLoopError(max_tool_rounds)

    parsed = response.parsed_output
    if parsed is None:
        raise IncompleteResponse(response.stop_reason, settings.max_tokens)
    return parsed


def _run_tool(tools: dict[str, Tool], block: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.id}
    tool = tools.get(block.name)
    if tool is None:
        return result | {"content": f"There is no tool called {block.name!r}.", "is_error": True}
    try:
        return result | {"content": tool.run(dict(block.input or {}))}
    except ToolError as exc:
        return result | {"content": str(exc), "is_error": True}


def _respond(
    client: Any,
    settings: Settings,
    schema: type[T],
    system: str,
    messages: list[dict[str, Any]],
    tools: Sequence[Tool],
    on_progress: Callable[[Progress], None] | None,
    on_usage: Callable[[Usage], None] | None,
) -> Any:
    """One request and its whole response, with usage reported and a refusal raised."""
    extra: dict[str, Any] = {}
    if tools:
        # Only when there are tools, so a plain call's request (and the recordings
        # fingerprinted from it) is exactly what it was before tools existed.
        extra["tools"] = [tool.param() for tool in tools]

    # Streamed whether or not anyone is watching, so there is one code path. It also
    # keeps a raised CONFIDANT_MAX_TOKENS working: the SDK refuses a non-streaming
    # request it expects to run past its ten-minute timeout.
    try:
        with client.messages.stream(
            model=settings.model,
            max_tokens=settings.max_tokens,
            system=cached_system(system),
            thinking={"type": "adaptive"},
            output_config={"effort": settings.effort},
            # A copy: the loop goes on appending to its own list, and the request that
            # was sent should stay what it was.
            messages=list(messages),
            output_format=schema,
            **extra,
        ) as stream:
            if on_progress is not None:
                tracker = ProgressTracker(on_progress)
                for event in stream:
                    tracker.feed(event)
            response = stream.get_final_message()
    except ValidationError as exc:
        # Only unfinished JSON means a cut-off answer. Valid JSON in the wrong shape is a
        # schema problem (a stale recording, usually) and should surface as itself.
        if all(error["type"] == "json_invalid" for error in exc.errors()):
            raise IncompleteResponse(None, settings.max_tokens) from exc
        raise

    usage = getattr(response, "usage", None)
    if on_usage is not None and usage is not None:
        # The model that answered, which is what is billed, rather than the one asked for.
        model = getattr(response, "model", None) or settings.model
        on_usage(Usage.from_response(usage, model))

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise ModelRefusal(
            category=getattr(details, "category", None),
            explanation=getattr(details, "explanation", None),
        )
    return response


def describe_api_error(exc: anthropic.APIError, settings: Settings) -> str:
    """Say what went wrong in terms of what the owner can do about it."""
    if isinstance(exc, anthropic.APIConnectionError):
        # Includes timeouts, which subclass it.
        return "Could not reach the Anthropic API. Check your connection and try again."
    if isinstance(exc, anthropic.AuthenticationError):
        return (
            "The Anthropic API did not accept these credentials. Check ANTHROPIC_API_KEY "
            "in .env or your shell."
        )
    if isinstance(exc, anthropic.NotFoundError):
        return (
            f"The Anthropic API does not know the model {settings.model!r}. Check "
            "CONFIDANT_MODEL, or unset it to use the default."
        )
    if isinstance(exc, anthropic.RateLimitError):
        return "The Anthropic API is rate-limiting this key. Wait a minute and try again."
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and status >= 500:
        return f"The Anthropic API is having trouble (HTTP {status}). Try again shortly."
    return f"The Anthropic API call failed: {exc}"
