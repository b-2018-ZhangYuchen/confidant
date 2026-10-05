"""Thin wrapper over the Anthropic SDK.

Every model call in Confidant goes through :func:`structured_call`, which keeps six
things consistent across the codebase: adaptive thinking is on, the system prompt is
cached, the response is streamed, it is validated against a Pydantic schema, what it
used is reported (see :mod:`confidant.usage`), and a refusal is surfaced as an exception
instead of quietly becoming an empty report.

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
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from confidant.config import ConfigError, Settings
from confidant.progress import Progress, ProgressTracker
from confidant.usage import Usage

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "IncompleteResponse",
    "ModelRefusal",
    "build_client",
    "cached_system",
    "structured_call",
]

# The default five-minute lifetime, renewed on every hit. The one-hour option costs twice
# as much to write, and the calls that share a prefix (one `profile --update`, or an
# `analyze` and a `flags` run back to back) land minutes apart, not hours.
_CACHE_CONTROL = {"type": "ephemeral"}


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
    """
    settings = settings or Settings.from_env()
    client = client or build_client(settings)

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
            messages=[{"role": "user", "content": user_content}],
            output_format=schema,
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

    parsed = response.parsed_output
    if parsed is None:
        raise IncompleteResponse(response.stop_reason, settings.max_tokens)
    return parsed
