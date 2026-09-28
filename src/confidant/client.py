"""Thin wrapper over the Anthropic SDK.

Every model call in Confidant goes through :func:`structured_call`, which keeps four
things consistent across the codebase: adaptive thinking is on, the response is streamed,
it is validated against a Pydantic schema, and a refusal is surfaced as an exception
instead of quietly becoming an empty report.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

import anthropic
from pydantic import BaseModel

from confidant.config import ConfigError, Settings
from confidant.progress import Progress, ProgressTracker

T = TypeVar("T", bound=BaseModel)

__all__ = ["ModelRefusal", "build_client", "structured_call"]


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
            "(`confidant stats` works without a key.)"
        )
    return anthropic.Anthropic(api_key=settings.api_key)


def structured_call(
    *,
    schema: type[T],
    system: str,
    user_content: str,
    settings: Settings | None = None,
    client: anthropic.Anthropic | None = None,
    on_progress: Callable[[Progress], None] | None = None,
) -> T:
    """Ask the model one question and get back a validated ``schema`` instance.

    ``client`` is normally left out and built from ``settings``. Tests pass a
    :class:`~confidant.recording.ReplayClient` here, so everything below runs against a
    recorded response instead of being patched away.

    ``on_progress`` is told when the model starts thinking, starts writing, and moves on
    to each part of the report. It never sees the report itself; see
    :mod:`confidant.progress` for why.
    """
    settings = settings or Settings.from_env()
    client = client or build_client(settings)

    # Streamed whether or not anyone is watching, so there is one code path. It also
    # keeps a raised CONFIDANT_MAX_TOKENS working: the SDK refuses a non-streaming
    # request it expects to run past its ten-minute timeout.
    with client.messages.stream(
        model=settings.model,
        max_tokens=settings.max_tokens,
        system=system,
        thinking={"type": "adaptive"},
        messages=[{"role": "user", "content": user_content}],
        output_format=schema,
    ) as stream:
        if on_progress is not None:
            tracker = ProgressTracker(on_progress)
            for event in stream:
                tracker.feed(event)
        response = stream.get_final_message()

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise ModelRefusal(
            category=getattr(details, "category", None),
            explanation=getattr(details, "explanation", None),
        )

    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError(
            f"Model returned no structured output (stop_reason={response.stop_reason!r})"
        )
    return parsed
