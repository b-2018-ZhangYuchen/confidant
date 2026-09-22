"""Runtime configuration, read once from the environment."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

Effort = Literal["low", "medium", "high", "xhigh", "max"]

DEFAULT_MODEL = "claude-opus-5"
DEFAULT_EFFORT: Effort = "high"
DEFAULT_MAX_TOKENS = 16_000

_VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ConfigError(RuntimeError):
    """Raised when Confidant is missing something it needs to run."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything Confidant needs to talk to the model."""

    model: str = DEFAULT_MODEL
    effort: Effort = DEFAULT_EFFORT
    max_tokens: int = DEFAULT_MAX_TOKENS
    api_key: str | None = None

    @classmethod
    def from_env(cls, *, env_file: str | Path | None = None) -> Settings:
        """Load settings from ``.env`` (if present) and the process environment."""
        load_dotenv(env_file) if env_file else load_dotenv()

        effort = os.getenv("CONFIDANT_EFFORT", DEFAULT_EFFORT).lower()
        if effort not in _VALID_EFFORTS:
            raise ConfigError(
                f"CONFIDANT_EFFORT={effort!r} is not one of {', '.join(_VALID_EFFORTS)}"
            )

        raw_max_tokens = os.getenv("CONFIDANT_MAX_TOKENS")
        try:
            max_tokens = int(raw_max_tokens) if raw_max_tokens else DEFAULT_MAX_TOKENS
        except ValueError as exc:
            raise ConfigError(f"CONFIDANT_MAX_TOKENS={raw_max_tokens!r} is not an integer") from exc

        return cls(
            model=os.getenv("CONFIDANT_MODEL", DEFAULT_MODEL),
            effort=effort,  # type: ignore[arg-type]
            max_tokens=max_tokens,
            # Left as None on purpose: the SDK resolves ANTHROPIC_API_KEY and `ant auth
            # login` profiles itself, and we should not get in the way of that.
            api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        )
