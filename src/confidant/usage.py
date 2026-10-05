"""What a model call used, and roughly what it cost.

Every call through :func:`confidant.client.structured_call` reports its token usage to
an ``on_usage`` callback, and the CLI prints one line about it on stderr. This is the
owner's money, and a read that re-sends a long transcript is the kind of thing that adds
up without anyone noticing.

The API's ``input_tokens`` counts only the uncached part of the input. The cached prefix
is reported separately, as tokens written to the cache (billed above the input rate) and
tokens read from it (billed far below it), so all three are kept apart here and priced
separately. Output tokens include adaptive thinking, which is billed as output whether or
not any of it is shown.

The prices are a table in this file, not something the API returns. A model that is not
in it gets its tokens counted and its cost left unknown, rather than a guess presented
as a number.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["PRICES", "Price", "Usage", "UsageMeter", "describe"]


@dataclass(frozen=True, slots=True)
class Price:
    """US dollars per million tokens, for one model, at the five-minute cache TTL."""

    input: float
    output: float
    cache_write: float
    cache_read: float


# Anthropic first-party list prices as of October 2026. A cache write is 1.25x the input
# rate at the five-minute TTL that confidant.client uses; a read is a fraction of it that
# varies by model. Matched on the exact model id: "claude-opus-5-5" starts with
# "claude-opus-5", so a prefix match would price one as the other.
PRICES: dict[str, Price] = {
    "claude-opus-5": Price(input=5.00, output=25.00, cache_write=6.25, cache_read=0.50),
    "claude-opus-5-5": Price(input=4.00, output=20.00, cache_write=5.00, cache_read=0.20),
    "claude-sonnet-5-5": Price(input=2.00, output=10.00, cache_write=2.50, cache_read=0.20),
    "claude-sonnet-5": Price(input=2.00, output=10.00, cache_write=2.50, cache_read=0.20),
    "claude-haiku-4-5": Price(input=1.00, output=5.00, cache_write=1.25, cache_read=0.10),
}


def _count(usage: Any, name: str) -> int:
    # The cache fields are None, not 0, on a response that touched no cache.
    value = getattr(usage, name, None)
    return int(value) if value else 0


@dataclass(frozen=True, slots=True)
class Usage:
    """The tokens one model call used."""

    model: str
    input_tokens: int = 0
    """Input that was neither read from nor written to the cache."""

    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    """Everything the model wrote, thinking included."""

    @classmethod
    def from_response(cls, usage: Any, model: str) -> Usage:
        """Read the ``usage`` object on an SDK response."""
        return cls(
            model=model,
            input_tokens=_count(usage, "input_tokens"),
            cache_write_tokens=_count(usage, "cache_creation_input_tokens"),
            cache_read_tokens=_count(usage, "cache_read_input_tokens"),
            output_tokens=_count(usage, "output_tokens"),
        )

    @property
    def total_input(self) -> int:
        return self.input_tokens + self.cache_write_tokens + self.cache_read_tokens

    @property
    def cost(self) -> float | None:
        """Estimated cost in US dollars, or ``None`` for a model not in :data:`PRICES`."""
        price = PRICES.get(self.model)
        if price is None:
            return None
        return (
            self.input_tokens * price.input
            + self.cache_write_tokens * price.cache_write
            + self.cache_read_tokens * price.cache_read
            + self.output_tokens * price.output
        ) / 1_000_000


@dataclass(slots=True)
class UsageMeter:
    """Collects the usage of every call it is handed; pass it as ``on_usage``."""

    calls: list[Usage] = field(default_factory=list)

    def __call__(self, usage: Usage) -> None:
        self.calls.append(usage)

    def __bool__(self) -> bool:
        return bool(self.calls)

    def describe(self) -> str:
        return describe(self.calls)


def _dollars(amount: float) -> str:
    # Most reads cost a few cents; two decimals would round many of them to $0.00 or make
    # a cheap cached read look the same as a full-price one.
    return f"${amount:.2f}" if amount >= 1 else f"${amount:.3f}"


def describe(calls: Iterable[Usage]) -> str:
    """One sentence on what ``calls`` used, for the end of a command's output."""
    calls = list(calls)
    total_input = sum(c.total_input for c in calls)
    written = sum(c.cache_write_tokens for c in calls)
    read = sum(c.cache_read_tokens for c in calls)
    output = sum(c.output_tokens for c in calls)

    cached = [f"{read:,} read from the cache"] if read else []
    if written:
        cached.append(f"{written:,} written to {'it' if read else 'the cache'}")
    text = f"{total_input:,} input tokens"
    if cached:
        text += f" ({', '.join(cached)})"
    text += f" and {output:,} output tokens"

    costs = [c.cost for c in calls]
    if all(cost is not None for cost in costs):
        text += f", about {_dollars(sum(costs))}"
    else:
        unpriced = sorted({c.model for c in calls if c.cost is None})
        text += f"; no price on file for {', '.join(unpriced)}, so no cost estimate"
    who = f"{len(calls)} calls used" if len(calls) > 1 else "used"
    return f"{who} {text}."
