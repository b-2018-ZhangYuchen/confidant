"""Crisis-resource routing: which numbers a safety notice lists, given where the owner is.

The notices in :mod:`confidant.safety` say what to do in words that are right
everywhere. This module adds the part that is not: the actual emergency number and the
lines worth calling, for the owner's region. Three rules shape it:

* **Explicit, never guessed.** The region comes from ``--region`` or
  ``CONFIDANT_REGION``. It is not inferred from the locale or the clock: plenty of
  people run ``en_US`` wherever they live, and a confident wrong number is worse than
  the general advice, which is still printed either way.
* **A short table, kept honest.** Only national services, free to call, that answer
  around the clock. Where a country has no single national line for something (Canada
  has no national domestic-violence line; it is provincial), the table leaves a gap
  rather than filling it with a regional one. These numbers change rarely, but they do
  change: check each against the service's own site before a release.
* **Routed by what was found.** The owner at risk is pointed at a crisis line. Danger
  from the match that could become physical is pointed at a domestic-abuse line. A
  threat also gets the crisis line, because a threat to self-harm is the match's crisis
  and they can be pointed there. Money pressure alone lists nothing: a crisis line is
  the wrong door for a scam.

Nothing here makes a network call. The table is data in this file.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "DIRECTORY",
    "REGIONS",
    "REGION_ENV",
    "Kind",
    "Region",
    "Resource",
    "Routing",
    "current_region",
    "normalize_region",
    "route",
    "using_region",
]

REGION_ENV = "CONFIDANT_REGION"

Kind = Literal["emergency", "crisis", "abuse"]

# The one fallback that is right in most places. Named in text only; Confidant never
# opens it.
DIRECTORY = "findahelpline.com lists free support lines in most countries."


class Resource(BaseModel):
    """One number worth calling, in words that fit on a line."""

    kind: Kind
    name: str
    how: str = Field(description="How to reach it, e.g. 'call or text 988'.")

    def to_text(self) -> str:
        return f"{self.name}: {self.how}."


@dataclass(frozen=True, slots=True)
class Region:
    code: str
    name: str
    resources: tuple[Resource, ...]


def _r(kind: Kind, name: str, how: str) -> Resource:
    return Resource(kind=kind, name=name, how=how)


# Emergency first in every region, so it is the first number on the page.
REGIONS: dict[str, Region] = {
    region.code: region
    for region in (
        Region(
            "US",
            "the United States",
            (
                _r("emergency", "Emergency services", "call 911"),
                _r("crisis", "988 Suicide & Crisis Lifeline", "call or text 988"),
                _r(
                    "abuse",
                    "National Domestic Violence Hotline",
                    "call 1-800-799-7233 or text START to 88788",
                ),
            ),
        ),
        Region(
            "CA",
            "Canada",
            (
                _r("emergency", "Emergency services", "call 911"),
                _r("crisis", "9-8-8 Suicide Crisis Helpline", "call or text 988"),
            ),
        ),
        Region(
            "GB",
            "the United Kingdom",
            (
                _r("emergency", "Emergency services", "call 999"),
                _r("crisis", "Samaritans", "call 116 123"),
                _r("crisis", "Shout", "text SHOUT to 85258"),
                # England only; Scotland, Wales, and Northern Ireland run their own, and
                # saying so beats implying this one covers them.
                _r("abuse", "National Domestic Abuse Helpline (England)", "call 0808 2000 247"),
            ),
        ),
        Region(
            "IE",
            "Ireland",
            (
                _r("emergency", "Emergency services", "call 112 or 999"),
                _r("crisis", "Samaritans", "call 116 123"),
                _r("crisis", "Text About It", "text HELLO to 50808"),
                _r("abuse", "Women's Aid", "call 1800 341 900"),
            ),
        ),
        Region(
            "AU",
            "Australia",
            (
                _r("emergency", "Emergency services", "call 000"),
                _r("crisis", "Lifeline", "call 13 11 14"),
                _r("abuse", "1800RESPECT", "call 1800 737 732"),
            ),
        ),
        Region(
            "NZ",
            "New Zealand",
            (
                _r("emergency", "Emergency services", "call 111"),
                _r("crisis", "Need to Talk?", "call or text 1737"),
                _r("abuse", "Women's Refuge", "call 0800 733 843"),
            ),
        ),
        Region(
            "DE",
            "Germany",
            (
                _r("emergency", "Emergency services", "call 112"),
                _r("crisis", "TelefonSeelsorge", "call 0800 111 0 111 or 0800 111 0 222"),
                _r("abuse", "Hilfetelefon Gewalt gegen Frauen", "call 116 016"),
            ),
        ),
        Region(
            "FR",
            "France",
            (
                _r("emergency", "Emergency services", "call 112"),
                _r("crisis", "Numéro national de prévention du suicide", "call 3114"),
                _r("abuse", "Violences Femmes Info", "call 3919"),
            ),
        ),
        Region(
            "NL",
            "the Netherlands",
            (
                _r("emergency", "Emergency services", "call 112"),
                _r("crisis", "113 Zelfmoordpreventie", "call 113 or 0800-0113"),
                _r("abuse", "Veilig Thuis", "call 0800-2000"),
            ),
        ),
        Region(
            "IN",
            "India",
            (
                _r("emergency", "Emergency services", "call 112"),
                _r("crisis", "Tele-MANAS", "call 14416 or 1-800-891-4416"),
            ),
        ),
    )
}

# What people actually type. Kept to unambiguous names; anything else falls through to
# "no numbers on file", which still prints the general advice.
_ALIASES: dict[str, str] = {
    "UK": "GB",
    "ENGLAND": "GB",
    "SCOTLAND": "GB",
    "WALES": "GB",
    "NORTHERN IRELAND": "GB",
    "BRITAIN": "GB",
    "GREAT BRITAIN": "GB",
    "UNITED KINGDOM": "GB",
    "USA": "US",
    "UNITED STATES": "US",
    "AMERICA": "US",
    "IRELAND": "IE",
    "AUSTRALIA": "AU",
    "NEW ZEALAND": "NZ",
    "AOTEAROA": "NZ",
    "GERMANY": "DE",
    "DEUTSCHLAND": "DE",
    "FRANCE": "FR",
    "NETHERLANDS": "NL",
    "THE NETHERLANDS": "NL",
    "HOLLAND": "NL",
    "INDIA": "IN",
    "CANADA": "CA",
}

_REGION: ContextVar[str | None] = ContextVar("confidant_region", default=None)


def normalize_region(raw: str | None) -> str | None:
    """``"uk"`` -> ``"GB"``; blank -> ``None``. Unknown codes come back upper-cased."""
    if raw is None:
        return None
    text = " ".join(raw.replace("_", " ").replace(".", "").split()).upper()
    if not text:
        return None
    return _ALIASES.get(text, text)


def current_region() -> str | None:
    """The region set for this run by :func:`using_region`, else ``CONFIDANT_REGION``."""
    return normalize_region(_REGION.get() or os.getenv(REGION_ENV))


@contextmanager
def using_region(region: str | None) -> Iterator[None]:
    """Route every notice built inside the block to ``region`` (``None`` defers to env).

    A context variable rather than a parameter, because notices are built deep inside
    the profile, timeline, nudge, and draft code, and threading a region through all of
    them would put a safety concern into every signature for one lookup.
    """
    token = _REGION.set(region)
    try:
        yield
    finally:
        _REGION.reset(token)


class Routing(BaseModel):
    """The numbers attached to one notice, or a note saying why there are none."""

    region: str | None = Field(default=None, description="The region code routed to.")
    region_name: str | None = None
    resources: list[Resource] = Field(default_factory=list)
    note: str | None = Field(
        default=None, description="Shown when there are no numbers: how to get them."
    )

    def heading_and_numbers(self) -> list[str]:
        """``["In Canada:", "  Emergency services: call 911.", ...]``, or ``[]``."""
        if not self.resources:
            return []
        return [f"In {self.region_name}:", *(f"  {r.to_text()}" for r in self.resources)]


def route(kinds: Iterable[Kind], region: str | None = None) -> Routing | None:
    """The resources of ``kinds`` for ``region`` (default: :func:`current_region`).

    Returns ``None`` when no kinds are asked for, so a notice that needs no numbers
    carries no routing at all. Emergency services are always listed alongside a crisis
    or abuse line: someone reading for one of those may need the other within the hour.
    """
    wanted = set(kinds)
    if not wanted:
        return None
    wanted.add("emergency")
    code = normalize_region(region) if region is not None else current_region()

    if code is None:
        return Routing(
            note=(
                "To list emergency and support numbers here, pass --region with your "
                f"country (US, GB, AU, ...) or set {REGION_ENV}. {DIRECTORY}"
            )
        )
    known = REGIONS.get(code)
    if known is None:
        return Routing(
            region=code, note=f"Confidant has no numbers on file for {code}. {DIRECTORY}"
        )

    resources = [r for r in known.resources if r.kind in wanted]
    # A gap in the table (no national line of that kind) is said out loud, so an
    # emergency number on its own does not read as "that is all there is".
    missing = wanted - {r.kind for r in resources}
    return Routing(
        region=code,
        region_name=known.name,
        resources=resources,
        note=DIRECTORY if missing else None,
    )
