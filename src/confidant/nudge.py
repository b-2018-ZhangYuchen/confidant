"""`confidant nudge`: who is worth a message today, and why.

The arithmetic lives in :mod:`confidant.contact`. This is how it is put in front of the
owner, and the choices here are about reading order, not scoring:

* A danger-tier notice comes before anything else, as it does in the profile and the
  timeline. A digest that opened with "Ada: your turn" and mentioned the threat from
  Casey third would be ordering by timing something that is not about timing.
* Only the threads worth a message get their reasons. Everyone else gets one line, so
  the digest stays short enough to read each morning; ``nudge NAME`` gives anyone's in
  full.
* The priority is never printed. It only puts the suggestions in order.

Nothing here touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from confidant.contact import Contact, ContactState, assess_everyone, assess_person
from confidant.store import Store

__all__ = ["CAVEAT", "Nudge", "build_nudge"]

CAVEAT = (
    "This goes by the timing of saved messages alone. It cannot see calls, plans made\n"
    "in person, or other apps, and whether to write is yours to decide."
)


@dataclass(frozen=True, slots=True)
class Nudge:
    contacts: list[Contact]
    """Most worth a message first, as :func:`confidant.contact.assess_everyone` orders."""

    single: bool = False
    """Whether this is ``nudge NAME``, which shows one person in full."""

    @property
    def safety(self) -> list[Contact]:
        return [c for c in self.contacts if c.state is ContactState.SAFETY]

    @property
    def suggested(self) -> list[Contact]:
        return [c for c in self.contacts if c.suggested]

    @property
    def others(self) -> list[Contact]:
        return [c for c in self.contacts if not c.suggested and c.state is not ContactState.SAFETY]

    def _caveat(self) -> list[str]:
        # Said only when there was timing to go on: "nothing saved yet" needs no caveat.
        if any(c.silence is not None for c in self.contacts):
            return ["", CAVEAT]
        return []

    def to_text(self) -> str:
        if not self.contacts:
            return "No one saved yet. Add someone with: confidant add NAME [TRANSCRIPT ...]"
        if self.single:
            return "\n".join([self.contacts[0].to_text(), *self._caveat()])

        out: list[str] = []
        for contact in self.safety:
            out += [contact.to_text(), ""]

        suggested = self.suggested
        if suggested:
            out.append("Worth a message today")
            for contact in suggested:
                out += ["", contact.to_text()]
        else:
            out.append("No one is due a message today.")

        others = self.others
        if others:
            out += ["", "Not today"]
            out += [f"  {contact.headline()}" for contact in others]
            out += ["", "'confidant nudge NAME' shows the reasons for anyone."]
        return "\n".join(out + self._caveat())


def build_nudge(store: Store, name: str | None = None, *, at: datetime | None = None) -> Nudge:
    """The digest for everyone saved, or for one person, now or as things stood ``at``.

    Local only.
    """
    # A moment the owner picked is "as things stood then", so later messages are left
    # out. The real clock is not: an export a few hours ahead of it should still count.
    now, until = (at, at) if at is not None else (datetime.now(), None)
    if name is not None:
        return Nudge([assess_person(store, name, now=now, until=until)], single=True)
    return Nudge(assess_everyone(store, now=now, until=until))
