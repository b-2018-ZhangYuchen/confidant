"""Adapters for the files chat apps actually export.

Nobody hand-copies three months of WhatsApp into Confidant's format, so ``read_export``
recognizes the common exports and turns them into the same
:class:`~confidant.models.Conversation` the plain-text parser makes:

* **WhatsApp** "Export chat" (without media), from iOS or Android, in any of the date
  layouts WhatsApp uses: ``[02/03/2026, 19:04:11] Robin: hey`` or
  ``2/3/26, 7:04 PM - Robin: hey``.
* **iMessage**, as the TXT output of ``imessage-exporter``, since Messages has no export
  of its own: a date line, a sender line (``Me`` for you), then the message.
* **Hinge**'s data download (``matches.json``) is recognized but not read, because it
  holds only the messages you sent. Every read Confidant makes is of the other person, and
  half a conversation would be read as someone who never answered.

What an export carries besides messages — media placeholders, deleted messages,
encryption notices, tapbacks — is left out and counted, so the owner is told what was
dropped rather than finding out from a read that never mentions the photo.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from confidant.ingest.transcript import TranscriptError, parse_transcript
from confidant.models import Conversation, Message, Role

__all__ = ["DateOrder", "ExportFormat", "Imported", "parse_export", "read_export"]


class ExportFormat(StrEnum):
    CONFIDANT = "confidant"
    WHATSAPP = "whatsapp"
    IMESSAGE = "imessage"
    HINGE = "hinge"

    @property
    def label(self) -> str:
        return {
            ExportFormat.CONFIDANT: "a Confidant transcript",
            ExportFormat.WHATSAPP: "a WhatsApp export",
            ExportFormat.IMESSAGE: "an iMessage export",
            ExportFormat.HINGE: "a Hinge data download",
        }[self]


class DateOrder(StrEnum):
    DMY = "dmy"
    MDY = "mdy"


_LEFT_OUT_WORDS = {
    "media": ("media message", "media messages"),
    "deleted": ("deleted message", "deleted messages"),
    "system": ("app notice", "app notices"),
    "tapback": ("tapback", "tapbacks"),
}


@dataclass(slots=True)
class Imported:
    """A conversation, which format it came from, and what was dropped on the way."""

    conversation: Conversation
    format: ExportFormat
    left_out: Counter[str] = field(default_factory=Counter)

    def note(self) -> str | None:
        """One line for the owner about how the file was read, or ``None`` for our own format."""
        if self.format is ExportFormat.CONFIDANT:
            return None
        n = len(self.conversation)
        line = f"read as {self.format.label}, {n} message{'' if n == 1 else 's'}"
        parts = []
        for kind, (one, many) in _LEFT_OUT_WORDS.items():
            if self.left_out[kind]:
                parts.append(f"{self.left_out[kind]} {one if self.left_out[kind] == 1 else many}")
        if parts:
            line += "; left out " + _join(parts)
        return line + "."


def _join(parts: list[str]) -> str:
    return parts[0] if len(parts) == 1 else ", ".join(parts[:-1]) + " and " + parts[-1]


# Exports are full of direction marks and no-break spaces that look like nothing on screen
# and break every pattern below.
_INVISIBLE = str.maketrans(
    {"\u200e": None, "\u200f": None, "\ufeff": None, "\u202f": " ", "\xa0": " "}
)


def _clean(text: str) -> str:
    return text.translate(_INVISIBLE)


# -- WhatsApp -----------------------------------------------------------------

# The comma after the date is what tells this apart from Confidant's own
# "[03/02/2026 19:04] Robin:" lines; every WhatsApp layout has it.
_WA_HEADER = re.compile(
    r"^\[?(?P<a>\d{1,2})[./-](?P<b>\d{1,2})[./-](?P<y>\d{2,4}),\s*"
    r"(?P<h>\d{1,2})[:.](?P<m>\d{2})(?:[:.](?P<s>\d{2}))?"
    r"(?:\s*(?P<ampm>[AaPp])\.?\s?[Mm]\.?)?"
    r"(?:\]\s*|\s+-\s+)(?P<rest>.*)$"
)
_WA_SENDER = re.compile(r"^(?P<sender>[^:]{1,60}?):\s(?P<text>.*)$|^(?P<bare>[^:]{1,60}?):$")
_WA_MEDIA = re.compile(
    r"^(?:<media omitted>|<attached: .+>|.+\.\w{2,4} \(file attached\)|"
    r"(?:.+ )?(?:image|video|audio|sticker|gif|document|contact card) omitted)$",
    re.IGNORECASE,
)
_WA_DELETED = re.compile(
    r"^(?:this message was deleted|you deleted this message)\.?$", re.IGNORECASE
)
# iOS files the app's own notices under the other person's name, so they arrive looking
# like messages; read as theirs, an encryption notice would be the first thing they "said".
_WA_NOTICE = re.compile(
    r"^(?:messages and calls are end-to-end encrypted|messages to this chat and calls are "
    r"now secured|your security code with .+ changed|.+ changed (?:their|your) phone number|"
    r"(?:missed )?(?:voice|video) call\b|.+ turned (?:on|off) disappearing messages|"
    r"disappearing messages|you (?:un)?blocked this contact)",
    re.IGNORECASE,
)
_WA_EDITED = re.compile(r"\s*<this message was edited>$", re.IGNORECASE)


def _year(raw: str) -> int:
    year = int(raw)
    return year + 2000 if year < 100 else year


def _wa_dates(headers: list[re.Match[str]], order: DateOrder | None) -> list[datetime]:
    """Work out the date layout and parse every header's timestamp.

    WhatsApp writes dates in the phone's locale with nothing to say which one, so
    01/02/2026 may be January or February. A day over 12 settles it; failing that, only
    one reading may keep the messages in order. If both do and they disagree, guessing
    would put every message in the wrong month, so the owner is asked.
    """

    def build(order: DateOrder) -> list[datetime] | None:
        stamps = []
        for h in headers:
            day, month = (h["a"], h["b"]) if order is DateOrder.DMY else (h["b"], h["a"])
            hour = int(h["h"])
            if h["ampm"]:
                hour = hour % 12 + (12 if h["ampm"].lower() == "p" else 0)
            try:
                stamps.append(
                    datetime(
                        _year(h["y"]), int(month), int(day), hour, int(h["m"]), int(h["s"] or 0)
                    )
                )
            except ValueError:
                return None
        return stamps

    if order is not None:
        stamps = build(order)
        if stamps is None:
            raise TranscriptError(
                f"These dates do not all read as {order.value}; try the other --dates order"
            )
        return stamps

    readings = {o: build(o) for o in DateOrder}
    readings = {o: s for o, s in readings.items() if s is not None}
    if len(readings) > 1:
        in_order = {o: s for o, s in readings.items() if s == sorted(s)}
        readings = in_order or readings
    if not readings:
        raise TranscriptError("The dates in this WhatsApp export do not read as any date layout")
    distinct = {tuple(s) for s in readings.values()}
    if len(distinct) > 1:
        raise TranscriptError(
            "Cannot tell whether the dates in this WhatsApp export are day-first or "
            "month-first. Pass --dates dmy (e.g. 02/03 is 2 March) or --dates mdy."
        )
    return next(iter(readings.values()))


def _parse_whatsapp(text: str, order: DateOrder | None) -> tuple[list[_Raw], Counter[str]]:
    left_out: Counter[str] = Counter()
    headers: list[re.Match[str]] = []
    # (header index, sender or None for a notice, body lines)
    entries: list[tuple[int, str | None, list[str]]] = []

    for raw_line in _clean(text).splitlines():
        header = _WA_HEADER.match(raw_line.strip())
        if header is None:
            # Multi-line messages arrive as bare lines after the first.
            if entries and raw_line.strip():
                entries[-1][2].append(raw_line.strip())
            continue
        headers.append(header)
        said = _WA_SENDER.match(header["rest"])
        if said is None:
            entries.append((len(headers) - 1, None, []))
        else:
            sender = (said["sender"] or said["bare"]).strip()
            entries.append((len(headers) - 1, sender, [said["text"] or ""]))

    stamps = _wa_dates(headers, order)
    messages: list[_Raw] = []
    for index, sender, lines in entries:
        if sender is None:
            left_out["system"] += 1
            continue
        body = _WA_EDITED.sub("", "\n".join(lines)).strip()
        if _WA_NOTICE.match(body):
            left_out["system"] += 1
        elif not body or _WA_MEDIA.match(body):
            left_out["media"] += 1
        elif _WA_DELETED.match(body):
            left_out["deleted"] += 1
        else:
            messages.append(_Raw(sender, body, stamps[index]))
    return messages, left_out


# -- iMessage (imessage-exporter TXT) ------------------------------------------

_IM_DATE = re.compile(
    r"^(?P<stamp>[A-Z][a-z]{2} \d{1,2}, \d{4}\s+\d{1,2}:\d{2}:\d{2}\s?[AP]M)(?:\s+\(.*\))?$"
)
_IM_ATTACHMENT = re.compile(
    r"^(?:attachment missing!|sticker from .+|.*\.(?:jpe?g|png|heic|heif|gif|webp|tiff?|mov|"
    r"mp4|m4v|caf|m4a|mp3|amr|pdf|vcf|pluginpayloadattachment))$",
    re.IGNORECASE,
)
_IM_ME = "Me"


def _parse_imessage(text: str) -> tuple[list[_Raw], Counter[str]]:
    left_out: Counter[str] = Counter()
    # Replies are indented under the message they answer; flattening them keeps the
    # order they were sent in, which is what every read cares about.
    lines = [line.strip() for line in _clean(text).splitlines()]
    starts = [
        i
        for i, line in enumerate(lines)
        if _IM_DATE.match(line) and i + 1 < len(lines) and lines[i + 1]
    ]
    messages: list[_Raw] = []
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        stamp = datetime.strptime(
            " ".join(_IM_DATE.match(lines[start])["stamp"].split()), "%b %d, %Y %I:%M:%S %p"
        )
        sender = lines[start + 1]
        body: list[str] = []
        in_tapbacks = False
        for line in lines[start + 2 : end]:
            if line == "Tapbacks:":
                in_tapbacks = True
                continue
            if in_tapbacks:
                if line:
                    left_out["tapback"] += 1
                    continue
                in_tapbacks = False
            if _IM_ATTACHMENT.match(line):
                left_out["media"] += 1
                continue
            body.append(line)
        text_body = "\n".join(body).strip()
        if not text_body:
            # A block with no text of its own was an attachment, or an announcement
            # ("You named the conversation ...") with no sender line under the date.
            if not any(lines[start + 2 : end]):
                left_out["system"] += 1
            continue
        messages.append(_Raw(sender, text_body, stamp))
    return messages, left_out


# -- Hinge ----------------------------------------------------------------------


def _hinge_refusal(data: list[dict], path: str) -> TranscriptError:
    chats = [m.get("chats") or [] for m in data]
    sent = sum(len(c) for c in chats)
    with_chat = sum(1 for c in chats if c)
    return TranscriptError(
        f"{path} is a Hinge data download. Hinge leaves the other person's messages out of "
        f"it, so it holds only the {sent} message{'' if sent == 1 else 's'} you sent, to "
        f"{with_chat} match{'' if with_chat == 1 else 'es'}, and Confidant has nothing of "
        "theirs to read. Copy the conversation from the app into a transcript instead "
        "('confidant stats --help' shows the format)."
    )


def _is_hinge(data: object) -> bool:
    keys = {"chats", "match", "like", "we_met", "block"}
    return (
        isinstance(data, list)
        and bool(data)
        and all(isinstance(m, dict) for m in data)
        and any(keys & m.keys() for m in data)
    )


# -- shared ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Raw:
    sender: str
    text: str
    timestamp: datetime | None


def detect_format(text: str) -> ExportFormat:
    """Which format ``text`` is in, from its first non-blank line."""
    stripped = _clean(text).lstrip()
    if stripped.startswith("["):
        # Could be a Hinge JSON list, or a bracketed transcript line.
        try:
            if _is_hinge(json.loads(stripped)):
                return ExportFormat.HINGE
        except ValueError:
            pass
    first = next((line.strip() for line in stripped.splitlines() if line.strip()), "")
    if _WA_HEADER.match(first):
        return ExportFormat.WHATSAPP
    if _IM_DATE.match(first):
        return ExportFormat.IMESSAGE
    return ExportFormat.CONFIDANT


def _assemble(
    raw: list[_Raw],
    fmt: ExportFormat,
    *,
    owner: str | None,
    match: str | None,
    source: str | None,
) -> Conversation:
    if not raw:
        raise TranscriptError(f"This looks like {fmt.label}, but it has no text messages in it")
    senders: list[str] = []
    for message in raw:
        if message.sender not in senders:
            senders.append(message.sender)

    if fmt is ExportFormat.IMESSAGE:
        # The exporter writes "Me" for every message you sent, so whose side is whose is
        # known without asking; --owner only says what to call you.
        owner_sender = _IM_ME
        owner_name = owner or _IM_ME
    else:
        if owner is None:
            raise TranscriptError(
                f"This is {fmt.label}. Say which speaker is you with --owner NAME "
                f"(speakers found: {senders})"
            )
        owner_sender = owner
        owner_name = owner

    others = [s for s in senders if s.casefold() != owner_sender.casefold()]
    if len(senders) == len(others) and fmt is not ExportFormat.IMESSAGE:
        raise TranscriptError(f"{owner!r} does not speak in this export (speakers: {senders})")
    if len(others) > 1:
        raise TranscriptError(
            "Confidant handles one-on-one conversations, but this export has more than two "
            f"people in it: {senders}. Export the one-on-one chat instead of a group."
        )
    if not others:
        raise TranscriptError(f"Only {owner_name!r} speaks in this export")

    messages = [
        Message(
            role=Role.OWNER if m.sender.casefold() == owner_sender.casefold() else Role.MATCH,
            text=m.text,
            sender=m.sender,
            timestamp=m.timestamp,
        )
        for m in raw
    ]
    return Conversation(
        match_name=match or others[0],
        owner_name=owner_name,
        messages=messages,
        source=source,
    )


def parse_export(
    text: str,
    *,
    owner: str | None = None,
    match: str | None = None,
    source: str | None = None,
    dates: DateOrder | str | None = None,
) -> Imported:
    """Parse ``text`` in whichever supported format it is in.

    ``dates`` settles whether a WhatsApp export's dates are day- or month-first when the
    export itself cannot; it is ignored for every other format.
    """
    fmt = detect_format(text)
    if fmt is ExportFormat.CONFIDANT:
        conversation = parse_transcript(text, owner=owner, match=match, source=source)
        return Imported(conversation, fmt)
    if fmt is ExportFormat.HINGE:
        raise _hinge_refusal(json.loads(_clean(text)), source or "This file")
    if fmt is ExportFormat.WHATSAPP:
        raw, left_out = _parse_whatsapp(text, DateOrder(dates) if dates else None)
    else:
        raw, left_out = _parse_imessage(text)
    conversation = _assemble(raw, fmt, owner=owner, match=match, source=source)
    return Imported(conversation, fmt, left_out)


def read_export(
    path: str | Path,
    *,
    owner: str | None = None,
    match: str | None = None,
    dates: DateOrder | str | None = None,
) -> Imported:
    """Read a transcript or a chat app's export from ``path``."""
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise TranscriptError(f"No transcript at {path}") from exc
    except IsADirectoryError as exc:
        raise TranscriptError(f"{path} is a folder; give the path to one transcript file") from exc
    except PermissionError as exc:
        raise TranscriptError(f"Not allowed to read {path}") from exc
    except UnicodeDecodeError as exc:
        # WhatsApp's export is a zip when media is included; the chat is the .txt inside.
        hint = " If it is a zip, unzip it and give the .txt inside." if _is_zip(path) else ""
        raise TranscriptError(
            f"{path} is not UTF-8 text. If it is a chat export, re-save it as UTF-8 plain "
            f"text.{hint}"
        ) from exc
    return parse_export(text, owner=owner, match=match, source=str(path), dates=dates)


def _is_zip(path: Path) -> bool:
    try:
        with path.open("rb") as f:
            return f.read(4) == b"PK\x03\x04"
    except OSError:
        return False
