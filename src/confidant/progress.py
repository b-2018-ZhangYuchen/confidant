"""Progress for a model call that takes a while, so the terminal is not blank meanwhile.

A careful read of a long transcript can take a minute or more. Streaming the request lets
Confidant say what stage it is at: sent and waiting, thinking, or writing a particular
part of the report.

What is streamed to the terminal is the *stage*, never the content. Every report is
checked before the owner sees it — quotes are grounded, severities floored, placeholders
swapped back, and a danger finding is put under a fixed safety notice — and a half-written
report has been through none of that. Printing it as it arrived would mean showing an
ungrounded flag, or the model's reasoning about someone, and then quietly withdrawing it.
So the status line names the section being written ("traits", "flags"), which is the same
for every conversation, and the report itself is printed only once it is whole.
"""

from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, TextIO

import jiter

__all__ = ["Phase", "Progress", "ProgressTracker", "StatusLine"]

Phase = Literal["waiting", "thinking", "writing"]


@dataclass(frozen=True, slots=True)
class Progress:
    """Where a model call has got to."""

    phase: Phase
    section: str | None = None
    """The top-level field of the report being written, e.g. ``"green_flags"``."""

    chars: int = 0
    """How much of the report has been written, in characters of JSON."""


class ProgressTracker:
    """Turns SDK stream events into :class:`Progress` updates.

    It reads three kinds of event and ignores the rest: a content block starting (which
    says whether the model is thinking or writing), a text delta with its accumulated
    snapshot, and the message stopping.
    """

    def __init__(self, on_progress: Callable[[Progress], None]) -> None:
        self._emit = on_progress
        self._last: Progress | None = None
        self.update(Progress("waiting"))

    def update(self, progress: Progress) -> None:
        # Only report changes, so a renderer is not redrawn on every token of a long field.
        if progress != self._last:
            self._last = progress
            self._emit(progress)

    def feed(self, event: Any) -> None:
        kind = getattr(event, "type", None)
        if kind == "content_block_start":
            block = getattr(event.content_block, "type", None)
            if block in ("thinking", "redacted_thinking"):
                self.update(Progress("thinking"))
            elif block == "text":
                self.update(Progress("writing"))
        elif kind == "text":
            snapshot = event.snapshot
            self.update(Progress("writing", current_section(snapshot), len(snapshot)))


def current_section(partial_json: str) -> str | None:
    """The last top-level key in a JSON object that is still being written."""
    try:
        value = jiter.from_json(partial_json.encode("utf-8"), partial_mode="trailing-strings")
    except ValueError:
        # Mid-way through a key or a number the prefix may not parse yet; the next delta
        # will. Not worth failing a report over.
        return None
    if isinstance(value, dict) and value:
        return next(reversed(value))
    return None


_PHASE_TEXT: dict[str, str] = {
    "waiting": "sent the redacted transcript, waiting for a reply",
    "thinking": "thinking it over",
    "writing": "writing the report",
}


class StatusLine:
    """A single self-rewriting line on stderr showing a :class:`Progress`.

    Use it as the ``on_progress`` callback, inside a ``with`` block. A background thread
    redraws it every ``interval`` seconds so the elapsed time keeps moving during a long
    think, when the stream itself can go quiet. Leaving the block erases the line, so the
    report that follows starts on a clean terminal.
    """

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        label: str = "confidant",
        clock: Callable[[], float] = time.monotonic,
        interval: float = 0.5,
    ) -> None:
        self._stream = stream if stream is not None else sys.stderr
        self._label = label
        self._clock = clock
        self._interval = interval
        self._started = clock()
        self._progress = Progress("waiting")
        self._width = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __call__(self, progress: Progress) -> None:
        with self._lock:
            self._progress = progress
            self._draw()

    def render(self) -> str:
        """The line as it would be drawn now."""
        progress = self._progress
        text = _PHASE_TEXT[progress.phase]
        if progress.phase == "writing" and progress.section:
            text += f": {progress.section.replace('_', ' ')}"
        elapsed = int(self._clock() - self._started)
        return f"{self._label}: {text}... {elapsed}s"

    def _draw(self) -> None:
        line = self.render()
        # Pad over the previous line rather than relying on an erase escape, so a
        # terminal without ANSI support still ends up with a clean line.
        padding = " " * max(0, self._width - len(line))
        self._stream.write(f"\r{line}{padding}")
        self._stream.flush()
        self._width = len(line)

    def _tick(self) -> None:
        while not self._stop.wait(self._interval):
            with self._lock:
                self._draw()

    def __enter__(self) -> StatusLine:
        self._started = self._clock()
        with self._lock:
            self._draw()
        self._thread = threading.Thread(target=self._tick, name="confidant-status", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
        with self._lock:
            self._stream.write("\r" + " " * self._width + "\r")
            self._stream.flush()
            self._width = 0
