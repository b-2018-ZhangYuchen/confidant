"""Recorded model responses, so the analysis code can be tested without the network.

Monkeypatching ``structured_call`` away tests the code on either side of the model but
skips the part in the middle: the refusal check, the schema validation, the handling of a
response that stopped early. A recording sits one layer lower, where the SDK would be, so
everything Confidant does with a response runs for real in a test.

A recording is one JSON file holding one model response and a fingerprint of the request
that produced it: the schema name and SHA-256 hashes of the system prompt and the
messages. Replaying it against a request whose fingerprint differs is an error, not a
warning. A prompt edit that silently kept passing against a response written for the old
prompt would make the whole suite a record of how things used to be.

Two kinds of recording live in ``tests/fixtures/recorded``:

* ``recorded`` — captured from a live call with ``python -m confidant.recording record``.
  When the prompt changes, the only honest fix is to record again.
* ``hand-written`` — written by a person to pin down a shape the model can produce (a
  refusal, a misquote) or because no API key was available. These can be re-stamped with
  ``python -m confidant.recording stamp`` after someone has read the response and agrees
  it still fits the new prompt.

The request is stored only as hashes. The response still quotes the transcript, which is
why ``record`` only accepts transcripts from ``examples/``: a recording is committed to
the repository, and nothing real belongs there.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

from pydantic import BaseModel

from confidant.analysis.flags import analyze_flags
from confidant.analysis.personality import analyze_personality
from confidant.client import ModelRefusal, build_client
from confidant.config import Settings
from confidant.ingest.transcript import read_transcript

__all__ = [
    "ANALYSES",
    "FORMAT_VERSION",
    "Fingerprint",
    "Recording",
    "RecordingClient",
    "RecordingError",
    "ReplayClient",
    "StaleRecording",
    "replay",
    "run_analysis",
]

FORMAT_VERSION = 1

Provenance = Literal["recorded", "hand-written"]


class RecordingError(RuntimeError):
    """A recording is missing, malformed, or used in a way it cannot support."""


class StaleRecording(RecordingError):
    """The request no longer matches the one the recording was made for."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Fingerprint:
    """What identifies a request, without storing any of its text."""

    schema: str
    system_sha256: str
    messages_sha256: str

    @classmethod
    def of(cls, request: dict[str, Any]) -> Fingerprint:
        """Fingerprint the keyword arguments of a ``messages.parse`` call."""
        schema = request["output_format"]
        return cls(
            schema=schema.__name__,
            system_sha256=_sha256(request["system"]),
            # sort_keys so that a dict built in a different order is the same request.
            messages_sha256=_sha256(json.dumps(request["messages"], sort_keys=True)),
        )

    def differences(self, other: Fingerprint) -> list[str]:
        """Plain descriptions of what changed between ``self`` and ``other``."""
        changed = []
        if self.schema != other.schema:
            changed.append(f"the schema is {other.schema}, not {self.schema}")
        if self.system_sha256 != other.system_sha256:
            changed.append("the system prompt changed")
        if self.messages_sha256 != other.messages_sha256:
            changed.append("the request (transcript or instructions) changed")
        return changed


@dataclass(slots=True)
class Recording:
    """One model response, and the request it answers."""

    fingerprint: Fingerprint
    stop_reason: str
    output: dict[str, Any] | None
    stop_details: dict[str, Any] | None = None
    provenance: Provenance = "recorded"
    model: str | None = None
    source: dict[str, str] = field(default_factory=dict)
    """How to make this recording again: ``{"analysis": ..., "transcript": ...}``."""

    note: str | None = None
    path: Path | None = None

    # -- files ---------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> Recording:
        path = Path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise RecordingError(f"No recording at {path}") from exc
        except json.JSONDecodeError as exc:
            raise RecordingError(f"{path} is not valid JSON: {exc}") from exc

        if data.get("format") != FORMAT_VERSION:
            raise RecordingError(
                f"{path} is format {data.get('format')!r}; this version reads {FORMAT_VERSION}"
            )
        try:
            request, response = data["request"], data["response"]
            return cls(
                fingerprint=Fingerprint(
                    schema=request["schema"],
                    system_sha256=request["system_sha256"],
                    messages_sha256=request["messages_sha256"],
                ),
                stop_reason=response["stop_reason"],
                output=response.get("output"),
                stop_details=response.get("stop_details"),
                provenance=data["provenance"],
                model=request.get("model"),
                source=data.get("source", {}),
                note=data.get("note"),
                path=path,
            )
        except KeyError as exc:
            raise RecordingError(f"{path} is missing the field {exc}") from exc

    def to_json(self) -> str:
        data: dict[str, Any] = {"format": FORMAT_VERSION, "provenance": self.provenance}
        if self.note:
            data["note"] = self.note
        if self.source:
            data["source"] = self.source
        data["request"] = {
            "schema": self.fingerprint.schema,
            "model": self.model,
            "system_sha256": self.fingerprint.system_sha256,
            "messages_sha256": self.fingerprint.messages_sha256,
        }
        data["response"] = {
            "stop_reason": self.stop_reason,
            "stop_details": self.stop_details,
            "output": self.output,
        }
        # ensure_ascii off so a quote with a curly apostrophe reads as the model wrote it.
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")
        self.path = path
        return path

    # -- replay --------------------------------------------------------------

    def response_for(self, schema: type[BaseModel]) -> SimpleNamespace:
        """Rebuild the parts of an SDK ``ParsedMessage`` that ``structured_call`` reads.

        The output is validated against the schema *now*, not trusted as it was stored,
        so a schema change that the recorded response no longer satisfies fails here the
        way a live response would.
        """
        parsed = schema.model_validate(self.output) if self.output is not None else None
        details = SimpleNamespace(**self.stop_details) if self.stop_details else None
        return SimpleNamespace(
            stop_reason=self.stop_reason,
            stop_details=details,
            parsed_output=parsed,
            model=self.model,
        )

    def _describe(self) -> str:
        return str(self.path) if self.path else "the recording"

    def _remedy(self) -> str:
        where = self.path or "<recording>"
        analysis = self.source.get("analysis", "<analysis>")
        transcript = self.source.get("transcript", "<transcript>")
        fix = (
            f"Record it again with `python -m confidant.recording record {analysis} "
            f"{transcript} {where}` (needs an API key)"
        )
        if self.provenance == "hand-written":
            fix += (
                f", or, since it was written by hand, read the response against the new "
                f"prompt and then run `python -m confidant.recording stamp {where}`"
            )
        return fix + "."


# -- clients ------------------------------------------------------------------


class ReplayClient:
    """Stands in for ``anthropic.Anthropic``, answering from recordings in order.

    Each ``messages.parse`` call consumes the next recording. A call with no recording
    left, or one whose fingerprint does not match, is an error: the point is to notice
    when the code starts asking the model something different. ``remaining`` shows the
    opposite case, a recording that nothing asked for.
    """

    def __init__(self, *recordings: Recording | str | Path, strict: bool = True) -> None:
        self._queue = [r if isinstance(r, Recording) else Recording.load(r) for r in recordings]
        self._strict = strict
        self.requests: list[dict[str, Any]] = []
        self.messages = SimpleNamespace(parse=self._parse)

    @property
    def remaining(self) -> int:
        return len(self._queue)

    def _parse(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        if not self._queue:
            raise RecordingError(
                f"Model call number {len(self.requests)} has no recording to answer it."
            )
        recording = self._queue.pop(0)
        actual = Fingerprint.of(request)
        changed = recording.fingerprint.differences(actual)
        if changed and (self._strict or recording.fingerprint.schema != actual.schema):
            raise StaleRecording(
                f"{recording._describe()} is stale: {'; '.join(changed)}. {recording._remedy()}"
            )
        return recording.response_for(request["output_format"])


class RecordingClient:
    """Wraps a real client and saves each response it gets as a recording."""

    def __init__(
        self,
        inner: Any,
        path: str | Path,
        *,
        source: dict[str, str] | None = None,
        note: str | None = None,
    ) -> None:
        self._inner = inner
        self._path = Path(path)
        self._source = source or {}
        self._note = note
        self.saved: list[Path] = []
        self.messages = SimpleNamespace(parse=self._parse)

    def _parse(self, **request: Any) -> Any:
        response = self._inner.messages.parse(**request)
        parsed = getattr(response, "parsed_output", None)
        details = getattr(response, "stop_details", None)
        recording = Recording(
            fingerprint=Fingerprint.of(request),
            stop_reason=response.stop_reason,
            output=parsed.model_dump(mode="json") if parsed is not None else None,
            stop_details=_as_dict(details),
            provenance="recorded",
            model=getattr(response, "model", None) or request.get("model"),
            source=self._source,
            note=self._note,
        )
        # A second call in one run gets its own numbered file rather than overwriting the
        # first; today every analysis makes exactly one.
        path = self._path
        if self.saved:
            path = path.with_name(f"{path.stem}.{len(self.saved) + 1}{path.suffix}")
        self.saved.append(recording.save(path))
        return response


def _as_dict(details: Any) -> dict[str, Any] | None:
    if details is None:
        return None
    if isinstance(details, BaseModel):
        return details.model_dump(mode="json", exclude_none=True)
    if isinstance(details, dict):
        return details
    return {k: v for k, v in vars(details).items() if v is not None}


# -- replaying an analysis ----------------------------------------------------

ANALYSES: dict[str, Callable[..., Any]] = {
    "analyze": analyze_personality,
    "flags": analyze_flags,
}


def run_analysis(
    analysis: str,
    transcript: str | Path,
    *,
    client: Any,
    settings: Settings | None = None,
) -> Any:
    """Run one of the CLI's analyses against ``transcript`` with ``client``."""
    if analysis not in ANALYSES:
        raise RecordingError(f"Unknown analysis {analysis!r}; expected one of {list(ANALYSES)}")
    conversation = read_transcript(transcript)
    # Settings() rather than from_env(): a replay must not change with CONFIDANT_MODEL.
    return ANALYSES[analysis](conversation, settings=settings or Settings(), client=client)


def replay(recording: Recording | str | Path, *, strict: bool = True) -> Any:
    """Replay a recording through the analysis it came from, and return the report."""
    if not isinstance(recording, Recording):
        recording = Recording.load(recording)
    return _run_source(recording, ReplayClient(recording, strict=strict))


def _run_source(recording: Recording, client: ReplayClient) -> Any:
    if not {"analysis", "transcript"} <= recording.source.keys():
        raise RecordingError(f"{recording._describe()} does not say how it was made ('source')")
    return run_analysis(recording.source["analysis"], recording.source["transcript"], client=client)


# -- command line ---------------------------------------------------------------


def _check_fictional(transcript: Path) -> None:
    if "examples" not in transcript.resolve().parts:
        raise RecordingError(
            f"{transcript} is not under examples/. Recordings are committed to the "
            "repository and quote the transcript, so only fictional examples can be recorded."
        )


def _portable(transcript: Path) -> str:
    # Stored relative to the repository root, so the recording replays on any checkout.
    try:
        return transcript.resolve().relative_to(Path.cwd()).as_posix()
    except ValueError:
        return transcript.as_posix()


def _record(analysis: str, transcript: Path, out: Path, note: str | None) -> Path:
    _check_fictional(transcript)
    settings = Settings.from_env()
    recorder = RecordingClient(
        build_client(settings),
        out,
        source={"analysis": analysis, "transcript": _portable(transcript)},
        note=note,
    )
    try:
        run_analysis(analysis, transcript, client=recorder, settings=settings)
    except ModelRefusal as exc:
        # A refusal is worth keeping as a recording; it was saved before the raise.
        print(f"recorded a refusal: {exc}", file=sys.stderr)
    return out


def _stamp(path: Path) -> list[str]:
    """Re-fingerprint a hand-written recording against the current prompts."""
    recording = Recording.load(path)
    if recording.provenance != "hand-written":
        raise RecordingError(
            f"{path} was recorded from a live call. Its response answers the old prompt, so "
            "re-stamping it would make it claim something that never happened. Record it again."
        )
    client = ReplayClient(recording, strict=False)
    try:
        _run_source(recording, client)
    except RecordingError:
        raise
    except (ModelRefusal, RuntimeError):
        # A hand-written refusal or truncation raises by design, after the request was
        # made. A response that no longer fits the schema raises ValidationError instead,
        # and that should stop the stamp: the fix is to edit the response, not the hash.
        pass
    [request] = client.requests
    fresh = Fingerprint.of(request)
    changed = recording.fingerprint.differences(fresh)
    recording.fingerprint = fresh
    recording.save(path)
    return changed


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m confidant.recording",
        description="Make and maintain the recorded model responses the tests replay.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    record = sub.add_parser("record", help="Make a live call and save the response.")
    record.add_argument("analysis", choices=ANALYSES)
    record.add_argument("transcript", type=Path, help="A fictional transcript in examples/.")
    record.add_argument("out", type=Path, help="Where to write the recording.")
    record.add_argument("--note", help="Why this recording exists.")

    stamp = sub.add_parser(
        "stamp", help="Re-fingerprint a hand-written recording after a prompt change."
    )
    stamp.add_argument("recordings", type=Path, nargs="+")

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "record":
            print(f"wrote {_record(args.analysis, args.transcript, args.out, args.note)}")
        else:
            for path in args.recordings:
                changed = _stamp(path)
                print(f"{path}: {'; '.join(changed) if changed else 'already current'}")
    except Exception as exc:
        print(f"confidant.recording: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
