"""Command line entry point.

confidant stats    examples/sample_chat.txt      # local, no API call
confidant redact   examples/details_chat.txt     # local: what would be sent
confidant analyze  examples/sample_chat.txt      # calls Claude
confidant flags    examples/sample_chat.txt      # calls Claude

confidant add      Robin examples/sample_chat.txt  # local store, no API call
confidant list
confidant show     Robin
confidant remove   Robin
"""

from __future__ import annotations

import argparse
import contextlib
import sys
from datetime import datetime

import anthropic

from confidant import __version__
from confidant.analysis.flags import analyze_flags
from confidant.analysis.personality import analyze_personality
from confidant.client import IncompleteResponse, ModelRefusal
from confidant.config import DEFAULT_MAX_TOKENS, DEFAULT_MODEL, ConfigError, Settings
from confidant.ingest.transcript import TranscriptError, read_transcript
from confidant.models import Conversation, Role
from confidant.progress import StatusLine
from confidant.prompts.common import render_conversation
from confidant.redaction import redact
from confidant.store import (
    SaveOutcome,
    Store,
    StoredConversation,
    StoreError,
    UnknownPerson,
    name_key,
)


def _describe(conversation: Conversation) -> str:
    """Everything we can say about a conversation without calling a model."""
    lines = [
        f"{conversation.owner_name} <-> {conversation.match_name}",
        f"  messages         {len(conversation)} "
        f"({len(conversation.owner_messages)} you / {len(conversation.match_messages)} them)",
        f"  avg words (you)  {conversation.mean_words(Role.OWNER):.1f}",
        f"  avg words (them) {conversation.mean_words(Role.MATCH):.1f}",
    ]

    ratio = conversation.effort_ratio
    if ratio is not None:
        lines.append(f"  effort ratio     {ratio:.2f} (their words per word of yours)")

    span = conversation.timespan
    if span is not None:
        lines.append(f"  spans            {span.days} days")
    if conversation.last_activity is not None:
        lines.append(f"  last message     {conversation.last_activity:%Y-%m-%d %H:%M}")

    return "\n".join(lines)


def _preview(conversation: Conversation) -> str:
    """The transcript exactly as the analyses would send it, then the key to it."""
    redaction = redact(conversation)
    lines = render_conversation(redaction.conversation)
    lines += ["", f"Replaced: {redaction.summary()}.", "The key, which stays on your machine:"]
    width = max(len(p) for p in redaction.originals)
    for placeholder, original in redaction.originals.items():
        lines.append(f"  {placeholder:<{width}}  {original}")
    return "\n".join(lines)


def _plural(count: int, word: str) -> str:
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def _when(moment: datetime | None) -> str:
    return f"{moment:%Y-%m-%d %H:%M}" if moment is not None else "-"


def _people_table(store: Store) -> str:
    people = store.people()
    if not people:
        return "No one saved yet. Add someone with: confidant add NAME [TRANSCRIPT ...]"
    rows = [("NAME", "CONVERSATIONS", "MESSAGES", "LAST MESSAGE")] + [
        (p.person.name, str(p.conversations), str(p.messages), _when(p.last_message))
        for p in people
    ]
    widths = [max(len(row[i]) for row in rows) for i in range(3)]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row[:3], widths, strict=True))
        + "  "
        + row[3]
        for row in rows
    )


def _conversation_heading(stored: StoredConversation) -> str:
    source = stored.source or "(no source)"
    heading = f"#{stored.id}  {source}, saved {_when(stored.imported_at)}"
    if stored.updated_at is not None:
        heading += f", updated {_when(stored.updated_at)}"
    return heading


def _show_person(store: Store, name: str) -> str:
    person = store.person(name)
    conversations = store.conversations(person)
    lines = [
        f"{person.name}: added {person.added_at:%Y-%m-%d}, "
        f"{_plural(len(conversations), 'conversation')}"
    ]
    if not conversations:
        lines += ["", f"Nothing saved yet. Add a transcript with: confidant add {person.name} FILE"]
    for stored in conversations:
        lines += ["", _conversation_heading(stored), _describe(store.load_conversation(stored.id))]
    return "\n".join(lines)


def _add(store: Store, args: argparse.Namespace) -> str:
    # Parse everything before writing anything, so a typo in the third file does not
    # leave a person half-added with the first two.
    parsed = [
        read_transcript(path, owner=args.owner, match=args.match) for path in args.transcripts
    ]
    for conversation in parsed:
        # Saving a chat with Casey under Robin would quietly mix two people's histories,
        # which every later read would build on. An export that spells the name
        # differently is fine, but the owner has to say so.
        if args.match is None and name_key(conversation.match_name) != name_key(args.name):
            raise StoreError(
                f"{conversation.source} is a conversation with {conversation.match_name!r}, "
                f"not {args.name!r}. If that is how {args.name} appears in this transcript, "
                f"pass --match {conversation.match_name!r}."
            )

    lines = []
    person = store.find_person(args.name)
    if person is None:
        person = store.add_person(args.name)
        lines.append(f"Added {person.name}.")
    elif not parsed:
        raise StoreError(
            f"{person.name!r} is already saved. To add a conversation: "
            f"confidant add {person.name} FILE"
        )
    for conversation in parsed:
        result = store.save_conversation(person, conversation)
        stored = result.conversation
        if result.outcome is SaveOutcome.NEW:
            lines.append(
                f"Saved {conversation.source} as #{stored.id} "
                f"({_plural(stored.messages, 'message')})."
            )
        elif result.outcome is SaveOutcome.EXTENDED:
            lines.append(
                f"Added {_plural(result.added, 'new message')} from {conversation.source} "
                f"to #{stored.id} ({stored.messages} in all)."
            )
        else:
            lines.append(f"{conversation.source} is already saved, as #{stored.id}.")
    return "\n".join(lines)


def _remove(store: Store, args: argparse.Namespace) -> str | None:
    person = store.person(args.name)
    count = len(store.conversations(person))
    what = f"{person.name} and {_plural(count, 'conversation')}"
    if not args.yes:
        if not sys.stdin.isatty():
            raise StoreError(f"Forgetting {what} cannot be undone; pass --yes to confirm")
        answer = input(f"Forget {what}? This cannot be undone. [y/N] ")
        if answer.strip().lower() not in ("y", "yes"):
            return None
    store.remove_person(person.name)
    return f"Forgot {what}."


_STORE_COMMANDS = ("add", "list", "show", "remove")


def _run_store_command(args: argparse.Namespace) -> int:
    try:
        with Store() as store:
            if args.command == "add":
                out = _add(store, args)
            elif args.command == "list":
                out = _people_table(store)
            elif args.command == "show":
                out = _show_person(store, args.name)
            else:
                out = _remove(store, args)
    except UnknownPerson as exc:
        print(f"confidant: {exc}. 'confidant list' shows who is.", file=sys.stderr)
        return 2
    except (StoreError, TranscriptError) as exc:
        print(f"confidant: {exc}", file=sys.stderr)
        return 2
    except (KeyboardInterrupt, EOFError):
        print("\nconfidant: cancelled.", file=sys.stderr)
        return 130
    if out is None:
        print("Nothing removed.")
    else:
        print(out)
    return 0


_EPILOG = f"""\
examples:
  confidant stats examples/sample_chat.txt
  confidant redact examples/details_chat.txt
  confidant analyze examples/sample_chat.txt
  confidant flags examples/pressure_chat.txt --json
  confidant add Robin examples/sample_chat.txt
  confidant show Robin

stats and redact never touch the network, and neither do add, list, show, and
remove, which keep people and their conversations in a local file. analyze and
flags send the redacted transcript to the Anthropic API and nowhere else.

environment (or put these in .env):
  ANTHROPIC_API_KEY     needed for analyze and flags
  CONFIDANT_DB          the local store, default ~/.confidant/confidant.db
  CONFIDANT_MODEL       default {DEFAULT_MODEL}
  CONFIDANT_EFFORT      low, medium, high (default), xhigh, or max
  CONFIDANT_MAX_TOKENS  default {DEFAULT_MAX_TOKENS}; raise it if a report is cut off

exit codes:
  0    success
  2    the transcript or configuration needs fixing
  3    Claude declined the request
  4    the API call failed, or its answer was incomplete
  130  cancelled with Ctrl-C
"""

_FORMAT = """\
transcript format:
  # owner: Sam                            which speaker is you (or --owner)
  [2026-03-02 19:04] Robin: hey!          timestamps are optional
  Sam: hi, how was the ride?
      indented lines continue the message above
"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="confidant",
        description="A second opinion on the people you are dating.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"confidant {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    for name, help_text in (
        ("stats", "Show conversation statistics. Runs locally, no API call."),
        ("redact", "Show what would be sent to Claude, after redaction. Runs locally."),
        ("analyze", "Ask Claude for a read on how the other person comes across."),
        ("flags", "Ask Claude to check the other person's messages for red flags."),
    ):
        sub = subcommands.add_parser(
            name,
            help=help_text,
            description=help_text,
            epilog=_FORMAT,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        sub.add_argument("transcript", help="Path to a plain-text transcript file.")
        sub.add_argument(
            "--owner", metavar="NAME", help="Your name in the transcript. Overrides '# owner:'."
        )
        sub.add_argument(
            "--match", metavar="NAME", help="Their name in the transcript. Overrides '# match:'."
        )

    for name in ("redact", "analyze", "flags"):
        subcommands.choices[name].add_argument(
            "--redact",
            action="append",
            default=[],
            metavar="NAME",
            dest="private_names",
            help="Another person's name to strip before sending. Repeat for more than one.",
        )
    for name in ("analyze", "flags"):
        subcommands.choices[name].add_argument(
            "--json", action="store_true", help="Print the raw report as JSON."
        )
        subcommands.choices[name].add_argument(
            "--no-progress",
            action="store_false",
            dest="progress",
            help="Do not show the progress line while Claude works.",
        )

    add = subcommands.add_parser(
        "add",
        help="Save someone you are seeing, and optionally conversations with them.",
        description=(
            "Save someone you are seeing. Any transcripts given are saved under them; "
            "run it again with more transcripts to add to someone already saved. A newer "
            "export of a chat already saved adds only the messages after the saved ones. "
            "Runs locally."
        ),
        epilog=_FORMAT,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    add.add_argument("name", help="What you call them. Case and spacing do not matter.")
    add.add_argument("transcripts", nargs="*", metavar="TRANSCRIPT", help="Transcript files.")
    add.add_argument(
        "--owner", metavar="NAME", help="Your name in the transcripts. Overrides '# owner:'."
    )
    add.add_argument(
        "--match",
        metavar="NAME",
        help="Their name in the transcripts, if it is not the name being added.",
    )

    subcommands.add_parser(
        "list",
        help="List everyone saved, with how much is saved for each. Runs locally.",
        description="List everyone saved, with how much is saved for each. Runs locally.",
    )
    show = subcommands.add_parser(
        "show",
        help="Show someone saved and statistics for each conversation. Runs locally.",
        description="Show someone saved and statistics for each conversation. Runs locally.",
    )
    show.add_argument("name")
    remove = subcommands.add_parser(
        "remove",
        help="Forget someone and every conversation saved with them.",
        description=(
            "Forget someone and every conversation saved with them. Their messages are "
            "overwritten in the store, not just unlinked."
        ),
    )
    remove.add_argument("name")
    remove.add_argument("--yes", action="store_true", help="Do not ask for confirmation.")
    return parser


def _api_error(exc: anthropic.APIError, settings: Settings) -> str:
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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command in _STORE_COMMANDS:
        return _run_store_command(args)

    try:
        conversation = read_transcript(args.transcript, owner=args.owner, match=args.match)
    except TranscriptError as exc:
        print(f"confidant: {exc}", file=sys.stderr)
        return 2

    if args.command == "stats":
        print(_describe(conversation))
        return 0

    conversation.private_names.extend(args.private_names)
    if args.command == "redact":
        print(_preview(conversation))
        return 0

    try:
        settings = Settings.from_env()
        analyze = analyze_flags if args.command == "flags" else analyze_personality
        # Progress goes to stderr, and only to a terminal: piped into a file or another
        # program, a line that rewrites itself is just noise.
        status = StatusLine() if args.progress and sys.stderr.isatty() else None
        with status or contextlib.nullcontext():
            report = analyze(conversation, settings=settings, on_progress=status)
    except ConfigError as exc:
        print(f"confidant: {exc}", file=sys.stderr)
        return 2
    except ModelRefusal as exc:
        print(f"confidant: {exc}", file=sys.stderr)
        return 3
    except IncompleteResponse as exc:
        print(f"confidant: {exc}", file=sys.stderr)
        return 4
    except anthropic.APIError as exc:
        print(f"confidant: {_api_error(exc, settings)}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        print("confidant: cancelled.", file=sys.stderr)
        return 130
    except ValueError as exc:
        print(f"confidant: {exc}", file=sys.stderr)
        return 2

    print(report.model_dump_json(indent=2) if args.json else report.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
