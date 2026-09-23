"""Command line entry point.

confidant stats    examples/sample_chat.txt      # local, no API call
confidant analyze  examples/sample_chat.txt      # calls Claude
confidant flags    examples/sample_chat.txt      # calls Claude
"""

from __future__ import annotations

import argparse
import sys

import anthropic

from confidant import __version__
from confidant.analysis.flags import analyze_flags
from confidant.analysis.personality import analyze_personality
from confidant.client import ModelRefusal
from confidant.config import ConfigError, Settings
from confidant.ingest.transcript import TranscriptError, read_transcript
from confidant.models import Conversation, Role


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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="confidant",
        description="A second opinion on the people you are dating.",
    )
    parser.add_argument("--version", action="version", version=f"confidant {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("stats", "Show conversation statistics. Runs locally, no API call."),
        ("analyze", "Ask Claude for a read on how the other person comes across."),
        ("flags", "Ask Claude to check the other person's messages for red flags."),
    ):
        sub = subcommands.add_parser(name, help=help_text, description=help_text)
        sub.add_argument("transcript", help="Path to a transcript file.")
        sub.add_argument("--owner", help="Your name in the transcript.")
        sub.add_argument("--match", help="Their name in the transcript.")

    for name in ("analyze", "flags"):
        subcommands.choices[name].add_argument(
            "--json", action="store_true", help="Print the raw report as JSON."
        )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    try:
        conversation = read_transcript(args.transcript, owner=args.owner, match=args.match)
    except TranscriptError as exc:
        print(f"confidant: {exc}", file=sys.stderr)
        return 2

    if args.command == "stats":
        print(_describe(conversation))
        return 0

    try:
        settings = Settings.from_env()
        analyze = analyze_flags if args.command == "flags" else analyze_personality
        report = analyze(conversation, settings=settings)
    except ConfigError as exc:
        print(f"confidant: {exc}", file=sys.stderr)
        return 2
    except ModelRefusal as exc:
        print(f"confidant: {exc}", file=sys.stderr)
        return 3
    except anthropic.APIError as exc:
        print(f"confidant: the Anthropic API call failed: {exc}", file=sys.stderr)
        return 4
    except ValueError as exc:
        print(f"confidant: {exc}", file=sys.stderr)
        return 2

    print(report.model_dump_json(indent=2) if args.json else report.to_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
