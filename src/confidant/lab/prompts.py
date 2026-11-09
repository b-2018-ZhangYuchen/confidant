"""Prompt variants for the local red-flag check, compared in step 4.

Each variant builds the messages for one conversation. All of them start from Claude's
own system prompt, so a difference in score is a difference in what was added:

- ``schema`` — the JSON Schema spelled out (the default, what the backend uses).
- ``compact`` — a short field-by-field description instead of the full schema, for
  small models that get lost in a long one.
- ``fewshot`` — ``compact`` plus worked examples from the *train* split, as previous
  turns of the chat.
- ``rag`` — ``compact`` plus the pattern cards most relevant to this transcript
  (:mod:`confidant.lab.rag`).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from collections.abc import Callable
from functools import cache
from pathlib import Path

from confidant.lab.backend import Messages, build_local_messages
from confidant.models import Conversation
from confidant.prompts.flags import FLAGS_SYSTEM, build_flags_request
from confidant.redaction import redact

__all__ = ["COMPACT_NOTE", "PROMPTS", "compact_messages"]

NOT_A_FLAG = Path("lab/cards/none.md")

COMPACT_NOTE = """\
Answer with one JSON object and nothing else:
{"flags": [{"category": one of threat, coercion, sexual_pressure, financial_pressure, \
isolation, monitoring, boundary_pushing, demeaning, guilt_or_blame, inconsistency, \
evasion, other; "severity": "watch" | "concern" | "danger"; "behavior": what the match \
did, in one or two sentences; "evidence": [{"quote": exact words from a MATCH line, \
"why_it_matters": one sentence}]; "innocent_reading": one sentence, or null for danger}], \
"summary": one or two sentences, "confidence": "low" | "medium" | "high"}
Use "flags": [] when nothing rises to a flag.\
"""

Builder = Callable[[Conversation], Messages]


def compact_messages(conversation: Conversation, extra: str = "") -> Messages:
    system = FLAGS_SYSTEM + "\n\n" + COMPACT_NOTE + (f"\n\n{extra}" if extra else "")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": build_flags_request(conversation)},
    ]


@cache
def _shots(count: int) -> tuple[tuple[str, str], ...]:
    from confidant.lab.data import load_examples

    train = load_examples(Path("lab/data/examples.jsonl"), "train")
    # One flagged and one clean first, then alternate, shortest first to save context.
    flagged = sorted((e for e in train if e.kind == "flag"), key=lambda e: len(e.transcript))
    other = sorted((e for e in train if e.kind != "flag"), key=lambda e: len(e.transcript))
    picked = [x for pair in zip(flagged, other, strict=False) for x in pair][:count]
    return tuple(
        (
            build_flags_request(redact(e.conversation()).conversation),
            json.dumps(e.label, ensure_ascii=False),
        )
        for e in picked
    )


def fewshot(count: int) -> Builder:
    def build(conversation: Conversation) -> Messages:
        system, user = compact_messages(conversation)
        turns: Messages = []
        for request, answer in _shots(count):
            turns += [
                {"role": "user", "content": request},
                {"role": "assistant", "content": answer},
            ]
        return [system, *turns, user]

    return build


def rag(cards: int, retriever: str) -> Builder:
    from confidant.lab.rag import CardIndex

    index = CardIndex.load(retriever=retriever)

    def build(conversation: Conversation) -> Messages:
        found = index.search([m.text for m in conversation.match_messages], k=cards)
        notes = "\n\n".join([*(c.text for c in found), NOT_A_FLAG.read_text().strip()])
        return compact_messages(
            conversation, "Background on patterns that may be relevant:\n\n" + notes
        )

    return build


def last_messages(builder: Builder, count: int) -> Builder:
    """``builder``, shown only the last ``count`` messages: the context-length experiment.

    Grounding still runs against the whole transcript, so a quote from a cut message is
    simply never produced, not wrongly accepted.
    """

    def build(conversation: Conversation) -> Messages:
        return builder(dataclasses.replace(conversation, messages=conversation.messages[-count:]))

    return build


PROMPTS: dict[str, Callable[[argparse.Namespace], Builder]] = {
    "schema": lambda args: build_local_messages,
    "compact": lambda args: compact_messages,
    "fewshot": lambda args: fewshot(args.shots),
    "rag": lambda args: rag(args.cards, args.retriever),
}
