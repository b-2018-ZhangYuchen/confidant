"""Pieces shared by every prompt that shows the model a conversation."""

from __future__ import annotations

from confidant.models import Conversation, Role

REDACTION_NOTE = """\
Before this transcript reached you, names, phone numbers, email and street addresses, \
links, and social handles were replaced with placeholders in square brackets: [OWNER] \
and [MATCH] for the two people, [NAME_1] for anyone else named, [PHONE_1], [EMAIL_1], \
[ADDRESS_1], and so on. The same detail always gets the same placeholder. Write the \
placeholders exactly as they appear, in quotes and in your own sentences; they are \
swapped back before the owner reads your answer. Do not guess what they stand for. A \
hidden detail is not itself a finding, but what someone did with it still is: asking \
for an address, or sending a number unprompted, is behavior you can describe.\
"""


def render_conversation(conversation: Conversation) -> list[str]:
    """The header and tagged transcript block, as lines for a user turn.

    Every analyzer sees the conversation in exactly this shape, so a finding from one
    can be compared with a finding from another without wondering whether they were
    shown different things. ``conversation`` is expected to be redacted already, so the
    sender's name is left off each line: it would only repeat the tag.
    """
    lines = [
        f"Owner (the person I am helping): {conversation.owner_name}",
        f"Match (the person to analyze): {conversation.match_name}",
        f"Messages: {len(conversation)} "
        f"({len(conversation.owner_messages)} owner / {len(conversation.match_messages)} match)",
    ]

    span = conversation.timespan
    if span is not None:
        lines.append(f"Spanning: {span.days} days")

    ratio = conversation.effort_ratio
    if ratio is not None:
        lines.append(
            f"Average message length: match writes {ratio:.2f} words per owner word "
            f"(context only — do not over-read it)"
        )

    lines.append("")
    lines.append("--- TRANSCRIPT ---")
    for message in conversation:
        tag = "OWNER" if message.role is Role.OWNER else "MATCH"
        stamp = f"[{message.timestamp:%Y-%m-%d %H:%M}] " if message.timestamp else ""
        body = message.text.replace("\n", "\n    ")
        lines.append(f"{stamp}{tag}: {body}")
    lines.append("--- END TRANSCRIPT ---")
    return lines
