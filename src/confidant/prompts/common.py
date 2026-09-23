"""Pieces shared by every prompt that shows the model a conversation."""

from __future__ import annotations

from confidant.models import Conversation, Role


def render_conversation(conversation: Conversation) -> list[str]:
    """The header and tagged transcript block, as lines for a user turn.

    Every analyzer sees the conversation in exactly this shape, so a finding from one
    can be compared with a finding from another without wondering whether they were
    shown different things.
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
        lines.append(f"{stamp}{tag} ({message.sender}): {body}")
    lines.append("--- END TRANSCRIPT ---")
    return lines
