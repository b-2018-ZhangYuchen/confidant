"""Confidant — an AI agent that helps you think clearly about the people you date.

Confidant reads a chat transcript and gives you a second opinion: how this person
communicates, what the conversation suggests about them, what deserves a closer
look, and when it might be worth reaching out again.

It is a thinking aid, not a verdict machine. See ``docs/principles.md``.
"""

from confidant.models import Conversation, Message, Role

__version__ = "0.1.0"
__all__ = ["Conversation", "Message", "Role", "__version__"]
