"""System prompts, kept in their own module so they can be reviewed as prose.

Prompts are the most load-bearing and least reviewable part of a project like this, so
they live apart from the code that calls them rather than buried in a function body.
"""

from confidant.prompts.flags import FLAGS_SYSTEM, build_flags_request
from confidant.prompts.personality import PERSONALITY_SYSTEM, build_personality_request

__all__ = [
    "FLAGS_SYSTEM",
    "PERSONALITY_SYSTEM",
    "build_flags_request",
    "build_personality_request",
]
