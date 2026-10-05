"""Prompt caching: what is marked for reuse, and what is deliberately not.

A cache hit needs a byte-identical prefix, so the failures here are silent ones: a
timestamp slipped into a system prompt, or a prompt trimmed below the minimum the model
will cache, costs money on every call and raises no error. These tests are where that
would show up instead.
"""

from __future__ import annotations

import json

import pytest

from confidant.analysis.flags import FlagScan
from confidant.client import cached_system
from confidant.prompts import FLAGS_SYSTEM, PERSONALITY_SYSTEM
from confidant.recording import Fingerprint, run_analysis

# Claude Opus 5 caches a prefix of 512 tokens or more and silently skips a shorter one.
# English prose runs about four characters a token; five keeps the estimate conservative.
_MIN_CACHEABLE_TOKENS = 512
_CHARS_PER_TOKEN_UPPER_BOUND = 5


def test_the_system_prompt_is_sent_as_one_cached_block(replay_client):
    client = replay_client("flags_sample")
    run_analysis("flags", "examples/sample_chat.txt", client=client)
    [request] = client.requests
    assert request["system"] == [
        {"type": "text", "text": FLAGS_SYSTEM, "cache_control": {"type": "ephemeral"}}
    ]


def test_the_transcript_is_never_marked_for_caching(replay_client):
    client = replay_client("analyze_sample")
    run_analysis("analyze", "examples/sample_chat.txt", client=client)
    [request] = client.requests
    assert "cache_control" not in request
    assert "cache_control" not in json.dumps(request["messages"])


def test_the_cached_prefix_does_not_depend_on_the_conversation(replay_client):
    client = replay_client("flags_sample", "flags_pressure")
    run_analysis("flags", "examples/sample_chat.txt", client=client)
    run_analysis("flags", "examples/pressure_chat.txt", client=client)
    first, second = client.requests
    assert first["messages"] != second["messages"]
    assert first["system"] == second["system"]
    assert first["output_format"] is second["output_format"]


def test_each_call_gets_its_own_block():
    # A shared dict would let one caller's edit leak into every later request's prefix.
    a, b = cached_system("x"), cached_system("x")
    a[0]["cache_control"]["ttl"] = "1h"
    assert b == [{"type": "text", "text": "x", "cache_control": {"type": "ephemeral"}}]


@pytest.mark.parametrize("name", ["PERSONALITY_SYSTEM", "FLAGS_SYSTEM"])
def test_system_prompts_are_long_enough_to_cache(name):
    prompt = {"PERSONALITY_SYSTEM": PERSONALITY_SYSTEM, "FLAGS_SYSTEM": FLAGS_SYSTEM}[name]
    estimated_tokens = len(prompt) / _CHARS_PER_TOKEN_UPPER_BOUND
    assert estimated_tokens > _MIN_CACHEABLE_TOKENS, (
        f"{name} is about {estimated_tokens:.0f} tokens, under the {_MIN_CACHEABLE_TOKENS} "
        "the model will cache. The breakpoint would be ignored without an error."
    )


def test_caching_does_not_change_a_recording_fingerprint():
    messages = [{"role": "user", "content": "x"}]
    plain = {"output_format": FlagScan, "system": "s", "messages": messages}
    cached = {"output_format": FlagScan, "system": cached_system("s"), "messages": messages}
    assert Fingerprint.of(plain) == Fingerprint.of(cached)


def test_a_prompt_edit_still_changes_the_fingerprint_when_cached():
    messages = [{"role": "user", "content": "x"}]
    before = {"output_format": FlagScan, "system": cached_system("s"), "messages": messages}
    after = {"output_format": FlagScan, "system": cached_system("t"), "messages": messages}
    assert Fingerprint.of(before).differences(Fingerprint.of(after)) == [
        "the system prompt changed"
    ]
