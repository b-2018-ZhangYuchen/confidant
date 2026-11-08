"""The red-flag check on a local model, through ``mlx-lm``.

Claude is held to the flag schema by the API's structured outputs. A local model is not:
it is asked for the same schema in the prompt and may answer with prose around the JSON,
a code fence, or JSON that does not fit. :func:`parse_scan` takes the first JSON object
out of whatever came back and validates it, and anything it cannot use raises
:class:`LocalOutputError` with the raw text attached. That is an error, not an empty
scan: "the model said nothing was wrong" and "the model said nothing usable" must never
look the same, and how often the second happens is one of the things the lab measures.

Everything after the scan (grounding, severity floors, the safety notice) is
:func:`~confidant.analysis.flags.analyze_flags`'s job, exactly as for Claude.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from confidant.analysis.flags import FlagScan
from confidant.models import Conversation
from confidant.progress import Progress
from confidant.prompts.flags import FLAGS_SYSTEM, build_flags_request
from confidant.usage import Usage

__all__ = [
    "GenerationStats",
    "LocalModel",
    "LocalOutputError",
    "MlxBackend",
    "build_local_messages",
    "parse_scan",
]

Messages = list[dict[str, str]]

SCHEMA_NOTE = """\
Answer with a single JSON object that fits this JSON Schema, and nothing else: no prose \
before or after it, and no code fence.

{schema}\
"""


_THINKING = re.compile(r"<think>.*?</think>", re.DOTALL)


class LocalOutputError(ValueError):
    """A local model answered, but not with a scan that fits the schema."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


def build_local_messages(conversation: Conversation) -> Messages:
    """The chat for a local red-flag check: Claude's prompt, plus the schema spelled out."""
    schema = json.dumps(FlagScan.model_json_schema(), indent=1)
    return [
        {"role": "system", "content": FLAGS_SYSTEM + "\n\n" + SCHEMA_NOTE.format(schema=schema)},
        {"role": "user", "content": build_flags_request(conversation)},
    ]


def _objects(text: str) -> tuple[list[dict[str, Any]], bool]:
    """Every JSON object that starts at a ``{`` in ``text``, and whether the first one did.

    Objects nested inside a larger one are found too, which is why :func:`parse_scan`
    prefers one with a ``flags`` key: in an answer cut off by ``max_tokens``, the outer
    object fails to parse but an evidence object inside it does not.
    """
    decoder = json.JSONDecoder()
    found: list[dict[str, Any]] = []
    first_ok: bool | None = None
    start = text.find("{")
    while start != -1:
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            value = None
        if first_ok is None:
            first_ok = isinstance(value, dict)
        if isinstance(value, dict):
            found.append(value)
        start = text.find("{", start + 1)
    return found, bool(first_ok)


def parse_scan(raw: str) -> FlagScan:
    """Read a :class:`FlagScan` out of a local model's answer, or say why not."""
    found, first_ok = _objects(_THINKING.sub("", raw))
    answer = next((obj for obj in found if "flags" in obj), None)
    if answer is None:
        if not first_ok:
            raise LocalOutputError(
                "the model's answer has no complete JSON object (it may have been cut off)", raw
            )
        answer = found[0]
    try:
        return FlagScan.model_validate(answer)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in err['loc']) or 'answer'}: {err['msg']}"
            for err in exc.errors()[:3]
        )
        message = f"the model's JSON does not fit the flag schema ({problems})"
        raise LocalOutputError(message, raw) from None


@dataclass
class GenerationStats:
    """What one local generation cost: tokens, speed, memory, and wall time."""

    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    seconds: float = 0.0
    finish_reason: str | None = None


class LocalModel:
    """A local chat model through mlx-lm: messages in, text out, loaded on first use.

    The red-flag backend and the lab's data generator both use it. ``last_stats`` holds
    what the most recent call cost.
    """

    def __init__(
        self,
        model: str,
        *,
        adapter: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.last_stats = GenerationStats()
        self._loaded: tuple[Any, Any] | None = None

    def load(self) -> tuple[Any, Any]:
        if self._loaded is None:
            try:
                from mlx_lm import load
            except ImportError as exc:
                raise RuntimeError(
                    "the local backend needs mlx-lm, which runs on Apple silicon: "
                    'pip install "confidant-agent[lab]"'
                ) from exc
            self._loaded = load(self.model, adapter_path=self.adapter)
        return self._loaded

    def __call__(self, messages: Messages) -> str:
        model, tokenizer = self.load()  # first, so a missing mlx-lm gets the helpful error
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
        started = time.perf_counter()
        parts: list[str] = []
        last = None
        for last in stream_generate(
            model,
            tokenizer,
            prompt=_chat_prompt(tokenizer, messages),
            max_tokens=self.max_tokens,
            sampler=sampler,
        ):
            parts.append(last.text)
        self.last_stats = GenerationStats(seconds=time.perf_counter() - started)
        if last is not None:
            self.last_stats.prompt_tokens = last.prompt_tokens
            self.last_stats.generation_tokens = last.generation_tokens
            self.last_stats.prompt_tps = last.prompt_tps
            self.last_stats.generation_tps = last.generation_tps
            self.last_stats.peak_memory_gb = last.peak_memory
            self.last_stats.finish_reason = last.finish_reason
        return "".join(parts)


class MlxBackend:
    """A :class:`~confidant.analysis.flags.FlagBackend` that runs a local model with mlx-lm.

    ``model`` is a Hugging Face repo id or a local path, optionally with a LoRA
    ``adapter`` directory on top. The model is loaded on the first scan, not here, so
    building a backend is cheap and works where mlx-lm is not installed. ``generate``
    replaces the model entirely (messages in, text out); tests and lab code without a
    model pass one. ``messages`` builds the prompt, so the lab can try other prompts (few
    shot, retrieved cards) through the same backend. ``last_raw`` keeps the most recent
    answer, usable or not.
    """

    name = "local"

    def __init__(
        self,
        model: str,
        *,
        adapter: str | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        generate: Callable[[Messages], str] | None = None,
        messages: Callable[[Conversation], Messages] = build_local_messages,
    ) -> None:
        self.model = model
        self.adapter = adapter
        self._generate = generate or LocalModel(
            model, adapter=adapter, max_tokens=max_tokens, temperature=temperature, top_p=top_p
        )
        self._messages = messages
        self.last_raw: str | None = None

    @property
    def last_stats(self) -> GenerationStats | None:
        return getattr(self._generate, "last_stats", None)

    def scan(
        self,
        conversation: Conversation,
        *,
        on_progress: Callable[[Progress], None] | None = None,
        on_usage: Callable[[Usage], None] | None = None,
    ) -> FlagScan:
        self.last_raw = self._generate(self._messages(conversation))
        return parse_scan(self.last_raw)


def _chat_prompt(tokenizer: Any, messages: Messages) -> str:
    """Apply the model's chat template, folding the system turn in where it has none."""
    # enable_thinking is read by Qwen3's template and ignored by the rest; the flag
    # check wants the JSON, not a reasoning trace in front of it.
    try:
        return tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
    except Exception:
        # Some templates (Mistral's, Gemma's) reject a system role outright.
        system, user = messages[0]["content"], messages[1]["content"]
        folded = [{"role": "user", "content": system + "\n\n" + user}]
        return tokenizer.apply_chat_template(
            folded, add_generation_prompt=True, tokenize=False, enable_thinking=False
        )
