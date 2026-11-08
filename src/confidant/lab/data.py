"""Step 2 of the lab: labelled, fictional transcripts for the red-flag check.

The pipeline, each stage its own function so it can be tested without a model:

1. :func:`load_scenarios` reads ``lab/scenarios.toml`` and validates it: danger
   categories filed as danger, an innocent reading wherever the tier needs one, and no
   planted line that appears in the gold set (``examples/eval/``).
2. :func:`generate_examples` has a local model write each scenario into a transcript,
   several times with different names, lengths, and tones. The model writes the opening
   and the continuation; the planted lines between them are placed by code
   (:func:`write_conversation`), so a label never describes a line that is not there. A
   transcript that comes out too short, one-sided, with a planted line repeated, or too
   close to another variant is rejected and counted.
3. :func:`assign_splits` splits by *scenario*, stratified by category and kind, so no
   paraphrase of a test scenario is ever trained on.
4. :func:`contamination` checks the result against the gold set message by message.
5. :func:`to_chat_records` turns an example into the chat a fine-tune trains on: the same
   redacted prompt the backend sends at inference, and the label as the answer.

The label is built from the scenario, not from the generator: the planted lines are the
evidence, and the category and tier are the ones written down in advance.
"""

from __future__ import annotations

import json
import random
import re
import tomllib
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, get_args

from confidant.analysis.flags import DANGER_CATEGORIES, Category, FlagScan
from confidant.ingest.transcript import parse_transcript
from confidant.lab.backend import Messages, build_local_messages
from confidant.models import Conversation
from confidant.redaction import redact

__all__ = [
    "GOLD_DIR",
    "SCENARIOS",
    "Example",
    "GenerationReport",
    "Scenario",
    "assign_splits",
    "build_label",
    "contamination",
    "generate_examples",
    "load_examples",
    "load_scenarios",
    "save_examples",
    "to_chat_records",
]

SCENARIOS = Path("lab/scenarios.toml")
GOLD_DIR = Path("examples/eval")
KINDS = ("flag", "hard_negative", "clean")
NO_FLAGS_SUMMARY = "Nothing in the match's messages rises to a red flag."

OWNER_NAMES = ("Sam", "Alex", "Priya", "Jordan", "Mei", "Tomás", "Aisha", "Noor", "Ben", "Lena")
MATCH_NAMES = ("Robin", "Casey", "Theo", "Kai", "Marisol", "Dev", "Ines", "Felix", "Hana", "Jo")
LENGTHS = {"short": (8, 12), "medium": (14, 20), "long": (24, 32)}
TONES = ("warm and chatty", "dry and witty", "brief and low-key", "flirty and playful")


class ScenarioError(ValueError):
    """``lab/scenarios.toml`` says something the pipeline cannot use."""


@dataclass(frozen=True)
class Scenario:
    id: str
    kind: str
    setting: str
    planted: tuple[str, ...] = ()
    category: str | None = None
    severity: str | None = None
    behavior: str | None = None
    why: str | None = None
    innocent: str | None = None
    looks_like: str | None = None

    @property
    def group(self) -> str:
        """What the split is stratified by: the category for flags, else the kind."""
        return self.category if self.kind == "flag" else self.kind


@dataclass
class Example:
    id: str
    scenario: str
    kind: str
    category: str | None
    severity: str | None
    transcript: str
    label: dict[str, Any]
    variant: dict[str, Any] = field(default_factory=dict)
    split: str | None = None

    def conversation(self) -> Conversation:
        return parse_transcript(self.transcript)


def _gold_messages(gold_dir: Path = GOLD_DIR) -> list[str]:
    lines = []
    for path in sorted(gold_dir.glob("*.txt")):
        conversation = parse_transcript(path.read_text(encoding="utf-8"))
        lines.extend(m.text for m in conversation.messages)
    return lines


def _fold(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", text.casefold())).strip()


def load_scenarios(path: Path = SCENARIOS, gold_dir: Path = GOLD_DIR) -> list[Scenario]:
    """Read and validate the scenario grid."""
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    categories = set(get_args(Category))
    gold = {_fold(m) for m in _gold_messages(gold_dir)}
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for entry in raw.get("scenario", []):
        s = Scenario(**{k: tuple(v) if k == "planted" else v for k, v in entry.items()})
        where = f"scenario {s.id!r}"
        if s.id in seen:
            raise ScenarioError(f"{where} appears twice")
        seen.add(s.id)
        if s.kind not in KINDS:
            raise ScenarioError(f"{where}: kind must be one of {', '.join(KINDS)}")
        if s.kind == "flag":
            if s.category not in categories or not s.planted or not s.behavior or not s.why:
                raise ScenarioError(f"{where}: a flag needs category, planted, behavior, why")
            if s.category in DANGER_CATEGORIES and s.severity != "danger":
                raise ScenarioError(f"{where}: {s.category} is always danger")
            if s.severity != "danger" and not s.innocent:
                raise ScenarioError(f"{where}: {s.severity} needs an innocent reading")
        elif s.kind == "hard_negative" and not s.planted:
            raise ScenarioError(f"{where}: a hard negative needs planted lines")
        elif s.kind == "clean" and s.planted:
            raise ScenarioError(f"{where}: a clean scenario plants nothing")
        for line in s.planted:
            if _fold(line) in gold:
                raise ScenarioError(f"{where}: {line!r} is in the gold set")
        scenarios.append(s)
    return scenarios


def build_label(scenario: Scenario) -> FlagScan:
    """What a careful read of any transcript written from ``scenario`` should return."""
    if scenario.kind != "flag":
        return FlagScan(flags=[], summary=NO_FLAGS_SUMMARY, confidence="medium")
    flag = {
        "category": scenario.category,
        "severity": scenario.severity,
        "behavior": scenario.behavior,
        "evidence": [{"quote": q, "why_it_matters": scenario.why} for q in scenario.planted],
        "innocent_reading": None if scenario.severity == "danger" else scenario.innocent,
    }
    return FlagScan.model_validate(
        {"flags": [flag], "summary": scenario.behavior, "confidence": "medium"}
    )


# -- generation -----------------------------------------------------------------------

GENERATOR_SYSTEM = """\
You write realistic, entirely fictional text-message conversations between two people \
who met on a dating app. Write only the messages, one per line, each starting with the \
sender's name and a colon. No narration, no titles, no blank lines, no emoji descriptions.\
"""


def _setting(scenario: Scenario, variant: dict[str, Any]) -> str:
    """The scenario's setting in the characters' names: "the owner" leaks otherwise."""
    text = re.sub(r"\bthe owner's\b", variant["owner"] + "'s", scenario.setting)
    text = re.sub(r"\bthe owner\b", variant["owner"], text)
    text = re.sub(r"\bthe match's\b", variant["match"] + "'s", text)
    return re.sub(r"\bthe match\b", variant["match"], text)


def _ask(content: str) -> Messages:
    return [
        {"role": "system", "content": GENERATOR_SYSTEM},
        {"role": "user", "content": content},
    ]


def _behaviour(scenario: Scenario, match: str) -> str:
    if scenario.kind == "flag":
        return f"Apart from what is already written, {match} behaves normally."
    return (
        f"Keep {match} kind and respectful: nothing pushy, controlling, insulting, or threatening."
    )


def generator_messages(scenario: Scenario, variant: dict[str, Any]) -> Messages:
    """The prompt for a whole conversation, used when nothing is planted."""
    owner, match = variant["owner"], variant["match"]
    low, high = LENGTHS[variant["length"]]
    setting = _setting(scenario, variant)
    return _ask(
        f"Write a conversation between {owner} and {match}. Context: {setting}.\n"
        f"Length: {low} to {high} messages. Tone: {variant['tone']}.\n"
        f"{_behaviour(scenario, match)}\n"
        f"Start with {match if variant['match_first'] else owner}."
    )


def opening_messages(scenario: Scenario, variant: dict[str, Any], count: int) -> Messages:
    """The prompt for the part of a conversation before the planted lines."""
    owner, match = variant["owner"], variant["match"]
    return _ask(
        f"Write the first {count} messages of a conversation between {owner} and {match}. "
        f"Context: {_setting(scenario, variant)}. Tone: {variant['tone']}.\n"
        f"Start with {match if variant['match_first'] else owner}, and end on a message from "
        f"{owner} that leads into the situation in the context. Keep it ordinary so far."
    )


def continuation_messages(
    scenario: Scenario, variant: dict[str, Any], so_far: list[tuple[str, str]], count: int
) -> Messages:
    """The prompt for the rest of a conversation, after the planted lines."""
    owner, match = variant["owner"], variant["match"]
    transcript = "\n".join(f"{n}: {t}" for n, t in so_far)
    return _ask(
        f"Here is a text conversation between {owner} and {match} "
        f"(context: {_setting(scenario, variant)}):\n\n{transcript}\n\n"
        f"Write only the next {count} messages, starting with {owner}'s reply to {match}'s "
        f"last message. {owner} reacts the way a real person would. "
        f"{_behaviour(scenario, match)} Do not repeat earlier messages."
    )


def write_conversation(
    scenario: Scenario, variant: dict[str, Any], generate: Callable[[Messages], str]
) -> list[tuple[str, str]]:
    """A conversation for ``variant``, with the planted lines placed by code, not by the model.

    A model asked to drop given lines into a conversation repeats them, gives them to the
    wrong person, or paraphrases them. So the model writes an opening, the planted lines are
    appended as the match's next messages, and the model continues from there. The label
    is then true by construction, and the owner's reaction is written in context.
    """
    owner, match = variant["owner"], variant["match"]
    if not scenario.planted:
        return parse_generated(generate(generator_messages(scenario, variant)), owner, match)
    low, high = LENGTHS[variant["length"]]
    target = (low + high) // 2
    n_open = max(3, target // 2)
    opening = parse_generated(generate(opening_messages(scenario, variant, n_open)), owner, match)
    so_far = opening[:n_open] + [(match, line) for line in scenario.planted]
    n_rest = max(3, target - len(so_far))
    rest = parse_generated(
        generate(continuation_messages(scenario, variant, so_far, n_rest)), owner, match
    )
    seen = {_fold(t) for _, t in so_far}
    return so_far + [(n, t) for n, t in rest[: n_rest + 2] if _fold(t) not in seen]


_LINE = re.compile(r"^[\W\d]*(?P<name>[^\W\d][\w\s.'-]{0,30}?)[*_\s]*:[*_\s]*(?P<text>.+)$")


def parse_generated(raw: str, owner: str, match: str) -> list[tuple[str, str]]:
    """The ``(speaker, text)`` messages in a generator's output; anything else is dropped."""
    messages = []
    for line in raw.splitlines():
        found = _LINE.match(line.strip())
        if not found:
            continue
        name = found["name"].strip()
        if name in (owner, match):
            text = found["text"].strip().strip('"').strip()
            if text and "](" not in text:  # markdown images and links are not messages
                messages.append((name, text))
    return messages


def _planted_count(
    messages: list[tuple[str, str]], match: str, planted: Iterable[str]
) -> list[int]:
    """How many of the match's messages carry each planted line."""
    theirs = [_fold(text) for name, text in messages if name == match]
    return [sum(_fold(p) in t for t in theirs) for p in planted]


def render_transcript(
    scenario: Scenario, messages: list[tuple[str, str]], owner: str, match: str
) -> str:
    head = [
        f"# Entirely fictional. Generated by confidant.lab from scenario {scenario.id}.",
        f"# owner: {owner}",
        f"# match: {match}",
        "",
    ]
    return "\n".join(head + [f"{name}: {text}" for name, text in messages]) + "\n"


def _shingles(text: str, n: int = 5) -> set[tuple[str, ...]]:
    words = _fold(text).split()
    return {tuple(words[i : i + n]) for i in range(max(len(words) - n + 1, 1))}


def _too_similar(a: str, b: str, threshold: float = 0.6) -> bool:
    sa, sb = _shingles(a), _shingles(b)
    return len(sa & sb) / max(len(sa | sb), 1) >= threshold


@dataclass
class GenerationReport:
    kept: int = 0
    rejected_missing_lines: int = 0
    rejected_repeated: int = 0
    rejected_too_short: int = 0
    rejected_duplicate: int = 0
    attempts: int = 0


def make_variants(scenario: Scenario, count: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(f"{seed}:{scenario.id}")
    variants = []
    lengths = list(LENGTHS)
    for i in range(count):
        owner, match = rng.choice(OWNER_NAMES), rng.choice(MATCH_NAMES)
        variants.append(
            {
                "owner": owner,
                "match": match,
                "length": lengths[(i + rng.randrange(3)) % 3],
                "tone": rng.choice(TONES),
                "match_first": rng.random() < 0.5,
            }
        )
    return variants


def generate_examples(
    scenarios: Sequence[Scenario],
    generate: Callable[[Messages], str],
    *,
    per_scenario: int = 3,
    retries: int = 2,
    seed: int = 0,
    on_example: Callable[[Example | None, GenerationReport], None] | None = None,
) -> tuple[list[Example], GenerationReport]:
    """Write each scenario out ``per_scenario`` times with ``generate``."""
    report = GenerationReport()
    examples: list[Example] = []
    for scenario in scenarios:
        label = build_label(scenario).model_dump()
        kept_here: list[str] = []
        for v, variant in enumerate(make_variants(scenario, per_scenario, seed)):
            example = None
            for _ in range(retries + 1):
                report.attempts += 1
                owner, match = variant["owner"], variant["match"]
                messages = write_conversation(scenario, variant, generate)
                if len(messages) < 6 or {n for n, _ in messages} != {owner, match}:
                    report.rejected_too_short += 1
                    continue
                counts = _planted_count(messages, match, scenario.planted)
                if 0 in counts:
                    report.rejected_missing_lines += 1
                    continue
                if any(c > 1 for c in counts):
                    # A repeated threat reads as a different, stranger conversation.
                    report.rejected_repeated += 1
                    continue
                text = render_transcript(scenario, messages, owner, match)
                if any(_too_similar(text, other) for other in kept_here):
                    report.rejected_duplicate += 1
                    continue
                example = Example(
                    id=f"{scenario.id}-v{v}",
                    scenario=scenario.id,
                    kind=scenario.kind,
                    category=scenario.category,
                    severity=scenario.severity,
                    transcript=text,
                    label=label,
                    variant=variant,
                )
                break
            if example is not None:
                report.kept += 1
                kept_here.append(example.transcript)
                examples.append(example)
            if on_example is not None:
                on_example(example, report)
    return examples, report


# -- splits, contamination, and export ------------------------------------------------


def assign_splits(
    examples: list[Example],
    scenarios: Sequence[Scenario],
    *,
    fractions: tuple[float, float] = (0.2, 0.2),
    seed: int = 0,
) -> dict[str, str]:
    """Split by scenario, stratified by group; returns scenario id -> split."""
    by_group: dict[str, list[str]] = defaultdict(list)
    for s in scenarios:
        by_group[s.group].append(s.id)
    rng = random.Random(seed)
    split_of: dict[str, str] = {}
    for group in sorted(by_group):
        ids = sorted(by_group[group])
        rng.shuffle(ids)
        n_val = max(1, round(len(ids) * fractions[0]))
        n_test = max(1, round(len(ids) * fractions[1]))
        for i, sid in enumerate(ids):
            split_of[sid] = "test" if i < n_test else "val" if i < n_test + n_val else "train"
    for e in examples:
        e.split = split_of[e.scenario]
    return split_of


def contamination(examples: Iterable[Example], gold_dir: Path = GOLD_DIR) -> list[str]:
    """Every generated message that repeats a gold-set message (longer than four words)."""
    gold = {_fold(m) for m in _gold_messages(gold_dir) if len(m.split()) > 4}
    hits = []
    for e in examples:
        for m in e.conversation().messages:
            if _fold(m.text) in gold:
                hits.append(f"{e.id}: {m.text}")
    return hits


def to_chat_records(
    examples: Iterable[Example], messages: Callable[[Conversation], Messages] = build_local_messages
) -> list[dict[str, Any]]:
    """Fine-tuning chats: the redacted prompt the backend sends, and the label as answer."""
    records = []
    for e in examples:
        prompt = messages(redact(e.conversation()).conversation)
        answer = json.dumps(e.label, ensure_ascii=False)
        records.append({"messages": [*prompt, {"role": "assistant", "content": answer}]})
    return records


def save_examples(examples: Iterable[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as out:
        for e in examples:
            out.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")


def load_examples(path: Path, split: str | None = None) -> list[Example]:
    examples = [
        Example(**json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    return [e for e in examples if split is None or e.split == split]


def describe(examples: Sequence[Example]) -> str:
    """Counts by split, kind, and category, for the datasheet."""
    rows = Counter(
        (e.split, e.group if hasattr(e, "group") else (e.category or e.kind)) for e in examples
    )
    splits = sorted({s for s, _ in rows})
    groups = sorted({g for _, g in rows})
    lines = ["| group | " + " | ".join(splits) + " |", "|---|" + "---:|" * len(splits)]
    for g in groups:
        lines.append(f"| {g} | " + " | ".join(str(rows[(s, g)]) for s in splits) + " |")
    totals = Counter(e.split for e in examples)
    lines.append("| **total** | " + " | ".join(str(totals[s]) for s in splits) + " |")
    return "\n".join(lines)
