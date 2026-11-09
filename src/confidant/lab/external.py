"""An outside test set: MentalManip (Wang et al., ACL 2024), human-written dialogues.

Everything else in the lab was written by one local model from one scenario grid. That
is the lab's biggest limitation. MentalManip is a check from outside: 2,915 two-person
dialogues adapted from movie scripts (the Cornell Movie-Dialogs Corpus), each labelled by
three annotators as manipulative or not, with the manipulation techniques used. The
consensus version (``mentalmanip_con.csv``) keeps only dialogues where all three agreed.
It is CC BY-NC 4.0, so it is downloaded into ``lab-cache/`` and never committed.

It is not Confidant's task, and the scoring says so rather than pretending:

- **Who is the match?** The dataset does not say which speaker manipulates. Each
  dialogue is checked twice, once with each speaker as the match, and counts as flagged
  if either direction raises a flag.
- **Manipulation is not the same as a red flag.** "Persuasion or seduction" in a movie
  scene need not be anything Confidant should flag, and a dialogue all three annotators
  call non-manipulative can still contain a threat. So the numbers here measure
  *agreement with the annotators*, not accuracy. The most telling ones are the false
  alarm rate on consensus-non-manipulative dialogues, and whether a flagged category is
  one a technique plausibly maps to.
- **Techniques map to sets of categories** (:data:`TECHNIQUE_CATEGORIES`), written before
  any model was run on the data. A flag on a manipulative dialogue "agrees" if its
  category is in the set of any technique the annotators named.
"""

from __future__ import annotations

import csv
import random
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from confidant.lab.data import Example
from confidant.lab.evaluate import Read

__all__ = [
    "MENTALMANIP",
    "TECHNIQUE_CATEGORIES",
    "Dialogue",
    "directions",
    "load_mentalmanip",
    "score_external",
]

MENTALMANIP = Path("lab-cache/external/mentalmanip-src/mentalmanip_con.csv")
SOURCE = "https://github.com/audreycs/MentalManip (CC BY-NC 4.0)"

# MentalManip's eleven techniques, and the Confidant categories a careful read could
# file each under. Fixed before any model saw the data.
TECHNIQUE_CATEGORIES: dict[str, frozenset[str]] = {
    "Shaming or Belittlement": frozenset({"demeaning"}),
    "Accusation": frozenset({"guilt_or_blame", "demeaning"}),
    "Intimidation": frozenset({"threat", "coercion"}),
    "Brandishing Anger": frozenset({"threat", "coercion", "demeaning"}),
    "Playing Victim Role": frozenset({"guilt_or_blame"}),
    "Rationalization": frozenset({"guilt_or_blame", "inconsistency", "other"}),
    "Denial": frozenset({"inconsistency", "evasion", "other"}),
    "Evasion": frozenset({"evasion"}),
    "Feigning Innocence": frozenset({"evasion", "inconsistency", "other"}),
    "Persuasion or Seduction": frozenset({"boundary_pushing", "sexual_pressure", "coercion"}),
    "Playing Servant Role": frozenset({"guilt_or_blame", "other"}),
}

_TURN = re.compile(r"^(Person\d+):\s*(.*)$")


@dataclass(frozen=True)
class Dialogue:
    id: str
    manipulative: bool
    techniques: tuple[str, ...]
    turns: tuple[tuple[str, str], ...]

    @property
    def speakers(self) -> list[str]:
        return sorted({s for s, _ in self.turns})

    @property
    def categories(self) -> frozenset[str]:
        return frozenset().union(*(TECHNIQUE_CATEGORIES[t] for t in self.techniques))


def _parse(text: str) -> tuple[tuple[str, str], ...] | None:
    turns = []
    for line in text.splitlines():
        if not line.strip():
            continue
        found = _TURN.match(line.strip())
        if not found or not found[2].strip():
            return None
        turns.append((found[1], found[2].strip()))
    return tuple(turns) if len({s for s, _ in turns}) == 2 else None


def load_mentalmanip(
    path: Path = MENTALMANIP, *, per_class: int = 60, seed: int = 0
) -> list[Dialogue]:
    """A balanced sample: ``per_class`` manipulative (with techniques) and non-manipulative.

    Manipulative dialogues are drawn round-robin across their first-listed technique, so
    rare techniques are not crowded out by "Persuasion or Seduction". Dialogues that are
    not two clean speakers are skipped.
    """
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    dialogues = []
    for row in rows:
        turns = _parse(row["Dialogue"])
        techniques = tuple(t.strip() for t in row["Technique"].split(",") if t.strip())
        if turns is None or any(t not in TECHNIQUE_CATEGORIES for t in techniques):
            continue
        manipulative = row["Manipulative"].strip() == "1"
        if manipulative and not techniques:
            continue  # no technique agreed on: nothing to check a category against
        dialogues.append(Dialogue(row["ID"], manipulative, techniques, turns))

    rng = random.Random(seed)
    clean = [d for d in dialogues if not d.manipulative]
    rng.shuffle(clean)
    by_technique: dict[str, list[Dialogue]] = {}
    for d in dialogues:
        if d.manipulative:
            by_technique.setdefault(d.techniques[0], []).append(d)
    for group in by_technique.values():
        rng.shuffle(group)
    picked: list[Dialogue] = []
    while len(picked) < per_class and any(by_technique.values()):
        for name in sorted(by_technique):
            if by_technique[name] and len(picked) < per_class:
                picked.append(by_technique[name].pop())
    return picked + clean[:per_class]


def directions(dialogue: Dialogue) -> list[Example]:
    """The dialogue as two Confidant transcripts, each speaker taking a turn as the match."""
    a, b = dialogue.speakers
    examples = []
    for owner, match in ((a, b), (b, a)):
        head = [
            f"# Fictional: adapted movie dialogue from MentalManip ({SOURCE}).",
            f"# owner: {owner}",
            f"# match: {match}",
            "",
        ]
        text = "\n".join(head + [f"{s}: {t}" for s, t in dialogue.turns]) + "\n"
        examples.append(
            Example(
                id=f"{dialogue.id}:{match}",
                scenario=dialogue.id,
                kind="flag" if dialogue.manipulative else "clean",
                category=None,
                severity=None,
                transcript=text,
                label={},
            )
        )
    return examples


def score_external(dialogues: Sequence[Dialogue], reads: Sequence[Read]) -> dict[str, Any]:
    """Agreement with the annotators, per dialogue (both directions pooled)."""
    by_dialogue: dict[str, list[Read]] = {}
    for r in reads:
        by_dialogue.setdefault(r.id.rsplit(":", 1)[0], []).append(r)
    tp = fp = fn = 0
    agree = flagged_manip = 0
    danger_on_clean = invalid = 0
    per_technique: dict[str, list[int]] = {t: [0, 0] for t in TECHNIQUE_CATEGORIES}
    for d in dialogues:
        rs = by_dialogue.get(d.id, [])
        invalid += sum(not r.valid for r in rs)
        found = {c for r in rs for c, _ in r.found}
        tiers = {s for r in rs for _, s in r.found}
        flagged = bool(found)
        if d.manipulative:
            tp, fn = tp + flagged, fn + (not flagged)
            for t in d.techniques:
                per_technique[t][0] += flagged
                per_technique[t][1] += 1
            if flagged:
                flagged_manip += 1
                agree += bool(found & d.categories)
        else:
            fp += flagged
            danger_on_clean += "danger" in tiers
    n_clean = sum(not d.manipulative for d in dialogues)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "dialogues": len(dialogues),
        "manipulative": tp + fn,
        "non_manipulative": n_clean,
        "reads": len(reads),
        "valid": 1 - invalid / len(reads) if reads else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_alarm": fp / n_clean if n_clean else 0.0,
        "danger_on_non_manipulative": danger_on_clean,
        "category_agreement": agree / flagged_manip if flagged_manip else 0.0,
        "recall_by_technique": {
            t: (hit / n if n else None, n) for t, (hit, n) in per_technique.items() if n
        },
    }
