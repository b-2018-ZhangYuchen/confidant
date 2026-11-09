"""The MentalManip loader and scoring, on a tiny invented CSV in the dataset's format."""

from __future__ import annotations

import csv
import json

from confidant.analysis.flags import FakeBackend, FlagScan
from confidant.lab.evaluate import evaluate
from confidant.lab.external import (
    TECHNIQUE_CATEGORIES,
    directions,
    load_mentalmanip,
    load_reament,
    score_external,
)


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Dialogue", "Manipulative", "Technique", "Vulnerability"])
        writer.writerows(rows)


ROWS = [
    [
        "1",
        "Person1: you never do anything right\nPerson2: that's not fair",
        "1",
        "Shaming or Belittlement",
        "",
    ],
    ["2", "Person1: lovely day\nPerson2: it is", "0", "", ""],
    ["3", "Person1: hi\nPerson2: hi\nPerson3: hi", "0", "", ""],  # three speakers: skipped
    ["4", "Person1: do it\nPerson2: no", "1", "", ""],  # manipulative, no technique: skipped
]


def test_only_clean_two_speaker_dialogues_are_kept(tmp_path):
    path = tmp_path / "m.csv"
    write_csv(path, ROWS)
    dialogues = load_mentalmanip(path, per_class=5)
    assert sorted(d.id for d in dialogues) == ["1", "2"]


def test_each_dialogue_is_checked_with_each_speaker_as_the_match(tmp_path):
    path = tmp_path / "m.csv"
    write_csv(path, ROWS)
    [first, _] = load_mentalmanip(path, per_class=5)
    a, b = directions(first)
    assert (a.conversation().match_name, b.conversation().match_name) == ("Person2", "Person1")
    assert a.transcript.startswith("# Outside test data from MentalManip")


def _flag(category, quote):
    return FlagScan.model_validate(
        {
            "flags": [
                {
                    "category": category,
                    "severity": "concern",
                    "behavior": "b",
                    "evidence": [{"quote": quote, "why_it_matters": "w"}],
                    "innocent_reading": "i",
                }
            ],
            "summary": "s",
            "confidence": "medium",
        }
    )


EMPTY = FlagScan(flags=[], summary="nothing", confidence="medium")


def test_either_direction_flagging_counts_and_categories_are_checked(tmp_path):
    path = tmp_path / "m.csv"
    write_csv(path, ROWS)
    dialogues = load_mentalmanip(path, per_class=5)
    examples = [e for d in dialogues for e in directions(d)]
    # Manipulative dialogue: Person2 as match says nothing; Person1 as match is demeaning.
    # Clean dialogue: nothing either way.
    backend = FakeBackend(EMPTY, _flag("demeaning", "you never do anything right"), EMPTY, EMPTY)
    scores = score_external(dialogues, evaluate(backend, examples))
    assert (scores["recall"], scores["false_alarm"], scores["category_agreement"]) == (
        1.0,
        0.0,
        1.0,
    )


def test_a_flag_in_an_unmapped_category_does_not_agree(tmp_path):
    path = tmp_path / "m.csv"
    write_csv(path, ROWS)
    dialogues = load_mentalmanip(path, per_class=5)
    examples = [e for d in dialogues for e in directions(d)]
    backend = FakeBackend(EMPTY, _flag("monitoring", "you never do anything right"), EMPTY, EMPTY)
    scores = score_external(dialogues, evaluate(backend, examples))
    assert scores["recall"] == 1.0 and scores["category_agreement"] == 0.0


def test_every_technique_maps_to_real_categories():
    from typing import get_args

    from confidant.analysis.flags import Category

    assert len(TECHNIQUE_CATEGORIES) == 11
    for categories in TECHNIQUE_CATEGORIES.values():
        assert categories <= set(get_args(Category))


def write_reament(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


REAMENT_ROWS = [
    {"id": "0001", "dialogue": "Person1: you owe me\nPerson2: I don't", "manipulative": "1"},
    {"id": "0002", "dialogue": "Person1: nice weather\nPerson2: lovely", "manipulative": "0"},
    {"id": "0003", "dialogue": "Person1: just me talking", "manipulative": "0"},  # one speaker
]


def test_reament_is_sampled_per_class_and_one_speaker_rows_are_skipped(tmp_path):
    path = tmp_path / "r.json"
    write_reament(path, REAMENT_ROWS)
    dialogues = load_reament(path, per_class=5)
    assert sorted((d.id, d.manipulative) for d in dialogues) == [("0001", True), ("0002", False)]
    assert all(d.source == "ReaMent" and d.techniques == () for d in dialogues)
    assert directions(dialogues[0])[0].transcript.startswith("# Outside test data from ReaMent")


def test_without_techniques_category_agreement_is_not_reported(tmp_path):
    path = tmp_path / "r.json"
    write_reament(path, REAMENT_ROWS)
    dialogues = load_reament(path, per_class=5)
    examples = [e for d in dialogues for e in directions(d)]
    backend = FakeBackend(EMPTY, _flag("demeaning", "you owe me"), EMPTY, EMPTY)
    scores = score_external(dialogues, evaluate(backend, examples))
    assert scores["recall"] == 1.0 and scores["category_agreement"] is None
