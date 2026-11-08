"""The lab's data pipeline, scoring, retrieval, and prompts: offline, no model."""

from __future__ import annotations

import json
import re

import pytest

from confidant.analysis.flags import FakeBackend, FlagScan
from confidant.lab.data import (
    Example,
    Scenario,
    ScenarioError,
    assign_splits,
    build_label,
    contamination,
    generate_examples,
    load_examples,
    load_scenarios,
    parse_generated,
    save_examples,
    to_chat_records,
)
from confidant.lab.evaluate import Read, evaluate, score
from confidant.lab.prompts import compact_messages
from confidant.lab.rag import CardIndex, load_cards, retrieval_recall

THREAT = Scenario(
    id="threat-x",
    kind="flag",
    category="threat",
    severity="danger",
    setting="a test",
    planted=("you will be sorry, I mean it",),
    behavior="Threatens the owner.",
    why="It is a threat.",
)
CLEAN = Scenario(id="clean-x", kind="clean", setting="a nice chat")


def writer(lines_for):
    """A fake generator: ``lines_for(prompt)`` returns the transcript lines to emit."""

    def generate(messages):
        return "\n".join(lines_for(messages[-1]["content"]))

    return generate


def chat(owner, match, planted=()):
    lines = [f"{owner}: hi", f"{match}: hey there", f"{owner}: how was your day"]
    lines += [f"{match}: {p}" for p in planted]
    lines += [f"{match}: pretty good", f"{owner}: nice", f"{match}: talk soon"]
    return lines


def names(prompt):
    found = re.search(r"between (\w+) and (\w+)", prompt)
    return found[1], found[2]


# -- scenarios ------------------------------------------------------------------------


def test_the_committed_scenarios_are_valid():
    scenarios = load_scenarios()
    assert len(scenarios) == 100
    assert {s.kind for s in scenarios} == {"flag", "hard_negative", "clean"}


def test_a_danger_category_filed_lower_is_refused(tmp_path):
    path = tmp_path / "s.toml"
    path.write_text(
        '[[scenario]]\nid = "x"\nkind = "flag"\ncategory = "monitoring"\nseverity = "watch"\n'
        'setting = "s"\nplanted = ["a line"]\nbehavior = "b"\nwhy = "w"\ninnocent = "i"\n'
    )
    with pytest.raises(ScenarioError, match="always danger"):
        load_scenarios(path)


def test_a_planted_line_from_the_gold_set_is_refused(tmp_path):
    path = tmp_path / "s.toml"
    path.write_text(
        '[[scenario]]\nid = "x"\nkind = "hard_negative"\nsetting = "s"\n'
        "planted = [\"I'd rather not this week, honestly. I'm wiped\"]\n"
    )
    with pytest.raises(ScenarioError, match="gold set"):
        load_scenarios(path)


def test_the_label_comes_from_the_scenario():
    label = build_label(THREAT)
    [flag] = label.flags
    assert (flag.category, flag.severity, flag.innocent_reading) == ("threat", "danger", None)
    assert [e.quote for e in flag.evidence] == list(THREAT.planted)
    assert build_label(CLEAN).flags == []


# -- generation -----------------------------------------------------------------------


def test_generated_lines_are_parsed_and_stray_text_dropped():
    raw = 'Here you go!\n**Sam:** hi\n2. Robin: "hey"\nNarrator: no\nSam: bye'
    assert parse_generated(raw, "Sam", "Robin") == [("Sam", "hi"), ("Robin", "hey"), ("Sam", "bye")]


def test_planted_lines_are_placed_by_code_whatever_the_model_writes():
    generate = writer(lambda prompt: chat(*names(prompt)))  # never says the planted line
    examples, report = generate_examples([THREAT], generate, per_scenario=1)
    [example] = examples
    theirs = [m.text for m in example.conversation().match_messages]
    assert theirs.count(THREAT.planted[0]) == 1
    assert report.attempts == 1


def test_a_continuation_that_repeats_a_planted_line_is_rejected():
    def lines(prompt):
        owner, match = names(prompt)
        extra = [f"{match}: like I said, {THREAT.planted[0]}"] if "Here is" in prompt else []
        return chat(owner, match) + extra

    examples, report = generate_examples([THREAT], writer(lines), per_scenario=1, retries=0)
    assert examples == [] and report.rejected_repeated == 1


def test_a_one_sided_conversation_is_rejected():
    generate = writer(lambda prompt: ["Robin: hi"] * 8)
    examples, report = generate_examples([CLEAN], generate, per_scenario=1, retries=0)
    assert examples == [] and report.rejected_too_short == 1


def test_kept_transcripts_are_fictional_parseable_and_labelled():
    generate = writer(lambda prompt: chat(*names(prompt)))
    examples, report = generate_examples([THREAT], generate, per_scenario=1)
    [example] = examples
    assert report.kept == 1
    assert example.transcript.startswith("# Entirely fictional.")
    conversation = example.conversation()
    assert any(THREAT.planted[0] in m.text for m in conversation.match_messages)
    assert example.label["flags"][0]["category"] == "threat"


def test_near_duplicate_variants_are_dropped():
    generate = writer(lambda prompt: chat("Sam", "Robin"))
    sam_robin = [
        {"owner": "Sam", "match": "Robin", "length": "short", "tone": "x", "match_first": 1}
    ]
    import confidant.lab.data as data

    original = data.make_variants
    data.make_variants = lambda s, count, seed: sam_robin * count
    try:
        examples, report = generate_examples([THREAT], generate, per_scenario=2, retries=0)
    finally:
        data.make_variants = original
    assert len(examples) == 1 and report.rejected_duplicate == 1


# -- splits, contamination, export ----------------------------------------------------


def _example(scenario: Scenario, v: int = 0) -> Example:
    return Example(
        id=f"{scenario.id}-v{v}",
        scenario=scenario.id,
        kind=scenario.kind,
        category=scenario.category,
        severity=scenario.severity,
        transcript="# Entirely fictional.\n# owner: Sam\n# match: Robin\n"
        + "\n".join(chat("Sam", "Robin", scenario.planted))
        + "\n",
        label=build_label(scenario).model_dump(),
    )


def test_splits_keep_every_variant_of_a_scenario_together():
    scenarios = load_scenarios()
    examples = [_example(s, v) for s in scenarios for v in range(3)]
    split_of = assign_splits(examples, scenarios)
    for e in examples:
        assert e.split == split_of[e.scenario]
    # Every category is represented in test and val.
    for group in {s.group for s in scenarios}:
        assert {split_of[s.id] for s in scenarios if s.group == group} == {"train", "val", "test"}


def test_contamination_finds_a_gold_message():
    bad = _example(CLEAN)
    bad.transcript += "Robin: this week is packed for me though, can we look at next week?\n"
    assert contamination([_example(CLEAN)]) == []
    assert len(contamination([bad])) == 1


def test_fine_tuning_chats_are_redacted_and_answer_with_the_label():
    [record] = to_chat_records([_example(THREAT)], compact_messages)
    system, user, answer = record["messages"]
    assert "Robin" not in user["content"] and "Sam" not in user["content"]
    assert FlagScan.model_validate_json(answer["content"]).flags[0].category == "threat"


def test_examples_round_trip(tmp_path):
    path = tmp_path / "e.jsonl"
    save_examples([_example(THREAT), _example(CLEAN)], path)
    assert [e.id for e in load_examples(path)] == ["threat-x-v0", "clean-x-v0"]


# -- scoring --------------------------------------------------------------------------


def test_scoring_through_the_real_flag_path():
    threat, clean = _example(THREAT), _example(CLEAN)
    false_alarm = build_label(THREAT).model_copy(deep=True)
    false_alarm.flags[0].evidence[0].quote = "pretty good"  # really in the clean chat
    reads = evaluate(FakeBackend(build_label(THREAT), false_alarm), [threat, clean])
    s = score(reads)
    assert (s.recall, s.precision, s.false_alarm_clean) == (1.0, 0.5, 1.0)
    assert s.danger_missed == 0 and s.valid == 1.0


def test_an_ungrounded_flag_is_not_a_catch():
    scan = build_label(THREAT).model_copy(deep=True)
    scan.flags[0].evidence[0].quote = "a line nobody wrote"
    [read] = evaluate(FakeBackend(scan), [_example(THREAT)])
    s = score([read])
    assert s.recall == 0.0 and s.grounded == 0.0 and s.danger_missed == 1


def test_an_invalid_answer_counts_as_no_flags_and_as_invalid():
    reads = [
        Read("a", "flag", "threat", "danger", valid=False),
        Read("b", "clean", None, None, valid=True),
    ]
    s = score(reads)
    assert s.valid == 0.5 and s.recall == 0.0 and s.danger_missed == 1


def test_severity_and_category_are_scored_separately():
    reads = [
        Read("a", "flag", "demeaning", "concern", valid=True, found=[("demeaning", "watch")]),
        Read("b", "flag", "evasion", "watch", valid=True, found=[("other", "watch")]),
    ]
    s = score(reads)
    assert s.recall == 1.0
    assert s.severity_accuracy == 0.5
    assert s.category_f1 == pytest.approx(0.5)


# -- retrieval and prompts ------------------------------------------------------------


def test_every_category_has_a_card():
    keys = {c.key for c in load_cards()}
    assert keys >= {s.category for s in load_scenarios() if s.kind == "flag"} | {"none"}


def test_bm25_finds_the_right_card_and_never_retrieves_none():
    index = CardIndex.load(retriever="bm25")
    found = [c.key for c in index.search(["turn on location sharing, nothing to hide"], k=3)]
    assert found[0] == "monitoring" and "none" not in found
    s = retrieval_recall(index, [_example(THREAT)], k=3)
    assert s["n"] == 1


def test_compact_prompt_names_every_category():
    system, _ = compact_messages(_example(CLEAN).conversation())
    from typing import get_args

    from confidant.analysis.flags import Category

    for category in get_args(Category):
        assert category in system["content"]
    json.loads('{"ok": 1}')  # the prompt promises JSON; nothing else to check offline
