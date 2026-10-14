"""Tests for the eval set. Offline: scoring is local, and runs replay recordings or fakes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from confidant import evals
from confidant.analysis.flags import Flag, FlagReport, FlagScan, ground_flags
from confidant.analysis.personality import Evidence, PersonalityReport, Trait
from confidant.evals import (
    DEFAULT_CASES,
    EvalError,
    Scorecard,
    load_case,
    load_cases,
    main,
    run_cases,
    score_read,
)
from confidant.ingest.transcript import read_transcript

SAMPLE = Path("examples/sample_chat.txt").resolve()


def write_case(tmp_path: Path, body: str, transcript: Path = SAMPLE) -> Path:
    path = tmp_path / "case.toml"
    path.write_text(
        f'about = "A test case."\ntranscript = "{transcript.as_posix()}"\n\n{body}',
        encoding="utf-8",
    )
    return path


def report(**overrides) -> PersonalityReport:
    fields = {
        "headline": "Robin comes across as curious and keeps the plans moving.",
        "communication_style": "Quick replies, usually ending on a question.",
        "traits": [
            Trait(
                name="remembers details",
                reading="Follows up on things Sam mentioned in passing.",
                confidence="high",
                evidence=[
                    Evidence(
                        quote="how did the presentation go? you said it was thursday",
                        why_it_matters="Recalls a detail and checks in.",
                    )
                ],
            )
        ],
        "what_they_seem_to_want": "To meet.",
        "green_flags": ["Followed up a day later."],
        "worth_watching": [],
        "questions_to_ask": ["Is the coffee cart still on?"],
        "confidence": "medium",
        "caveat": "Seventeen messages is a small sample.",
    }
    return PersonalityReport(**{**fields, **overrides})


def flag_report(*flags: Flag, discarded: int = 0) -> FlagReport:
    return FlagReport(
        flags=list(flags), summary="A summary.", confidence="medium", discarded=discarded
    )


def flag(category: str, severity: str, quote: str = "who are you with right now") -> Flag:
    return Flag(
        category=category,
        severity=severity,
        behavior="Asked where the owner was.",
        evidence=[Evidence(quote=quote, why_it_matters="It matters.")],
        innocent_reading=None,
    )


def failures(checks) -> dict[str, str]:
    return {c.name: c.detail for c in checks if not c.passed}


# -- the committed set ------------------------------------------------------------


def test_every_committed_case_loads():
    cases = load_cases()
    assert len(cases) >= 8
    # Both analyses, and both kinds of transcript: some with danger in them, some without.
    assert {a for c in cases for a in c.sections} == {"analyze", "flags"}
    assert any(c.sections.get("flags") and c.sections["flags"].escalation for c in cases)
    assert any(c.sections.get("flags") and c.sections["flags"].max_severity for c in cases)


def test_every_committed_transcript_says_it_is_fictional():
    for path in DEFAULT_CASES.glob("*.txt"):
        assert "fictional" in path.read_text(encoding="utf-8").splitlines()[0].casefold(), path


def test_check_lists_the_cases_without_calling_anything(capsys):
    assert main(["check"]) == 0
    out = capsys.readouterr().out
    assert "terse_texter" in out
    assert "Nothing was sent." in out


def test_run_without_credentials_exits_2(capsys, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "/nonexistent")
    assert main(["run", "--only", "thin_transcript"]) == 2
    assert "No Anthropic credentials found" in capsys.readouterr().err


def test_run_names_an_unknown_case(capsys):
    assert main(["run", "--only", "no_such_case"]) == 2
    assert "No case named no_such_case" in capsys.readouterr().err


# -- validating a case ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ("[analyze]\nconfidense = 'low'\n", "unknown keys: confidense"),
        ("[analyse]\n", "unknown keys: analyse"),
        ("", "checks nothing"),
        ("[analyze]\nconfidence = 'certain'\n", "certain is not a level"),
        ("[analyze]\nnonempty = ['headline']\n", "headline is not a list field"),
        ("[[analyze.cite]]\nwhy = 'x'\nany = ['you are not allowed to go']\n", "is not in Robin"),
        ("[[analyze.mention]]\nany = ['plan']\n", "needs a 'why'"),
        ("[[analyze.mention]]\nwhy = 'x'\nany = []\n", "non-empty list of strings"),
        ("[[analyze.mention]]\nwhy = 'x'\nany = ['a']\nfield = 'quote'\n", "field 'quote'"),
        ("[flags]\nmax_severity = 'bad'\n", "max_severity should be one of"),
        ("[[flags.expect]]\nwhy = 'x'\ncategory = 'stalking'\n", "stalking is not a flag"),
        ("[[flags.expect]]\nwhy = 'x'\ncategory = 'threat'\nmin_severity = 'none'\n", "watch"),
        ("[flags]\nescalation = 'yes'\n", "true or false"),
        ("[flags]\nnonempty = ['flags']\n", "unknown keys: nonempty"),
    ],
)
def test_a_malformed_case_is_refused(tmp_path, body, message):
    with pytest.raises(EvalError, match=message):
        load_case(write_case(tmp_path, body))


def test_a_transcript_that_does_not_say_it_is_fictional_is_refused(tmp_path):
    transcript = tmp_path / "chat.txt"
    transcript.write_text("# owner: Sam\n# match: Robin\nRobin: hey\nSam: hi\n", encoding="utf-8")
    with pytest.raises(EvalError, match="does not say it is fictional"):
        load_case(write_case(tmp_path, "[flags]\n", transcript))


def test_a_missing_transcript_is_refused(tmp_path):
    with pytest.raises(EvalError, match="no transcript at"):
        load_case(write_case(tmp_path, "[flags]\n", tmp_path / "gone.txt"))


def test_cite_terms_may_be_part_of_a_line(tmp_path):
    case = load_case(write_case(tmp_path, "[[analyze.cite]]\nwhy = 'x'\nany = ['first date']\n"))
    assert case.sections["analyze"].cite[0].terms == ("first date",)


# -- scoring a personality read --------------------------------------------------------


def test_a_good_read_passes_everything(tmp_path):
    case = load_case(
        write_case(
            tmp_path,
            """
[analyze]
confidence = ["medium", "high"]
nonempty = ["green_flags"]

[[analyze.cite]]
why = "quotes the follow-up"
any = ["how did the presentation go?"]

[[analyze.mention]]
why = "mentions plans"
any = ["plan"]

[[analyze.avoid]]
why = "no verdicts"
terms = ["emotionally unavailable"]
""",
        )
    )
    checks = score_read(case, "analyze", report())
    assert failures(checks) == {}
    assert len(checks) == 7


def test_mention_reads_prose_not_quotes(tmp_path):
    # "presentation" is in the quote, and only there: a read that quotes the line has not
    # thereby said anything about it.
    case = load_case(
        write_case(tmp_path, "[[analyze.mention]]\nwhy = 'says it'\nany = ['presentation']\n")
    )
    assert failures(score_read(case, "analyze", report())) == {"says it": 'no "presentation"'}


def test_mention_matches_at_the_start_of_a_word_only(tmp_path):
    case = load_case(write_case(tmp_path, "[[analyze.mention]]\nwhy = 'says it'\nany = ['plan']\n"))
    explained = report(headline="An explanation.", communication_style="Brief.")
    assert "says it" in failures(score_read(case, "analyze", explained))


def test_mention_can_be_held_to_one_field(tmp_path):
    case = load_case(
        write_case(
            tmp_path,
            "[[analyze.mention]]\nwhy = 'x'\nany = ['sample']\nfield = 'worth_watching'\n",
        )
    )
    # "sample" is in the caveat, which is not the field asked about.
    assert failures(score_read(case, "analyze", report())) == {"x": 'no "sample" in worth_watching'}


def test_avoid_fails_on_any_use(tmp_path):
    case = load_case(
        write_case(tmp_path, "[[analyze.avoid]]\nwhy = 'no verdicts'\nterms = ['cold']\n")
    )
    read = report(communication_style="Not cold, just busy.")
    assert failures(score_read(case, "analyze", read)) == {"no verdicts": 'said "cold"'}


def test_a_misquote_and_a_label_fail_without_being_asked_for(tmp_path):
    case = load_case(write_case(tmp_path, "[analyze]\n"))
    trait = Trait(
        name="self-focused",
        reading="Reads as narcissistic.",
        confidence="low",
        evidence=[Evidence(quote="I never said this", why_it_matters="x")],
    )
    failed = failures(score_read(case, "analyze", report(traits=[trait])))
    assert failed == {
        "every claim quotes the match": 'not in the transcript: "I never said this"',
        "no diagnostic labels": 'used "narcissis"',
    }


def test_a_quote_from_the_owner_does_not_count_as_the_match(tmp_path):
    case = load_case(write_case(tmp_path, "[analyze]\n"))
    trait = Trait(
        name="x",
        reading="x",
        confidence="low",
        evidence=[Evidence(quote="the river path, mostly", why_it_matters="x")],
    )
    assert "every claim quotes the match" in failures(
        score_read(case, "analyze", report(traits=[trait]))
    )


def test_confidence_and_nonempty(tmp_path):
    case = load_case(
        write_case(tmp_path, "[analyze]\nconfidence = 'low'\nnonempty = ['worth_watching']\n")
    )
    assert failures(score_read(case, "analyze", report())) == {
        "confidence is low": "it was medium",
        "worth watching is not empty": "it was empty",
    }


# -- scoring a red-flag check ----------------------------------------------------------

PRESSURE = Path("examples/pressure_chat.txt").resolve()


def test_expected_flags_and_escalation(tmp_path):
    case = load_case(
        write_case(
            tmp_path,
            """
[flags]
escalation = true

[[flags.expect]]
why = "names the tracking"
category = "monitoring"
min_severity = "danger"

[[flags.expect]]
why = "names the isolation"
category = ["isolation", "coercion"]
""",
            PRESSURE,
        )
    )
    found = flag_report(flag("monitoring", "concern"))
    failed = failures(score_read(case, "flags", found))
    assert failed == {
        "names the tracking": "found monitoring (concern)",
        "names the isolation": "found monitoring (concern)",
        "opens with the safety notice": "it did not",
    }


def test_no_flags_means_none_at_all(tmp_path):
    case = load_case(write_case(tmp_path, "[flags]\nmax_severity = 'none'\n"))
    assert failures(score_read(case, "flags", flag_report())) == {}
    watched = flag_report(flag("evasion", "watch", "no apology needed"))
    assert failures(score_read(case, "flags", watched)) == {"no flags": "found evasion (watch)"}


def test_a_ceiling_allows_flags_up_to_it(tmp_path):
    case = load_case(write_case(tmp_path, "[flags]\nmax_severity = 'watch'\n"))
    watched = flag_report(flag("evasion", "watch", "no apology needed"))
    assert failures(score_read(case, "flags", watched)) == {}


def test_a_flag_lost_to_grounding_counts_against_the_check(tmp_path):
    case = load_case(write_case(tmp_path, "[flags]\n"))
    assert failures(score_read(case, "flags", flag_report(discarded=2))) == {
        "no flag lost to a misquote": "2 dropped by grounding"
    }


def test_the_fixed_safety_notice_is_not_the_model_speaking(tmp_path):
    # The notice names tracking in words written in advance. A mention check must not be
    # satisfied by them, or every danger case would "say" what the notice says.
    case = load_case(
        write_case(tmp_path, "[[flags.mention]]\nwhy = 'x'\nany = ['trust']\n", PRESSURE)
    )
    scan = FlagScan(
        flags=[flag("monitoring", "danger")], summary="Something is wrong.", confidence="high"
    )
    grounded = ground_flags(scan, read_transcript(PRESSURE))
    assert grounded.escalation is not None
    assert "trust" in grounded.escalation.to_text()
    assert "x" in failures(score_read(case, "flags", grounded))


# -- running ---------------------------------------------------------------------------


def test_run_cases_against_recordings(tmp_path, replay_client):
    # The real analyzers, redaction and grounding included, answered by the recordings
    # the rest of the suite uses for the same transcript.
    case = load_case(
        write_case(
            tmp_path,
            """
[analyze]
nonempty = ["green_flags"]

[[analyze.cite]]
why = "quotes the follow-up"
any = ["how did the presentation go? you said it was thursday"]

[[analyze.mention]]
why = "mentions the coffee cart"
any = ["coffee cart"]

[flags]
max_severity = "none"
""",
        )
    )
    client = replay_client("analyze_sample", "flags_sample")
    card = run_cases([case], client=client)
    assert client.remaining == 0
    assert [(r.case, r.analysis) for r in card.reads] == [("case", "analyze"), ("case", "flags")]
    assert card.passed == card.total == 8
    # Names are put back before scoring, as they are before the owner reads anything.
    assert "Robin" in card.reads[0].report["headline"]


def test_a_refused_read_fails_every_check(tmp_path, replay_client):
    case = load_case(write_case(tmp_path, "[analyze]\nconfidence = 'low'\n"))
    card = run_cases([case], client=replay_client("analyze_refusal"))
    (read,) = card.reads
    assert read.error is not None and "declined" in read.error
    assert (read.passed, len(read.checks)) == (0, 3)
    assert "ERROR" in card.to_text()


def test_repeat_and_the_scorecard(tmp_path, monkeypatch):
    reads = iter([report(), report(confidence="high")])
    monkeypatch.setitem(evals.ANALYSES, "analyze", (lambda *a, **k: next(reads), "system"))
    case = load_case(write_case(tmp_path, "[analyze]\nconfidence = 'medium'\n"))
    seen = []
    card = run_cases([case], client=object(), repeat=2, on_read=seen.append)
    assert len(seen) == 2
    assert (card.passed, card.total) == (5, 6)

    text = card.to_text()
    assert "case  analyze  5/6" in text
    assert "FAIL confidence is medium (1 of 2 runs): it was high" in text
    assert "Score: 5/6 checks (83%) across 1 cases, 2 runs each" in text

    data = json.loads(card.to_json())
    assert set(data["prompts"]) == {"analyze", "flags"}
    assert data["prompts"]["analyze"] == evals.hashlib.sha256(b"system").hexdigest()
    assert data["reads"][1]["report"]["confidence"] == "high"


def test_an_empty_scorecard_renders():
    card = Scorecard([], model="m", effort="high")
    assert card.to_text().endswith("Score: 0/0 checks (0%) across 0 cases, m at high effort.")


# -- comparing -------------------------------------------------------------------------


def scorecard(tmp_path, monkeypatch, *reports, system="system", body=None, name="card"):
    """Run a case against fixed reports and save the scorecard, as `run --out` would."""
    reads = iter(reports)
    monkeypatch.setitem(evals.ANALYSES, "analyze", (lambda *a, **k: next(reads), system))
    body = (
        body
        or """
[analyze]
confidence = "medium"

[[analyze.mention]]
why = "mentions the weekend"
any = ["weekend"]

[[analyze.avoid]]
why = "no verdict on who Robin is"
terms = ["needy"]
"""
    )
    case = load_case(write_case(tmp_path, body))
    card = run_cases([case], client=object(), analyses=["analyze"], repeat=len(reports))
    path = tmp_path / f"{name}.json"
    path.write_text(card.to_json(), encoding="utf-8")
    return path


def test_checks_carry_their_kind(tmp_path):
    case = load_case(write_case(tmp_path, "[analyze]\nconfidence = 'medium'\n"))
    kinds = [c.kind for c in score_read(case, "analyze", report())]
    assert kinds == ["confidence", "grounding", "labels"]


def test_compare_names_what_was_won_and_lost(tmp_path, monkeypatch, capsys):
    before = scorecard(tmp_path, monkeypatch, report(), report(), name="before")
    weekend = "Robin asks about the weekend."
    after = scorecard(
        tmp_path,
        monkeypatch,
        report(communication_style=weekend, confidence="high"),
        report(communication_style=weekend),
        system="a better system",
        name="after",
    )
    result = evals.compare_scorecards(evals.load_scorecard(before), evals.load_scorecard(after))
    assert result.prompts_changed == ["analyze"]
    assert [(c.name, c.before, c.after) for c in result.gained] == [
        ("mentions the weekend", (0, 2), (2, 2))
    ]
    assert [(c.name, c.before, c.after, c.one_run) for c in result.lost] == [
        ("confidence is medium", (2, 2), (1, 2), True)
    ]
    assert result.shared == 5 and not result.guards_lost

    assert main(["compare", str(before), str(after)]) == 0
    out = capsys.readouterr().out
    assert "Prompts changed: analyze" in out
    assert "confidence is medium  2/2 -> 1/2  ~" in out
    assert "now: it was high" in out
    assert "mentions the weekend  0/2 -> 2/2\n" in out
    assert "~ 1 of 2 changes moved by a single run" in out
    assert "Score on the 5 checks both made: 80% -> 90% (+10 points)." in out
    # Lost is listed before gained: it is what to read first.
    assert out.index("Lost") < out.index("Gained")


def test_a_lost_guard_fails_the_comparison_whatever_the_score(tmp_path, monkeypatch, capsys):
    before = scorecard(tmp_path, monkeypatch, report(), name="before")
    after = scorecard(
        tmp_path,
        monkeypatch,
        report(
            communication_style="The weekend again. A bit needy, and avoidant too.",
        ),
        system="worse",
        name="after",
    )
    result = evals.compare_scorecards(evals.load_scorecard(before), evals.load_scorecard(after))
    assert result.rate_after > result.rate_before - 0.5
    assert {c.name for c in result.guards_lost} == {
        "no verdict on who Robin is",
        "no diagnostic labels",
    }
    assert main(["compare", str(before), str(after)]) == 1
    out = capsys.readouterr().out
    assert "  GUARD case  analyze  no diagnostic labels  1/1 -> 0/1  ~" in out
    assert "2 guard checks lost." in out


def test_unchanged_prompts_mean_the_difference_is_sampling(tmp_path, monkeypatch, capsys):
    before = scorecard(tmp_path, monkeypatch, report(), name="before")
    after = scorecard(tmp_path, monkeypatch, report(confidence="low"), name="after")
    assert main(["compare", str(before), str(after)]) == 0
    out = capsys.readouterr().out
    assert "Prompts changed: none" in out
    assert "every difference below is sampling" in out


def test_identical_scorecards_change_nothing(tmp_path, monkeypatch, capsys):
    card = scorecard(tmp_path, monkeypatch, report())
    assert main(["compare", str(card), str(card)]) == 0
    out = capsys.readouterr().out
    assert "No check changed." in out
    assert "80% -> 80% (+0 points)" in out


def test_a_refused_read_counts_as_failing_every_check(tmp_path, monkeypatch):
    before = scorecard(tmp_path, monkeypatch, report(), name="before")

    def refuse(*args, **kwargs):
        raise evals.ModelRefusal(None, "declined to read this one")

    monkeypatch.setitem(evals.ANALYSES, "analyze", (refuse, "refused"))
    case = load_case(write_case(tmp_path, "[analyze]\nconfidence = 'medium'\n"))
    after = tmp_path / "after.json"
    after.write_text(run_cases([case], client=object(), analyses=["analyze"]).to_json())

    result = evals.compare_scorecards(evals.load_scorecard(before), evals.load_scorecard(after))
    # The refusal is not a check of its own; it is a failed run of every check the other
    # side's read made, the ones only the before case file asked for included.
    assert {c.name for c in result.lost} == {
        "confidence is medium",
        "no verdict on who Robin is",
        "every claim quotes the match",
        "no diagnostic labels",
    }
    assert all("declined" in c.detail for c in result.lost)
    assert (result.rate_after, result.unmatched) == (0.0, 0)


def test_different_repeats_are_compared_as_rates(tmp_path, monkeypatch):
    before = scorecard(tmp_path, monkeypatch, report(), report(confidence="high"), name="b")
    after = scorecard(tmp_path, monkeypatch, report(), name="a")
    result = evals.compare_scorecards(evals.load_scorecard(before), evals.load_scorecard(after))
    ((change,),) = [result.gained]
    assert (change.before, change.after, change.one_run) == ((1, 2), (1, 1), True)


def test_scorecards_from_different_models_are_not_compared(tmp_path, monkeypatch, capsys):
    before = scorecard(tmp_path, monkeypatch, report(), name="before")
    data = json.loads(before.read_text())
    data["effort"] = "low"
    after = tmp_path / "after.json"
    after.write_text(json.dumps(data))
    assert main(["compare", str(before), str(after)]) == 2
    assert "different efforts" in capsys.readouterr().err


def test_older_scorecards_without_kinds_still_compare(tmp_path, monkeypatch):
    before = scorecard(tmp_path, monkeypatch, report(), name="before")
    after = scorecard(
        tmp_path, monkeypatch, report(headline="Robin seems narcissistic."), name="after"
    )
    stripped = []
    for path in (before, after):
        data = json.loads(path.read_text())
        for read in data["reads"]:
            for check in read["checks"]:
                del check["kind"]
        stripped.append(data)
    result = evals.compare_scorecards(*stripped)
    # The fixed names are recovered; the case author's own avoid check cannot be.
    assert [c.name for c in result.guards_lost] == ["no diagnostic labels"]


def test_two_checks_with_the_same_name_stay_apart(tmp_path, monkeypatch):
    body = """
[analyze]
[[analyze.mention]]
why = "says something"
any = ["coffee cart"]

[[analyze.mention]]
why = "says something"
any = ["presentation"]
"""
    before = scorecard(tmp_path, monkeypatch, report(), body=body, name="before")
    after = scorecard(
        tmp_path,
        monkeypatch,
        report(communication_style="Mentions the presentation."),
        body=body,
        name="after",
    )
    result = evals.compare_scorecards(evals.load_scorecard(before), evals.load_scorecard(after))
    assert len(result.gained) == 1 and not result.lost


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (None, "No scorecard at"),
        ("not json", "is not JSON"),
        ('{"model": "m"}', "is not a scorecard"),
    ],
)
def test_a_bad_scorecard_is_refused(tmp_path, capsys, content, message):
    path = tmp_path / "card.json"
    if content is not None:
        path.write_text(content)
    assert main(["compare", str(path), str(path)]) == 2
    assert message in capsys.readouterr().err
