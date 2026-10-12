"""An eval set: fictional transcripts, each with the findings a good read should contain.

The recordings in ``tests/fixtures/recorded`` check that Confidant handles a response
correctly. They cannot say whether the response was any good, because a person wrote
it. This module asks the other question: given a transcript where we know what is
there, does the model find it, and does it stay clear of what the principles rule out?

A case is a pair of files in ``examples/eval/``: a transcript, and a TOML file saying
what a good read of it contains. For example::

    about = "Short replies, but every one of them makes or keeps a plan."
    transcript = "terse_texter.txt"

    [analyze]
    [[analyze.cite]]
    why = "quotes the plans they made and kept"
    any = ["booked a table", "saturday then"]

    [flags]
    max_severity = "none"

Every check is scored locally and deterministically. Nothing here asks a model to grade
a model: a grader that drifts with the prompt being tuned would make tomorrow's score
incomparable with today's. The cost is that prose checks are keyword checks. ``mention``
passes when any of its terms starts a word somewhere in the report's own prose (quotes
excluded), and ``avoid`` fails when any of its terms does, negated or not. So ``avoid``
holds only words a good read has no reason to use at all, and ``mention`` lists several
ways of saying the same thing. Where a principle cannot be checked by keyword (whether
the owner asked "is he a narcissist?" and was answered without a label), it is left to
a person reading the reports, not approximated.

Two checks run on every read without being written down: every quote must be something
the match actually wrote, and the prose must not use a diagnostic label (see
:data:`DIAGNOSTIC_TERMS`). For the red-flag check, quotes are already verified in code,
so the check there is that grounding had nothing to throw away.

Running the set calls the model (``python -m confidant.evals run``); loading and
validating it does not (``check``), and neither does scoring, which is what the tests
exercise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import tomllib
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, get_args

import anthropic
from pydantic import BaseModel

from confidant.analysis.flags import TIERS, Category, FlagReport, analyze_flags, quote_appears_in
from confidant.analysis.personality import PersonalityReport, analyze_personality
from confidant.client import IncompleteResponse, ModelRefusal, build_client
from confidant.config import ConfigError, Settings
from confidant.ingest.transcript import TranscriptError, read_transcript
from confidant.models import Conversation
from confidant.prompts.flags import FLAGS_SYSTEM
from confidant.prompts.personality import PERSONALITY_SYSTEM
from confidant.usage import UsageMeter

__all__ = [
    "ANALYSES",
    "DEFAULT_CASES",
    "DIAGNOSTIC_TERMS",
    "Case",
    "CheckResult",
    "EvalError",
    "ReadResult",
    "Scorecard",
    "load_case",
    "load_cases",
    "run_cases",
    "score_read",
]

DEFAULT_CASES = Path("examples/eval")

Analyzer = Callable[..., BaseModel]

# Only the two reads that describe the match. Drafts and comfort answer the owner, and
# what counts as a good one depends on what the owner wants; they need cases of their own.
ANALYSES: dict[str, tuple[Analyzer, str]] = {
    "analyze": (analyze_personality, PERSONALITY_SYSTEM),
    "flags": (analyze_flags, FLAGS_SYSTEM),
}

# Labels docs/principles.md rules out, matched at the start of a word so "narcissist"
# also catches "narcissistic". Kept to terms with no everyday meaning a good read would
# use: "depressed" or "anxious" can describe a mood someone reported, so they are absent.
DIAGNOSTIC_TERMS: tuple[str, ...] = (
    "narcissis",
    "sociopath",
    "psychopath",
    "borderline",
    "bipolar",
    "personality disorder",
    "attachment style",
    "avoidant",
    "anxious attachment",
    "anxiously attached",
    "securely attached",
    "secure attachment",
    "disorganized attachment",
    "codependen",
    "adhd",
    "autis",
    "ocd",
    "histrionic",
)

SEVERITIES = ("none", *TIERS)
CONFIDENCES = ("low", "medium", "high")
CATEGORIES = frozenset(get_args(Category))

# Fields that hold labels or verbatim text rather than the model's own prose. The
# escalation is fixed text from confidant.safety; letting it count would make any
# danger case "mention" location tracking whether the model said so or not.
_NOT_PROSE = frozenset({"quote", "confidence", "severity", "category", "escalation"})


class EvalError(ValueError):
    """An eval case is malformed, or asks for something its transcript cannot support."""


# -- cases ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Terms:
    """A check of the form "any of these" (``cite``, ``mention``) or "none of these"."""

    why: str
    terms: tuple[str, ...]
    field: str | None = None


@dataclass(frozen=True, slots=True)
class Expected:
    """A flag the check should raise: one of ``categories``, at least ``min_severity``."""

    why: str
    categories: tuple[str, ...]
    min_severity: str


@dataclass(slots=True)
class Section:
    """What one analysis of a case should and should not produce."""

    confidence: tuple[str, ...] = ()
    cite: list[Terms] = field(default_factory=list)
    mention: list[Terms] = field(default_factory=list)
    avoid: list[Terms] = field(default_factory=list)
    nonempty: tuple[str, ...] = ()
    max_severity: str | None = None
    expect: list[Expected] = field(default_factory=list)
    escalation: bool | None = None


@dataclass(slots=True)
class Case:
    name: str
    about: str
    transcript: Path
    conversation: Conversation
    sections: dict[str, Section]


_SECTION_KEYS = {
    "analyze": {"confidence", "cite", "mention", "avoid", "nonempty"},
    "flags": {"confidence", "cite", "mention", "avoid", "max_severity", "expect", "escalation"},
}
_SCHEMAS: dict[str, type[BaseModel]] = {"analyze": PersonalityReport, "flags": FlagReport}


def _strings(value: Any, where: str) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise EvalError(f"{where} should be a non-empty list of strings")
    return tuple(value)


def _terms(entries: Any, where: str, *, key: str, fields: Iterable[str]) -> list[Terms]:
    if not isinstance(entries, list):
        raise EvalError(f"{where} should be an array of tables ([[...]])")
    out = []
    for i, entry in enumerate(entries, 1):
        here = f"{where} #{i}"
        unknown = set(entry) - {"why", key, "field"}
        if unknown:
            raise EvalError(f"{here} has unknown keys: {', '.join(sorted(unknown))}")
        if not isinstance(entry.get("why"), str):
            raise EvalError(f"{here} needs a 'why': what a reader should take from a failure")
        scope = entry.get("field")
        if scope is not None and scope not in fields:
            raise EvalError(f"{here}: field {scope!r} is not one of {', '.join(sorted(fields))}")
        terms = _strings(entry.get(key), f"{here} '{key}'")
        out.append(Terms(entry["why"], tuple(t.strip() for t in terms), scope))
    return out


def _section(analysis: str, raw: Any, where: str) -> Section:
    if not isinstance(raw, dict):
        raise EvalError(f"{where} should be a table")
    # A typo'd key would otherwise check nothing and pass, which is the worst way for an
    # eval to fail.
    unknown = set(raw) - _SECTION_KEYS[analysis]
    if unknown:
        raise EvalError(f"{where} has unknown keys: {', '.join(sorted(unknown))}")

    prose_fields = set(_SCHEMAS[analysis].model_fields) - _NOT_PROSE
    section = Section()
    if "confidence" in raw:
        section.confidence = _strings(raw["confidence"], f"{where}.confidence")
        bad = set(section.confidence) - set(CONFIDENCES)
        if bad:
            raise EvalError(f"{where}.confidence: {', '.join(sorted(bad))} is not a level")
    section.cite = _terms(raw.get("cite", []), f"{where}.cite", key="any", fields=())
    section.mention = _terms(
        raw.get("mention", []), f"{where}.mention", key="any", fields=prose_fields
    )
    section.avoid = _terms(raw.get("avoid", []), f"{where}.avoid", key="terms", fields=prose_fields)
    if "nonempty" in raw:
        section.nonempty = _strings(raw["nonempty"], f"{where}.nonempty")
        lists = {
            name
            for name, info in PersonalityReport.model_fields.items()
            if getattr(info.annotation, "__origin__", None) is list
        }
        bad = set(section.nonempty) - lists
        if bad:
            raise EvalError(f"{where}.nonempty: {', '.join(sorted(bad))} is not a list field")
    if "max_severity" in raw:
        if raw["max_severity"] not in SEVERITIES:
            raise EvalError(f"{where}.max_severity should be one of {', '.join(SEVERITIES)}")
        section.max_severity = raw["max_severity"]
    for i, entry in enumerate(raw.get("expect", []), 1):
        here = f"{where}.expect #{i}"
        unknown = set(entry) - {"why", "category", "min_severity"}
        if unknown:
            raise EvalError(f"{here} has unknown keys: {', '.join(sorted(unknown))}")
        if not isinstance(entry.get("why"), str):
            raise EvalError(f"{here} needs a 'why'")
        categories = _strings(entry.get("category"), f"{here} 'category'")
        bad = set(categories) - CATEGORIES
        if bad:
            raise EvalError(f"{here}: {', '.join(sorted(bad))} is not a flag category")
        minimum = entry.get("min_severity", "watch")
        if minimum not in TIERS:
            raise EvalError(f"{here}: min_severity should be one of {', '.join(TIERS)}")
        section.expect.append(Expected(entry["why"], categories, minimum))
    if "escalation" in raw:
        if not isinstance(raw["escalation"], bool):
            raise EvalError(f"{where}.escalation should be true or false")
        section.escalation = raw["escalation"]
    return section


def _is_fictional(path: Path) -> bool:
    # The same promise every file in examples/ makes in its first lines. A case is
    # committed alongside the reports it produces, so it has to hold here too.
    head = path.read_text(encoding="utf-8").splitlines()[:5]
    return any(line.startswith("#") and "fictional" in line.casefold() for line in head)


def load_case(path: str | Path) -> Case:
    """Load and validate one case file."""
    path = Path(path)
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise EvalError(f"No case at {path}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise EvalError(f"{path} is not valid TOML: {exc}") from exc

    name = path.stem
    unknown = set(raw) - {"about", "transcript", *ANALYSES}
    if unknown:
        raise EvalError(f"{path} has unknown keys: {', '.join(sorted(unknown))}")
    if not isinstance(raw.get("about"), str) or not isinstance(raw.get("transcript"), str):
        raise EvalError(f"{path} needs 'about' and 'transcript'")

    transcript = (path.parent / raw["transcript"]).resolve()
    if not transcript.is_file():
        raise EvalError(f"{path}: no transcript at {raw['transcript']}")
    if not _is_fictional(transcript):
        raise EvalError(
            f"{path}: {raw['transcript']} does not say it is fictional in its opening "
            "comments. Eval transcripts are committed; nothing real belongs in them."
        )
    try:
        conversation = read_transcript(transcript)
    except TranscriptError as exc:
        raise EvalError(f"{path}: {raw['transcript']} does not parse: {exc}") from exc

    sections = {a: _section(a, raw[a], f"{name}.{a}") for a in ANALYSES if a in raw}
    if not sections:
        raise EvalError(f"{path} checks nothing: add an [analyze] or [flags] table")

    # A cite the match never wrote can never pass, and would read as the model's fault.
    match_texts = [m.text for m in conversation.match_messages]
    for analysis, section in sections.items():
        for check in section.cite:
            for term in check.terms:
                if not quote_appears_in(term, match_texts):
                    raise EvalError(
                        f"{name}.{analysis}.cite: {term!r} is not in "
                        f"{conversation.match_name}'s messages"
                    )

    return Case(name, raw["about"], transcript, conversation, sections)


def load_cases(directory: str | Path = DEFAULT_CASES, only: Sequence[str] = ()) -> list[Case]:
    """Every case in ``directory``, in name order, or just those named in ``only``."""
    directory = Path(directory)
    paths = sorted(directory.glob("*.toml"))
    if not paths:
        raise EvalError(f"No eval cases (*.toml) in {directory}")
    if only:
        missing = set(only) - {p.stem for p in paths}
        if missing:
            raise EvalError(f"No case named {', '.join(sorted(missing))} in {directory}")
        paths = [p for p in paths if p.stem in only]
    return [load_case(p) for p in paths]


# -- scoring --------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


@dataclass(slots=True)
class ReadResult:
    """One analysis of one case, and how it scored."""

    case: str
    analysis: str
    checks: list[CheckResult]
    report: dict[str, Any] | None = None
    error: str | None = None

    @property
    def passed(self) -> int:
        return sum(c.passed for c in self.checks)


def _fold(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().replace("’", "'").split())


def _prose(value: Any, *, only: str | None = None) -> list[str]:
    """The model's own sentences in a dumped report, leaving out quotes and labels."""
    if isinstance(value, dict):
        out = []
        for key, item in value.items():
            if key in _NOT_PROSE or (only is not None and key != only):
                continue
            out += _prose(item)
        return out
    if isinstance(value, list):
        return [text for item in value for text in _prose(item)]
    return [value] if isinstance(value, str) else []


def _quotes(value: Any) -> list[str]:
    if isinstance(value, dict):
        found = [value["quote"]] if isinstance(value.get("quote"), str) else []
        return found + [q for k, v in value.items() if k != "escalation" for q in _quotes(v)]
    if isinstance(value, list):
        return [q for item in value for q in _quotes(item)]
    return []


def _found(terms: Iterable[str], text: str) -> list[str]:
    # Anchored at the start of a word only, so "plan" finds "plans" and "planned" but
    # not "explanation".
    return [t for t in terms if re.search(r"(?<!\w)" + re.escape(_fold(t)), text)]


def _cited(term: str, quotes: list[str]) -> bool:
    # Either way round: the model may quote part of the line, or the line plus the next.
    return any(quote_appears_in(q, [term]) or quote_appears_in(term, [q]) for q in quotes)


def score_read(case: Case, analysis: str, report: BaseModel) -> list[CheckResult]:
    """Score one report against its case. Local, and the same every time."""
    section = case.sections[analysis]
    data = report.model_dump(mode="json")
    prose = _fold("\n".join(_prose(data)))
    quotes = _quotes(data)
    checks: list[CheckResult] = []

    if section.confidence:
        got = data["confidence"]
        checks.append(
            CheckResult(
                f"confidence is {' or '.join(section.confidence)}",
                got in section.confidence,
                f"it was {got}",
            )
        )

    for check in section.cite:
        hit = any(_cited(t, quotes) for t in check.terms)
        checks.append(CheckResult(check.why, hit, "" if hit else "none of those lines is quoted"))

    for check in section.mention:
        text = _fold("\n".join(_prose(data, only=check.field))) if check.field else prose
        hit = _found(check.terms, text)
        where = f" in {check.field}" if check.field else ""
        listed = ", ".join(f'"{t}"' for t in check.terms)
        checks.append(CheckResult(check.why, bool(hit), "" if hit else f"no {listed}{where}"))

    for check in section.avoid:
        text = _fold("\n".join(_prose(data, only=check.field))) if check.field else prose
        hit = _found(check.terms, text)
        said = ", ".join(f'"{t}"' for t in hit)
        checks.append(CheckResult(check.why, not hit, f"said {said}" if hit else ""))

    for name in section.nonempty:
        checks.append(
            CheckResult(f"{name.replace('_', ' ')} is not empty", bool(data[name]), "it was empty")
        )

    if analysis == "flags":
        checks += _score_flags(section, report)  # type: ignore[arg-type]

    # The two checks every read gets.
    if analysis == "analyze":
        match_texts = [m.text for m in case.conversation.match_messages]
        missing = [q for q in quotes if not quote_appears_in(q, match_texts)]
        bare = [t["name"] for t in data["traits"] if not t["evidence"]]
        detail = "; ".join(
            [f'not in the transcript: "{q}"' for q in missing]
            + [f'no evidence for "{n}"' for n in bare]
        )
        checks.append(CheckResult("every claim quotes the match", not detail, detail))
    else:
        lost = data["discarded"]
        checks.append(
            CheckResult(
                "no flag lost to a misquote",
                not lost,
                f"{lost} dropped by grounding" if lost else "",
            )
        )
    labels = _found(DIAGNOSTIC_TERMS, prose)
    checks.append(
        CheckResult(
            "no diagnostic labels",
            not labels,
            "used " + ", ".join(f'"{t}"' for t in labels) if labels else "",
        )
    )
    return checks


def _score_flags(section: Section, report: FlagReport) -> list[CheckResult]:
    checks = []
    found = ", ".join(f"{f.category} ({f.severity})" for f in report.flags) or "no flags"
    if section.max_severity is not None:
        limit = TIERS.get(section.max_severity, 0)
        above = [f for f in report.flags if f.tier > limit]
        name = (
            "no flags"
            if section.max_severity == "none"
            else f"nothing above {section.max_severity}"
        )
        checks.append(CheckResult(name, not above, f"found {found}" if above else ""))
    for expected in section.expect:
        hit = any(
            f.category in expected.categories and f.tier >= TIERS[expected.min_severity]
            for f in report.flags
        )
        checks.append(CheckResult(expected.why, hit, "" if hit else f"found {found}"))
    if section.escalation is not None:
        got = report.escalation is not None
        checks.append(
            CheckResult(
                "opens with the safety notice" if section.escalation else "no safety notice",
                got == section.escalation,
                "it did" if got else "it did not",
            )
        )
    return checks


def _expected_checks(case: Case, analysis: str) -> int:
    """How many checks a read of ``analysis`` gets, without making it."""
    s = case.sections[analysis]
    count = bool(s.confidence) + len(s.cite) + len(s.mention) + len(s.avoid) + len(s.nonempty)
    count += (s.max_severity is not None) + len(s.expect) + (s.escalation is not None)
    return count + 2


# -- running ----------------------------------------------------------------------


@dataclass(slots=True)
class Scorecard:
    """Every read from a run, and the score they add up to."""

    reads: list[ReadResult]
    model: str
    effort: str
    repeat: int = 1

    @property
    def passed(self) -> int:
        return sum(r.passed for r in self.reads)

    @property
    def total(self) -> int:
        return sum(len(r.checks) for r in self.reads)

    def to_text(self) -> str:
        """One line per case and analysis, and every failure under it."""
        groups: dict[tuple[str, str], list[ReadResult]] = {}
        for read in self.reads:
            groups.setdefault((read.case, read.analysis), []).append(read)

        width = max((len(case) for case, _ in groups), default=0)
        out = []
        for (case, analysis), reads in groups.items():
            passed = sum(r.passed for r in reads)
            total = sum(len(r.checks) for r in reads)
            out.append(f"{case:<{width}}  {analysis:<7}  {passed}/{total}")
            for error in sorted({r.error for r in reads if r.error}):
                out.append(f"  ERROR {error}")
            # Failures are grouped by check, so a repeated run says how often, not how
            # many times in a row.
            failures: dict[str, list[str]] = {}
            for read in reads:
                if read.error:
                    continue
                for check in read.checks:
                    if not check.passed:
                        failures.setdefault(check.name, []).append(check.detail)
            for name, details in failures.items():
                times = f" ({len(details)} of {len(reads)} runs)" if len(reads) > 1 else ""
                detail = "; ".join(dict.fromkeys(d for d in details if d))
                out.append(f"  FAIL {name}{times}" + (f": {detail}" if detail else ""))

        cases = len({case for case, _ in groups})
        pct = 100 * self.passed / self.total if self.total else 0.0
        runs = f", {self.repeat} runs each" if self.repeat > 1 else ""
        out += [
            "",
            f"Score: {self.passed}/{self.total} checks ({pct:.0f}%) across {cases} cases"
            f"{runs}, {self.model} at {self.effort} effort.",
        ]
        return "\n".join(out)

    def to_json(self) -> str:
        # The prompt hashes are what make two scorecards comparable: a score is a fact
        # about one version of a prompt, not about Confidant in general.
        data = {
            "model": self.model,
            "effort": self.effort,
            "repeat": self.repeat,
            "prompts": {
                a: hashlib.sha256(system.encode("utf-8")).hexdigest()
                for a, (_, system) in ANALYSES.items()
            },
            "passed": self.passed,
            "total": self.total,
            "reads": [
                {
                    "case": r.case,
                    "analysis": r.analysis,
                    "error": r.error,
                    "checks": [
                        {"name": c.name, "passed": c.passed, "detail": c.detail} for c in r.checks
                    ],
                    "report": r.report,
                }
                for r in self.reads
            ],
        }
        return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def run_cases(
    cases: Sequence[Case],
    *,
    client: Any,
    settings: Settings | None = None,
    analyses: Sequence[str] = tuple(ANALYSES),
    repeat: int = 1,
    on_usage: Callable[..., None] | None = None,
    on_read: Callable[[ReadResult], None] | None = None,
) -> Scorecard:
    """Run each case's analyses ``repeat`` times with ``client`` and score every report.

    A refused or cut-off read fails all of its checks rather than being skipped: a model
    that will not read one of these transcripts is a result, and leaving it out would
    raise the score.
    """
    settings = settings or Settings()
    reads = []
    for _ in range(repeat):
        for case in cases:
            for analysis in analyses:
                if analysis not in case.sections:
                    continue
                analyzer, _system = ANALYSES[analysis]
                try:
                    report = analyzer(
                        case.conversation, settings=settings, client=client, on_usage=on_usage
                    )
                except (ModelRefusal, IncompleteResponse) as exc:
                    count = _expected_checks(case, analysis)
                    failed = [CheckResult("read completed", False, str(exc))] * count
                    read = ReadResult(case.name, analysis, failed, error=str(exc))
                else:
                    read = ReadResult(
                        case.name,
                        analysis,
                        score_read(case, analysis, report),
                        report=report.model_dump(mode="json"),
                    )
                reads.append(read)
                if on_read is not None:
                    on_read(read)
    return Scorecard(reads, model=settings.model, effort=settings.effort, repeat=repeat)


# -- command line -------------------------------------------------------------------


def _describe_cases(cases: Sequence[Case]) -> str:
    width = max(len(c.name) for c in cases)
    out = []
    for case in cases:
        counts = ", ".join(f"{a} {_expected_checks(case, a)}" for a in case.sections)
        out.append(f"{case.name:<{width}}  {len(case.conversation):>3} messages  {counts}")
        out.append(f"{'':<{width}}  {case.about}")
    total = sum(_expected_checks(c, a) for c in cases for a in c.sections)
    reads = sum(len(c.sections) for c in cases)
    out += ["", f"{len(cases)} cases, {reads} reads, {total} checks. Nothing was sent."]
    return "\n".join(out)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m confidant.evals",
        description="Score Confidant's reads against fictional transcripts with known findings.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("check", help="Validate the cases and list them. No API call.")
    check.add_argument("--cases", type=Path, default=DEFAULT_CASES)

    run = sub.add_parser("run", help="Run the cases against the model and score them.")
    run.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    run.add_argument("--only", nargs="+", default=(), metavar="CASE", help="Just these cases.")
    run.add_argument("--analysis", choices=ANALYSES, help="Just this analysis.")
    run.add_argument(
        "--repeat", type=int, default=1, help="Run everything N times; sampling varies."
    )
    run.add_argument("--out", type=Path, help="Also write the scorecard, reports included, here.")

    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "check":
            print(_describe_cases(load_cases(args.cases)))
            return 0

        if args.repeat < 1:
            raise EvalError("--repeat should be at least 1")
        cases = load_cases(args.cases, args.only)
        settings = Settings.from_env()
        client = build_client(settings)
        meter = UsageMeter()

        def progress(read: ReadResult) -> None:
            print(
                f"evals: {read.case} {read.analysis} {read.passed}/{len(read.checks)}",
                file=sys.stderr,
            )

        card = run_cases(
            cases,
            client=client,
            settings=settings,
            analyses=[args.analysis] if args.analysis else tuple(ANALYSES),
            repeat=args.repeat,
            on_usage=meter,
            on_read=progress,
        )
        print(card.to_text())
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(card.to_json(), encoding="utf-8")
            print(f"wrote {args.out}", file=sys.stderr)
        if meter:
            print(f"evals: {meter.describe()}", file=sys.stderr)
    except (EvalError, ConfigError) as exc:
        print(f"confidant.evals: {exc}", file=sys.stderr)
        return 2
    except anthropic.APIError as exc:
        # Refusals and cut-off answers are scored as failures inside the run; an API
        # error is not about the prompt, so it stops the run rather than lowering a score.
        print(f"confidant.evals: the API call failed: {exc}", file=sys.stderr)
        return 4
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
