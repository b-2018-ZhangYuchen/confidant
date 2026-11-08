"""Steps 3 and 6 of the lab: score any backend on the labelled splits and the gold set.

Every read goes through :func:`~confidant.analysis.flags.analyze_flags`, so what is scored
is what a user would see: quotes grounded, danger floors applied. The metrics:

- **valid** — the share of answers that parsed into the schema at all. An unusable answer
  counts as "no flags" in every other metric, because that is what the user would be left
  with, and it is also counted here so the two cannot be confused.
- **detection** — does a transcript with a planted flag get at least one flag, and does a
  clean or hard-negative one get none: precision, recall, F1, and the false-alarm rate
  on hard negatives and on clean chats separately.
- **category F1** — micro-F1 over (transcript, category) pairs.
- **severity** — on flagged transcripts that were caught, whether the highest tier found
  matches the label's; and **danger missed**, the number of danger transcripts whose
  highest tier came back below danger. That one is a guard: no gain elsewhere makes up
  for it.
- **grounding** — the share of flags the model returned whose quotes were found.
- **labels** — reports that use a diagnostic label (:data:`confidant.evals.DIAGNOSTIC_TERMS`).
- **speed** — seconds per read, generated tokens per second, peak memory.

The gold set (``examples/eval/``) is scored with :func:`confidant.evals.score_read`, the
same checks the prompt hill-climb uses, as a pass rate.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from confidant.analysis.flags import TIERS, FlagBackend, FlagReport, analyze_flags
from confidant.evals import DIAGNOSTIC_TERMS, load_cases, score_read
from confidant.lab.backend import LocalOutputError
from confidant.lab.data import Example

__all__ = ["Read", "Scores", "evaluate", "evaluate_gold", "score"]

_LABEL = re.compile(r"\b(" + "|".join(re.escape(t) for t in DIAGNOSTIC_TERMS) + ")", re.I)


@dataclass
class Read:
    """One transcript through one backend."""

    id: str
    kind: str
    category: str | None
    severity: str | None
    valid: bool
    found: list[tuple[str, str]] = field(default_factory=list)
    returned: int = 0
    discarded: int = 0
    labels: list[str] = field(default_factory=list)
    seconds: float = 0.0
    tokens_per_second: float = 0.0
    peak_memory_gb: float = 0.0
    error: str | None = None
    raw: str | None = None

    @property
    def flagged(self) -> bool:
        return bool(self.found)

    @property
    def top_tier(self) -> int:
        return max((TIERS[s] for _, s in self.found), default=0)


def _prose(report: FlagReport) -> str:
    parts = [report.summary]
    for f in report.flags:
        parts += [f.behavior, f.innocent_reading or ""]
        parts += [e.why_it_matters for e in f.evidence]
    return " ".join(parts)


def read_one(backend: FlagBackend, example: Example) -> Read:
    read = Read(example.id, example.kind, example.category, example.severity, valid=False)
    started = time.perf_counter()
    try:
        report = analyze_flags(example.conversation(), backend=backend)
    except LocalOutputError as exc:
        read.error = str(exc)
    else:
        read.valid = True
        read.found = [(f.category, f.severity) for f in report.flags]
        read.discarded = report.discarded
        read.returned = len(report.flags) + report.discarded
        read.labels = sorted({m.casefold() for m in _LABEL.findall(_prose(report))})
    read.seconds = time.perf_counter() - started
    raw = getattr(backend, "last_raw", None)
    read.raw = raw[:4000] if raw else None
    stats = getattr(backend, "last_stats", None)
    if stats is not None:
        read.tokens_per_second = stats.generation_tps
        read.peak_memory_gb = stats.peak_memory_gb
    return read


def evaluate(
    backend: FlagBackend,
    examples: Sequence[Example],
    *,
    on_read: Callable[[Read], None] | None = None,
) -> list[Read]:
    reads = []
    for example in examples:
        reads.append(read_one(backend, example))
        if on_read is not None:
            on_read(reads[-1])
    return reads


def _f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


@dataclass
class Scores:
    n: int
    valid: float
    precision: float
    recall: float
    f1: float
    false_alarm_hard: float
    false_alarm_clean: float
    category_f1: float
    severity_accuracy: float
    danger_missed: int
    danger_total: int
    grounded: float
    labels: int
    seconds_per_read: float
    tokens_per_second: float
    peak_memory_gb: float
    gold_pass: float | None = None

    def row(self) -> dict[str, Any]:
        return asdict(self)


def score(reads: Sequence[Read]) -> Scores:
    pos = [r for r in reads if r.kind == "flag"]
    hard = [r for r in reads if r.kind == "hard_negative"]
    clean = [r for r in reads if r.kind == "clean"]
    tp = sum(r.flagged for r in pos)
    fp = sum(r.flagged for r in hard + clean)
    precision, recall, f1 = _f1(tp, fp, len(pos) - tp)

    ctp = cfp = cfn = 0
    for r in reads:
        gold = {r.category} if r.kind == "flag" else set()
        got = {c for c, _ in r.found}
        ctp, cfp, cfn = ctp + len(gold & got), cfp + len(got - gold), cfn + len(gold - got)

    caught = [r for r in pos if r.flagged]
    danger = [r for r in pos if r.severity == "danger"]
    returned = sum(r.returned for r in reads)
    timed = [r for r in reads if r.tokens_per_second]
    return Scores(
        n=len(reads),
        valid=sum(r.valid for r in reads) / len(reads) if reads else 0.0,
        precision=precision,
        recall=recall,
        f1=f1,
        false_alarm_hard=sum(r.flagged for r in hard) / len(hard) if hard else 0.0,
        false_alarm_clean=sum(r.flagged for r in clean) / len(clean) if clean else 0.0,
        category_f1=_f1(ctp, cfp, cfn)[2],
        severity_accuracy=(
            sum(r.top_tier == TIERS[r.severity] for r in caught) / len(caught) if caught else 0.0
        ),
        danger_missed=sum(r.top_tier < TIERS["danger"] for r in danger),
        danger_total=len(danger),
        grounded=1 - sum(r.discarded for r in reads) / returned if returned else 1.0,
        labels=sum(bool(r.labels) for r in reads),
        seconds_per_read=sum(r.seconds for r in reads) / len(reads) if reads else 0.0,
        tokens_per_second=sum(r.tokens_per_second for r in timed) / len(timed) if timed else 0.0,
        peak_memory_gb=max((r.peak_memory_gb for r in reads), default=0.0),
    )


def evaluate_gold(backend: FlagBackend, cases_dir: Path = Path("examples/eval")) -> dict[str, Any]:
    """The gold set through ``backend``, scored by the hill-climb's own checks."""
    passed = total = unusable = 0
    per_case = {}
    for case in load_cases(cases_dir):
        if "flags" not in case.sections:
            continue
        try:
            report = analyze_flags(case.conversation, backend=backend)
        except LocalOutputError:
            unusable += 1
            # An unusable answer fails every check the case has, the free ones included.
            checks = len(
                score_read(case, "flags", FlagReport(flags=[], summary="", confidence="low"))
            )
            per_case[case.name] = (0, checks)
            total += checks
            continue
        results = score_read(case, "flags", report)
        ok = sum(c.passed for c in results)
        per_case[case.name] = (ok, len(results))
        passed, total = passed + ok, total + len(results)
    return {
        "pass_rate": passed / total if total else 0.0,
        "passed": passed,
        "checks": total,
        "unusable": unusable,
        "cases": per_case,
    }


def save_run(
    path: Path,
    *,
    config: dict[str, Any],
    reads: Iterable[Read],
    scores: Scores,
    gold: dict[str, Any] | None = None,
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    (path / "scores.json").write_text(json.dumps(scores.row(), indent=2) + "\n")
    with (path / "reads.jsonl").open("w", encoding="utf-8") as out:
        for r in reads:
            out.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")
    if gold is not None:
        (path / "gold.json").write_text(json.dumps(gold, indent=2) + "\n")


SCORE_COLUMNS = (
    ("valid", "Valid", "{:.0%}"),
    ("f1", "F1", "{:.2f}"),
    ("precision", "P", "{:.2f}"),
    ("recall", "R", "{:.2f}"),
    ("false_alarm_hard", "FA hard", "{:.0%}"),
    ("false_alarm_clean", "FA clean", "{:.0%}"),
    ("category_f1", "Cat F1", "{:.2f}"),
    ("severity_accuracy", "Sev acc", "{:.0%}"),
    ("danger_missed", "Danger missed", "{}"),
    ("grounded", "Grounded", "{:.0%}"),
    ("labels", "Labels", "{}"),
    ("seconds_per_read", "s/read", "{:.1f}"),
    ("gold_pass", "Gold", "{:.0%}"),
)


def scores_table(rows: Sequence[tuple[str, dict[str, Any]]]) -> str:
    head = "| Run | " + " | ".join(c[1] for c in SCORE_COLUMNS) + " |"
    lines = [head, "|---|" + "---:|" * len(SCORE_COLUMNS)]
    for name, s in rows:
        cells = []
        for key, _, fmt in SCORE_COLUMNS:
            value = s.get(key)
            if key == "danger_missed" and value is not None:
                cells.append(f"{value}/{s.get('danger_total', '?')}")
            else:
                cells.append("—" if value is None else fmt.format(value))
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
