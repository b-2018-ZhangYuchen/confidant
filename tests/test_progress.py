"""Tests for streamed progress: what the status line says, and what it never says."""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from confidant.analysis.flags import FlagScan, analyze_flags
from confidant.analysis.personality import PersonalityReport, analyze_personality
from confidant.cli import main
from confidant.client import ModelRefusal
from confidant.ingest.transcript import read_transcript
from confidant.progress import Progress, ProgressTracker, StatusLine, current_section
from confidant.recording import RecordingClient
from confidant.redaction import redact


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def watch(analysis, transcript: str, client) -> tuple[list[Progress], object]:
    seen: list[Progress] = []
    report = analysis(read_transcript(transcript), client=client, on_progress=seen.append)
    return seen, report


# -- reading the partial JSON -------------------------------------------------


@pytest.mark.parametrize(
    ("partial", "section"),
    [
        ("", None),
        ("{", None),
        ('{"headline": "Warm and curi', "headline"),
        ('{"headline": "x", "traits": [{"name": "asks', "traits"),
        # A key cut off mid-name is not reported until it is whole.
        ('{"headline": "x", "traits": [], "green_fl', "traits"),
        ('{"headline": "x", "confidence": "low"}', "confidence"),
        ("not json at all", None),
    ],
)
def test_current_section_is_the_last_top_level_key(partial, section):
    assert current_section(partial) == section


def test_a_nested_key_is_not_mistaken_for_a_section():
    partial = '{"traits": [{"name": "a", "evidence": [{"quote": "hey'
    assert current_section(partial) == "traits"


# -- the tracker --------------------------------------------------------------


def test_progress_runs_through_the_phases_in_order(replay_client):
    seen, _ = watch(
        analyze_personality, "examples/sample_chat.txt", replay_client("analyze_sample")
    )
    phases = [p.phase for p in seen]
    assert phases[:3] == ["waiting", "thinking", "writing"]
    assert set(phases[3:]) == {"writing"}


def test_every_section_is_reported_in_schema_order(replay_client):
    seen, _ = watch(
        analyze_personality, "examples/sample_chat.txt", replay_client("analyze_sample")
    )
    sections = list(dict.fromkeys(p.section for p in seen if p.section))
    assert sections == list(PersonalityReport.model_fields)


def test_progress_carries_the_shape_of_the_report_and_none_of_its_content(replay_client):
    # The flags report is the one where this matters: a flag shown before grounding
    # could be one that grounding then throws away.
    seen, report = watch(
        analyze_flags, "examples/pressure_chat.txt", replay_client("flags_pressure")
    )
    assert {p.section for p in seen} <= set(FlagScan.model_fields) | {None}
    assert Progress.__slots__ == ("phase", "section", "chars")
    assert report.flags, "the recording should have something to find"


def test_only_changes_are_reported(replay_client):
    seen, _ = watch(analyze_flags, "examples/sample_chat.txt", replay_client("flags_sample"))
    assert all(a != b for a, b in zip(seen, seen[1:], strict=False))


def test_watching_does_not_change_the_report(replay_client):
    conversation = read_transcript("examples/sample_chat.txt")
    quiet = analyze_personality(conversation, client=replay_client("analyze_sample"))
    _, watched = watch(
        analyze_personality, "examples/sample_chat.txt", replay_client("analyze_sample")
    )
    assert watched == quiet


def test_a_refusal_stops_after_thinking(replay_client):
    seen: list[Progress] = []
    with pytest.raises(ModelRefusal):
        analyze_personality(
            read_transcript("examples/sample_chat.txt"),
            client=replay_client("analyze_refusal"),
            on_progress=seen.append,
        )
    assert [p.phase for p in seen] == ["waiting", "thinking"]


def test_events_confidant_does_not_read_are_ignored():
    seen: list[Progress] = []
    tracker = ProgressTracker(seen.append)
    for event in (
        SimpleNamespace(type="message_start"),
        SimpleNamespace(type="content_block_start", content_block=SimpleNamespace(type="odd")),
        SimpleNamespace(type="thinking", thinking="...", snapshot="..."),
        SimpleNamespace(type="message_stop"),
    ):
        tracker.feed(event)
    assert seen == [Progress("waiting")]


def test_a_recording_client_passes_the_live_stream_through(tmp_path, recording):
    live = recording("analyze_sample")
    response = live.response_for(PersonalityReport)
    sdk = SimpleNamespace(
        messages=SimpleNamespace(stream=lambda **_: _Stream(list(live.events()), response))
    )
    out = tmp_path / "analyze.json"
    recorder = RecordingClient(sdk, out, source={"analysis": "analyze", "transcript": "x"})
    seen, _ = watch(analyze_personality, "examples/sample_chat.txt", recorder)
    assert "traits" in {p.section for p in seen}
    assert recorder.saved == [out]


class _Stream:
    def __init__(self, events, response):
        self._events, self._response = events, response

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def __iter__(self):
        return iter(self._events)

    def get_final_message(self):
        return self._response


# -- the status line ----------------------------------------------------------


def test_the_status_line_names_the_phase_section_and_time():
    clock = Clock()
    line = StatusLine(io.StringIO(), clock=clock)
    assert line.render() == "confidant: sent the redacted transcript, waiting for a reply... 0s"
    clock.now += 12.7
    line(Progress("thinking"))
    assert line.render() == "confidant: thinking it over... 12s"
    line(Progress("writing", "green_flags", 900))
    assert line.render() == "confidant: writing the report: green flags... 12s"


def test_the_status_line_overwrites_itself_and_cleans_up():
    out = io.StringIO()
    with StatusLine(out, interval=60) as line:
        line(Progress("writing", "questions_to_ask"))
        line(Progress("writing", "caveat"))
    written = out.getvalue()
    # The shorter line is padded over the longer one, and the last write blanks it.
    frames = written.split("\r")
    assert frames[-3].rstrip() == "confidant: writing the report: caveat... 0s"
    assert len(frames[-3]) >= len(frames[-4].rstrip())
    assert frames[-2] == " " * len(frames[-3].rstrip()) and frames[-1] == ""


def test_the_status_line_keeps_time_moving_while_the_stream_is_quiet():
    out = io.StringIO()
    with StatusLine(out, interval=0.01):
        time.sleep(0.1)
    assert out.getvalue().count("\r") > 4


# -- the CLI ------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch, replay_client):
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    for var in ("CONFIDANT_MODEL", "CONFIDANT_EFFORT", "CONFIDANT_MAX_TOKENS"):
        monkeypatch.delenv(var, raising=False)

    def use(*names: str):
        client = replay_client(*names)
        monkeypatch.setattr("confidant.client.build_client", lambda *_: client)

    return use


def _without_usage(err: str) -> str:
    """stderr minus the usage line every model command ends with, which is not progress."""
    progress, usage = err.rsplit("confidant: used ", 1)
    assert usage.endswith(".\n") and "\n" not in usage[:-1]
    return progress


def test_a_terminal_sees_progress_on_stderr_and_a_clean_report_on_stdout(cli, capsys, monkeypatch):
    cli("analyze_sample")
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    assert main(["analyze", "examples/sample_chat.txt"]) == 0
    captured = capsys.readouterr()
    assert "confidant: thinking it over..." in captured.err
    assert "confidant: writing the report: traits..." in captured.err
    # The status line clears itself before the usage line is printed below it.
    assert _without_usage(captured.err).endswith("\r")
    assert "confidant:" not in captured.out


def test_no_progress_when_stderr_is_not_a_terminal(cli, capsys):
    cli("flags_pressure")
    assert main(["flags", "examples/pressure_chat.txt"]) == 0
    assert _without_usage(capsys.readouterr().err) == ""


def test_no_progress_flag_silences_a_terminal(cli, capsys, monkeypatch):
    cli("analyze_sample")
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    assert main(["analyze", "examples/sample_chat.txt", "--no-progress", "--json"]) == 0
    assert _without_usage(capsys.readouterr().err) == ""


def test_placeholders_never_reach_the_status_line(cli, capsys, monkeypatch):
    # Sections are schema field names, so nothing from the transcript, redacted or not,
    # can end up on the status line.
    cli("flags_pressure")
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    assert main(["flags", "examples/pressure_chat.txt"]) == 0
    err = capsys.readouterr().err
    conversation = read_transcript("examples/pressure_chat.txt")
    for name in (conversation.owner_name, conversation.match_name, "[MATCH]", "[OWNER]"):
        assert name not in err
    assert redact(conversation).conversation.match_name not in err


def test_readme_progress_line_matches_what_the_status_line_prints():
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    marker = "rewrites itself as it goes:\n\n```\n"
    documented = readme[readme.index(marker) + len(marker) :].split("\n```", 1)[0]

    clock = Clock()
    line = StatusLine(io.StringIO(), clock=clock)
    clock.now += 23
    line(Progress("writing", "green_flags"))
    assert line.render() == documented
