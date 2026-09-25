"""Tests for the recorded-fixture harness, and the analyses replayed through it.

Unlike the tests that monkeypatch ``structured_call``, these run everything Confidant does
with a model response: the refusal check, schema validation, grounding, escalation, and
rendering. Only the SDK itself is replaced.
"""

from __future__ import annotations

import json
import shutil
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from confidant.analysis.flags import FlagReport, FlagScan, quote_appears_in
from confidant.analysis.personality import PersonalityReport
from confidant.cli import main
from confidant.client import ModelRefusal
from confidant.ingest.transcript import read_transcript
from confidant.recording import (
    Fingerprint,
    Recording,
    RecordingClient,
    RecordingError,
    ReplayClient,
    StaleRecording,
    replay,
    run_analysis,
)
from confidant.recording import main as recording_main

RECORDINGS = Path(__file__).parent / "fixtures" / "recorded"

ALL_RECORDINGS = sorted(RECORDINGS.glob("*.json"))


# -- every recording in the repository ----------------------------------------


def test_there_are_recordings_to_replay():
    assert len(ALL_RECORDINGS) >= 5


@pytest.mark.parametrize("path", ALL_RECORDINGS, ids=lambda p: p.stem)
def test_every_recording_is_current_and_replays(path):
    # Fails with StaleRecording when a prompt changes without the recordings following.
    recording = Recording.load(path)
    assert "examples" in Path(recording.source["transcript"]).parts
    assert recording.note, "say why the recording exists"
    if recording.stop_reason == "refusal":
        with pytest.raises(ModelRefusal):
            replay(recording)
    elif recording.output is None:
        with pytest.raises(RuntimeError, match="no structured output"):
            replay(recording)
    else:
        assert replay(recording) is not None


@pytest.mark.parametrize("path", ALL_RECORDINGS, ids=lambda p: p.stem)
def test_every_recording_is_stored_canonically(path):
    # So that a re-stamp or re-record shows up in a diff as the lines that changed.
    assert Recording.load(path).to_json() == path.read_text(encoding="utf-8")


# -- the analyses, replayed ---------------------------------------------------


def test_pressure_example_is_grounded_and_escalated(recording):
    report = replay(recording("flags_pressure"))
    assert isinstance(report, FlagReport)
    assert [(f.category, f.severity) for f in report.flags] == [
        ("monitoring", "danger"),
        ("isolation", "danger"),
        ("guilt_or_blame", "concern"),
    ]
    # The paraphrased boundary_pushing quote is the one grounding throws out.
    assert report.discarded == 1
    assert report.discarded_danger == 0
    assert report.escalation.categories == ["monitoring", "isolation"]


def test_a_quiet_transcript_has_nothing_to_flag(recording):
    report = replay(recording("flags_sample"))
    assert report.flags == []
    assert report.escalation is None
    assert "Nothing in this transcript rises to a red flag." in report.to_text()


def test_personality_replay_renders(recording):
    report = replay(recording("analyze_sample"))
    assert isinstance(report, PersonalityReport)
    text = report.to_text()
    assert text.startswith("Robin comes across as curious and playful")
    assert "Overall confidence: medium" in text


def test_personality_quotes_are_robins_own_words(recording):
    # The personality read has no grounding pass yet, so the recording itself is held to
    # the standard: a hand-written response must not put words in anyone's mouth.
    report = replay(recording("analyze_sample"))
    robin = [m.text for m in read_transcript("examples/sample_chat.txt").match_messages]
    for trait in report.traits:
        for item in trait.evidence:
            assert quote_appears_in(item.quote, robin), item.quote


def test_the_request_is_what_structured_call_promises(replay_client):
    client = replay_client("flags_sample")
    run_analysis("flags", "examples/sample_chat.txt", client=client)
    [request] = client.requests
    assert request["model"] == "claude-opus-5"
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_format"] is FlagScan
    assert client.remaining == 0


def test_a_refusal_is_surfaced_with_its_explanation(recording):
    with pytest.raises(ModelRefusal, match="profile a third party") as caught:
        replay(recording("analyze_refusal"))
    assert caught.value.category == "general_harms"


# -- staleness ----------------------------------------------------------------


def test_a_prompt_edit_makes_the_recording_stale(monkeypatch, recording):
    monkeypatch.setattr("confidant.analysis.flags.FLAGS_SYSTEM", "A different prompt.")
    with pytest.raises(StaleRecording, match="the system prompt changed") as caught:
        replay(recording("flags_pressure"))
    message = str(caught.value)
    assert "flags_pressure.json" in message
    assert "python -m confidant.recording record flags examples/pressure_chat.txt" in message
    assert "stamp" in message  # hand-written, so re-stamping is offered


def test_a_different_transcript_makes_the_recording_stale(recording):
    client = ReplayClient(recording("flags_pressure"))
    with pytest.raises(StaleRecording, match="the request .* changed"):
        run_analysis("flags", "examples/sample_chat.txt", client=client)


def test_live_recordings_are_not_offered_a_restamp(monkeypatch, recording):
    live = recording("flags_pressure")
    live.provenance = "recorded"
    monkeypatch.setattr("confidant.analysis.flags.FLAGS_SYSTEM", "A different prompt.")
    with pytest.raises(StaleRecording) as caught:
        replay(live)
    assert "stamp" not in str(caught.value)


def test_non_strict_replay_tolerates_prompt_drift_but_not_a_schema_change(monkeypatch, recording):
    monkeypatch.setattr("confidant.analysis.flags.FLAGS_SYSTEM", "A different prompt.")
    assert replay(recording("flags_pressure"), strict=False).has_danger
    client = ReplayClient(recording("flags_pressure"), strict=False)
    with pytest.raises(StaleRecording, match="schema is PersonalityReport"):
        run_analysis("analyze", "examples/pressure_chat.txt", client=client)


def test_a_schema_change_the_response_no_longer_fits_fails_validation(recording):
    broken = recording("flags_sample")
    del broken.output["confidence"]
    with pytest.raises(ValidationError, match="confidence"):
        replay(broken)


def test_an_extra_model_call_is_an_error(replay_client):
    client = replay_client("flags_sample")
    run_analysis("flags", "examples/sample_chat.txt", client=client)
    with pytest.raises(RecordingError, match="call number 2 has no recording"):
        run_analysis("flags", "examples/sample_chat.txt", client=client)


def test_fingerprints_ignore_dict_order():
    a = {"output_format": FlagScan, "system": "s", "messages": [{"role": "user", "content": "x"}]}
    b = {"output_format": FlagScan, "system": "s", "messages": [{"content": "x", "role": "user"}]}
    assert Fingerprint.of(a) == Fingerprint.of(b)
    assert Fingerprint.of(a).differences(Fingerprint.of(b)) == []


# -- loading ------------------------------------------------------------------


def test_a_missing_recording_is_a_clear_error(tmp_path):
    with pytest.raises(RecordingError, match="No recording at"):
        Recording.load(tmp_path / "nope.json")


def test_an_unknown_format_is_rejected(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"format": 99}), encoding="utf-8")
    with pytest.raises(RecordingError, match="format 99"):
        Recording.load(path)


def test_a_missing_field_is_named(tmp_path):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"format": 1, "provenance": "recorded"}), encoding="utf-8")
    with pytest.raises(RecordingError, match="missing the field 'request'"):
        Recording.load(path)


# -- recording ----------------------------------------------------------------


class FakeSDK:
    """Answers ``messages.parse`` the way the SDK does, from a canned object."""

    def __init__(self, response):
        self.response = response
        self.messages = SimpleNamespace(parse=lambda **_: self.response)


def test_a_recorded_response_replays_to_the_same_report(tmp_path, recording):
    scan = recording("flags_pressure").response_for(FlagScan).parsed_output
    sdk = FakeSDK(
        SimpleNamespace(
            stop_reason="end_turn", stop_details=None, parsed_output=scan, model="claude-opus-5"
        )
    )
    out = tmp_path / "flags.json"
    source = {"analysis": "flags", "transcript": "examples/pressure_chat.txt"}
    recorder = RecordingClient(sdk, out, source=source, note="test")
    live = run_analysis("flags", "examples/pressure_chat.txt", client=recorder)

    saved = Recording.load(out)
    assert recorder.saved == [out]
    assert saved.provenance == "recorded"
    assert saved.model == "claude-opus-5"
    assert replay(saved) == live


def test_a_refusal_is_recorded_before_it_is_raised(tmp_path):
    details = SimpleNamespace(category="general_harms", explanation="no", extra=None)
    sdk = FakeSDK(SimpleNamespace(stop_reason="refusal", stop_details=details, parsed_output=None))
    out = tmp_path / "refusal.json"
    with pytest.raises(ModelRefusal):
        run_analysis("analyze", "examples/sample_chat.txt", client=RecordingClient(sdk, out))
    saved = Recording.load(out)
    assert saved.stop_details == {"category": "general_harms", "explanation": "no"}
    assert saved.output is None


def test_a_second_call_gets_its_own_file(tmp_path, recording):
    scan = recording("flags_sample").response_for(FlagScan).parsed_output
    sdk = FakeSDK(SimpleNamespace(stop_reason="end_turn", stop_details=None, parsed_output=scan))
    recorder = RecordingClient(sdk, tmp_path / "r.json")
    for _ in range(2):
        run_analysis("flags", "examples/sample_chat.txt", client=recorder)
    assert [p.name for p in recorder.saved] == ["r.json", "r.2.json"]


# -- the maintenance command --------------------------------------------------


def test_record_refuses_a_transcript_outside_examples(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(
        "confidant.recording.build_client", lambda *_: pytest.fail("should not get this far")
    )
    private = tmp_path / "chat.txt"
    private.write_text("# owner: Sam\nRobin: hey\nSam: hi\n", encoding="utf-8")
    assert recording_main(["record", "flags", str(private), str(tmp_path / "out.json")]) == 2
    assert "only fictional examples can be recorded" in capsys.readouterr().err
    assert not (tmp_path / "out.json").exists()


def test_stamp_refreshes_a_hand_written_recording(tmp_path, capsys, recording):
    path = tmp_path / "flags_pressure.json"
    shutil.copy(RECORDINGS / "flags_pressure.json", path)
    stale = Recording.load(path)
    stale.fingerprint = Fingerprint(stale.fingerprint.schema, "0" * 64, "0" * 64)
    stale.save(path)

    assert recording_main(["stamp", str(path)]) == 0
    assert "the system prompt changed" in capsys.readouterr().out
    assert Recording.load(path).fingerprint == recording("flags_pressure").fingerprint

    assert recording_main(["stamp", str(path)]) == 0
    assert "already current" in capsys.readouterr().out


def test_stamp_refreshes_a_refusal(tmp_path, capsys):
    path = tmp_path / "refusal.json"
    shutil.copy(RECORDINGS / "analyze_refusal.json", path)
    assert recording_main(["stamp", str(path)]) == 0
    assert "already current" in capsys.readouterr().out


def test_stamp_refuses_a_live_recording(tmp_path, capsys):
    path = tmp_path / "live.json"
    data = json.loads((RECORDINGS / "flags_sample.json").read_text(encoding="utf-8"))
    data["provenance"] = "recorded"
    path.write_text(json.dumps(data), encoding="utf-8")
    assert recording_main(["stamp", str(path)]) == 2
    assert "Record it again" in capsys.readouterr().err


def test_stamp_will_not_bless_a_response_that_no_longer_fits(tmp_path, capsys):
    path = tmp_path / "broken.json"
    data = json.loads((RECORDINGS / "flags_sample.json").read_text(encoding="utf-8"))
    del data["response"]["output"]["summary"]
    data["request"]["system_sha256"] = "0" * 64
    path.write_text(json.dumps(data), encoding="utf-8")
    assert recording_main(["stamp", str(path)]) == 2
    assert json.loads(path.read_text(encoding="utf-8"))["request"]["system_sha256"] == "0" * 64


# -- end to end ---------------------------------------------------------------


@pytest.fixture
def cli_replaying(monkeypatch, tmp_path, replay_client):
    """Run the real CLI with the SDK client swapped for a replay."""
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    for var in ("CONFIDANT_MODEL", "CONFIDANT_EFFORT", "CONFIDANT_MAX_TOKENS"):
        monkeypatch.delenv(var, raising=False)

    def use(*names: str) -> ReplayClient:
        client = replay_client(*names)
        monkeypatch.setattr("confidant.client.build_client", lambda *_: client)
        return client

    return use


def test_cli_flags_end_to_end(cli_replaying, capsys):
    cli_replaying("flags_pressure")
    assert main(["flags", "examples/pressure_chat.txt"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("!! Something in Casey's messages is serious")
    assert "[CONCERN] guilt or blame" in out
    assert "(1 flag left out: the quotes behind them could not be found" in out


def test_cli_refusal_exits_3(cli_replaying, capsys):
    cli_replaying("analyze_refusal")
    assert main(["analyze", "examples/sample_chat.txt"]) == 3
    assert "declined this request (general_harms)" in capsys.readouterr().err


def test_readme_safety_notice_matches_what_the_cli_prints(cli_replaying, capsys):
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    marker = "the report\nbegins:\n\n```\n"
    documented = readme[readme.index(marker) + len(marker) :].split("\n```", 1)[0]

    cli_replaying("flags_pressure")
    assert main(["flags", "examples/pressure_chat.txt"]) == 0
    assert capsys.readouterr().out.startswith(documented + "\n")


# -- the guard ----------------------------------------------------------------


def test_the_suite_cannot_reach_the_network():
    with pytest.raises(AssertionError, match="The suite is offline"):
        socket.create_connection(("192.0.2.1", 443), timeout=1)
