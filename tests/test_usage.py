"""Token and cost accounting: what a call used, what that cost, and where it is reported.

The usage in the recordings is hand-written, so these tests pin down the arithmetic and
the plumbing rather than any claim about what a real read of the examples costs.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from confidant.analysis.flags import FlagScan
from confidant.analysis.personality import analyze_personality
from confidant.cli import main
from confidant.client import IncompleteResponse, ModelRefusal
from confidant.config import Settings
from confidant.ingest.transcript import read_transcript
from confidant.recording import Recording, RecordingClient, ReplayClient, run_analysis
from confidant.usage import PRICES, Usage, UsageMeter, describe


def _usage(**counts) -> SimpleNamespace:
    """An SDK ``usage`` object; the cache fields are None when no cache was touched."""
    fields = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_creation_input_tokens": None,
        "cache_read_input_tokens": None,
    }
    return SimpleNamespace(**{**fields, **counts})


# -- arithmetic -----------------------------------------------------------------


def test_missing_cache_fields_count_as_zero():
    usage = Usage.from_response(_usage(input_tokens=10, output_tokens=3), "claude-opus-5")
    assert (usage.cache_write_tokens, usage.cache_read_tokens) == (0, 0)
    assert usage.total_input == 10


def test_each_kind_of_token_is_priced_at_its_own_rate():
    usage = Usage(
        model="claude-opus-5",
        input_tokens=1_000_000,
        cache_write_tokens=1_000_000,
        cache_read_tokens=1_000_000,
        output_tokens=1_000_000,
    )
    assert usage.cost == pytest.approx(5.00 + 6.25 + 0.50 + 25.00)


def test_a_cache_read_is_far_cheaper_than_sending_the_prefix_again():
    fresh = Usage(model="claude-opus-5", input_tokens=1000)
    cached = Usage(model="claude-opus-5", cache_read_tokens=1000)
    assert cached.cost == pytest.approx(fresh.cost / 10)


def test_models_are_priced_by_exact_id_not_prefix():
    # "claude-opus-5-5" starts with "claude-opus-5"; a prefix match would misprice it.
    tokens = {"input_tokens": 1_000_000}
    assert Usage(model="claude-opus-5-5", **tokens).cost == pytest.approx(4.00)
    assert Usage(model="claude-opus-5", **tokens).cost == pytest.approx(5.00)
    assert Usage(model="claude-opus-5-20990101", **tokens).cost is None


def test_cache_writes_are_priced_at_the_five_minute_ttl():
    # confidant.client asks for the default TTL; the one-hour write would be 2x input.
    for price in PRICES.values():
        assert price.cache_write == pytest.approx(price.input * 1.25)


def test_an_unpriced_model_gets_counts_and_no_invented_cost():
    text = describe([Usage(model="claude-someday", input_tokens=1200, output_tokens=300)])
    assert text == (
        "used 1,200 input tokens and 300 output tokens; no price on file for "
        "claude-someday, so no cost estimate."
    )


# -- the sentence -----------------------------------------------------------------


def test_one_call_with_a_cache_hit():
    usage = Usage(
        model="claude-opus-5", input_tokens=327, cache_read_tokens=1052, output_tokens=1388
    )
    assert describe([usage]) == (
        "used 1,379 input tokens (1,052 read from the cache) and 1,388 output tokens, about $0.037."
    )


def test_reads_and_writes_together():
    usage = Usage(
        model="claude-opus-5", cache_read_tokens=900, cache_write_tokens=100, output_tokens=10
    )
    assert "(900 read from the cache, 100 written to it)" in describe([usage])


def test_several_calls_are_added_up():
    calls = [
        Usage(model="claude-opus-5", input_tokens=100_000, output_tokens=20_000),
        Usage(model="claude-opus-5", input_tokens=100_000, output_tokens=20_000),
    ]
    assert describe(calls) == (
        "2 calls used 200,000 input tokens and 40,000 output tokens, about $2.00."
    )


def test_one_unpriced_call_withholds_the_total():
    # A total that silently left one call out would be an underestimate presented as fact.
    calls = [Usage(model="claude-opus-5", input_tokens=10), Usage(model="mystery", input_tokens=5)]
    assert describe(calls).endswith("; no price on file for mystery, so no cost estimate.")


def test_a_meter_collects_what_it_is_handed():
    meter = UsageMeter()
    assert not meter
    meter(Usage(model="claude-opus-5", input_tokens=1))
    assert meter and len(meter.calls) == 1


# -- structured_call reports it -----------------------------------------------------


def _analyze(rec: Recording, settings: Settings | None = None) -> UsageMeter:
    meter = UsageMeter()
    analyze_personality(
        read_transcript("examples/sample_chat.txt"),
        settings=settings,
        client=ReplayClient(rec),
        on_usage=meter,
    )
    return meter


def test_a_successful_call_reports_its_usage(recording):
    [usage] = _analyze(recording("analyze_sample")).calls
    assert usage == Usage(
        model="claude-opus-5",
        input_tokens=498,
        cache_write_tokens=671,
        cache_read_tokens=0,
        output_tokens=1964,
    )


@pytest.mark.parametrize(
    ("name", "error"),
    [("analyze_refusal", ModelRefusal), ("analyze_truncated", IncompleteResponse)],
)
def test_a_failed_answer_is_still_counted(recording, name, error):
    # A refusal or a cut-off answer is billed, so it is reported before the raise.
    meter = UsageMeter()
    with pytest.raises(error):
        analyze_personality(
            read_transcript("examples/sample_chat.txt"),
            client=ReplayClient(recording(name)),
            on_usage=meter,
        )
    [usage] = meter.calls
    assert usage.total_input == 1169


def test_the_answering_model_is_what_gets_priced(recording):
    rec = recording("analyze_sample")
    rec.model = "claude-opus-5-5"
    meter = _analyze(rec, Settings(model="claude-opus-5"))
    assert [u.model for u in meter.calls] == ["claude-opus-5-5"]


def test_a_response_without_usage_reports_nothing(recording):
    rec = recording("analyze_sample")
    rec.usage = None
    assert not _analyze(rec)


# -- recordings keep it ----------------------------------------------------------------


class _LiveStream:
    def __init__(self, response):
        self.response = response
        self.messages = SimpleNamespace(stream=lambda **_: self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self.response


def test_a_live_recording_keeps_the_usage_counts(tmp_path, recording):
    scan = recording("flags_sample").response_for(FlagScan).parsed_output
    response = SimpleNamespace(
        stop_reason="end_turn",
        stop_details=None,
        parsed_output=scan,
        model="claude-opus-5",
        usage=_usage(input_tokens=509, cache_read_input_tokens=1052, output_tokens=806),
    )
    out = tmp_path / "flags.json"
    run_analysis(
        "flags", "examples/sample_chat.txt", client=RecordingClient(_LiveStream(response), out)
    )
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["response"]["usage"] == {
        "input_tokens": 509,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 1052,
        "output_tokens": 806,
    }
    assert Recording.load(out).usage == saved["response"]["usage"]


def test_usage_is_not_part_of_the_fingerprint(recording):
    # It says what a call cost, not what was asked, so editing it never makes a recording
    # stale.
    rec = recording("flags_sample")
    rec.usage = {**rec.usage, "output_tokens": 1}
    run_analysis("flags", "examples/sample_chat.txt", client=ReplayClient(rec))


# -- the CLI ------------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch, replay_client):
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    for var in ("CONFIDANT_MODEL", "CONFIDANT_EFFORT", "CONFIDANT_MAX_TOKENS"):
        monkeypatch.delenv(var, raising=False)

    def use(*names: str):
        client = replay_client(*names)
        monkeypatch.setattr("confidant.client.build_client", lambda *_: client)

    return use


def test_analyze_ends_with_a_usage_line_on_stderr(cli, capsys):
    cli("analyze_sample")
    assert main(["analyze", "examples/sample_chat.txt"]) == 0
    captured = capsys.readouterr()
    assert captured.err == (
        "confidant: used 1,169 input tokens (671 written to the cache) and 1,964 output "
        "tokens, about $0.056.\n"
    )
    assert "tokens" not in captured.out


def test_json_output_stays_one_json_document(cli, capsys):
    cli("flags_pressure")
    assert main(["flags", "examples/pressure_chat.txt", "--json"]) == 0
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert captured.err.startswith("confidant: used ")


def test_a_refusal_still_says_what_it_cost(cli, capsys):
    cli("analyze_refusal")
    assert main(["analyze", "examples/sample_chat.txt"]) == 3
    first, second = capsys.readouterr().err.splitlines()
    assert "declined" in first
    assert second.startswith("confidant: used 1,169 input tokens")


def test_no_call_means_no_usage_line(capsys):
    assert main(["stats", "examples/sample_chat.txt"]) == 0
    assert capsys.readouterr().err == ""


def test_the_readme_usage_line_is_what_analyze_prints(cli, capsys):
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    cli("analyze_sample")
    main(["analyze", "examples/sample_chat.txt"])
    line = capsys.readouterr().err
    assert f"```\n{line}```\n" in readme
