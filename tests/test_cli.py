"""Tests for configuration, the credential preflight, and CLI exit codes.

Everything here stays offline. The one test that reaches the model layer asserts that we
refuse to get there without credentials.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import anthropic
import pytest
from pydantic import TypeAdapter, ValidationError

from confidant.analysis.personality import PersonalityReport
from confidant.cli import _EPILOG, main
from confidant.client import IncompleteResponse, ModelRefusal, build_client, structured_call
from confidant.config import DEFAULT_MAX_TOKENS, ConfigError, Settings


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """Isolate every test from the developer's real keys and `.env` file."""
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CONFIDANT_MODEL",
        "CONFIDANT_EFFORT",
        "CONFIDANT_MAX_TOKENS",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))


def test_settings_defaults():
    settings = Settings.from_env()
    assert settings.model == "claude-opus-5"
    assert settings.effort == "high"
    assert settings.max_tokens == DEFAULT_MAX_TOKENS
    assert settings.api_key is None


def test_settings_read_the_environment(monkeypatch):
    monkeypatch.setenv("CONFIDANT_MODEL", "claude-sonnet-5")
    monkeypatch.setenv("CONFIDANT_EFFORT", "LOW")
    monkeypatch.setenv("CONFIDANT_MAX_TOKENS", "2048")
    settings = Settings.from_env()
    assert (settings.model, settings.effort, settings.max_tokens) == (
        "claude-sonnet-5",
        "low",
        2048,
    )


def test_invalid_effort_is_rejected(monkeypatch):
    monkeypatch.setenv("CONFIDANT_EFFORT", "enormous")
    with pytest.raises(ConfigError, match="CONFIDANT_EFFORT"):
        Settings.from_env()


def test_non_numeric_max_tokens_is_rejected(monkeypatch):
    monkeypatch.setenv("CONFIDANT_MAX_TOKENS", "lots")
    with pytest.raises(ConfigError, match="not an integer"):
        Settings.from_env()


def test_missing_credentials_fail_before_any_request():
    with pytest.raises(ConfigError, match="No Anthropic credentials found"):
        build_client(Settings.from_env())


def test_an_api_key_satisfies_the_preflight(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    build_client(Settings.from_env())  # constructs without touching the network


def test_model_refusal_message_is_readable():
    error = ModelRefusal(category="privacy", explanation="declined to profile a third party")
    assert "privacy" in str(error)
    assert "declined to profile a third party" in str(error)


# -- CLI ------------------------------------------------------------------


def test_stats_runs_offline_and_succeeds(capsys):
    assert main(["stats", "examples/sample_chat.txt"]) == 0
    out = capsys.readouterr().out
    assert "Sam <-> Robin" in out
    assert "effort ratio" in out


def test_a_bad_transcript_exits_2(capsys):
    assert main(["stats", "does-not-exist.txt"]) == 2
    assert "No transcript at" in capsys.readouterr().err


def test_analyze_without_credentials_exits_2(capsys):
    assert main(["analyze", "examples/sample_chat.txt"]) == 2
    assert "No Anthropic credentials found" in capsys.readouterr().err


def test_a_transcript_needing_an_owner_exits_2(capsys, tmp_path):
    path = tmp_path / "chat.txt"
    path.write_text("Robin: hey\nAlex: hi\n", encoding="utf-8")
    assert main(["stats", str(path)]) == 2
    assert "which side of this conversation is yours" in capsys.readouterr().err


def test_owner_can_be_supplied_on_the_command_line(capsys, tmp_path):
    path = tmp_path / "chat.txt"
    path.write_text("Robin: hey\nAlex: hi\n", encoding="utf-8")
    assert main(["stats", str(path), "--owner", "Alex"]) == 0
    assert "Alex <-> Robin" in capsys.readouterr().out


# -- what reaches the API -----------------------------------------------------


def test_the_effort_setting_is_sent(replay_client):
    from confidant.analysis.personality import analyze_personality
    from confidant.ingest.transcript import read_transcript

    client = replay_client("analyze_sample")
    analyze_personality(
        read_transcript("examples/sample_chat.txt"), settings=Settings(effort="low"), client=client
    )
    [request] = client.requests
    assert request["output_config"] == {"effort": "low"}
    assert request["thinking"] == {"type": "adaptive"}


# -- a response that stops short ----------------------------------------------


class _BrokenStream:
    """A live stream whose text stops mid-JSON, the way a token-limited answer does."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.messages = SimpleNamespace(stream=lambda **_: self)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        # What the SDK does at content_block_stop.
        TypeAdapter(PersonalityReport).validate_json(self.text)
        raise AssertionError("unreachable")


def _ask(client):
    return structured_call(
        schema=PersonalityReport, system="s", user_content="u", settings=Settings(), client=client
    )


def test_a_cut_off_stream_is_reported_as_incomplete():
    with pytest.raises(IncompleteResponse, match="CONFIDANT_MAX_TOKENS=16000"):
        _ask(_BrokenStream('{"headline": "Warm, curious, and'))


def test_valid_json_in_the_wrong_shape_is_not_mistaken_for_a_cut_off():
    with pytest.raises(ValidationError):
        _ask(_BrokenStream('{"headline": 1}'))


def test_a_non_token_stop_says_what_stopped_it():
    assert "stop_reason='pause_turn'" in str(IncompleteResponse("pause_turn", 100))


# -- errors the CLI turns into sentences --------------------------------------


@pytest.fixture
def failing_analysis(monkeypatch):
    def fail_with(exc: BaseException) -> None:
        def raise_it(*_, **__):
            raise exc

        monkeypatch.setattr("confidant.cli.analyze_personality", raise_it)

    return fail_with


def _status_error(cls, status: int):
    response = SimpleNamespace(request=None, status_code=status, headers={})
    return cls("error", response=response, body=None)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (lambda: anthropic.APIConnectionError(request=None), "Could not reach the Anthropic API"),
        (lambda: anthropic.APITimeoutError(request=None), "Could not reach the Anthropic API"),
        (
            lambda: _status_error(anthropic.AuthenticationError, 401),
            "did not accept these credentials",
        ),
        (lambda: _status_error(anthropic.NotFoundError, 404), "does not know the model"),
        (lambda: _status_error(anthropic.RateLimitError, 429), "rate-limiting"),
        (lambda: _status_error(anthropic.InternalServerError, 529), "(HTTP 529)"),
        (lambda: _status_error(anthropic.BadRequestError, 400), "The Anthropic API call failed"),
    ],
)
def test_api_failures_exit_4_with_a_useful_sentence(failing_analysis, capsys, error, expected):
    failing_analysis(error())
    assert main(["analyze", "examples/sample_chat.txt"]) == 4
    err = capsys.readouterr().err
    assert expected in err
    assert "Traceback" not in err


def test_a_cut_off_report_exits_4(failing_analysis, capsys):
    failing_analysis(IncompleteResponse("max_tokens", 16_000))
    assert main(["analyze", "examples/sample_chat.txt"]) == 4
    assert "Raise CONFIDANT_MAX_TOKENS" in capsys.readouterr().err


def test_ctrl_c_exits_130_quietly(failing_analysis, capsys):
    failing_analysis(KeyboardInterrupt())
    assert main(["analyze", "examples/sample_chat.txt"]) == 130
    assert capsys.readouterr().err == "confidant: cancelled.\n"


def test_transcript_errors_speak_in_command_line_terms(capsys, tmp_path):
    path = tmp_path / "chat.txt"
    path.write_text("Robin: hey\nAlex: hi\n", encoding="utf-8")
    assert main(["stats", str(path)]) == 2
    err = capsys.readouterr().err
    assert "--owner NAME" in err
    assert "owner=..." not in err


def test_a_file_that_is_not_a_transcript_says_what_one_looks_like(capsys, tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("just some notes\n", encoding="utf-8")
    assert main(["stats", str(path)]) == 2
    assert "Messages look like 'Name: text'" in capsys.readouterr().err


# -- help text ------------------------------------------------------------------


def test_help_lists_examples_environment_and_exit_codes(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    for expected in ("examples:", "CONFIDANT_MAX_TOKENS", "exit codes:", "130"):
        assert expected in out


def test_subcommand_help_shows_the_transcript_format(capsys):
    with pytest.raises(SystemExit):
        main(["flags", "--help"])
    out = capsys.readouterr().out
    assert "transcript format:" in out
    assert "--no-progress" in out


def test_readme_exit_codes_match_the_help():
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    documented = _EPILOG[_EPILOG.index("exit codes:\n") :]
    assert documented in readme


@pytest.mark.parametrize(
    ("argv", "marker"),
    [
        (["stats", "examples/sample_chat.txt"], "`stats` gives you the cheap signals:\n\n```\n"),
        (["redact", "examples/details_chat.txt"], "```bash\nconfidant redact examples/"),
    ],
)
def test_readme_examples_match_what_the_cli_prints(capsys, argv, marker):
    readme = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    after = readme[readme.index(marker) + len(marker) :]
    if marker.startswith("```bash"):
        # The command block, then "prints", then the output block.
        after = after.split("```\n\nprints\n\n```\n", 1)[1]
    documented = after.split("\n```", 1)[0]

    assert main(argv) == 0
    assert capsys.readouterr().out == documented + "\n"
