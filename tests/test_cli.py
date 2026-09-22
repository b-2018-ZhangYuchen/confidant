"""Tests for configuration, the credential preflight, and CLI exit codes.

Everything here stays offline. The one test that reaches the model layer asserts that we
refuse to get there without credentials.
"""

from __future__ import annotations

import pytest

from confidant.cli import main
from confidant.client import ModelRefusal, build_client
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
