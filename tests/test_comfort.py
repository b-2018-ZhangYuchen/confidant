"""Tests for `confidant comfort`: what Confidant says when it has gone badly.

Offline, like the rest of the suite. The model's side is two hand-written recordings in
``tests/fixtures/recorded``: a slow fade in ``examples/faded_chat.txt``, and the owner
ending things in ``examples/pressure_chat.txt``, with one piece of praise written to be
dropped.
"""

from __future__ import annotations

import functools
import json
from datetime import datetime
from pathlib import Path

import pytest

from confidant.analysis.comfort import (
    FOOTER,
    ComfortScan,
    Moment,
    comfort,
    ground_comfort,
    silence_note,
    unanswered,
)
from confidant.cli import main
from confidant.config import Settings
from confidant.ingest.transcript import read_transcript
from confidant.models import Conversation, Message, Role
from confidant.profile import read_conversation
from confidant.recording import Recording, ReplayClient
from confidant.redaction import redact
from confidant.safety import SUPPORT_HEADLINE, SUPPORT_STEPS
from confidant.store import ReadingKind, Store

README = Path(__file__).parents[1] / "README.md"
RECORDINGS = Path(__file__).parent / "fixtures" / "recorded"
FADED = "examples/faded_chat.txt"
PRESSURE = "examples/pressure_chat.txt"
FADED_WHAT = "two good dates, then Jamie went quiet"
NOW = datetime(2026, 10, 9, 9, 0)


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path) -> Path:
    """A private store, no real credentials, and a clock that does not move."""
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CONFIDANT_MODEL"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "empty-config"))
    path = tmp_path / "store" / "confidant.db"
    monkeypatch.setenv("CONFIDANT_DB", str(path))
    monkeypatch.setattr("confidant.cli.Store", functools.partial(Store, clock=lambda: NOW))
    return path


@pytest.fixture
def answer_with(monkeypatch):
    """Make the API answer from recordings, in order, for the rest of the test."""

    def install(*names: str, strict: bool = True) -> ReplayClient:
        client = ReplayClient(*(RECORDINGS / f"{name}.json" for name in names), strict=strict)
        monkeypatch.setattr("confidant.client.build_client", lambda settings=None: client)
        return client

    return install


def run(capsys, *argv: str) -> tuple[int, str, str]:
    code = main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _sent(client: ReplayClient) -> str:
    [request] = client.requests
    return request["messages"][0]["content"]


def _loose(name: str = "comfort_faded") -> ReplayClient:
    # Answers a request that differs from the recorded one, so a test can look at what a
    # variation of it sends without needing a recording per variation.
    return ReplayClient(RECORDINGS / f"{name}.json", strict=False)


def _at_risk(name: str = "comfort_faded") -> Recording:
    recording = Recording.load(RECORDINGS / f"{name}.json")
    recording.output = {**recording.output, "owner_at_risk": True}
    return recording


def _scan(*moments: Moment, at_risk: bool = False) -> ComfortScan:
    return ComfortScan(
        what_happened="[MATCH] stopped replying.",
        what_it_says="The messages do not say why.",
        you_did_well=list(moments),
        worth_knowing=None,
        next_steps=["Sleep on it."],
        owner_at_risk=at_risk,
    )


# -- what comes back ------------------------------------------------------------


def test_a_fade_comes_back_in_plain_words_with_names_restored():
    client = ReplayClient(RECORDINGS / "comfort_faded.json")
    report = comfort(read_transcript(FADED), what=FADED_WHAT, settings=Settings(), client=client)

    assert len(report.you_did_well) == 3
    assert report.discarded == 0
    assert report.support is None and report.escalation is None
    text = report.to_text()
    assert "[OWNER]" not in text and "[MATCH]" not in text
    assert text.startswith("After two dates that you both enjoyed, Jamie's replies")
    assert '"how did the interview go?"' in text
    assert text.endswith(FOOTER)


def test_praise_quoting_the_matchs_line_is_dropped():
    client = ReplayClient(RECORDINGS / "comfort_pressure.json")
    report = comfort(
        read_transcript(PRESSURE),
        what="I ended it this morning and I feel awful about it",
        settings=Settings(),
        client=client,
    )
    assert [m.quote for m in report.you_did_well] == [
        "I'm not really comfortable sharing that yet",
        "I'm going out with friends tonight",
    ]
    assert report.discarded == 1
    text = report.to_text()
    assert "I guess I just care more" not in text
    assert "(1 thing left out of what you did well" in text


def test_only_the_owners_messages_count_as_their_own_words():
    redaction = redact(read_transcript(FADED))
    report = ground_comfort(
        _scan(
            Moment(point="Asked about the interview.", quote="how did the interview go?"),
            Moment(point="Was kind.", quote="you were not terrible. you were medium"),
            Moment(point="Made something up.", quote="I had the best time"),
        ),
        redaction.conversation,
    )
    assert [m.point for m in report.you_did_well] == ["Asked about the interview."]
    assert report.discarded == 2


def test_with_nothing_to_praise_the_section_is_left_out():
    report = ground_comfort(_scan(), redact(read_transcript(FADED)).conversation)
    text = report.to_text()
    assert "WHAT YOU DID WELL" not in text
    assert "left out" not in text


# -- the silence ----------------------------------------------------------------


def test_the_unanswered_stretch_is_counted_locally_and_sent():
    client = _loose()
    comfort(read_transcript(FADED), settings=Settings(), client=client)
    assert "My last 2 messages in this thread have had no reply." in _sent(client)


def test_the_silence_note_is_fixed_text_with_the_matchs_last_date():
    client = ReplayClient(RECORDINGS / "comfort_faded.json")
    report = comfort(read_transcript(FADED), what=FADED_WHAT, settings=Settings(), client=client)
    assert report.unanswered == 2
    assert report.match_last_wrote == datetime(2026, 5, 15, 9, 5)
    assert "Your last 2 messages have had no reply. Jamie last wrote on 2026-05-15." in (
        report.to_text()
    )


def test_when_the_match_wrote_last_nothing_is_said_about_silence():
    client = _loose("comfort_pressure")
    report = comfort(read_transcript(PRESSURE), settings=Settings(), client=client)
    assert "have had no reply" not in _sent(client)
    assert report.unanswered == 0
    assert "had no reply" not in report.to_text()


@pytest.mark.parametrize(
    ("count", "when", "expected"),
    [
        (0, None, None),
        (1, None, "Your last message has had no reply."),
        (3, None, "Your last 3 messages have had no reply."),
        (
            1,
            datetime(2026, 5, 1),
            "Your last message has had no reply. Robin last wrote on 2026-05-01.",
        ),
    ],
)
def test_silence_note_wording(count, when, expected):
    assert silence_note(count, "Robin", when) == expected


def test_a_thread_they_never_answered_counts_every_message():
    conversation = Conversation(
        match_name="Robin",
        owner_name="Sam",
        messages=[
            Message(role=Role.OWNER, text="hey!", sender="Sam"),
            Message(role=Role.OWNER, text="loved your profile", sender="Sam"),
        ],
    )
    assert unanswered(conversation) == 2
    report = ground_comfort(_scan(), conversation)
    assert report.match_last_wrote is None
    assert "Your last 2 messages have had no reply.\n" in report.to_text()


# -- what the owner says happened -------------------------------------------------


def test_what_happened_is_redacted_with_the_transcript():
    client = _loose()
    comfort(read_transcript(FADED), what=FADED_WHAT, settings=Settings(), client=client)
    sent = _sent(client)
    assert "What happened, in my words: two good dates, then [MATCH] went quiet" in sent
    assert "Jamie" not in sent and "Alex" not in sent


def test_an_empty_what_is_left_out():
    client = _loose()
    comfort(read_transcript(FADED), what="  ", settings=Settings(), client=client)
    assert "What happened, in my words" not in _sent(client)


# -- the owner's own safety ---------------------------------------------------------


def test_when_the_model_sees_risk_the_support_notice_comes_first():
    client = ReplayClient(_at_risk(), strict=False)
    report = comfort(read_transcript(FADED), settings=Settings(), client=client)
    text = report.to_text()
    assert text.startswith("!! Some of what you wrote sounds like")
    assert text.startswith(report.support.to_text() + "\n\nAfter two dates")
    assert report.support.steps == list(SUPPORT_STEPS)


def test_the_owners_own_words_raise_the_notice_even_if_the_model_does_not():
    report = comfort(
        read_transcript(FADED),
        what="honestly I don't want to be alive right now",
        settings=Settings(),
        client=_loose(),
    )
    assert report.support is not None
    assert report.support.headline == SUPPORT_HEADLINE


def test_ordinary_heartbreak_does_not_raise_the_notice():
    report = comfort(
        read_transcript(FADED),
        what="this is killing me, I wanted to end things on a good note",
        settings=Settings(),
        client=_loose(),
    )
    assert report.support is None


def test_the_support_notice_goes_above_a_danger_notice(capsys, monkeypatch):
    assert run(capsys, "add", "Casey", PRESSURE)[0] == 0
    _save_danger_check()
    client = ReplayClient(_at_risk("comfort_pressure"), strict=False)
    monkeypatch.setattr("confidant.client.build_client", lambda settings=None: client)
    code, out, _ = run(capsys, "comfort", "Casey")
    assert code == 0
    assert out.index("!! Some of what you wrote") < out.index("!! Something in Casey's")


# -- the command line ---------------------------------------------------------------


def _save_danger_check() -> None:
    with Store(clock=lambda: NOW) as store:
        [stored] = store.conversations("Casey")
        read_conversation(
            store,
            stored,
            ReadingKind.FLAGS,
            settings=Settings(),
            client=ReplayClient(RECORDINGS / "flags_pressure.json"),
        )


def test_comfort_from_a_transcript_file(capsys, answer_with):
    answer_with("comfort_faded")
    code, out, err = run(capsys, "comfort", FADED, "--what", FADED_WHAT)
    assert code == 0
    assert "WHAT YOU DID WELL" in out
    assert err.startswith("confidant: used ")


def test_comfort_json_is_one_document(capsys, answer_with):
    answer_with("comfort_pressure")
    code, out, _ = run(
        capsys,
        "comfort",
        PRESSURE,
        "--what",
        "I ended it this morning and I feel awful about it",
        "--json",
    )
    assert code == 0
    data = json.loads(out)
    assert data["discarded"] == 1
    assert data["support"] is None
    assert data["escalation"] is None


def test_comfort_for_someone_saved_carries_their_safety_notice(capsys, answer_with):
    assert run(capsys, "add", "Casey", PRESSURE)[0] == 0
    _save_danger_check()
    answer_with("comfort_pressure", strict=False)
    code, out, _ = run(capsys, "comfort", "casey")
    assert code == 0
    assert out.startswith("!! Something in Casey's messages is serious")
    assert out.index("!!") < out.index("WHAT YOU DID WELL")


def test_comfort_without_credentials_exits_2(capsys):
    code, _, err = run(capsys, "comfort", FADED)
    assert code == 2
    assert "No Anthropic credentials found" in err


def test_comfort_for_someone_unknown_points_at_list(capsys):
    code, _, err = run(capsys, "comfort", "Nobody")
    assert code == 2
    assert "'confidant list'" in err


def test_help_names_comfort(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "confidant comfort examples/faded_chat.txt" in out
    assert "draft, comfort, and profile --update" in out


def test_readme_comfort_example_matches_what_the_cli_prints(capsys, answer_with):
    answer_with("comfort_faded")
    _, out, err = run(capsys, "comfort", FADED, "--what", FADED_WHAT)
    readme = README.read_text()
    command = f'confidant comfort {FADED} --what "{FADED_WHAT}"\n```\n\nprints\n\n```\n'
    start = readme.index(command)
    block = readme[start:].split("```\n", 3)[2]
    assert out == block
    assert err.strip() in readme
