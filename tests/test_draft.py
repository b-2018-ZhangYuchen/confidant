"""Tests for `confidant draft`: drafting the owner's next message, and the checks on it.

Offline, like the rest of the suite. The model's side is two hand-written recordings in
``tests/fixtures/recorded``: an easy turn in ``examples/sample_chat.txt``, and a firm
reply to ``examples/pressure_chat.txt`` with two drafts written to be dropped.
"""

from __future__ import annotations

import functools
import json
import typing
from datetime import datetime
from pathlib import Path

import pytest

from confidant.analysis.draft import (
    FOOTER,
    WAITING_NOTE,
    DraftScan,
    Reply,
    Tone,
    draft_reply,
    ground_drafts,
)
from confidant.cli import _latest, main
from confidant.config import Settings
from confidant.ingest.transcript import read_transcript
from confidant.models import Conversation, Message, Role
from confidant.profile import read_conversation
from confidant.prompts.draft import TONES
from confidant.recording import RecordingError, ReplayClient, StaleRecording, run_analysis
from confidant.redaction import redact
from confidant.store import ReadingKind, Store, StoredConversation

README = Path(__file__).parents[1] / "README.md"
RECORDINGS = Path(__file__).parent / "fixtures" / "recorded"
SAMPLE = "examples/sample_chat.txt"
PRESSURE = "examples/pressure_chat.txt"
NOW = datetime(2026, 10, 8, 9, 0)


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


def _loose(name: str) -> ReplayClient:
    # Answers a request that differs from the recorded one, so a test can look at what a
    # variation of it sends without needing a recording per variation.
    return ReplayClient(RECORDINGS / f"{name}.json", strict=False)


# -- tones ----------------------------------------------------------------------


def test_every_tone_has_a_description_and_every_description_a_tone():
    assert set(typing.get_args(Tone)) == set(TONES)


def test_an_unknown_tone_is_refused_before_anything_is_sent():
    client = _loose("draft_sample")
    with pytest.raises(ValueError, match="Unknown tone 'sultry'"):
        draft_reply(read_transcript(SAMPLE), tone="sultry", settings=Settings(), client=client)
    assert client.requests == []


def test_the_tone_is_part_of_the_request():
    with pytest.raises(StaleRecording):
        run_analysis(
            "draft",
            SAMPLE,
            client=ReplayClient(RECORDINGS / "draft_sample.json"),
            options={"tone": "playful"},
        )
    client = _loose("draft_sample")
    draft_reply(read_transcript(SAMPLE), tone="playful", settings=Settings(), client=client)
    assert f"Tone: playful. {TONES['playful']}" in _sent(client)


# -- what comes back ------------------------------------------------------------


def test_drafts_come_back_in_the_owners_names_with_their_quotes():
    client = ReplayClient(RECORDINGS / "draft_sample.json")
    report = draft_reply(read_transcript(SAMPLE), settings=Settings(), client=client)

    assert report.tone == "natural"
    assert len(report.replies) == 3
    assert report.discarded == 0
    assert not report.waiting
    assert "Sam's turn" in report.where_it_stands
    text = report.to_text()
    assert "[OWNER]" not in text and "[MATCH]" not in text
    assert "re: \"that's a first date now, I don't make the rules\"" in text
    assert text.endswith(FOOTER)


def test_a_reply_can_build_on_the_owners_own_line():
    report = draft_reply(
        read_transcript(SAMPLE),
        settings=Settings(),
        client=ReplayClient(RECORDINGS / "draft_sample.json"),
    )
    assert "distracted please" in report.replies[2].picks_up


def test_misquoted_drafts_and_invented_placeholders_are_dropped():
    client = ReplayClient(RECORDINGS / "draft_pressure_firm.json")
    report = draft_reply(read_transcript(PRESSURE), tone="firm", settings=Settings(), client=client)

    assert len(report.replies) == 2
    assert report.discarded == 2
    text = report.to_text()
    assert "who are you with tonight" not in text
    assert "[PHONE_2]" not in text
    assert "(2 drafts left out" in text
    assert text.index("Before sending:") < text.index(FOOTER)


def test_quotes_that_are_not_in_the_thread_are_pruned_and_good_ones_kept():
    redaction = redact(read_transcript(SAMPLE))
    scan = DraftScan(
        where_it_stands="[MATCH] made a joke.",
        replies=[
            Reply(
                text="ha",
                picks_up=["unacceptable", "a quote nobody wrote"],
                why="Answers the joke.",
            )
        ],
        before_sending=None,
    )
    report = ground_drafts(scan, redaction, tone="natural", waiting=False)
    assert report.replies[0].picks_up == ["unacceptable"]
    assert report.discarded == 0


def test_a_quote_cannot_stitch_two_messages_together():
    redaction = redact(read_transcript(SAMPLE))
    stitched = "distracted please excellent"
    scan = DraftScan(
        where_it_stands="",
        replies=[Reply(text="ok", picks_up=[stitched], why="")],
        before_sending=None,
    )
    assert ground_drafts(scan, redaction, tone="natural", waiting=False).discarded == 1


def test_placeholders_the_request_issued_are_filled_back_in():
    conversation = read_transcript("examples/details_chat.txt")
    conversation.private_names.append("Maya")
    redaction = redact(conversation)
    scan = DraftScan(
        where_it_stands="[MATCH] asked to be texted near [ADDRESS_1].",
        replies=[
            Reply(
                text="will do, and [NAME_1] says hi back",
                picks_up=["tell [NAME_1] I said hi"],
                why="Passes the greeting along.",
            )
        ],
        before_sending=None,
    )
    report = redaction.restore(ground_drafts(scan, redaction, tone="natural", waiting=False))
    assert report.replies[0].text == "will do, and Maya says hi back"
    assert report.replies[0].picks_up == ["tell Maya I said hi"]
    assert "214 Linden Street" in report.where_it_stands


def test_with_nothing_left_it_says_so_rather_than_printing_an_empty_list():
    redaction = redact(read_transcript(SAMPLE))
    scan = DraftScan(
        where_it_stands="[MATCH] wrote last.",
        replies=[Reply(text="hi [NAME_7]", picks_up=["excellent"], why="")],
        before_sending=None,
    )
    report = ground_drafts(scan, redaction, tone="brief", waiting=False)
    assert report.replies == []
    assert "No draft held up against the thread" in report.to_text()


# -- whose turn it is ---------------------------------------------------------------


def _owner_wrote_last() -> Conversation:
    conversation = read_transcript(SAMPLE)
    last = conversation.messages[-1]
    conversation.messages.append(
        Message(
            role=Role.OWNER,
            text="ok go, carpet opinion",
            sender="Sam",
            timestamp=last.timestamp,
        )
    )
    return conversation


def test_when_the_owner_wrote_last_the_request_and_the_report_both_say_so():
    client = _loose("draft_sample")
    report = draft_reply(_owner_wrote_last(), settings=Settings(), client=client)
    assert "I sent the last message and have not had an answer" in _sent(client)
    assert report.waiting
    assert WAITING_NOTE in report.to_text()


def test_when_it_is_the_owners_turn_there_is_no_waiting_note():
    client = ReplayClient(RECORDINGS / "draft_sample.json")
    report = draft_reply(read_transcript(SAMPLE), settings=Settings(), client=client)
    assert "I sent the last message" not in _sent(client)
    assert WAITING_NOTE not in report.to_text()


def test_a_thread_they_never_wrote_in_has_nothing_to_reply_to():
    conversation = Conversation(
        match_name="Robin",
        owner_name="Sam",
        messages=[Message(role=Role.OWNER, text="hey!", sender="Sam")],
    )
    with pytest.raises(ValueError, match="nothing to reply to"):
        draft_reply(conversation, settings=Settings(), client=_loose("draft_sample"))


# -- what the owner asks to say -------------------------------------------------------


def test_what_the_owner_wants_to_say_is_redacted_with_the_transcript():
    conversation = read_transcript("examples/details_chat.txt")
    conversation.private_names.append("Maya")
    client = _loose("draft_sample")
    draft_reply(
        conversation,
        say="ask if Maya can pick me up from 214 Linden Street",
        settings=Settings(),
        client=client,
    )
    sent = _sent(client)
    assert "What I want to say: ask if [NAME_1] can pick me up from [ADDRESS_1]" in sent
    assert "Maya" not in sent and "Linden" not in sent


def test_a_note_does_not_renumber_the_transcript():
    conversation = read_transcript("examples/details_chat.txt")
    plain = redact(conversation)
    noted = redact(conversation, notes=["call 555-010-9999 or email a@example.com"])
    assert noted.conversation.messages == plain.conversation.messages
    assert noted.notes == ["call [PHONE_2] or email [EMAIL_2]"]


def test_an_empty_say_is_left_out():
    client = _loose("draft_sample")
    draft_reply(read_transcript(SAMPLE), say="   ", settings=Settings(), client=client)
    assert "What I want to say" not in _sent(client)


# -- recordings -------------------------------------------------------------------


def test_a_recording_keeps_the_options_it_was_made_with():
    source = json.loads((RECORDINGS / "draft_pressure_firm.json").read_text())["source"]
    assert source == {"analysis": "draft", "transcript": PRESSURE, "tone": "firm"}


def test_an_option_an_analysis_does_not_take_is_an_error():
    with pytest.raises(RecordingError, match="flags does not take tone"):
        run_analysis("flags", PRESSURE, client=_loose("flags_pressure"), options={"tone": "firm"})


# -- the command line -------------------------------------------------------------


def test_draft_from_a_transcript_file(capsys, answer_with):
    answer_with("draft_sample")
    code, out, err = run(capsys, "draft", SAMPLE)
    assert code == 0
    assert out.startswith("Sam asked to be distracted")
    assert "DRAFTS (natural)" in out
    assert err.startswith("confidant: used ")


def test_draft_json_is_one_document(capsys, answer_with):
    answer_with("draft_pressure_firm")
    code, out, _ = run(capsys, "draft", PRESSURE, "--tone", "firm", "--json")
    assert code == 0
    data = json.loads(out)
    assert data["tone"] == "firm"
    assert data["discarded"] == 2
    assert data["escalation"] is None


def test_draft_for_someone_saved_uses_their_conversation(capsys, answer_with):
    assert run(capsys, "add", "Robin", SAMPLE)[0] == 0
    answer_with("draft_sample")
    code, out, _ = run(capsys, "draft", "robin")
    assert code == 0
    assert "DRAFTS (natural)" in out


def test_a_saved_danger_flag_puts_the_safety_notice_above_the_drafts(capsys, answer_with):
    assert run(capsys, "add", "Casey", PRESSURE)[0] == 0
    with Store(clock=lambda: NOW) as store:
        [stored] = store.conversations("Casey")
        read_conversation(
            store,
            stored,
            ReadingKind.FLAGS,
            settings=Settings(),
            client=ReplayClient(RECORDINGS / "flags_pressure.json"),
        )
    answer_with("draft_pressure_firm")
    code, out, _ = run(capsys, "draft", "Casey", "--tone", "firm")
    assert code == 0
    assert out.startswith("!! Something in Casey's messages is serious")
    assert out.index("!!") < out.index("DRAFTS (firm)")


def test_someone_neither_saved_nor_a_file_points_at_list(capsys):
    code, _, err = run(capsys, "draft", "Nobody")
    assert code == 2
    assert "not a transcript file, and no one by that name is saved" in err
    assert "'confidant list'" in err


def test_someone_saved_with_nothing_to_draft_into(capsys):
    run(capsys, "add", "Robin")
    code, _, err = run(capsys, "draft", "Robin")
    assert code == 2
    assert "confidant add Robin FILE" in err


def test_owner_and_match_are_only_for_files(capsys):
    run(capsys, "add", "Robin", SAMPLE)
    code, _, err = run(capsys, "draft", "Robin", "--match", "Robyn")
    assert code == 2
    assert "--owner and --match apply to a transcript file" in err


def test_draft_without_credentials_exits_2(capsys):
    code, _, err = run(capsys, "draft", SAMPLE)
    assert code == 2
    assert "No Anthropic credentials found" in err


def test_a_draft_from_a_file_never_opens_the_store(capsys, answer_with, isolated):
    answer_with("draft_sample")
    run(capsys, "draft", SAMPLE)
    assert not isolated.exists()


def test_an_unknown_tone_is_a_usage_error(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["draft", SAMPLE, "--tone", "sultry"])
    assert exc.value.code == 2
    assert "invalid choice: 'sultry'" in capsys.readouterr().err


def test_the_most_recent_conversation_is_the_one_drafted_into():
    def stored(id: int, last: datetime | None) -> StoredConversation:
        return StoredConversation(
            id=id,
            person_id=1,
            source=None,
            imported_at=NOW,
            messages=1,
            first_message=last,
            last_message=last,
        )

    older, newer = stored(1, datetime(2026, 3, 1)), stored(2, datetime(2026, 4, 1))
    assert _latest([newer, older]) is newer
    undated = [stored(3, None), stored(4, None)]
    assert _latest(undated).id == 4


def test_help_names_draft_and_says_it_never_sends(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    out = capsys.readouterr().out
    assert "confidant draft examples/sample_chat.txt --tone playful" in out
    assert "draft never sends a message" in out


def test_readme_draft_example_matches_what_the_cli_prints(capsys, answer_with):
    answer_with("draft_sample")
    _, out, err = run(capsys, "draft", SAMPLE)
    readme = README.read_text()
    start = readme.index("confidant draft examples/sample_chat.txt\n```\n\nprints\n\n```\n")
    block = readme[start:].split("```\n", 3)[2]
    assert out == block
    assert err.strip() in readme
