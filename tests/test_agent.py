"""Tests for `confidant ask`: the tool loop, the tools, and what is checked after it.

Offline, like the rest of the suite. Three hand-written recording sets in
``tests/fixtures/recorded`` drive the agent end to end: ``ask_pressure`` (a search, the
statistics, the red-flag check, and an answer with one misquote) and ``ask_sample`` (an
answer from the messages alone, so the red-flag check runs afterwards), and for `confidant
chat`, ``chat_pressure``: ``ask_pressure`` call for call, then a follow-up that searches
once and answers without a second red-flag check. The loop itself is tested against a
scripted fake, where each round's response is spelled out.
"""

from __future__ import annotations

import functools
import io
import json
import shutil
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from confidant.agent import (
    HISTORY_LIMIT,
    READ_LIMIT,
    SEARCH_LIMIT,
    AgentAnswer,
    AskReport,
    Chat,
    _Session,
    ask,
    chat,
)
from confidant.analysis.flags import FlagScan
from confidant.analysis.personality import Evidence
from confidant.cli import main
from confidant.client import (
    DEFAULT_TOOL_ROUNDS,
    ModelRefusal,
    Tool,
    ToolError,
    ToolLoopError,
    block_param,
    structured_call,
)
from confidant.config import Settings
from confidant.ingest.transcript import read_transcript
from confidant.models import Conversation, Message, Role
from confidant.prompts.agent import FOLLOW_UP_NOTE, TOOL_DESCRIPTIONS
from confidant.recording import (
    Fingerprint,
    Recording,
    RecordingClient,
    RecordingError,
    ReplayClient,
    recording_set,
    replay,
    run_analysis,
)
from confidant.recording import main as recording_main
from confidant.redaction import redact
from confidant.safety import STILL_APPLIES, escalate
from confidant.store import Store
from confidant.usage import UsageMeter

README = Path(__file__).parents[1] / "README.md"
RECORDINGS = Path(__file__).parent / "fixtures" / "recorded"
SAMPLE = "examples/sample_chat.txt"
PRESSURE = "examples/pressure_chat.txt"
PRESSURE_QUESTION = "Casey says everyone shares their location. Is that normal?"
SAMPLE_QUESTION = "Does Robin ever ask about my life, or is it all them?"
PRESSURE_NOW = datetime(2026, 4, 12, 10, 0)
SAMPLE_NOW = datetime(2026, 3, 6, 9, 0)


def _set(name: str) -> ReplayClient:
    return ReplayClient(*recording_set(RECORDINGS / f"{name}.json"))


def _ask_pressure(**kwargs) -> tuple[AskReport, ReplayClient]:
    client = kwargs.pop("client", None) or _set("ask_pressure")
    report = ask(
        read_transcript(PRESSURE),
        PRESSURE_QUESTION,
        settings=Settings(),
        client=client,
        now=PRESSURE_NOW,
        **kwargs,
    )
    return report, client


# -- a scripted model, for the loop itself ------------------------------------


def _use(id_: str, name: str, inputs: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", id=id_, name=name, input=inputs or {})


def _tool_round(*blocks, thinking: bool = True) -> SimpleNamespace:
    content = list(blocks)
    if thinking:
        content.insert(0, SimpleNamespace(type="thinking", thinking="hmm", signature="sig"))
    return SimpleNamespace(
        stop_reason="tool_use", stop_details=None, parsed_output=None, content=content
    )


def _final(parsed) -> SimpleNamespace:
    return SimpleNamespace(stop_reason="end_turn", stop_details=None, parsed_output=parsed)


ANSWER = AgentAnswer(answer="ok", evidence=[], confidence="low", caveat="thin")


class Scripted:
    """Answers each ``messages.stream`` call with the next scripted response."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests: list[dict] = []
        self.messages = SimpleNamespace(stream=self._stream)

    def _stream(self, **request):
        self.requests.append(request)
        return _Stream(self.responses.pop(0))


class _Stream:
    def __init__(self, response):
        self.response = response

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return None

    def __iter__(self):
        return iter(())

    def get_final_message(self):
        return self.response


def _echo(name: str = "echo") -> Tool:
    return Tool(name, "Repeat the input.", lambda args: f"you said {args.get('x')}")


def _call(client, *tools: Tool, **kwargs):
    return structured_call(
        schema=AgentAnswer,
        system="system",
        user_content="question",
        settings=Settings(),
        client=client,
        tools=tools,
        **kwargs,
    )


def test_tool_results_go_back_with_the_round_that_asked_for_them():
    client = Scripted(
        _tool_round(_use("a", "echo", {"x": 1}), _use("b", "echo", {"x": 2})), _final(ANSWER)
    )
    assert _call(client, _echo()) == ANSWER

    first, second = client.requests
    assert first["tools"] == [_echo().param()]
    assert len(first["messages"]) == 1, "each request keeps the messages it was sent with"
    _, assistant, results = second["messages"]
    assert assistant["role"] == "assistant"
    # Thinking goes back exactly as it came, or the API rejects the next round.
    assert assistant["content"][0] == {"type": "thinking", "thinking": "hmm", "signature": "sig"}
    assert results == {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "a", "content": "you said 1"},
            {"type": "tool_result", "tool_use_id": "b", "content": "you said 2"},
        ],
    }


def test_a_tool_error_and_an_unknown_tool_go_back_as_failed_results():
    def broken(_):
        raise ToolError("not like that")

    client = Scripted(_tool_round(_use("a", "broken"), _use("b", "missing")), _final(ANSWER))
    _call(client, Tool("broken", "Fails.", broken))
    results = client.requests[1]["messages"][2]["content"]
    assert results[0] == {
        "type": "tool_result",
        "tool_use_id": "a",
        "content": "not like that",
        "is_error": True,
    }
    assert results[1]["is_error"] and "no tool called 'missing'" in results[1]["content"]


def test_any_other_exception_from_a_tool_is_a_bug_and_propagates():
    def buggy(_):
        raise KeyError("oops")

    with pytest.raises(KeyError):
        _call(Scripted(_tool_round(_use("a", "buggy"))), Tool("buggy", "Bug.", buggy))


def test_a_model_that_never_stops_calling_tools_is_stopped():
    rounds = [_tool_round(_use(str(i), "echo")) for i in range(3)]
    client = Scripted(*rounds)
    with pytest.raises(ToolLoopError, match="2 rounds of tools"):
        _call(client, _echo(), max_tool_rounds=2)
    assert len(client.requests) == 3


def test_the_round_limit_has_room_for_a_careful_answer():
    assert 4 <= DEFAULT_TOOL_ROUNDS <= 12


def test_every_round_is_billed():
    usage = SimpleNamespace(
        input_tokens=10, cache_creation_input_tokens=0, cache_read_input_tokens=0, output_tokens=5
    )
    tool_round = _tool_round(_use("a", "echo"))
    tool_round.usage = usage
    final = _final(ANSWER)
    final.usage = usage
    meter = UsageMeter()
    _call(Scripted(tool_round, final), _echo(), on_usage=meter)
    assert len(meter.calls) == 2


def test_a_call_without_tools_sends_no_tools_field():
    client = Scripted(_final(ANSWER))
    _call(client)
    assert "tools" not in client.requests[0]


def test_tool_use_without_tools_is_not_looped_on():
    # Cannot happen against the API, but a stray stop reason should not be retried forever.
    client = Scripted(_tool_round(_use("a", "echo")))
    with pytest.raises(RuntimeError, match="stop_reason='tool_use'"):
        _call(client)


def test_block_param_sends_back_only_what_the_api_accepts():
    text = SimpleNamespace(type="text", text="hi", parsed_output=object(), citations=None)
    assert block_param(text) == {"type": "text", "text": "hi"}
    assert block_param(SimpleNamespace(type="redacted_thinking", data="x")) == {
        "type": "redacted_thinking",
        "data": "x",
    }
    with pytest.raises(ValueError, match="'server_tool_use'"):
        block_param(SimpleNamespace(type="server_tool_use"))


# -- the tools ----------------------------------------------------------------


def _session(conversation: Conversation | None = None, client=None) -> _Session:
    conversation = conversation or read_transcript(PRESSURE)
    return _Session(
        redact(conversation).conversation,
        settings=Settings(),
        client=client,
        now=PRESSURE_NOW,
        on_progress=None,
        on_usage=None,
    )


def test_every_tool_is_described_in_the_prompt_module():
    assert [tool.name for tool in _session().tools()] == list(TOOL_DESCRIPTIONS)


def test_search_finds_messages_with_their_numbers_and_only_from_who_was_asked():
    session = _session()
    out = session.search({"text": "Location", "who": "both"})
    assert out.splitlines() == [
        "#3 [2026-04-10 20:16] MATCH: send me your location next time you're out, "
        "I like knowing where you are",
    ]
    assert "#6" in session.search({"text": "friends"})
    assert "#6" not in session.search({"text": "friends", "who": "match"})
    assert session.search({"text": "pizza"}) == "No message contains 'pizza'."
    with pytest.raises(ToolError):
        session.search({"text": "  "})


def test_search_results_are_capped():
    messages = [Message(Role.MATCH, f"hello {i}", "Robin") for i in range(SEARCH_LIMIT + 5)]
    session = _session(Conversation("Robin", "Sam", messages))
    lines = session.search({"text": "hello"}).splitlines()
    assert len(lines) == SEARCH_LIMIT + 1
    assert lines[-1] == "(5 more not shown; search for something narrower.)"


def test_read_is_numbered_from_one_bounded_and_capped():
    session = _session()
    assert session.read({"first": 1, "last": 1}).splitlines() == [
        "#1 [2026-04-10 20:02] MATCH: you looked amazing in that last photo. where was that taken?",
        "(messages 2 to 10 not shown)",
    ]
    assert session.read({"first": 9, "last": 500}).splitlines()[-1].startswith("#10 ")
    with pytest.raises(ToolError, match="numbered 1 to 10"):
        session.read({"first": 11, "last": 12})
    with pytest.raises(ToolError, match="message numbers"):
        session.read({"first": "soon"})

    long = [Message(Role.MATCH, f"line {i}", "Robin") for i in range(READ_LIMIT * 2)]
    lines = _session(Conversation("Robin", "Sam", long)).read({"first": 1, "last": 999})
    assert len(lines.splitlines()) == READ_LIMIT + 1


def test_a_multi_line_message_reads_as_one_numbered_message():
    out = _session().read({"first": 7, "last": 7})
    assert out.startswith("#7 [2026-04-11 18:41] MATCH: your friends again?")
    assert "\n    you should be spending that time with me" in out


def test_stats_are_local_and_placeholdered():
    out = _session().stats({})
    assert out.splitlines() == [
        "Messages: 10 (4 owner / 6 match)",
        "Average words per message: owner 7.8, match 11.2",
        "First message: 2026-04-10 20:02",
        "Last message: 2026-04-12 09:20",
        "Stretches of conversation (split at 8+ hours of quiet) started by: owner 2, match 1",
        "Longest quiet stretch: 22 hours",
        "Wrote last: match",
    ]


def test_stats_without_timestamps_say_so():
    undated = Conversation("Robin", "Sam", [Message(Role.MATCH, "hi", "Robin")])
    assert "No message has a timestamp" in _session(undated).stats({})


def test_keep_in_touch_uses_the_redacted_name_and_a_danger_finding_once_known(recording):
    session = _session(client=ReplayClient(recording("flags_pressure")))
    assert session.keep_in_touch({}).startswith("[MATCH]: nothing due.")
    session.check_red_flags({})
    assert session.keep_in_touch({}).startswith("[MATCH]: not suggested.")


def test_check_red_flags_tells_the_model_a_notice_is_shown_but_not_its_words(recording):
    session = _session(client=ReplayClient(recording("flags_pressure")))
    data = json.loads(session.check_red_flags({}))
    assert data["safety_notice_shown_to_owner"] is True
    assert "escalation" not in data
    assert data["flags"][0]["category"] == "monitoring"
    # The model works in placeholders: no name comes back in a tool result.
    assert "Casey" not in json.dumps(data) and "Jordan" not in json.dumps(data)


def test_each_analyzer_runs_once_however_often_it_is_asked_for(recording):
    client = ReplayClient(recording("flags_pressure"), recording("analyze_sample"), strict=False)
    session = _session(client=client)
    assert session.check_red_flags({}) == session.check_red_flags({})
    session.read_personality({})
    session.read_personality({})
    assert client.remaining == 0
    assert len(client.requests) == 2


def test_a_check_that_declines_becomes_a_tool_error_and_is_not_retried(recording):
    client = ReplayClient(recording("analyze_refusal"), strict=False)
    session = _session(client=client)
    with pytest.raises(ToolError, match="declined"):
        session.read_personality({})

    refusal = recording("analyze_refusal")
    refusal.fingerprint = Fingerprint("FlagScan", "", "")
    session = _session(client=ReplayClient(refusal, strict=False))
    with pytest.raises(ToolError, match="declined"):
        session.run_flags()
    with pytest.raises(ToolError, match="declined"):
        session.run_flags()


# -- the agent, end to end ------------------------------------------------------


def test_pressure_answer_leads_with_the_notice_and_drops_the_misquote():
    report, client = _ask_pressure()
    assert client.remaining == 0
    assert report.escalation is not None
    assert report.escalation.categories == ["monitoring", "isolation"]
    assert report.consulted == ["search_messages", "conversation_stats", "check_red_flags"]
    assert report.discarded == 1
    assert [e.quote for e in report.evidence] == [
        "send me your location next time you're out, I like knowing where you are",
        "come on, it's not a big deal. everyone does it",
    ]
    text = report.to_text()
    assert text.startswith("!! Something in Casey's messages is serious")
    assert "Casey asked for your location" in text
    assert "Jordan's no" in text
    assert "(1 quote left out: not found in the transcript as written.)" in text
    assert "Looked at: a search, the statistics, the red-flag check." in text


def test_nothing_identifying_is_sent_in_any_request():
    _, client = _ask_pressure()
    assert len(client.requests) == 4
    for request in client.requests:
        sent = json.dumps(request["messages"]) + json.dumps(request.get("tools", []))
        assert "Casey" not in sent and "Jordan" not in sent
    # The question is redacted with the transcript's own placeholders.
    first = client.requests[0]["messages"][0]["content"]
    assert first.endswith(
        "My question: [MATCH] says everyone shares their location. Is that normal?"
    )
    assert "--- TRANSCRIPT ---" not in first


def test_the_nested_check_sends_exactly_what_flags_sends(recording):
    _, client = _ask_pressure()
    nested = Fingerprint.of(client.requests[2])
    assert nested == recording("flags_pressure").fingerprint


def test_the_check_runs_after_an_answer_that_did_not_ask_for_it():
    client = _set("ask_sample")
    report = ask(
        read_transcript(SAMPLE), SAMPLE_QUESTION, settings=Settings(), client=client, now=SAMPLE_NOW
    )
    assert client.remaining == 0
    assert client.requests[-1]["output_format"] is FlagScan
    assert report.consulted == ["read_messages"]
    assert report.escalation is None
    assert report.discarded == 0
    assert report.to_text().splitlines()[0].startswith("They ask, and they follow up.")


def test_a_saved_notice_leads_when_this_conversation_has_none(recording):
    pressure = read_transcript(PRESSURE)
    flags = recording("flags_pressure").response_for(FlagScan).parsed_output.flags
    saved = escalate(flags, pressure.match_name)
    report = ask(
        read_transcript(SAMPLE),
        SAMPLE_QUESTION,
        settings=Settings(),
        client=_set("ask_sample"),
        now=SAMPLE_NOW,
        escalation=saved,
    )
    assert report.escalation == saved


def test_a_declined_safety_check_is_said_outright(recording):
    refusal = recording("analyze_refusal")
    refusal.fingerprint = Fingerprint("FlagScan", "", "")
    first, answer, _ = recording_set(RECORDINGS / "ask_sample.json")
    client = ReplayClient(first, answer, refusal, strict=False)
    report = ask(
        read_transcript(SAMPLE), SAMPLE_QUESTION, settings=Settings(), client=client, now=SAMPLE_NOW
    )
    assert report.escalation is None
    assert report.unchecked is not None
    assert "has not been checked for anything dangerous" in report.to_text()


def test_an_empty_question_or_a_silent_match_is_refused_before_anything_is_sent():
    client = ReplayClient()
    with pytest.raises(ValueError, match="Ask a question"):
        ask(read_transcript(SAMPLE), "   ", settings=Settings(), client=client)
    silent = Conversation("Robin", "Sam", [Message(Role.OWNER, "hello?", "Sam")])
    with pytest.raises(ValueError, match="no messages"):
        ask(silent, "why so quiet", settings=Settings(), client=client)
    assert client.requests == []


# -- recordings of a tool loop ------------------------------------------------


def test_a_set_replays_from_its_first_file_only():
    assert len(recording_set(RECORDINGS / "ask_pressure.json")) == 4
    assert isinstance(replay(RECORDINGS / "ask_pressure.json"), AskReport)
    with pytest.raises(RecordingError, match="use the set's first file, ask_pressure.json"):
        recording_set(RECORDINGS / "ask_pressure.2.json")


def test_the_moment_asked_is_part_of_the_source():
    first = Recording.load(RECORDINGS / "ask_pressure.json")
    assert first.source["now"] == PRESSURE_NOW.isoformat(timespec="minutes")
    report = run_analysis(
        "ask",
        PRESSURE,
        client=_set("ask_pressure"),
        options={"question": PRESSURE_QUESTION, "now": "2026-04-12T10:00"},
    )
    assert report.escalation is not None


def test_a_tool_round_is_recorded_with_its_content(tmp_path):
    rounds = Scripted(_tool_round(_use("a", "conversation_stats")), _final(ANSWER))
    recorder = RecordingClient(rounds, tmp_path / "ask.json", source={"analysis": "ask"})
    structured_call(
        schema=AgentAnswer,
        system="s",
        user_content="q",
        settings=Settings(),
        client=recorder,
        tools=[Tool("conversation_stats", "Stats.", lambda _: "10 messages")],
    )
    first, second = (Recording.load(p) for p in recorder.saved)
    assert first.stop_reason == "tool_use"
    assert first.content == [
        {"type": "thinking", "thinking": "hmm", "signature": "sig"},
        {"type": "tool_use", "id": "a", "name": "conversation_stats", "input": {}},
    ]
    assert second.content is None
    assert [p.name for p in recorder.saved] == ["ask.json", "ask.2.json"]


def test_tool_definitions_are_part_of_the_fingerprint():
    request = {"output_format": AgentAnswer, "system": "s", "messages": []}
    plain = Fingerprint.of(request)
    with_tools = Fingerprint.of(request | {"tools": [_echo().param()]})
    renamed = Fingerprint.of(request | {"tools": [_echo("other").param()]})
    assert plain.system_sha256 != with_tools.system_sha256 != renamed.system_sha256


def test_stamp_refreshes_a_whole_set(tmp_path, capsys):
    for path in recording_set(RECORDINGS / "ask_pressure.json"):
        shutil.copy(path.path, tmp_path / path.path.name)
    stale = Recording.load(tmp_path / "ask_pressure.4.json")
    stale.fingerprint = Fingerprint("AgentAnswer", "0" * 64, "0" * 64)
    stale.save(tmp_path / "ask_pressure.4.json")

    assert recording_main(["stamp", str(tmp_path / "ask_pressure.json")]) == 0
    out = capsys.readouterr().out
    assert "ask_pressure.4.json: the system prompt changed" in out
    assert "ask_pressure.2.json" not in out
    assert Recording.load(tmp_path / "ask_pressure.4.json").fingerprint == (
        Recording.load(RECORDINGS / "ask_pressure.4.json").fingerprint
    )


def test_stamp_refuses_a_set_the_code_no_longer_walks_through(tmp_path, capsys):
    for path in recording_set(RECORDINGS / "ask_sample.json"):
        shutil.copy(path.path, tmp_path / path.path.name)
    shutil.copy(tmp_path / "ask_sample.3.json", tmp_path / "ask_sample.4.json")
    assert recording_main(["stamp", str(tmp_path / "ask_sample.json")]) == 2
    assert "made 3 calls" in capsys.readouterr().err


# -- the command line -----------------------------------------------------------


@pytest.fixture
def cli(monkeypatch, tmp_path):
    for var in ("ANTHROPIC_API_KEY", "CONFIDANT_MODEL", "CONFIDANT_EFFORT", "CONFIDANT_MAX_TOKENS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr("confidant.config.load_dotenv", lambda *a, **k: False)
    monkeypatch.setenv("CONFIDANT_DB", str(tmp_path / "store" / "confidant.db"))
    monkeypatch.setattr("confidant.agent.datetime", _Frozen)
    monkeypatch.setattr("confidant.cli.Store", functools.partial(Store, clock=lambda: PRESSURE_NOW))

    def use(name: str) -> ReplayClient:
        client = _set(name)
        monkeypatch.setattr("confidant.client.build_client", lambda *_: client)
        return client

    return use


class _Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        return PRESSURE_NOW


def test_cli_ask_on_a_transcript(cli, capsys):
    cli("ask_pressure")
    assert main(["ask", PRESSURE, PRESSURE_QUESTION]) == 0
    captured = capsys.readouterr()
    assert captured.out.startswith("!! Something in Casey's messages is serious")
    assert captured.err.startswith("confidant: 4 calls used ")


def test_cli_ask_json(cli, capsys):
    cli("ask_pressure")
    assert main(["ask", PRESSURE, PRESSURE_QUESTION, "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["question"] == PRESSURE_QUESTION
    assert data["discarded"] == 1


def test_cli_ask_about_someone_saved(cli, capsys):
    assert main(["add", "Casey", PRESSURE]) == 0
    capsys.readouterr()
    cli("ask_pressure")
    assert main(["ask", "Casey", PRESSURE_QUESTION]) == 0
    assert "Casey asked for your location" in capsys.readouterr().out


def test_cli_ask_reports_a_runaway_loop_as_exit_4(cli, capsys, monkeypatch):
    def runaway(*args, **kwargs):
        raise ToolLoopError(8)

    monkeypatch.setattr("confidant.agent.structured_call", runaway)
    cli("ask_pressure")
    assert main(["ask", PRESSURE, PRESSURE_QUESTION]) == 4
    assert "8 rounds of tools" in capsys.readouterr().err


def test_readme_ask_example_matches_what_the_cli_prints(cli, capsys):
    cli("ask_pressure")
    assert main(["ask", PRESSURE, PRESSURE_QUESTION]) == 0
    captured = capsys.readouterr()
    readme = README.read_text(encoding="utf-8")
    start = readme.index(f'confidant ask {PRESSURE} \\\n    "{PRESSURE_QUESTION}"\n```\n\nprints')
    block = readme[start:].split("```\n", 3)[2]
    assert captured.out == block
    assert captured.err.strip() in readme


# -- chat: follow-up questions ----------------------------------------------------

FOLLOW_UP = "And what about when I went out with my friends?"
NO_FLAGS = FlagScan(flags=[], summary="Nothing of concern.", confidence="low")


def _answer(text: str = "ok", *quotes: str) -> AgentAnswer:
    evidence = [Evidence(quote=q, why_it_matters="because") for q in quotes]
    return AgentAnswer(answer=text, evidence=evidence, confidence="low", caveat="thin")


def _chat_pressure():
    client = ReplayClient(*recording_set(RECORDINGS / "chat_pressure.json"))
    reports = chat(
        read_transcript(PRESSURE),
        [PRESSURE_QUESTION, FOLLOW_UP],
        settings=Settings(),
        client=client,
        now=PRESSURE_NOW,
    )
    return reports, client


def _scripted_chat(*responses, conversation: Conversation | None = None) -> tuple[Chat, Scripted]:
    client = Scripted(*responses)
    session = Chat(
        conversation or read_transcript(SAMPLE),
        settings=Settings(),
        client=client,
        now=PRESSURE_NOW,
    )
    return session, client


def _first_turn(request: dict) -> str:
    return request["messages"][0]["content"]


# -- the recorded chat ----------------------------------------------------------


def test_the_first_question_sends_exactly_what_ask_sends():
    _, client = _chat_pressure()
    ask_set = recording_set(RECORDINGS / "ask_pressure.json")
    sent = [Fingerprint.of(request) for request in client.requests[: len(ask_set)]]
    assert sent == [recording.fingerprint for recording in ask_set]


def test_the_follow_up_reuses_the_red_flag_check():
    (first, second), client = _chat_pressure()
    assert client.remaining == 0
    assert [r["output_format"] for r in client.requests].count(FlagScan) == 1
    assert first.consulted == ["search_messages", "conversation_stats", "check_red_flags"]
    assert second.consulted == ["search_messages"]
    assert second.escalation == first.escalation
    assert second.question == FOLLOW_UP
    assert "your friends again? they don't even like me" in [e.quote for e in second.evidence]


def test_the_follow_up_carries_the_earlier_exchange_redacted_and_grounded():
    _, client = _chat_pressure()
    follow_up = _first_turn(client.requests[4])
    assert FOLLOW_UP_NOTE in follow_up
    assert "Earlier question 1: [MATCH] says everyone shares their location." in follow_up
    assert "Your answer: Some couples do share locations" in follow_up
    assert '  quoted: "come on, it\'s not a big deal. everyone does it"' in follow_up
    # The misquote dropped from the first answer is not handed on as checked.
    assert "I need to know where you are at all times" not in follow_up
    assert follow_up.endswith(f"My follow-up question: {FOLLOW_UP}")
    for request in client.requests:
        sent = json.dumps(request["messages"])
        assert "Casey" not in sent and "Jordan" not in sent


def test_a_replayed_chat_returns_every_answer():
    reports = run_analysis(
        "chat",
        PRESSURE,
        client=ReplayClient(*recording_set(RECORDINGS / "chat_pressure.json")),
        options={"questions": f"{PRESSURE_QUESTION}\n{FOLLOW_UP}", "now": "2026-04-12T10:00"},
    )
    assert [r.question for r in reports] == [PRESSURE_QUESTION, FOLLOW_UP]


def test_the_reminder_names_the_danger_without_repeating_the_steps():
    (first, second), _ = _chat_pressure()
    full = second.to_text()
    short = second.to_text(full_notice=False)
    assert full.startswith(first.escalation.to_text())
    notice, rest = short.split("\n\n", 1)
    assert notice.startswith("!! Something in Casey's messages is serious")
    assert " ".join(notice.split()).endswith(STILL_APPLIES)
    assert "You do not have to share your location" not in short
    assert full.endswith(rest)


# -- the session, against a scripted model -------------------------------------


def test_an_analyzer_runs_once_for_the_whole_chat():
    session, client = _scripted_chat(
        _final(_answer()),
        _final(NO_FLAGS),
        _tool_round(_use("a", "check_red_flags")),
        _final(_answer()),
    )
    first = session.ask("does Robin ask about me?")
    second = session.ask("is there anything to worry about?")
    assert [r["output_format"] for r in client.requests].count(FlagScan) == 1
    assert first.consulted == []
    assert second.consulted == ["check_red_flags"]
    assert client.responses == []


def test_older_exchanges_are_dropped_past_the_limit():
    asked = HISTORY_LIMIT + 3
    answers = [_final(_answer(f"answer {n}")) for n in range(asked)]
    session, client = _scripted_chat(answers[0], _final(NO_FLAGS), *answers[1:])
    for n in range(asked):
        session.ask(f"question {n + 1}")
    last = _first_turn(client.requests[-1])
    assert "(2 earlier questions not shown.)" in last
    assert "Earlier question 2:" not in last
    assert "Earlier question 3: question 3" in last
    assert f"Earlier question {asked - 1}: question {asked - 1}" in last
    assert last.count("Your answer:") == HISTORY_LIMIT


def test_a_question_that_fails_leaves_no_trace_in_the_next():
    refusal = SimpleNamespace(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category="cyber", explanation="no"),
        parsed_output=None,
    )
    session, client = _scripted_chat(refusal, _final(_answer()), _final(NO_FLAGS))
    with pytest.raises(ModelRefusal):
        session.ask("first try")
    session.ask("second try")
    second = _first_turn(client.requests[1])
    assert "Earlier question" not in second
    assert second.endswith("My question: second try")


def test_a_follow_up_is_redacted_with_the_conversations_own_key():
    conversation = read_transcript(SAMPLE)
    conversation.private_names.append("Maya")
    session, client = _scripted_chat(
        _final(_answer()),
        _final(NO_FLAGS),
        _final(_answer("Call [PHONE_1] if [MATCH] means it, and ask [NAME_1].")),
        conversation=conversation,
    )
    session.ask("does Robin like me?")
    report = session.ask("Robin gave me 555-201-3344, should I call? Maya thinks so")
    sent = _first_turn(client.requests[2])
    assert "[MATCH] gave me [PHONE_1], should I call? [NAME_1] thinks so" in sent
    assert "Robin" not in sent and "555" not in sent and "Maya" not in sent
    assert report.answer == "Call 555-201-3344 if Robin means it, and ask Maya."


def test_a_saved_notice_leads_every_answer():
    (first, _), _ = _chat_pressure()
    session, _ = _scripted_chat(_final(_answer()), _final(NO_FLAGS), _final(_answer()))
    session.escalation = first.escalation
    assert session.ask("one").escalation == first.escalation
    assert session.ask("two").escalation == first.escalation


def test_a_silent_match_is_refused_before_anything_is_sent():
    silent = Conversation("Robin", "Sam", [Message(Role.OWNER, "hello?", "Sam")])
    with pytest.raises(ValueError, match="no messages"):
        Chat(silent, settings=Settings(), client=Scripted())
    session, client = _scripted_chat()
    with pytest.raises(ValueError, match="Ask a question"):
        session.ask("  ")
    assert client.requests == []


# -- recording a chat -------------------------------------------------------------


def test_ask_cannot_be_recorded_with_follow_ups(tmp_path, capsys):
    args = ["record", "ask", PRESSURE, str(tmp_path / "x.json"), "--question", "a"]
    assert recording_main([*args, "--question", "b"]) == 2
    assert "record a chat for follow-ups" in capsys.readouterr().err


# -- the command line -------------------------------------------------------------


def _stdin(monkeypatch, text: str) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def test_cli_chat_shows_the_notice_in_full_once(cli, capsys, monkeypatch):
    cli("chat_pressure")
    _stdin(monkeypatch, f"{PRESSURE_QUESTION}\n\n{FOLLOW_UP}\n")
    assert main(["chat", PRESSURE]) == 0
    captured = capsys.readouterr()
    assert captured.out.count("You do not have to share your location") == 1
    assert captured.out.count("!! Something in Casey's messages is serious") == 2
    assert " ".join(captured.out.split()).count(STILL_APPLIES) == 1
    assert captured.err.startswith("confidant: 6 calls used ")
    # Piped input gets no prompt and no greeting.
    assert "> " not in captured.err and "Ctrl-D" not in captured.err


def test_cli_chat_json_is_one_answer_per_line(cli, capsys, monkeypatch):
    cli("chat_pressure")
    _stdin(monkeypatch, f"{PRESSURE_QUESTION}\n{FOLLOW_UP}\n")
    assert main(["chat", PRESSURE, "--json"]) == 0
    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line)["question"] for line in lines] == [PRESSURE_QUESTION, FOLLOW_UP]


def test_cli_chat_stops_at_a_quit_word(cli, capsys, monkeypatch):
    client = cli("ask_pressure")
    _stdin(monkeypatch, f"{PRESSURE_QUESTION}\nQuit\n{FOLLOW_UP}\n")
    assert main(["chat", PRESSURE]) == 0
    assert client.remaining == 0
    assert "Casey asked for your location" in capsys.readouterr().out


def test_cli_chat_goes_on_after_a_question_fails(cli, capsys, monkeypatch):
    refusal = SimpleNamespace(
        stop_reason="refusal",
        stop_details=SimpleNamespace(category=None, explanation="not this one"),
        parsed_output=None,
    )
    client = Scripted(refusal, _final(_answer("They do ask.")), _final(NO_FLAGS))
    monkeypatch.setattr("confidant.client.build_client", lambda *_: client)
    _stdin(monkeypatch, "first\nsecond\n")
    assert main(["chat", SAMPLE]) == 0
    captured = capsys.readouterr()
    assert "not this one" in captured.err
    assert captured.out.startswith("They do ask.")


def test_cli_chat_about_someone_saved(cli, capsys, monkeypatch):
    assert main(["add", "Casey", PRESSURE]) == 0
    capsys.readouterr()
    cli("chat_pressure")
    _stdin(monkeypatch, f"{PRESSURE_QUESTION}\n{FOLLOW_UP}\n")
    assert main(["chat", "Casey"]) == 0
    assert "your friends again?" in capsys.readouterr().out


def test_readme_chat_example_matches_what_the_cli_prints(cli, capsys, monkeypatch):
    cli("chat_pressure")
    _stdin(monkeypatch, f"{PRESSURE_QUESTION}\n{FOLLOW_UP}\n")
    assert main(["chat", PRESSURE]) == 0
    captured = capsys.readouterr()
    readme = README.read_text(encoding="utf-8")
    start = readme.index("> And what about when I went out with my friends?\n```\n\n")
    block = readme[start:].split("```\n", 3)[2]
    second = captured.out.split("\n\n!! ", 1)[1]
    assert "!! " + second.rstrip("\n") + "\n" == block
    assert captured.err.strip() in readme
