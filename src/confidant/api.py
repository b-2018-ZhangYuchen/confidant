"""An HTTP backend over the same core the command line uses.

``confidant serve`` runs it. Every route calls the functions the CLI calls — the same
parser, redaction, analyzers, grounding, safety notices, and store — so the API cannot
drift into a second, looser way of reading a conversation. What is new here is only the
shape: JSON in, JSON out, and HTTP status codes in place of exit codes.

Three decisions follow from the API holding the same private data the CLI does:

* **It only listens on this machine.** There is no sign-in, because there is no one else
  for it to serve: ``confidant serve`` refuses any host but a loopback address. Listening
  locally is not enough by itself, since any web page open in the owner's browser can
  send requests to ``localhost``. So a request must name a local ``Host`` (which defeats
  DNS rebinding, where an attacker's domain is pointed at 127.0.0.1), and a request a
  browser marks with a non-local ``Origin`` is refused before it reaches a route.
* **It never guesses the region.** ``?region=`` works as ``--region`` does, and falls
  back to ``CONFIDANT_REGION``; a safety notice is built in full either way.
* **What a model call used is reported, failure or not.** A refused or cut-off answer is
  billed, so error bodies carry the usage too.

Every report comes back twice: as data, and as the ``text`` the CLI prints, so a client
that wants the reviewed wording — safety notice first — does not have to re-implement it.
A report that carries a safety notice also gets it at the top level as ``escalation``,
with its own ``text``, so a page can put that block first without assembling it.

``/`` is the web UI, a page in ``confidant/web`` that only talks to the routes here. It
loads nothing from anywhere else, and says so to the browser in a Content-Security-Policy
that would refuse a script, style, or request to another site even if one were added.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime
from importlib.resources import files
from typing import Any
from urllib.parse import urlsplit

import anthropic
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

from confidant import __version__
from confidant.agent import ask
from confidant.analysis.comfort import comfort
from confidant.analysis.draft import DEFAULT_TONE, Tone, draft_reply
from confidant.analysis.flags import analyze_flags
from confidant.analysis.personality import analyze_personality
from confidant.client import (
    IncompleteResponse,
    ModelRefusal,
    ToolLoopError,
    describe_api_error,
)
from confidant.config import ConfigError, Settings
from confidant.ingest.exports import DateOrder, parse_export
from confidant.ingest.transcript import TranscriptError
from confidant.models import Conversation, Role
from confidant.nudge import Nudge, build_nudge
from confidant.profile import Profile, build_profile, read_conversation
from confidant.prompts.common import render_conversation
from confidant.redaction import redact
from confidant.resources import using_region
from confidant.safety import Escalation
from confidant.store import (
    DuplicatePerson,
    SaveOutcome,
    Store,
    StoredConversation,
    StoreError,
    UnknownPerson,
    latest_conversation,
    name_key,
)
from confidant.timeline import Period, build_timeline
from confidant.usage import UsageMeter

__all__ = [
    "LOCAL_HOSTS",
    "MAX_TRANSCRIPT_CHARS",
    "UI_ASSETS",
    "UI_HEADERS",
    "create_app",
    "is_local_host",
]

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

# Several years of daily texting is a few megabytes. Anything far past that is a wrong
# file, and reading it would hold the server up for nothing.
MAX_TRANSCRIPT_CHARS = 20_000_000


# The web UI's files, and the only paths under /static. A fixed list rather than a
# directory listing, so nothing else that ends up in the package is served by accident.
UI_ASSETS = {
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
}

# Same-origin only, for everything. connect-src 'self' is the line that matters most: the
# page holds a transcript, and this stops any script on it sending one anywhere but here.
# frame-ancestors stops another site framing the page to click its buttons for the owner.
UI_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; "
        "img-src 'self'; form-action 'none'; frame-ancestors 'none'; base-uri 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    # A newer install should not run against an older cached script.
    "Cache-Control": "no-cache",
}


def _ui_file(name: str, media_type: str) -> Response:
    content = files("confidant.web").joinpath(name).read_bytes()
    return Response(content, media_type=media_type, headers=UI_HEADERS)


def is_local_host(host: str | None) -> bool:
    """Whether ``host`` (a hostname, without a port) is this machine."""
    return host is not None and host.strip("[]").lower() in LOCAL_HOSTS


def _host_header(value: str) -> str:
    # "[::1]:8000" -> "::1", "localhost:8000" -> "localhost". urlsplit does the IPv6
    # brackets properly, where splitting on ":" would not.
    return urlsplit(f"//{value}").hostname or ""


# -- request bodies -----------------------------------------------------------------


class TranscriptIn(BaseModel):
    """A transcript or a chat app's export, as text, with the options the CLI takes."""

    text: str = Field(max_length=MAX_TRANSCRIPT_CHARS, description="The file's contents.")
    filename: str | None = Field(
        default=None, description="Where it came from, kept as the conversation's source."
    )
    owner: str | None = Field(default=None, description="Your name in it, like --owner.")
    match: str | None = Field(default=None, description="Their name in it, like --match.")
    dates: DateOrder | None = Field(
        default=None, description="Day- or month-first, for an ambiguous WhatsApp export."
    )
    redact: list[str] = Field(
        default_factory=list, description="Other people's names to strip, like --redact."
    )


class PersonIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class ThreadIn(BaseModel):
    """What ``draft``, ``comfort``, and ``ask`` work on: a transcript, or someone saved."""

    transcript: TranscriptIn | None = None
    person: str | None = Field(
        default=None, description="Someone saved; their most recent conversation is used."
    )
    redact: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _one_source(self) -> ThreadIn:
        if (self.transcript is None) == (self.person is None):
            raise ValueError("give exactly one of 'transcript' and 'person'")
        return self


class DraftIn(ThreadIn):
    tone: Tone = DEFAULT_TONE
    say: str | None = Field(default=None, max_length=2000)


class ComfortIn(ThreadIn):
    what: str | None = Field(default=None, max_length=4000)


class AskIn(ThreadIn):
    question: str = Field(min_length=1, max_length=4000)


# -- failures -----------------------------------------------------------------------


class ApiFailure(Exception):
    """Anything a route gives up on, with the status and body to answer with."""

    def __init__(
        self, status: int, kind: str, message: str, usage: UsageMeter | None = None
    ) -> None:
        super().__init__(message)
        self.status, self.kind, self.message, self.usage = status, kind, message, usage

    def body(self) -> dict[str, Any]:
        body: dict[str, Any] = {"error": {"kind": self.kind, "message": self.message}}
        if self.usage:
            body["usage"] = _usage(self.usage)
        return body


def _model_failure(exc: Exception, settings: Settings | None, meter: UsageMeter) -> ApiFailure:
    """The HTTP form of the CLI's exit codes 2, 3, and 4."""
    if isinstance(exc, ModelRefusal):
        return ApiFailure(422, "refused", str(exc), meter)
    if isinstance(exc, IncompleteResponse | ToolLoopError):
        return ApiFailure(502, "incomplete", str(exc), meter)
    if isinstance(exc, anthropic.APIError):
        message = describe_api_error(exc, settings or Settings())
        return ApiFailure(502, "api", message, meter)
    if isinstance(exc, ConfigError):
        # Missing credentials are the server's configuration, not the request's.
        return ApiFailure(503, "configuration", str(exc), meter)
    return ApiFailure(422, "invalid", str(exc), meter)


_MODEL_ERRORS = (
    ConfigError,
    ModelRefusal,
    IncompleteResponse,
    ToolLoopError,
    anthropic.APIError,
    ValueError,
)


# -- shaping responses --------------------------------------------------------------


def _usage(meter: UsageMeter) -> dict[str, Any]:
    costs = [call.cost for call in meter.calls]
    return {
        "calls": len(meter.calls),
        "input_tokens": sum(c.input_tokens for c in meter.calls),
        "cache_write_tokens": sum(c.cache_write_tokens for c in meter.calls),
        "cache_read_tokens": sum(c.cache_read_tokens for c in meter.calls),
        "output_tokens": sum(c.output_tokens for c in meter.calls),
        # None rather than a partial sum: a total that silently skips a model is a guess.
        "cost_usd": sum(costs) if costs and all(c is not None for c in costs) else None,
        "summary": meter.describe() if meter else None,
    }


def _escalation(escalation: Escalation | None) -> dict[str, Any] | None:
    if escalation is None:
        return None
    routing = escalation.routing
    lines = routing.heading_and_numbers() if routing is not None else []
    if routing is not None and routing.note:
        lines.append(routing.note)
    return {
        **escalation.model_dump(mode="json"),
        "text": escalation.to_text(),
        # The routing as the notice words it, one line each and unwrapped: ``text`` is
        # wrapped for a terminal, and a page re-wrapping it would break it mid-sentence.
        "routing_lines": [line.strip() for line in lines],
    }


def _stats(conversation: Conversation) -> dict[str, Any]:
    span = conversation.timespan
    return {
        "owner": conversation.owner_name,
        "match": conversation.match_name,
        "messages": len(conversation),
        "owner_messages": len(conversation.owner_messages),
        "match_messages": len(conversation.match_messages),
        "owner_mean_words": round(conversation.mean_words(Role.OWNER), 2),
        "match_mean_words": round(conversation.mean_words(Role.MATCH), 2),
        "effort_ratio": conversation.effort_ratio,
        "span_days": span.days if span is not None else None,
        "last_message": conversation.last_activity,
        "text": conversation.describe(),
    }


def _stored(stored: StoredConversation) -> dict[str, Any]:
    return {
        "id": stored.id,
        "source": stored.source,
        "saved_at": stored.imported_at,
        "updated_at": stored.updated_at,
        "messages": stored.messages,
        "first_message": stored.first_message,
        "last_message": stored.last_message,
    }


def _profile(profile: Profile) -> dict[str, Any]:
    def current(read) -> dict[str, Any] | None:
        if read is None:
            return None
        return {
            "report": read.report.model_dump(mode="json"),
            "read_at": read.reading.read_at,
            "messages_read": read.reading.messages_read,
            "stale": read.stale,
        }

    return {
        "person": profile.person.name,
        "conversations": [
            {
                **_stored(entry.stored),
                "personality": current(entry.personality),
                "flags": current(entry.flags),
            }
            for entry in profile.entries
        ],
        "flags": [
            {"conversation": t.conversation_id, **t.flag.model_dump(mode="json")}
            for t in profile.flags
        ],
        "pending": len(profile.pending),
        "escalation": _escalation(profile.escalation),
        "text": profile.to_text(),
    }


def _nudge(nudge: Nudge) -> dict[str, Any]:
    return {
        "contacts": [
            {
                "name": c.name,
                "state": c.state.value,
                "suggested": c.suggested,
                "headline": c.headline(),
                "reasons": list(c.reasons),
                "last_message": c.last_message.timestamp if c.last_message else None,
                "escalation": _escalation(c.escalation),
            }
            # The priority is left out on purpose, as it is from the CLI: it orders the
            # list and is not a score of anyone. The order is kept.
            for c in nudge.contacts
        ],
        "text": nudge.to_text(),
    }


# -- the app ------------------------------------------------------------------------


def create_app(
    *,
    store: Callable[[], Store] = Store,
    settings: Callable[[], Settings] = Settings.from_env,
) -> FastAPI:
    """Build the app. ``store`` and ``settings`` are factories, so tests can swap them.

    A store is opened per request and closed with it: SQLite connections belong to the
    thread that made them, and FastAPI runs each request on a worker thread of its own.
    """
    app = FastAPI(
        title="Confidant",
        version=__version__,
        description="A second opinion on the people you are dating. Local only.",
        # FastAPI's /docs and /redoc pages load their scripts from a CDN, which would be a
        # third party in the loop. The schema itself is served from here, at /openapi.json.
        docs_url=None,
        redoc_url=None,
    )

    @app.middleware("http")
    async def local_only(request: Request, call_next):
        host = _host_header(request.headers.get("host", ""))
        if not is_local_host(host):
            return JSONResponse(
                {"error": {"kind": "forbidden", "message": "Confidant only answers on localhost."}},
                status_code=403,
            )
        origin = request.headers.get("origin")
        # A browser sends Origin on cross-site requests, and "null" from sandboxed pages
        # and local files. Either way, not a page this server or the owner's tools made.
        if origin is not None and not is_local_host(urlsplit(origin).hostname):
            return JSONResponse(
                {
                    "error": {
                        "kind": "forbidden",
                        "message": "Requests from other websites are refused.",
                    }
                },
                status_code=403,
            )
        return await call_next(request)

    @app.exception_handler(ApiFailure)
    async def failure(request: Request, exc: ApiFailure) -> JSONResponse:
        return JSONResponse(exc.body(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        # One error shape for the whole API, with FastAPI's detail kept for a client to
        # point at the field.
        problems = "; ".join(_problem(e) for e in exc.errors())
        return JSONResponse(
            {"error": {"kind": "invalid", "message": problems}, "detail": _plain(exc.errors())},
            status_code=422,
        )

    @contextmanager
    def opened() -> Iterator[Store]:
        try:
            with store() as s:
                yield s
        except UnknownPerson as exc:
            raise ApiFailure(404, "unknown-person", str(exc)) from exc
        except DuplicatePerson as exc:
            raise ApiFailure(409, "duplicate-person", str(exc)) from exc
        except (StoreError, TranscriptError) as exc:
            raise ApiFailure(422, "invalid", str(exc)) from exc

    def read(body: TranscriptIn) -> tuple[Conversation, str | None]:
        try:
            imported = parse_export(
                body.text,
                owner=body.owner,
                match=body.match,
                source=body.filename,
                dates=body.dates,
            )
        except TranscriptError as exc:
            raise ApiFailure(422, "invalid-transcript", str(exc)) from exc
        conversation = imported.conversation
        conversation.private_names.extend(body.redact)
        return conversation, imported.note()

    def thread(body: ThreadIn) -> tuple[Conversation, Escalation | None]:
        if body.transcript is not None:
            conversation, _ = read(body.transcript)
            escalation = None
        else:
            with opened() as s:
                person = s.person(body.person or "")
                conversations = s.conversations(person)
                if not conversations:
                    raise ApiFailure(
                        422, "nothing-saved", f"Nothing is saved for {person.name} yet."
                    )
                conversation = s.load_conversation(latest_conversation(conversations).id)
                # Every saved check counts, as in the CLI: a threat made in an older chat
                # is still the first thing to know before writing back.
                escalation = build_profile(s, person).escalation
        conversation.private_names.extend(body.redact)
        return conversation, escalation

    def call_model(run: Callable[[Settings, UsageMeter], BaseModel]) -> dict[str, Any]:
        meter = UsageMeter()
        loaded: Settings | None = None
        try:
            loaded = settings()
            report = run(loaded, meter)
        except _MODEL_ERRORS as exc:
            raise _model_failure(exc, loaded, meter) from exc
        body: dict[str, Any] = {
            "report": report.model_dump(mode="json"),
            "text": report.to_text(),
            "escalation": _escalation(getattr(report, "escalation", None)),
            "usage": _usage(meter),
        }
        if hasattr(report, "footnotes"):
            body["footnotes"] = report.footnotes()
        return body

    # -- the web UI -------------------------------------------------------------------

    @app.get("/", include_in_schema=False)
    def index() -> Response:
        return _ui_file("index.html", "text/html; charset=utf-8")

    @app.get("/static/{name}", include_in_schema=False)
    def static(name: str) -> Response:
        if name not in UI_ASSETS:
            raise ApiFailure(404, "not-found", f"No file called {name!r} here.")
        return _ui_file(name, UI_ASSETS[name])

    # -- local: nothing here touches the network --------------------------------------

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "version": __version__}

    @app.post("/api/stats")
    def stats(body: TranscriptIn) -> dict[str, Any]:
        conversation, note = read(body)
        return {**_stats(conversation), "note": note}

    @app.post("/api/redact")
    def redact_preview(body: TranscriptIn) -> dict[str, Any]:
        conversation, note = read(body)
        redaction = redact(conversation)
        return {
            "transcript": "\n".join(render_conversation(redaction.conversation)),
            "replaced": redaction.summary(),
            "key": dict(redaction.originals),
            "note": note,
        }

    @app.get("/api/people")
    def people() -> list[dict[str, Any]]:
        with opened() as s:
            return [
                {
                    "name": p.person.name,
                    "added_at": p.person.added_at,
                    "conversations": p.conversations,
                    "messages": p.messages,
                    "last_message": p.last_message,
                }
                for p in s.people()
            ]

    @app.post("/api/people", status_code=201)
    def add_person(body: PersonIn) -> dict[str, Any]:
        with opened() as s:
            person = s.add_person(body.name)
            return {"name": person.name, "added_at": person.added_at}

    @app.get("/api/people/{name}")
    def show_person(name: str) -> dict[str, Any]:
        with opened() as s:
            person = s.person(name)
            return {
                "name": person.name,
                "added_at": person.added_at,
                "conversations": [
                    {**_stored(c), "stats": _stats(s.load_conversation(c.id))}
                    for c in s.conversations(person)
                ],
            }

    @app.delete("/api/people/{name}")
    def remove_person(name: str) -> dict[str, Any]:
        with opened() as s:
            person = s.person(name)
            count = len(s.conversations(person))
            s.remove_person(person.name)
            return {"removed": person.name, "conversations": count}

    @app.post("/api/people/{name}/conversations")
    def save_conversation(name: str, body: TranscriptIn) -> JSONResponse:
        conversation, note = read(body)
        # The same guard as `confidant add`: a chat with Casey filed under Robin would
        # quietly mix two histories, so a different name has to be said out loud.
        if body.match is None and name_key(conversation.match_name) != name_key(name):
            raise ApiFailure(
                422,
                "wrong-person",
                f"This is a conversation with {conversation.match_name!r}, not {name!r}. If "
                f"that is how {name} appears in it, send 'match': "
                f"{conversation.match_name!r}.",
            )
        with opened() as s:
            result = s.save_conversation(s.person(name), conversation)
        status = 201 if result.outcome is SaveOutcome.NEW else 200
        return JSONResponse(
            {
                "outcome": result.outcome.value,
                "added": result.added,
                "conversation": _plain(_stored(result.conversation)),
                "note": note,
            },
            status_code=status,
        )

    @app.get("/api/people/{name}/profile")
    def profile(name: str, region: str | None = None) -> dict[str, Any]:
        with using_region(region), opened() as s:
            return _profile(build_profile(s, name))

    @app.get("/api/people/{name}/timeline")
    def timeline(name: str, by: Period = Period.WEEK, region: str | None = None):
        with using_region(region), opened() as s:
            result = build_timeline(s, name, by)
            return {
                "person": result.person.name,
                "period": result.period.value,
                "buckets": [
                    {
                        "start": bucket.start,
                        **{
                            role.value: {
                                "messages": side.messages,
                                "mean_words": side.mean_words,
                                "starts": side.starts,
                                "median_reply_seconds": (
                                    side.median_reply.total_seconds()
                                    if side.median_reply is not None
                                    else None
                                ),
                            }
                            for role, side in (
                                (Role.OWNER, bucket.owner),
                                (Role.MATCH, bucket.match),
                            )
                        },
                    }
                    for bucket in result.buckets
                ],
                "undated": result.undated,
                "escalation": _escalation(result.escalation),
                "text": result.to_text(),
            }

    @app.get("/api/nudge")
    def nudge(at: datetime | None = None, region: str | None = None) -> dict[str, Any]:
        # `at` answers as things stood then, as `confidant nudge --at` does.
        with using_region(region), opened() as s:
            return _nudge(build_nudge(s, at=at))

    @app.get("/api/people/{name}/nudge")
    def nudge_person(
        name: str, at: datetime | None = None, region: str | None = None
    ) -> dict[str, Any]:
        with using_region(region), opened() as s:
            return _nudge(build_nudge(s, name, at=at))

    # -- these send the redacted transcript to the Anthropic API ----------------------

    @app.post("/api/analyze")
    def analyze(body: TranscriptIn) -> dict[str, Any]:
        conversation, _ = read(body)
        return call_model(
            lambda st, meter: analyze_personality(conversation, settings=st, on_usage=meter)
        )

    @app.post("/api/flags")
    def flags(body: TranscriptIn, region: str | None = None) -> dict[str, Any]:
        conversation, _ = read(body)
        with using_region(region):
            return call_model(
                lambda st, meter: analyze_flags(conversation, settings=st, on_usage=meter)
            )

    @app.post("/api/draft")
    def draft(body: DraftIn, region: str | None = None) -> dict[str, Any]:
        with using_region(region):
            conversation, escalation = thread(body)

            def run(st: Settings, meter: UsageMeter) -> BaseModel:
                report = draft_reply(
                    conversation, tone=body.tone, say=body.say, settings=st, on_usage=meter
                )
                return report.model_copy(update={"escalation": escalation})

            return call_model(run)

    @app.post("/api/comfort")
    def comfort_route(body: ComfortIn, region: str | None = None) -> dict[str, Any]:
        with using_region(region):
            conversation, escalation = thread(body)

            def run(st: Settings, meter: UsageMeter) -> BaseModel:
                report = comfort(conversation, what=body.what, settings=st, on_usage=meter)
                return report.model_copy(update={"escalation": escalation})

            return call_model(run)

    @app.post("/api/ask")
    def ask_route(body: AskIn, region: str | None = None) -> dict[str, Any]:
        with using_region(region):
            conversation, escalation = thread(body)
            return call_model(
                lambda st, meter: ask(
                    conversation,
                    body.question,
                    settings=st,
                    escalation=escalation,
                    on_usage=meter,
                )
            )

    @app.post("/api/people/{name}/profile/update")
    def update_profile(name: str, region: str | None = None) -> dict[str, Any]:
        meter = UsageMeter()
        loaded: Settings | None = None
        done = 0
        with using_region(region), opened() as s:
            current = build_profile(s, name)
            pending = current.pending
            if pending:
                try:
                    loaded = settings()
                    for stored, kind in pending:
                        read_conversation(s, stored, kind, settings=loaded, on_usage=meter)
                        done += 1
                except _MODEL_ERRORS as exc:
                    failure = _model_failure(exc, loaded, meter)
                    if done:
                        # Each read is saved as it finishes; say so, or the owner will
                        # assume they have to pay for them again.
                        failure.message += f" Kept the {done} read(s) that finished before this."
                    raise failure from exc
                current = build_profile(s, current.person)
            return {"updated": done, "profile": _profile(current), "usage": _usage(meter)}

    return app


def _plain(value: Any) -> Any:
    """``value`` with datetimes and the like made JSON-ready, for hand-built responses."""
    return jsonable_encoder(value)


def _problem(error: dict[str, Any]) -> str:
    where = ".".join(str(part) for part in error.get("loc", ()) if part != "body")
    return f"{where}: {error.get('msg')}" if where else str(error.get("msg"))
