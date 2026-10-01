# Confidant

**A second opinion on the people you are dating.**

Dating apps are very good at producing matches and very bad at helping you think about
them. You end up with six conversations, a vague sense that one of them is going
somewhere, and no idea whether the one who takes three days to reply is busy or bored.

Confidant reads a chat transcript and gives you a careful, evidence-grounded read: how
this person communicates, what stands out, what is worth watching, and what to ask next.
It quotes the transcript for everything it claims and tells you when it does not know.

> It is a thinking aid, not a verdict machine. It does not diagnose people, it does not
> score them, and it will tell you when a transcript is too short to support a
> conclusion. See [`docs/principles.md`](docs/principles.md).

---

## Status

Early and honest about it. Working today:

| | |
|---|---|
| ✅ | Transcript parsing (plain text, timestamps optional, multi-line messages) |
| ✅ | Local conversation statistics — no API call, no data leaves your machine |
| ✅ | Redaction of names, phone numbers, and addresses before anything is sent |
| ✅ | Personality read backed by Claude, with quoted evidence and explicit confidence |
| ✅ | Red-flag detection with severity tiers, every quote checked against the transcript |
| ✅ | A fixed safety notice, printed first, whenever a danger-tier flag is found |
| ✅ | A live progress line while Claude works, so a long read is not a blank terminal |
| ✅ | Remembering who you are seeing — `add`, `list`, `show`, and `remove`, in a private local file |
| ✅ | Incremental saves — a newer export of a saved chat adds only the new messages |
| 🔜 | A per-person profile that builds up across conversations |
| 🔜 | Keep-in-touch suggestions |
| 🔜 | Comfort mode for when it goes badly |

The plan for getting from here to there is in [`ROADMAP.md`](ROADMAP.md).

## Install

```bash
git clone git@github.com:b-2018-ZhangYuchen/confidant.git
cd confidant
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Then set your API key:

```bash
cp .env.example .env    # add your ANTHROPIC_API_KEY
```

## Use

Write a transcript. The format is deliberately something you can paste together by hand:

```
# owner: Sam
# match: Robin

[2026-03-02 19:04] Robin: hey! your profile said you bike to work — which route?
[2026-03-02 19:31] Sam: the river path, mostly. takes longer but it's worth it
[2026-03-02 19:33] Robin: the one past the old mill? I did that every morning last summer
    there's a coffee cart near the second bridge that is criminally underrated
```

Timestamps are optional, indented lines continue the message above them, and `#` lines
are comments. Then:

```bash
# Local only — never touches the network. Good for checking your transcript parsed right.
confidant stats examples/sample_chat.txt

# Also local: exactly what would be sent to Claude, after redaction.
confidant redact examples/details_chat.txt

# Ask Claude for a read.
confidant analyze examples/sample_chat.txt
confidant analyze examples/sample_chat.txt --json

# A separate check for red flags, each with a severity tier.
confidant flags examples/pressure_chat.txt
confidant flags examples/pressure_chat.txt --json
```

`stats` gives you the cheap signals:

```
Sam <-> Robin
  messages         17 (8 you / 9 them)
  avg words (you)  8.9
  avg words (them) 14.3
  effort ratio     1.62 (their words per word of yours)
  spans            3 days
  last message     2026-03-05 21:36
```

While `analyze` or `flags` is waiting on Claude, a single line on stderr says what stage
it has reached, and rewrites itself as it goes:

```
confidant: writing the report: green flags... 23s
```

It names the stage and the part of the report being written, never the content. The
report is printed only once it is whole, because a flag is not shown to you until its
quotes have been checked against the transcript, and a half-streamed report has not been
checked yet. The line appears only when stderr is a terminal, so piping `--json` into
another program gets clean output; `--no-progress` turns it off anyway.

`analyze` gives you the considered read — traits with quoted evidence, green flags,
things worth watching, questions worth asking, and a stated confidence level.

`flags` is a separate pass that looks only for behavior that could matter for your
wellbeing or safety, and gives each finding one of three tiers:

| Tier | Meaning |
|---|---|
| `watch` | Worth noticing, quite possibly innocent. Shown with the most plausible mundane reading. |
| `concern` | A pattern that would matter in any relationship — pushing past a "no", belittling, guilt-tripping, asking for money. |
| `danger` | Threats, coercion, pressure around sex or money, isolation from friends, tracking where you are. Named plainly, with no charitable gloss. |

It runs as its own request so the warmth of the personality read cannot dilute it, and
two things are enforced in code rather than left to the prompt: every quote is checked
against what the other person actually wrote (a flag whose quotes cannot be found is
dropped, and the report says so), and threats, coercion, sexual pressure, isolation, and
monitoring are always tier 3. Most transcripts have no flags, and the report says that
plainly too. `examples/pressure_chat.txt` is a fictional conversation written to have
something to find.

When a `danger` flag survives that check, the report opens with a safety notice, above
the model's own summary. The model decides whether something is dangerous; what you are
told to do about it is fixed text, written in advance in
[`src/confidant/safety.py`](src/confidant/safety.py), so it is the same every time and a
mild-sounding summary cannot talk you out of it. It is shown whatever the confidence
level. If the check finds tracking and isolation in the pressure example, the report
begins:

```
!! Something in Casey's messages is serious: tracking where you are and pressure to pull
   away from friends or family.

   - You do not have to share your location or account for where you are. If you have
     already shared it in an app, you can turn that off without explaining.
   - Keep your plans with friends and family. They are the people to talk this through
     with.
   - You do not owe them a reply, and you do not have to decide anything right away.
     Talk it through with someone you trust first.
   - If you ever feel physically unsafe, call your local emergency number.
```

With `--json` the same notice is in the `escalation` field. If a flag filed as danger is
dropped because its quotes could not be found, the report says so rather than going
quiet about it. Region-specific crisis resources are on the roadmap; until then the
notice points only at what is right everywhere.

### Keeping track of people

Confidant can remember who you are seeing and the conversations you have had with them.
None of this touches the network:

```bash
confidant add Robin examples/sample_chat.txt    # save Robin, with a conversation
confidant add Casey examples/pressure_chat.txt
confidant add Robin later_chat.txt              # add another to someone already saved
confidant list
confidant show Robin                            # the stats for each saved conversation
confidant remove Casey                          # asks first; --yes to skip the question
```

`list` shows everyone at a glance:

```
NAME   CONVERSATIONS  MESSAGES  LAST MESSAGE
Casey  1              10        2026-04-12 09:20
Robin  1              17        2026-03-05 21:36
```

`add` without a transcript saves just the name. Chats keep going, so saving is
incremental: export the conversation again next week and `add` it, and only the messages
after the ones already saved are added, to the same conversation rather than a second
copy of it:

```
Added 7 new messages from robin_week2.txt to #1 (17 in all).
```

The new export can start anywhere, as long as its first messages are the last ones
already saved; `confidant show` then says when that conversation was last added to.
Saving a chat whose messages are all saved already — the same file twice, or an older,
shorter export — adds nothing, so re-running a command is harmless. Messages are matched
on who sent them, what they said, and when, exactly, so an edited message starts a new
conversation rather than being merged into the old one. Without timestamps, at least
three messages have to line up before Confidant treats them as the same chat, because
plenty of different chats open with the same "hey" and "hi". A transcript whose other speaker is someone
else is refused rather than filed under the wrong person — a chat with Casey saved under
Robin would quietly mix two histories. If an export spells their name differently, say
so with `--match`:

```bash
confidant add Robin robin_export.txt --match "Robin 🌻"
```

### What gets sent

Before `analyze` or `flags` builds a request, the transcript is redacted on your machine.
Both names become `[OWNER]` and `[MATCH]`; phone numbers, long account-like numbers,
email addresses, street addresses, links, and social handles become numbered
placeholders. The same detail always gets the same placeholder, so the model can still
see that someone sent their number twice, and it answers in placeholders that are
swapped back before you read the report. Money amounts, times, and dates are kept,
because a request for $1500 is exactly what a red-flag check needs to see.

Anyone else named in the conversation is only redacted if you say who they are, with a
`# redact:` line in the transcript or `--redact NAME` on the command line. `confidant
redact` shows you the result without calling anything, so you do not have to take any of
this on trust:

```bash
confidant redact examples/details_chat.txt
```

prints

```
Owner (the person I am helping): [OWNER]
Match (the person to analyze): [MATCH]
Messages: 6 (2 owner / 4 match)
Spanning: 0 days
Average message length: match writes 0.80 words per owner word (context only — do not over-read it)

--- TRANSCRIPT ---
[2026-05-08 18:02] MATCH: dinner saturday still on? I booked the place at [ADDRESS_1]
[2026-05-08 18:10] OWNER: yes! [NAME_1] is dropping me off, she wants to vet you from the car
[2026-05-08 18:11] MATCH: fair. text me when you're close, my number is [PHONE_1]
[2026-05-08 18:12] MATCH: or instagram, [HANDLE_1]. I'm slow on email ([EMAIL_1])
[2026-05-08 18:20] OWNER: ha, got it. I'll bring the $40 I owe you from the concert
[2026-05-08 18:21] MATCH: keep it, you got the drinks. tell [NAME_1] I said hi. see you at 7:30
--- END TRANSCRIPT ---

Replaced: 3 names, 1 phone number, 1 email address, 1 street address, 1 handle.
The key, which stays on your machine:
  [OWNER]      Priya
  [MATCH]      Theo
  [ADDRESS_1]  214 Linden Street
  [NAME_1]     Maya
  [PHONE_1]    (555) 010-4477
  [HANDLE_1]   @theo.cooks
  [EMAIL_1]    theo.m@example.com
```

This is pattern matching, not understanding. It catches the shapes that identify people
in a chat; it does not catch "the bakery on the corner by my work". Read the preview
before sending anything you would mind being seen.

## Your data

Your chat history is about as private as data gets, and this repo is built around that:

- `.gitignore` blocks `data/`, `transcripts/`, `conversations/`, `*.db`, and `.env`
  before you can make a mistake with them. The only conversations in this repository are
  `examples/sample_chat.txt`, `examples/pressure_chat.txt`, and
  `examples/details_chat.txt`, all fictional.
- `confidant stats` and `confidant redact` never make a network call.
- `confidant add`, `list`, `show`, and `remove` never make a network call either. They
  keep people and conversations in one SQLite file at `~/.confidant/confidant.db`, or
  wherever `CONFIDANT_DB` points. It is in your home directory rather than wherever you
  ran the command, so it cannot end up inside a git checkout; it is readable only by you
  (`0600`, in a `0700` folder); and `confidant remove` overwrites a person's messages
  instead of leaving them recoverable in the file. It holds the conversations as you
  wrote them — redaction happens on the way to the API, not on the way to disk.
- `confidant analyze` and `confidant flags` send the redacted transcript to the Anthropic
  API and nothing else — no telemetry, no analytics, no third parties. Names and contact
  details are replaced before the request is built (see [What gets sent](#what-gets-sent)).

## Settings and exit codes

Everything is set through the environment, or a `.env` file in the directory you run
from:

| Variable | Default | |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Needed for `analyze` and `flags`; `stats` and `redact` never use it. |
| `CONFIDANT_MODEL` | `claude-opus-5` | |
| `CONFIDANT_EFFORT` | `high` | How hard Claude thinks: `low`, `medium`, `high`, `xhigh`, or `max`. |
| `CONFIDANT_MAX_TOKENS` | `16000` | Raise it if a long transcript's report comes back cut off. |
| `CONFIDANT_DB` | `~/.confidant/confidant.db` | Where `add`, `list`, `show`, and `remove` keep people and conversations. |

`confidant --help` lists the same, with examples, and every subcommand's `--help` shows
the transcript format. Every failure is one sentence on stderr, never a traceback, and
the exit code says what kind it was:

```
exit codes:
  0    success
  2    the transcript or configuration needs fixing
  3    Claude declined the request
  4    the API call failed, or its answer was incomplete
  130  cancelled with Ctrl-C
```

## Development

```bash
pytest          # tests, none of which call the API
ruff check .
```

Tests are offline by design, and the suite enforces it: any attempt to open a network
connection fails the test. Anything that needs the model runs against a recorded response
in `tests/fixtures/recorded/`, replayed where the SDK would be, so the refusal check,
schema validation, grounding, and escalation all run for real.

A recording stores the response and a hash of the request that produced it. Change a
prompt and the recordings for it fail as stale, with the command that fixes them:

```bash
# Make a live call and save the response (needs an API key; examples/ only).
python -m confidant.recording record flags examples/pressure_chat.txt \
    tests/fixtures/recorded/flags_pressure.json

# Re-fingerprint a hand-written recording, after reading it against the new prompt.
python -m confidant.recording stamp tests/fixtures/recorded/flags_pressure.json
```

The recordings in the repository today are hand-written — marked `"provenance":
"hand-written"` in the file — because they were made without an API key. They pin down
the shapes that matter, including a refusal and a misquote that grounding has to catch.
A live recording cannot be re-stamped: its response answers the old prompt, so the only
honest fix is to record it again.

## License

MIT — see [`LICENSE`](LICENSE).
