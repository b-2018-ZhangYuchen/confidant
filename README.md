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
| ✅ | Personality read backed by Claude, with quoted evidence and explicit confidence |
| ✅ | Red-flag detection with severity tiers, every quote checked against the transcript |
| ✅ | A fixed safety notice, printed first, whenever a danger-tier flag is found |
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

## Your data

Your chat history is about as private as data gets, and this repo is built around that:

- `.gitignore` blocks `data/`, `transcripts/`, `conversations/`, `*.db`, and `.env`
  before you can make a mistake with them. The only conversations in this repository are
  `examples/sample_chat.txt` and `examples/pressure_chat.txt`, both fictional.
- `confidant stats` never makes a network call.
- `confidant analyze` and `confidant flags` send the transcript to the Anthropic API and
  nothing else — no telemetry, no analytics, no third parties. A local redaction layer is
  on the roadmap.

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
