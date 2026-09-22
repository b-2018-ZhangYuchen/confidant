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
| 🔜 | Red-flag detection with severity tiers |
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

## Your data

Your chat history is about as private as data gets, and this repo is built around that:

- `.gitignore` blocks `data/`, `transcripts/`, `conversations/`, `*.db`, and `.env`
  before you can make a mistake with them. The only conversation in this repository is
  `examples/sample_chat.txt`, which is fictional.
- `confidant stats` never makes a network call.
- `confidant analyze` sends the transcript to the Anthropic API and nothing else — no
  telemetry, no analytics, no third parties. A local redaction layer is on the roadmap.

## Development

```bash
pytest          # tests, none of which call the API
ruff check .
```

Tests are offline by design. Anything that needs the model is exercised against recorded
fixtures rather than live calls, so the suite is fast and free to run.

## License

MIT — see [`LICENSE`](LICENSE).
