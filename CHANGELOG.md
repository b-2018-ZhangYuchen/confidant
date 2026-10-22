# Changelog

Each release lists what someone using Confidant would notice. The day-by-day record is
in [`ROADMAP.md`](ROADMAP.md) and the git history.

## [0.2.0] — 2026-10-22

The first release meant to be installed rather than cloned.

### Install

- Distributed as **`confidant-agent`**, since `confidant` on PyPI is an unrelated project.
  The command and the import are still `confidant`. Install with
  `pipx install confidant-agent`, or `pipx install "confidant-agent[api]"` for
  `confidant serve`.
- Settings are read from the shell, then a `.env` in (or above) the working directory,
  then `~/.confidant/.env`. That last file is new, and it is what a pipx install
  uses: before this release, a `.env` was only found next to the source checkout.

### More than one conversation

- A private local store of the people you are seeing: `add`, `list`, `show`, `remove`.
- Incremental saves, so a newer export of a saved chat adds only the new messages.
- A per-person `profile` built up across conversations, with each read kept so it is
  paid for once, and a week-by-week `timeline`.
- Prompt caching on Confidant's own instructions, and a token and cost line after every
  command that calls Claude.

### Knowing when to reach out

- `confidant nudge`: who is worth a message today, and why. It never pushes you towards
  someone you are waiting on, or towards anyone a danger flag was found with.
- `confidant draft`: replies in a tone you pick, each tied to the line it answers, and
  never sent for you.
- `confidant comfort`, for when it has gone badly, with a fixed support notice first if
  anything you wrote suggests you may be at risk.
- `--region` / `CONFIDANT_REGION`: emergency and support numbers under safety notices.
- An eval set of fictional transcripts, and `evals compare` for measuring prompt changes.

### An agent

- `confidant ask` and `confidant chat`: Claude chooses which tools to use (search, read,
  the personality read, the red-flag check), every quote is checked, and the red-flag
  check runs whatever was asked.
- WhatsApp, iMessage, and Hinge exports are read as they are.
- `confidant serve`: the same commands as a JSON API and a web UI, on this machine only.

## [0.1.0] — 2026-09-28

A single conversation, read carefully: transcript parsing, local `stats`, redaction
before anything is sent, a personality read with quoted evidence and explicit confidence,
red-flag detection with severity tiers, a fixed safety notice whenever a danger-tier flag
is found, and streamed progress while Claude works.
