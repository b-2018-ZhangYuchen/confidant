# Roadmap

One meaningful change a day. Not a wishlist — each line is a day's work that ends
with something that runs and something that tests it.

Dates are targets, not promises. Items get reordered when a day's work turns out to
need something else first, and that reordering is part of the record.

## Week 1 — A conversation, read carefully

- [x] **Tue Sep 22** — Core models, transcript parser, personality read, CLI, CI
- [ ] **Wed Sep 23** — Red-flag detection with severity tiers, separate from the personality read
- [ ] **Thu Sep 24** — Safety escalation path: what happens when a tier-3 flag fires
- [ ] **Fri Sep 25** — Recorded-fixture test harness so analysis logic is testable offline
- [ ] **Sat Sep 26** — Redaction layer — strip names, numbers, and addresses before the API call
- [ ] **Sun Sep 27** — Streaming output, so a long analysis is not a blank terminal
- [ ] **Mon Sep 28** — Week 1 cleanup: error messages, `--help` text, README accuracy pass

## Week 2 — More than one conversation

- [ ] **Tue Sep 29** — SQLite store for people and conversations
- [ ] **Wed Sep 30** — `confidant add` / `list` / `show` — manage who you are seeing
- [ ] **Thu Oct 01** — Incremental ingest: append new messages without re-reading everything
- [ ] **Fri Oct 02** — Per-person profile that accumulates across conversations
- [ ] **Sat Oct 03** — Timeline view — how the dynamic changed over weeks
- [ ] **Sun Oct 04** — Prompt caching on the stable parts of the request
- [ ] **Mon Oct 05** — Token and cost accounting per analysis

## Week 3 — Knowing when to reach out

- [ ] **Tue Oct 06** — Keep-in-touch model: decay over last contact, weighted by reciprocity
- [ ] **Wed Oct 07** — `confidant nudge` — who is worth a message today, and why
- [ ] **Thu Oct 08** — Draft-a-reply with tone control, grounded in the thread so far
- [ ] **Fri Oct 09** — Comfort mode: what Confidant says when it has gone badly
- [ ] **Sat Oct 10** — Crisis-resource routing, with its own test suite
- [ ] **Sun Oct 11** — Eval set of synthetic transcripts with expected findings
- [ ] **Mon Oct 12** — Hill-climb the personality prompt against that eval

## Week 4 — An agent, not a script

- [ ] **Tue Oct 13** — Turn the analyzers into tools and put Claude in the loop
- [ ] **Wed Oct 14** — Conversational interface: ask follow-up questions about a person
- [ ] **Thu Oct 15** — Import adapters — WhatsApp, iMessage, and Hinge exports
- [ ] **Fri Oct 16** — FastAPI backend over the same core
- [ ] **Sat Oct 17** — Web UI: transcript upload and report view
- [ ] **Sun Oct 18** — Web UI: the people list and the nudge digest
- [ ] **Mon Oct 19** — Packaging, `pipx install confidant`, and a v0.2 release

## Later

Ideas that are real but not scheduled:

- Local-model mode, so the transcript never leaves the machine at all
- Voice-note transcription for the conversations that are not text
- A "why did you say that" command that shows the evidence chain behind any claim
- Multi-language transcripts, with the cultural-context caveats that requires
- Calibration study: how often is a low-confidence read actually wrong?
