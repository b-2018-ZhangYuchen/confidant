# Roadmap

One meaningful change a day. Not a wishlist — each line is a day's work that ends
with something that runs and something that tests it.

Dates are targets, not promises. Items get reordered when a day's work turns out to
need something else first, and that reordering is part of the record.

## Week 1 — A conversation, read carefully

- [x] **Tue Sep 22** — Core models, transcript parser, personality read, CLI, CI
- [x] **Wed Sep 23** — Red-flag detection with severity tiers, separate from the personality read
- [x] **Thu Sep 24** — Safety escalation path: what happens when a tier-3 flag fires
- [x] **Fri Sep 25** — Recorded-fixture test harness so analysis logic is testable offline
- [x] **Sat Sep 26** — Redaction layer — strip names, numbers, and addresses before the API call
- [x] **Sun Sep 27** — Streaming output, so a long analysis is not a blank terminal
- [x] **Mon Sep 28** — Week 1 cleanup: error messages, `--help` text, README accuracy pass

## Week 2 — More than one conversation

- [x] **Tue Sep 29** — SQLite store for people and conversations
- [x] **Wed Sep 30** — `confidant add` / `list` / `show` — manage who you are seeing
- [x] **Thu Oct 01** — Incremental ingest: append new messages without re-reading everything
- [x] **Fri Oct 02** — Per-person profile that accumulates across conversations
- [x] **Sat Oct 03** — Timeline view — how the dynamic changed over weeks
- [x] **Sun Oct 04** — Prompt caching on the stable parts of the request
- [x] **Mon Oct 05** — Token and cost accounting per analysis

## Week 3 — Knowing when to reach out

- [x] **Tue Oct 06** — Keep-in-touch model: decay over last contact, weighted by reciprocity
- [x] **Wed Oct 07** — `confidant nudge` — who is worth a message today, and why
- [x] **Thu Oct 08** — Draft-a-reply with tone control, grounded in the thread so far
- [x] **Fri Oct 09** — Comfort mode: what Confidant says when it has gone badly
- [x] **Sat Oct 10** — Crisis-resource routing, with its own test suite
- [x] **Sun Oct 11** — Eval set of synthetic transcripts with expected findings
- [x] **Mon Oct 12** — `evals compare`: which checks a prompt change won and lost, so a hill-climb has a step to repeat

## Week 4 — An agent, not a script

- [x] **Tue Oct 13** — Turn the analyzers into tools and put Claude in the loop
- [x] **Wed Oct 14** — Conversational interface: ask follow-up questions about a person
- [x] **Thu Oct 15** — Import adapters — WhatsApp, iMessage, and Hinge exports
- [x] **Fri Oct 16** — FastAPI backend over the same core
- [x] **Sat Oct 17** — Web UI: transcript upload and report view
- [x] **Sun Oct 18** — Web UI: the people list and the nudge digest
- [x] **Mon Oct 19** — Packaging, `pipx install confidant-agent`, and a v0.2 release

## Weeks 5–7 — A model of our own

The course asks for more than an API integration: pick a pretrained model, adapt it, and
show with numbers what each change did. The task is the red-flag check, because it has
labels that can be scored (category, severity, quote) and because running it on a local
model means the transcript never leaves the machine at all.

The plan follows the course's eight steps: model selection, dataset preparation, baseline,
customization (continued pre-training, LoRA fine-tuning, RAG, prompts), parameter
experiments, comparison, ablation, and reproducibility.

Rules for these weeks:

- Everything lives in `confidant.lab`, behind a `lab` extra (`mlx-lm` and friends). The
  core install, CI, and `pytest` never import it. Lab code is tested against a fake
  backend and tiny fixtures, so the suite stays offline and fast.
- Days that run a model commit the config and the results (`experiments/<name>/`) with
  the code. Models, adapters, and downloaded corpora are never committed. A run that
  will not finish in one session is split into its own day.
- All training and test data is fictional or public. `examples/eval/` is the **gold set**
  and is never trained on. A contamination check enforces that.
- Machine budget: M3 Pro with 18 GB, so models up to ~4B at full LoRA, and ~8B only at
  4-bit.

All of steps 1–8 were run in one go on Nov 8, ahead of these dates. The results are in
[`docs/report.md`](docs/report.md) and [`docs/lab/`](docs/lab/). Where the work differed
from these lines, the record is:
- 300 examples rather than ~400, split 60/20/20 by scenario so every category has
  whole scenarios in val and test.
- Continued pre-training was stopped at 200 of 600 iterations.
- The sweep ran 150 iterations per setting, which turned out to be too short to compare
  ranks.
- Card chunking and constrained JSON decoding were not tried.
- The CLI and web backend switch (Nov 21) is still to do.

### Week 5 — Choosing a model, building the data (steps 1–2)

- [x] **Thu Nov 05** — A `Backend` protocol for the red-flag check (Claude, local
  `mlx-lm`, fake), with grounding, minimum severities, and the safety notice applied the
  same way whatever produced the report
- [x] **Fri Nov 06** — Model selection: `lab models` loads each candidate (small current
  Qwen, Llama, and Mistral instruct models that fit in 18 GB). It measures peak memory,
  tokens/s, context length, and how often the output is valid JSON for the flag schema.
  The table and the reasoning for the choice go in `docs/lab/model-selection.md`
- [x] **Sat Nov 07** — Labelled-data spec: a scenario grid in `lab/scenarios.toml`
  (category × severity × length × tone, plus hard negatives such as "I'll track the
  parcel"), the example schema, and a validator. The validator checks that planted lines
  appear verbatim, that names are fictional, and that nothing overlaps the gold set
- [x] **Sun Nov 08** — Generator: a local instruct model writes each scenario into a
  transcript around its planted lines. Labels come from the scenario, not from the
  generator. Rejected and near-duplicate outputs are dropped, with a count of each
- [x] **Mon Nov 09** — Generate ~400 examples. Split train/val/test 70/15/15, stratified by
  category and severity and grouped by scenario so paraphrases never straddle a split.
  Run the contamination check against the gold set. Write the datasheet in
  `docs/lab/data.md`
- [x] **Tue Nov 10** — Continued-pre-training corpus: download and clean scripts for
  DailyDialog, EmpatheticDialogues, PersonaChat, and the Wikipedia articles on
  intimate-partner abuse and coercive control. Normalize them into Confidant's transcript
  format, deduplicate, redact, tokenize, and report token counts and licenses. Only the
  scripts and the stats are committed
- [x] **Wed Nov 11** — Baseline (step 3): `lab eval` runs any backend over a split.
  Metrics: flag/no-flag precision, recall, F1; category micro-F1; severity accuracy and
  the rate of under-rated danger; quote grounding; JSON validity; diagnostic labels;
  latency and memory. Run the unmodified model on val, test, and gold

### Week 6 — Adapting it (steps 4–5)

- [x] **Thu Nov 12** — Continued pre-training: LoRA with causal-LM loss on the dialogue
  corpus, config in `experiments/configs/`, with held-out perplexity on dialogue and on the
  val transcripts before and after
- [x] **Fri Nov 13** — Supervised LoRA fine-tune on train (transcript → flag report JSON),
  with val for early stopping, from both the base model and the continued-pre-training
  checkpoint
- [x] **Sat Nov 14** — Training sweep on val: learning rate, LoRA rank, epochs, plus
  QLoRA vs LoRA for memory and speed
- [x] **Sun Nov 15** — RAG: hand-written pattern cards in `lab/cards/`, one per category
  (what it looks like, innocent readings, cultural notes), bound by
  `docs/principles.md`. A local embedding index retrieves cards into the prompt
- [x] **Mon Nov 16** — Retrieval experiments: top-k, card chunking, and two embedding
  models, with retrieval recall scored against the labelled categories
- [x] **Tue Nov 17** — Inference experiments on val: temperature, top-p, and context
  length, including windowing long transcripts (window size, stride, merging flags across
  windows)
- [x] **Wed Nov 18** — Prompt experiments for the local model: zero-shot vs few-shot,
  schema in the prompt vs constrained JSON decoding, scored with `lab eval`

### Week 7 — Showing what worked (steps 6–8)

- [x] **Thu Nov 19** — Comparison on test and gold: base, +pre-training, +fine-tune,
  both, and +RAG, on every metric plus latency, memory, and cost per read. The table goes
  in `docs/lab/results.md`
- [x] **Fri Nov 20** — Ablation: take out each component in turn (pre-training,
  fine-tune, RAG, few-shot, grounding check). Error analysis of 20 failures by category,
  and every under-rated danger case listed by name
- [ ] **Sat Nov 21** — `confidant flags --backend local` and a backend switch in the web
  UI, with the safety notice and grounding unchanged and tests on the fake backend
- [x] **Sun Nov 22** — Reproducibility: `lab reproduce <experiment>` reruns any result
  from its config with fixed seeds and pinned versions. A model card for the adapter and a
  Lab section in the README
- [x] **Mon Nov 23** — Final report draft in `docs/report.md` from the results, the
  ablation, and the error analysis, plus a demo script

## Week 8 — Showing the work

A "why did you say that" command: every claim Confidant makes can be traced back to the
messages behind it.

- [ ] **Tue Nov 24** — Locate every quote: map each evidence quote to the stored message it
  came from (conversation, position, time), and ground personality quotes the way flags
  already are, dropping traits whose quotes cannot be found
- [ ] **Wed Nov 25** — Stable claim IDs (`t1`, `f2`, …) on stored readings and the profile,
  printed beside each claim in `analyze`, `show`, and the report
- [ ] **Thu Nov 26** — `confidant why <person> <claim>`: the claim, its confidence, each
  quote with the messages around it and their dates, and a plain note when a quote no
  longer matches what is stored
- [ ] **Fri Nov 27** — Evidence chain in the API and web UI: click a claim in the report
  to see the messages behind it

## Needs a live run, or the owner

The daily build has no API key and no PyPI account, so it cannot do these; they wait
for a machine, or a person, that can.

- [ ] **Needs the owner** — Add `confidant-agent` on PyPI as a trusted publisher for
  `release.yml` (environment `pypi`), then push the `v0.2.0` tag
- [ ] **Needs a key** — Hill-climb the personality prompt against the eval: baseline, change, `compare`, keep or revert
- [ ] **Needs a key** — Claude as the commercial row in the Week 7 comparison: the same
  test and gold splits through `lab eval --backend claude`, with its cost per read
- [ ] **Needs the owner** — Read 50 sampled generated examples and mark any whose labels
  are wrong, so the datasheet can report a label-error rate

## Later

Ideas that are real but not scheduled:

- From the lab's error analysis: a second pass that re-checks every flag against the
  danger categories, since a danger item filed under a non-danger category escapes the
  floor
- Hard negatives that reuse each danger scenario's own vocabulary innocently ("I promise
  to share a secret with you" was flagged as a threat)
- From the MentalManip test: hard negatives built from terse, ordinary replies (the main
  source of false alarms on human dialogue), and transcripts where someone *reports* a
  third party's threat rather than making one
- Fine-tune with retrieval in the loop (training prompts that include retrieved guidance),
  since adding guidance only at inference pushed the model toward danger flags on real
  speech
- More outside test sets from the survey of real data: LoveFraud02 or ConScamBench for
  scams, and the UCL Tinder corpus as real negatives (needs the authors)
- A longer training sweep (300+ iterations per setting) to compare LoRA rank and lower
  learning rates on F1, and constrained JSON decoding for the local model
- Voice-note transcription for the conversations that are not text
- Multi-language transcripts, with the cultural-context caveats that requires
- Calibration study: how often is a low-confidence read actually wrong?
