# Confidant: adapting a local model for evidence-grounded red-flag detection

Yuchen Zhang — GenAI course project report (draft)

## 1. Problem

Confidant reads a dating-app conversation and gives a careful second opinion. Its
red-flag check names concerning behavior (threats, coercion, monitoring, financial
pressure, and eight other categories) at one of three tiers, with verbatim quotes as
evidence. Every quote is checked against the transcript in code, danger categories can
never be filed below danger, and no diagnostic labels are allowed (`docs/principles.md`).

Until now, that check ran on Claude through its API. This project asks whether an open
model, small enough to run on the student's laptop, can be adapted to do the same job.
That would give two things the API cannot: **the transcript never leaves the machine**,
and **a read costs nothing**. The work follows the course's eight steps. Each has its
own page under [`docs/lab/`](lab/), with every number traceable to a saved run in
`experiments/`.

## 2. Model selection (step 1)

Seven 4-bit instruct models (Llama 3.2 1B/3B, Qwen2.5 1.5B/3B/7B, Qwen3 4B, Mistral 7B)
were run through the real red-flag path on an M3 Pro with 18 GB.

- Below 3B, no model could hold the JSON schema.
- **Qwen2.5-3B-Instruct** was the smallest model that answered in the schema every time,
  and the fastest of those (3.8 s per read, 2.8 GB). It became the model to adapt.
- **Qwen3-4B**, the best model out of the box, is kept as a reference.
- **Qwen2.5-7B** writes the training data.

→ [lab/model-selection.md](lab/model-selection.md)

## 3. Data (step 2)

No real chat logs may be used, so the labelled data is synthetic by design:

- **Scenario grid:** 100 hand-written scenarios. 55 plant a flag (5 per category), 25
  are hard negatives that look like flags ("text me when you get home safe?"), and 20
  are clean.
- **Generation:** each scenario was written out three times by the 7B model, with
  different names, lengths, and tones.
- **Labels are true by construction.** The model writes the conversation around the
  planted lines, and code inserts them. Asking the model to include the lines itself
  kept 1 transcript in 11; inserting them kept 300 of 303.
- **Splits are by scenario** (180/60/60), so no paraphrase of a test case is trained on.
  A contamination check keeps the 9 hand-written gold cases out of everything.
- **Pre-training corpus:** 6,107 documents (~947k words) for continued pre-training,
  from DailyDialog, EmpatheticDialogues, Synthetic-Persona-Chat, and 13 Wikipedia
  articles on abuse and scams. It is redacted and formatted the way the red-flag prompt
  shows a chat.

→ [lab/data.md](lab/data.md)

## 4. Baseline (step 3)

With Confidant's own prompt, the unmodified 3B model answered in the schema every time.
Every time it said there was nothing to flag, with high confidence, including when the
match threatened someone. **F1 0.00; 33 of 33 danger cases missed** on val and test.
Out of the box, it gives exactly the false reassurance Confidant exists to avoid.

## 5. Customization (step 4)

Four approaches were tried, alone and combined:

| Approach | Effect (val) |
|---|---|
| Prompting: compact schema | fewer tokens; validity 100% → 80% |
| Prompting: few-shot (2 train examples) | F1 0.00 → 0.47, no false alarms |
| RAG: pattern cards, per-line BM25 retrieval | alone: no change. On the fine-tuned model: category F1 0.42 → 0.56, danger missed 10 → 6 of 18 |
| Continued pre-training (QLoRA, ~400k tokens) | dialogue perplexity 5.75 → 3.98, but valid JSON 80% → 13% (catastrophic forgetting) |
| Fine-tuning (QLoRA r=8, 180 examples, 2 epochs) | F1 0.81 at the chosen checkpoint |
| Quote repair (split quotes that stitch whole messages) | final adapter: F1 0.47 → 0.57, grounding 67% → 87% |

Two findings shaped everything after:

1. **Validation loss chose the wrong checkpoint.** The lowest-loss checkpoint (iter 100)
   flagged *nothing*. Over 40% of answers are "no flags", the cheapest output to learn,
   so loss was lowest while the model said it everywhere. F1 by checkpoint: 0.00 at 100,
   0.17 at 200, **0.81 at 300**, 0.57 at 360. All later choices were made on val F1.
2. **Stitched quotes.** The fine-tuned model found the right lines but quoted two
   consecutive messages as one. Grounding correctly refuses that, because a stitched
   quote can say what neither message said. The repair splits a quote only when it is
   *exactly* a run of whole messages, so every quote is still verbatim.

→ [lab/experiments.md](lab/experiments.md)

## 6. Parameter experiments (step 5)

- **Learning rate:** 2e-4 diverged (val loss 0.70 → 1.81, degenerate output). 5e-5 and
  ranks 4 and 16, run for 150 iterations, were all still in the "no flags" phase.
  Training length dominated, and the short sweep could not separate rank. That is the
  experiment's main gap.
- **Temperature:** 0.3 matched greedy. 0.7 improved severity but doubled hard-negative
  false alarms. Top-p 0.9 was worse on every metric. Greedy was kept, because it is
  deterministic.
- **Context:** showing only the last 8 or 14 messages cut F1 to 0.42 and 0.53, because
  the evidence was cut too. Long exports need windowing, not truncation.
- **Retrieval:** per-line BM25 (58% recall at k=3) beat MiniLM (45%) and BGE-small (48%)
  embeddings. The planted lines share vocabulary with the cards, and scoring each line on
  its own keeps filler from drowning the one line that matters.

## 7. Comparison (step 6) — test split, unseen scenarios

| System | F1 | Danger missed | Gold set | s/read |
|---|---:|---:|---:|---:|
| Qwen2.5-3B as released | 0.00 | 15/15 | 64% | 3.6 |
| Qwen2.5-3B + few-shot | 0.30 | 12/15 | 72% | 4.3 |
| Qwen3-4B as released | 0.42 | 14/15 | 83% | 6.5 |
| CPT → fine-tune | 0.43 | 9/15 | 61% | 3.5 |
| **Fine-tune + repair** | **0.78** | **4/15** | **92%** | 3.9 |
| **Fine-tune + repair + RAG** | 0.75 | **4/15** | 89% | 4.5 |
| Claude | needs an API key; see the roadmap | | | |

The fine-tuned 3B model beats the larger Qwen3-4B out of the box on every measure, at
0.6× its latency. RAG trades 0.03 F1 for better categories (0.34 → 0.46) and no
hard-negative false alarms. No system used a diagnostic label.

→ [lab/results.md](lab/results.md)

## 8. Ablation and error analysis (step 7)

Removing one component at a time from the full system, on test:

- **Fine-tuning:** −0.45 F1, compared with few-shot.
- **Checkpoint selection:** −0.28.
- **RAG:** +0.03 F1 without it, but −0.12 category F1.
- **Grounding:** +0.04 F1 without it. Every flag it discarded on test was a near-misquote
  of a real line, which grounding is right to refuse.
- **Quote repair:** 0 on test (the chosen checkpoint rarely stitches).
- **Adding continued pre-training:** −0.32.

The errors that matter:

1. **Danger under a non-danger category.** A threat filed as "evasion" escapes the
   danger floor, which applies by category.
2. **Surface over-generalization.** "I promise to share a secret with you" was flagged
   as a danger-tier threat, because a training scenario threatened to share photos.
3. **Implicature.** "You know what that means" was missed every time.

The fixes they point to are a danger-category second pass, hard negatives that reuse each
danger scenario's own vocabulary, and keeping Claude as the default for nuance.

## 9. An outside test: MentalManip

All the data above came from one generator. To check what that hides, the same systems
were run on 120 dialogues from [MentalManip](https://github.com/audreycs/MentalManip)
(ACL 2024, CC BY-NC 4.0). These are human-written movie dialogues labelled as
manipulative or not by three annotators who all agreed. Half are manipulative, drawn
across all 11 techniques. Each dialogue is checked with each speaker as the match.

| System | F1 | Recall | False alarms (any / concern+) | Category agreement |
|---|---:|---:|---:|---:|
| Qwen2.5-3B as released | 0.00 | 0.00 | 0 / 0 of 60 | — |
| + few-shot | 0.67 | 0.65 | 18 / 16 | 0.51 |
| Qwen3-4B as released | 0.66 | 0.70 | 25 / 20 | 0.50 |
| Fine-tune + repair | **0.74** | **0.93** | 35 / 14 | 0.39 |
| Fine-tune + repair + RAG | **0.74** | 0.88 | 31 / 13 | 0.47 |

- **The fine-tuned model is still the best detector, but its lead over Qwen3-4B shrinks**
  from 0.36 F1 to 0.08. Part of the synthetic advantage was learning the generator.
- **It over-flags human dialogue.** It flags 58% of the non-manipulative dialogues,
  against 17% of clean synthetic chats. Most of these are watch-tier "evasion" on terse
  replies, a habit learned from the synthetic evasion scenarios.
- **Every system raised danger-tier flags on non-manipulative dialogues.** For the
  fine-tuned model, about a quarter of these were real threats in thriller plots, and the
  rest were errors.
- **RAG improved category agreement**, as on the synthetic test.

**ReaMent** (WWW 2026, CC BY-NC-ND 4.0) has the same labels on *unscripted* speech
transcribed from web videos. It is the closest thing here to real conversations. On the
same 60 + 60 protocol:

| System | F1 | P | R | False alarms (any / danger) |
|---|---:|---:|---:|---:|
| + few-shot | **0.71** | 0.66 | 0.77 | 24 / 21 |
| Qwen3-4B as released | 0.65 | **0.67** | 0.63 | **19 / 3** |
| Fine-tune + repair | 0.67 | 0.52 | **0.93** | 52 / 12 |
| Fine-tune + repair + cards | 0.70 | 0.57 | 0.92 | 42 / 10 |

On real speech, **the fine-tune no longer wins.** Its precision falls at each step away
from the synthetic data (0.88 → 0.62 → 0.52) while recall stays at 0.93. What it learned
is to find *something* in the match's lines.

→ [lab/external.md](lab/external.md)

## 10. RAG over relationship guidance

The pattern cards were my own words, and picking 2 of 11 is barely retrieval. So RAG was
rebuilt on a real corpus of relationship guidance: 39 pages (US government health sites,
public domain, and 24 Wikipedia articles, CC BY-SA), about 116k words. wikiHow was
dropped because it blocks Anthropic's crawler.

- **Grid on val** (chunking × retriever × k): embeddings beat BM25, and more passages made
  the model more cautious. Paragraphs + MiniLM + k=4 scored F1 0.80 at precision 1.00.
- **Synthetic test:** the best danger recall of any system (2 of 15 missed, against 4)
  and the best F1 (0.79).
- **Outside sets:** it did not fix false alarms. ReaMent went 87% → 78%; the cards
  reached 70%. Danger-tier false alarms rose from 12 to 15.
- **A tested explanation, refuted:** the corpus is 46% abuse material, which looked like
  the cause. Retrieving only ordinary-relationship passages made danger false alarms
  *worse* (18). Any retrieved text pushes this fine-tuned model toward danger, most
  likely because it was trained without retrieval in its prompts.

→ [lab/guidance.md](lab/guidance.md)

## 11. Reproducibility (step 8)

- **Code:** everything is in `confidant.lab` (`python -m confidant.lab --help`), behind
  the `lab` extra, tested offline: the core suite runs without it.
- **Configs:** every training run is a YAML file in `experiments/configs/`.
- **Records:** every evaluation saves its config, its exact command line, every raw model
  answer, and its scores. `python -m confidant.lab reproduce <run>` reruns it beside the
  original; `rescore` replays saved answers under different checks with no model call.
- **What is committed:** the generated dataset, scenario grid, and pattern cards. Models,
  adapters, and the downloaded corpus are rebuilt from scripts.
- **Model card:** [lab/model-card.md](lab/model-card.md).

## 12. Conclusions and limits

A 6.7M-parameter adapter, trained in 42 minutes on a laptop from 180 synthetic examples,
turns a 3B model that saw nothing into one that catches 70% of planted flags in unseen
scenarios. It also passes 92% of the hand-written gold checks. Its errors are the ones
that matter most for this tool: a quarter of danger cases are not filed as danger.

On human-written dialogue from outside the task, its lead shrinks: to 0.08 F1 on movie
dialogue, and to nothing on real unscripted speech. There it flags 87% of ordinary
conversations, mostly at watch tier, and Qwen3-4B out of the box is far better
calibrated.

So the local model is a credible **private option** beside Claude, not a replacement:
the user who will not send a chat anywhere gets a real check, with the same grounding
and safety floors as before.

The limits are real:
- the test set is small (one scenario is 9 points of recall)
- all training and synthetic test data comes from one generator in the student's own
  model family. The MentalManip test shows this overstated the advantage: 0.78 vs 0.42
  F1 against Qwen3-4B on synthetic data, 0.74 vs 0.66 on human-written data
- the negatives are not yet audited by a person
- each configuration was run once
- the Claude comparison is still waiting on an API key
