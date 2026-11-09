# Steps 3–5 — Baseline, customization, and parameter experiments

Every number on this page is from the **val** split: 60 transcripts, 33 of them with a
planted flag, 18 of those at danger tier. Checkpoints and settings were chosen here; the
test split was not touched until step 6 ([results.md](results.md)).

All runs use Qwen2.5-3B-Instruct (4-bit) on an M3 Pro and go through the real red-flag
path, so redaction, quote grounding, and severity floors apply to every number. Each run's
config, every answer the model gave, and its scores are saved under
`experiments/eval/val/<run>/`. Any run can be redone with
`python -m confidant.lab reproduce experiments/eval/val/<run>`.

## How to read the columns

- **Valid**: answers that parsed into the schema.
- **F1 / P / R**: did a flagged transcript get a flag, and a clean one none.
- **FA hard / FA clean**: false alarms on hard negatives and on ordinary chats.
- **Cat F1**: was it the right category.
- **Sev acc**: on caught transcripts, was the highest tier right.
- **Danger missed**: danger transcripts whose highest flag came back below danger. This
  is the safety number.
- **Grounded**: share of the model's flags whose quotes were found in the transcript.
- **Labels**: reports using a diagnostic label. It was 0 in every run.

Val is small: one transcript is 3 points of recall. Differences of a few points are noise.
Each sampled setting (temperature above 0) was run once.

## Step 3 — Baseline

| Run | Valid | F1 | P | R | FA hard | FA clean | Cat F1 | Sev acc | Danger missed | Grounded | s/read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base, schema prompt | 100% | 0.00 | 0.00 | 0.00 | 0% | 0% | 0.00 | 0% | 18/18 | 100% | 3.7 |

The unmodified model answers in the schema every time, and every time it says there is
nothing to flag, with high confidence. That includes a transcript where the match
threatens the owner's ex. Out of the box, a 3B model following Confidant's prompt is a
false-reassurance machine. That is the failure the principles worry about most.

## Step 4 — Customization

| Run | Valid | F1 | P | R | FA hard | FA clean | Cat F1 | Sev acc | Danger missed | Grounded | s/read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base, compact prompt | 80% | 0.00 | 0.00 | 0.00 | 0% | 0% | 0.00 | 0% | 18/18 | 100% | 2.8 |
| base, few-shot (2) | 95% | 0.47 | 1.00 | 0.30 | 0% | 0% | 0.21 | 80% | 11/18 | 93% | 4.1 |
| base, RAG | 100% | 0.00 | 0.00 | 0.00 | 0% | 0% | 0.00 | 0% | 18/18 | 100% | 3.2 |
| continued pre-training (CPT) only | 13% | 0.00 | 0.00 | 0.00 | 0% | 0% | 0.00 | 0% | 18/18 | 100% | 2.5 |
| fine-tune (SFT), iter 360 | 97% | 0.47 | 1.00 | 0.30 | 0% | 0% | 0.37 | 60% | 13/18 | 67% | 3.8 |
| SFT 360 + quote repair | 97% | 0.57 | 1.00 | 0.39 | 0% | 0% | 0.43 | 69% | 10/18 | 87% | 3.8 |
| CPT → SFT, iter 300 + repair | 100% | 0.31 | 1.00 | 0.18 | 0% | 0% | 0.21 | 67% | 14/18 | 100% | 3.2 |
| CPT → SFT, iter 360 | 100% | 0.53 | 1.00 | 0.36 | 0% | 0% | 0.22 | 75% | 9/18 | 100% | 3.3 |
| **SFT, iter 300 + repair** | 100% | **0.81** | 0.96 | 0.70 | 7% | 0% | 0.42 | 39% | 10/18 | 96% | 3.8 |
| **SFT 300 + repair + RAG** | 100% | 0.77 | 0.92 | 0.67 | 13% | 0% | **0.56** | **64%** | **6/18** | 96% | 4.5 |

### Prompting

- **The compact prompt** (a field list instead of the full JSON Schema) is 30% shorter. It
  costs validity, from 100% to 80%: the 3B model needs the schema spelled out.
- **Few-shot**, with two worked examples from the train split, is the only change that
  helps *without training*: F1 0.47 with no false alarms. Still, 11 of 18 danger cases
  go unflagged.
- **RAG alone does nothing.** The base model reads the retrieved pattern card ("threat:
  always danger...") and still answers that there is nothing to flag. Retrieval adds
  knowledge, and this model's problem is not missing knowledge.

### Continued pre-training (CPT)

- **Config:** QLoRA (rank 8, 16 layers), causal-LM loss on the 6,107-document dialogue and
  background corpus, batch 4 × 512 tokens, learning rate 1e-4.
- **Stopped early:** planned for 600 iterations, stopped at 200 (about 400k tokens). It
  shared the GPU with data generation at about 130 tokens/s. Validation loss went
  1.947 → 1.363 (iter 100) → 1.468 (iter 200).

Perplexity before and after ([`experiments/eval/perplexity.json`](../../experiments/eval/perplexity.json)):

| Model | Held-out dialogue | Val transcripts |
|---|---:|---:|
| base | 5.75 | 7.49 |
| after CPT | **3.98** | 7.88 |

CPT taught the model the corpus: perplexity on held-out dialogue dropped by 31%. It did
not transfer to dating-app chats, where perplexity rose slightly. It also damaged
instruction-following. On its own, the CPT model produced valid JSON only **13%** of the
time, usually by leaving out required fields (`{"flags": [], "summary": "..."}` with no
`confidence`). That is catastrophic forgetting: a few hundred thousand tokens of plain
dialogue were enough to erode the chat-format behavior the instruct model was tuned for.

CPT followed by fine-tuning recovered completely (100% valid). It also started
fine-tuning from a lower loss (1.212 vs 1.606) and never stitched quotes (100%
grounded). But its best validation loss was worse (0.633 vs 0.561), its category F1 was
lower, and at the chosen checkpoint it did not beat SFT alone. Step 7 measures this
directly.

### Fine-tuning (SFT)

- **Config:** QLoRA (rank 8, 16 layers, scale 20), on the 180 train transcripts as chats:
  redacted compact prompt in, label JSON out, prompt masked from the loss. Batch 1 with 4
  gradient-accumulation steps, learning rate 1e-4, 360 iterations (2 epochs), 42 minutes.
- **The validation loss curve was misleading.** It fell to its minimum at iter 100
  (0.561) and then rose steadily to 0.967. On loss alone, early stopping would pick
  iter 100. But that checkpoint **flags nothing**, on every val transcript, and so does
  the CPT→SFT checkpoint at iter 100. Over 40% of answers are "no flags", which is the
  cheapest thing to predict, so the model learns it first. Loss is lowest while the model
  says it everywhere, and loss rises as the model learns to name flags, because a wrong
  flag costs more tokens than a wrong empty list.

  | Checkpoint | val loss | val F1 |
  |---|---:|---:|
  | iter 100 | **0.561** | 0.00 |
  | iter 200 | 0.728 | 0.17 |
  | iter 300 | 0.742 | **0.81** |
  | iter 360 (final) | 0.967 | 0.57 |

  **Checkpoints are chosen on val F1, not loss.** Iter 300 is the fine-tuned model from
  here on.

### Quote repair

The fine-tuned model often found the right lines and then quoted two consecutive
messages as one string. One example is "tell him if he texts you again I'll make sure he
can't, I'm not joking about this", where those are two separate messages. Grounding
refuses stitched quotes on purpose, because a stitched quote can say something neither
message said. So these true detections were thrown away: only 67% of the final model's
flags were grounded.

The fix is `split_stitched` in [`backend.py`](../../src/confidant/lab/backend.py), applied
before grounding. A quote that matches no single message, but is exactly a run of whole
messages from the match, becomes one quote per message. Anything else (a partial stitch,
an owner's line, an invented word) is left for grounding to discard. Every quote is
still a verbatim message, so grounding is as strict as it was.

Because every run saves the model's raw answers, the effect was measured by replaying
them (`python -m confidant.lab rescore`), with no model call. On the final SFT adapter,
repair raised F1 from 0.47 to 0.57 and grounding from 67% to 87%. It changed nothing for
models that never stitched.

### RAG

- **Cards:** 11 hand-written pattern cards in `lab/cards/`, one per category: what it
  looks like, what it is not, and how to file it. A twelfth card, "what is not a flag",
  is always included.
- **Retrieval:** from the match's lines only. Recall of the gold category's card on
  flagged val transcripts ([`experiments/retrieval/recall.json`](../../experiments/retrieval/recall.json)):

  | Retriever | k=1 | k=2 | k=3 |
  |---|---:|---:|---:|
  | BM25 over all match lines | 30% | 36% | 45% |
  | **BM25, best single line** | **45%** | **52%** | **58%** |
  | MiniLM-L6 embeddings, best line | 15% | 33% | 45% |
  | BGE-small embeddings, best line | 18% | 39% | 48% |

  Plain word overlap, scored line by line, beat both embedding models. The planted lines
  share vocabulary with the cards ("location", "password", "gift cards"), and in a long
  chat, scoring each line on its own keeps the filler from drowning out the one line
  that matters. All RAG runs use per-line BM25 with k=2.
- **Effect:** on the base model, none (see above). On the fine-tuned model, RAG barely
  moves detection (F1 0.81 → 0.77), but it is the best setting for everything after
  detection: category F1 0.42 → **0.56**, severity accuracy 39% → **64%**, danger missed
  10 → **6** of 18. The cards say how each category is filed, which is exactly what the
  fine-tuned model was getting wrong. It costs 0.7 s per read and one more false alarm
  on the hard negatives.

## Step 5 — Parameter experiments

### Training: learning rate and LoRA rank

Four short runs (150 iterations, the SFT config otherwise unchanged):

| Run | val loss at 50 / 100 / 150 | val F1 at 150 | Notes |
|---|---|---:|---|
| lr 5e-5, rank 8 | 0.623 / **0.497** / 0.609 | 0.00 | lowest loss of any run; still in the "no flags" phase |
| lr 2e-4, rank 8 | 0.696 / 1.593 / 1.808 | 0.00 (5% valid) | diverged; output degenerated into repeating JSON fragments |
| lr 1e-4, rank 4 | 0.624 / 0.537 / 0.646 | 0.00 | still in the "no flags" phase |
| lr 1e-4, rank 16 | 0.646 / 0.595 / 0.767 | 0.00 | still in the "no flags" phase |
| *reference: lr 1e-4, rank 8* | *0.627 / 0.561 / 0.694* | *(0.17 at 200, 0.81 at 300)* | the main run |

Each run took 18 minutes. Loss curves: `experiments/runs/sweep-*/result.json`.

What this shows: **2e-4 is unstable** for this setup, and **training length dominates
everything else**. The run with the lowest validation loss of the whole project (5e-5,
0.497) is again a model that flags nothing, which confirms the step 4 finding about loss. Every run that survived was still in the phase where the model says
"no flags" everywhere, the same phase the main run was in at iters 100–200. What it does
not show: whether rank 4, rank 16, or 5e-5 would beat rank 8 at 1e-4 if trained to 300.
150 iterations was too short to tell. With the laptop budget (18 minutes per
150-iteration run), this is the biggest gap in the experiments.

### Inference: temperature and top-p (SFT iter 300 + repair)

| Setting | F1 | P | R | FA hard | Cat F1 | Sev acc | Danger missed |
|---|---:|---:|---:|---:|---:|---:|---:|
| greedy (T=0) | **0.81** | 0.96 | 0.70 | 7% | 0.42 | 39% | 10/18 |
| T=0.3 | **0.81** | 0.96 | 0.70 | 7% | 0.42 | 43% | 9/18 |
| T=0.7 | 0.79 | 0.92 | 0.70 | 13% | 0.48 | 57% | 7/18 |
| T=0.7, top-p 0.9 | 0.74 | 0.88 | 0.64 | 20% | 0.32 | 38% | 12/18 |

Sampling at 0.7 raised severity accuracy and lowered danger misses, but it doubled false
alarms on hard negatives. Adding top-p 0.9 made everything worse. These are single
sampled runs, so the T=0.7 gain may be luck. Greedy decoding stays the default: it is
deterministic, so a user who asks twice gets the same answer.

### Context length (SFT iter 300 + repair)

| Messages shown | F1 | R | Danger missed |
|---|---:|---:|---:|
| last 8 | 0.42 | 0.27 | 13/18 |
| last 14 | 0.53 | 0.36 | 13/18 |
| **all (mean 18.7)** | **0.81** | 0.70 | 10/18 |

The planted lines sit mid-conversation, so cutting the chat cuts the evidence. Every
transcript here fits easily in the model's 32k context, so there is no reason to
truncate. A real WhatsApp export can run to thousands of messages, though; for those, the
roadmap's windowing (overlapping chunks, flags merged) is the right design, not keeping
only the tail.

## What was carried into step 6

- **Fine-tuned model:** SFT at iter 300 (best val F1), with quote repair and greedy
  decoding.
- **With and without RAG:** RAG trades a little detection for much better severity, so
  both go to test.
- **CPT → SFT** at its best val checkpoint, iter 360 (val F1 0.53, against 0.31 at iter
  300), to measure what CPT adds.
- **References:** the base model with the schema prompt and with few-shot, the CPT-only
  model, and Qwen3-4B out of the box.
