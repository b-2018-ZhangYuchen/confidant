# Outside tests: MentalManip and ReaMent

Every other number in the lab comes from transcripts one local model wrote from one
scenario grid. That is the lab's biggest limitation: a model can learn a generator's
habits and look better than it is. This page tests the same systems on data written by
people for a different purpose.

## The dataset

[MentalManip](https://github.com/audreycs/MentalManip) (Wang, Yang, Hassanpour, and
Vosoughi, ACL 2024; CC BY-NC 4.0) is 4,000 two-person dialogues adapted from movie
scripts in the Cornell Movie-Dialogs Corpus. Three annotators labelled each dialogue as
manipulative or not, and named the manipulation techniques used. I use the consensus
version, where all three annotators agreed (2,915 dialogues).

The data is downloaded into `lab-cache/` and never committed. The saved reads keep the
model's answers, not the dialogue text.

The sample is 120 dialogues, chosen with seed 0:
- **60 manipulative**, drawn round-robin across all 11 techniques.
- **60 that all three annotators called non-manipulative.**

`python -m confidant.lab external` runs it; the loader is
[`external.py`](../../src/confidant/lab/external.py).

## Why it is not a straight accuracy test

- **The dataset never says who manipulates.** Each dialogue is checked twice, once with
  each speaker as the match. It counts as flagged if either direction raises a flag.
  That doubles the chance of a false alarm.
- **Manipulation is not the same as a red flag.** "Persuasion or seduction" in a film
  scene need not be anything Confidant should name. A dialogue every annotator calls
  non-manipulative can still contain a real threat. Movie dialogue, especially thrillers,
  has plenty of those.
- **Categories are checked through a mapping.** I wrote down, before running any model,
  which Confidant categories each technique could fairly be filed under (for example,
  Intimidation → threat or coercion; Accusation → guilt_or_blame or demeaning). A flag
  "agrees" if its category is in the set for a technique the annotators named.

So the scores below measure **agreement with the annotators**, not accuracy. The false
alarms were read one by one to tell model errors from disagreements about what counts.

## Results

Raw: `experiments/eval/mentalmanip/<run>/`.

| System | Valid | F1 | P | R | False alarms | at concern+ | at danger | Category agreement |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B as released | 100% | 0.00 | — | 0.00 | 0/60 | 0/60 | 0/60 | — |
| Qwen2.5-3B + few-shot | 91% | 0.67 | 0.68 | 0.65 | 18/60 | 16/60 | 16/60 | 0.51 |
| Qwen3-4B as released | 100% | 0.66 | 0.63 | 0.70 | 25/60 | 20/60 | 8/60 | 0.50 |
| **Fine-tune (iter 300) + repair** | 100% | **0.74** | 0.62 | **0.93** | 35/60 | **14/60** | 14/60 | 0.39 |
| **Fine-tune + repair + RAG** | 100% | **0.74** | 0.63 | 0.88 | 31/60 | **13/60** | 13/60 | 0.47 |

How these compare with the synthetic test split:

| | Synthetic test (step 6) | MentalManip |
|---|---|---|
| Fine-tune vs Qwen3-4B, F1 | 0.78 vs 0.42 | 0.74 vs 0.66 |
| Fine-tune, false alarms on clean / non-manipulative chats | 17% | 58% |

## What it shows

1. **The detection gain transfers, but much of the gap was the generator's.** On
   human-written dialogue, the fine-tuned model still has the best F1 and by far the
   best recall (0.93). Its lead over Qwen3-4B shrinks from 0.36 F1 to 0.08. Some of the
   synthetic advantage was learning what Qwen2.5-7B's planted lines look like. This is
   the effect the datasheet warned about, now measured.
2. **Calibration does not transfer.** The fine-tuned model raises a flag on 58% of
   non-manipulative dialogues, against 17% on clean synthetic chats. Most of the extra
   false alarms are **watch-tier "evasion" on terse replies**: "Oh.", "Yes, sir.", "He's
   completely out of his mind." Movie dialogue is clipped, and the synthetic evasion
   scenarios ("a bit of this and that", "around") taught the model that a short answer
   means dodging. At concern or above, its false-alarm rate (14/60) is close to the
   others' (13–20/60).
3. **Danger-tier false alarms are the real problem, for every system.** I read all 21
   danger flags the fine-tuned model raised on non-manipulative dialogues:
   - About **5 are defensible**. "I could force you." "Destroy his credibility…
     neutralize him." A neighbour's house watched "around the clock." These are real
     threats or surveillance in thriller plots, rightly not called manipulation by the
     annotators, and fair for Confidant to flag.
   - **The rest are errors.** "Wow, you are smart." was filed as sexual pressure, "Thank
     you man" as a threat, and a landlady's "I want you out now" as isolation. Several
     quotes attribute to the speaker a threat they were *describing* someone else making
     ("he said that if I did…").

   Few-shot (16/60) and Qwen3-4B (8/60) also do this. No system here should show a
   safety notice on its own say-so.
4. **RAG helps categories here too.** Category agreement goes from 0.39 to 0.47, and
   false alarms fall from 35 to 31, at a small cost in recall. That is the same direction
   as on the synthetic test, which makes it the most robust finding about RAG.
5. **The base model still flags nothing**, on any of the 240 reads. The out-of-the-box
   false reassurance is not an artifact of the synthetic data.

## What it changes

- **On the report's limitations:** the synthetic results overstate the fine-tuned
  model's advantage. The honest headline is "best recall and F1 on outside data too,
  but over-eager there", not "0.78 vs 0.42".
- **On the fixes:**
  - Hard negatives built from *terse, ordinary* replies would target the main false-alarm
    source.
  - Training transcripts where someone *reports* a third party's threat would teach the
    difference between describing a threat and making one.
  - Danger-tier flags from the local model need a second check before a safety notice:
    the danger-category re-check already in the roadmap, or Claude.
- **On the data question:** a dataset like this one, human-written but adjacent to the
  task, is the right kind of test set, not a training set. Training on it would teach
  "manipulation as annotated in movie scenes", which is not Confidant's task.

## A second outside test: ReaMent, unscripted speech

[ReaMent](https://huggingface.co/datasets/YSGao/ReaMent) (Gao et al., WWW 2026, released
with [MentalMAD](https://github.com/Yuansheng-Gao/MentalMAD); CC BY-NC-ND 4.0) uses the
same kind of label as MentalManip, but on **unscripted** speech: dialogues transcribed from
web videos (interviews, group talk, everyday scenes). The consensus version has 3,355
dialogues, 2,362 manipulative and 993 not.

There are no technique labels, so only detection and false alarms are scored. The sample,
scoring, and both-directions protocol are the same as for MentalManip (60 + 60, seed 0;
`python -m confidant.lab external --dataset reament`). The license forbids sharing
adaptations, so ReaMent is used only as a local test set. It is never committed and never
trained on, and the saved reads that quote it are ignored by git.

| System | Valid | F1 | P | R | False alarms | at concern+ | at danger |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B as released | 100% | 0.00 | — | 0.00 | 0/60 | 0/60 | 0/60 |
| Qwen2.5-3B + few-shot | 90% | **0.71** | 0.66 | 0.77 | 24/60 | 21/60 | 21/60 |
| Qwen3-4B as released | 100% | 0.65 | **0.67** | 0.63 | **19/60** | **10/60** | **3/60** |
| Fine-tune (iter 300) + repair | 99% | 0.67 | 0.52 | **0.93** | 52/60 | 13/60 | 12/60 |
| Fine-tune + repair + cards | 100% | 0.70 | 0.57 | 0.92 | 42/60 | **10/60** | 10/60 |

### What changes on real speech

- **The ranking turns over.** On synthetic test data, fine-tuning won by 0.36 F1. On
  scripted movie dialogue, it won by 0.08. On unscripted speech, it **does not win**:
  few-shot has the best F1, and Qwen3-4B has the best precision and by far the fewest
  danger-tier false alarms (3, against 12).
- **The fine-tuned model flags almost everything.** It raised a flag on 52 of 60
  dialogues the annotators all called non-manipulative. Most are watch tier: at concern
  or above it is 13/60, close to the others. Real speech is full of what the synthetic
  data taught it to call evasion and inconsistency: hedges, short answers, topic
  changes, self-corrections.
- **What it learned is "find something".** 55% of training transcripts had a planted
  flag, and every one of them had the flag in the match's lines. Recall stays at 0.93 on
  every dataset. Precision falls as the data gets more natural: 0.88 synthetic, 0.62
  movie dialogue, 0.52 real speech.
- **The cards help on every dataset.** False alarms dropped from 52 to 42 here and from
  35 to 31 on MentalManip, and F1 rose. That is the direction the guidance corpus is
  meant to push further.

### Across all three test sets

| System | Synthetic test F1 | MentalManip F1 | ReaMent F1 | ReaMent danger-tier false alarms |
|---|---:|---:|---:|---:|
| Qwen2.5-3B + few-shot | 0.30 | 0.67 | **0.71** | 21 |
| Qwen3-4B as released | 0.42 | 0.66 | 0.65 | **3** |
| Fine-tune + repair | **0.78** | **0.74** | 0.67 | 12 |
| Fine-tune + repair + cards | 0.75 | **0.74** | 0.70 | 10 |

The fine-tune's advantage shrinks at every step away from the generator that wrote its
training data. Its synthetic test score is not a forecast of real-world performance; the
ReaMent row is the closest thing this project has to one.
