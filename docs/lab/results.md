# Steps 6–7 — Comparison, ablation, and error analysis

Every system here was configured on val ([experiments.md](experiments.md)) and is scored
on the **test** split, touched only for this page. The test split has 60 transcripts from
20 scenarios that no training run saw: 33 with a planted flag (15 at danger tier), 15
hard negatives, and 12 clean chats. Each system is also scored on the **gold set**: the 9
hand-written cases in `examples/eval/`, scored by the same 36 checks the prompt
hill-climb uses.

Raw answers and scores are in `experiments/eval/test/<run>/`. The table is
`python -m confidant.lab table experiments/eval/test/*`.

## Step 6 — Comparison

| System | Valid | F1 | P | R | FA hard | FA clean | Cat F1 | Sev acc | Danger missed | Grounded | s/read | Gold |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B, as released (schema prompt) | 100% | 0.00 | 0.00 | 0.00 | 0% | 0% | 0.00 | 0% | 15/15 | 100% | 3.6 | 64% |
| Qwen2.5-3B + few-shot | 97% | 0.30 | 0.86 | 0.18 | 0% | 8% | 0.17 | 50% | 12/15 | 76% | 4.3 | 72% |
| Qwen3-4B, as released (schema prompt) | 100% | 0.42 | 0.90 | 0.27 | 0% | 8% | 0.09 | 22% | 14/15 | 100% | 6.5 | 83% |
| Qwen2.5-3B + continued pre-training | 5% | 0.00 | 0.00 | 0.00 | 0% | 0% | 0.00 | 0% | 15/15 | 100% | 2.5 | 0% |
| CPT → fine-tune (val-chosen, iter 360) | 100% | 0.43 | 0.71 | 0.30 | 27% | 0% | 0.13 | 60% | 9/15 | 100% | 3.5 | 61% |
| **Fine-tune (iter 300) + quote repair** | 100% | **0.78** | 0.88 | **0.70** | 7% | 17% | 0.34 | 65% | **4/15** | 96% | 3.9 | **92%** |
| **Fine-tune + quote repair + RAG** | 100% | 0.75 | **0.91** | 0.64 | **0%** | 17% | **0.46** | **71%** | **4/15** | 92% | 4.5 | 89% |
| Claude (commercial reference) | — | — | — | — | — | — | — | — | — | — | — | — |

No report used a diagnostic label, in any run.

The Claude row needs an API key, which the project machine does not have. It is the
first item under "Needs a key" in the roadmap: `python -m confidant.lab eval` with a
Claude backend on the same splits.

### What the comparison shows

- **Fine-tuning is what works.** On transcripts from scenarios it never saw, the
  fine-tuned 3B model catches 70% of planted flags at 88% precision. Out of the box, the
  same model caught none. On the 9 hand-written gold cases, which share nothing with
  the training data but the task, it passes 33 of 36 checks (92%), against 23 (64%) for
  the base model. So it did not just learn the generator's style.
- **A fine-tuned 3B model beats a bigger model out of the box.** Qwen3-4B was the best
  off-the-shelf model in step 1, and it is better than the base 3B: F1 0.42, gold 83%.
  But it misses 14 of 15 danger cases, and it is 1.7× slower. The adapter is 6.7M
  parameters, 0.2% of the model.
- **RAG is a trade, not a gain.** It moves categories from 0.34 to 0.46, severity from
  65% to 71%, and hard-negative false alarms from 7% to 0%. Recall drops from 0.70 to
  0.64. Danger misses are the same 4. On test the trade is smaller than on val (6 vs 10
  danger missed there). Which to ship depends on whether a missed flag or a mislabelled
  one is worse. I would ship RAG: Confidant's principles care more about *how*
  something is named than about catching every watch-tier item.
- **Continued pre-training hurt.** On its own it broke the output format (5% valid).
  Fine-tuned on top, it was worse than fine-tuning alone on every metric that matters,
  including 27% false alarms on hard negatives and 9 danger misses. See the ablation.
- **Checkpoint choice mattered on test too.** The final iter-360 adapter, the one you
  get by just training to the end, scores F1 0.47 and misses 11 danger cases. That
  confirms on unseen data what val showed: it was right to choose by F1 rather than by
  "last" or "lowest loss".
- **Cost:** every local system costs $0 per read and keeps the transcript on the machine.
  A read takes 3.9 s (4.5 with RAG) on an M3 Pro, with a peak of 2.7 GB.

## Step 7 — Ablation

Starting from the full system (fine-tune iter 300 + quote repair + RAG), one component
is taken out at a time, on test. Rows marked *replay* re-score the saved answers without
calling the model (`python -m confidant.lab rescore`), so the only difference is the
component removed.

| Variant | F1 | R | FA hard | Cat F1 | Sev acc | Danger missed | Gold | Δ F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Full system** | 0.75 | 0.64 | 0% | 0.46 | 71% | 4/15 | 89% | — |
| − RAG | 0.78 | 0.70 | 7% | 0.34 | 65% | 4/15 | 92% | +0.03 |
| − quote repair *(replay)* | 0.75 | 0.64 | 0% | 0.46 | 71% | 4/15 | — | 0.00 |
| − grounding *(replay)* | 0.79 | 0.70 | 0% | 0.48 | 70% | 3/15 | — | +0.04 |
| − checkpoint selection (train to the end, iter 360, no RAG) | 0.47 | 0.30 | 0% | 0.05 | 40% | 11/15 | 75% | −0.28 |
| − fine-tuning (base + few-shot) | 0.30 | 0.18 | 0% | 0.17 | 50% | 12/15 | 72% | −0.45 |
| − fine-tuning, − few-shot (base) | 0.00 | 0.00 | 0% | 0.00 | 0% | 15/15 | 64% | −0.75 |
| + continued pre-training (CPT → fine-tune, no RAG) | 0.43 | 0.30 | 27% | 0.13 | 60% | 9/15 | 61% | −0.32 |

What each component is worth:

- **Fine-tuning** is nearly all of it: +0.45 F1 over the best prompt-only setup, and 8
  fewer danger misses.
- **Checkpoint selection on val F1** is the second biggest effect (+0.28). The same
  training run, stopped at the wrong place, keeps only a third of its recall.
- **RAG** does not improve detection. It is what makes the *categories* right (+0.12 cat
  F1) and removes hard-negative false alarms.
- **Quote repair** did nothing on test. The iter-300 checkpoint rarely stitches quotes.
  On val, it recovered 10 points of F1 for the final adapter, which stitched often. It is
  a safeguard for one failure mode, not a general gain.
- **Grounding** costs 0.04 F1 and saves nothing *on this test set*. Every flag it
  discarded here was a near-quote of a real line ("you're *bringing* saturday" for
  "you're *coming* saturday"; a trailing "else" added), not an invented one. It stays,
  and not because of this number: a misquote puts words in someone's mouth, and on
  real chats, from a model that was not trained on exact quotes, the invented ones are
  the ones that matter. The ablation shows its price is low.
- **Continued pre-training** is worth −0.32 F1 in this setup. With 400k tokens of
  general dialogue (stopped at a third of the plan), it shifted the model toward the
  corpus's register, eroded the instruct behavior, and gave fine-tuning a worse starting
  point for this task, even though it started from a lower loss. A larger, in-domain
  corpus (real dating-app style chats, which this project cannot use) might change that.
  This run gives no reason to expect it.

## Error analysis

From the full system's 60 test reads (`experiments/eval/ablation/errors-sft-rag.md`;
the RAG-free system's list is beside it):

**Misses (12 of 33)** concentrate in the subtle, low-tier categories:

| Category | Missed | What was planted |
|---|---:|---|
| demeaning (watch) | 3/3 | "I'll eat before I come over then, I've seen your cooking" |
| boundary_pushing (concern) | 3/3 | "I already told everyone you're coming saturday, so you kind of have to now" |
| inconsistency (watch) | 2/3 | "I'm in the city, ten minutes from you" … later "driving two hours each way is a lot for me" |
| sexual_pressure (danger) | 2/3 | "I'll only drive out there if it's at your place and you know what that means" |
| financial_pressure (concern) | 1/3 | "can you get this one too? payday is next week, I swear" |
| monitoring (danger) | 1/3 | "why would you change it? I need your password…" |

- **Watch-tier digs** ("I've seen your cooking") are written as teasing, and the
  scenario's own innocent reading says so. Missing them is close to the right call.
- **Inconsistency** needs two lines far apart. The model reads lines one at a time.
- **The two danger misses are the serious ones.** "You know what that means" is an
  implication, not a statement. The password demand was *found*, but the quote picked up
  an extra word ("anyone else") and grounding discarded it.

**Wrong category (8):**
- Coercion was filed as isolation (3×). That is still danger, so the floor held.
- Guilt-tripping was filed as evasion (2×).
- Evasion was filed as monitoring or isolation (2×), which turns a watch item into a
  danger notice.
- One threat was filed as evasion at watch tier. That is the one way a caught danger
  item reaches the user as a mild note, because the floors apply by category.

**False alarms (2 of 27 negatives).** Both are the same clean scenario (a picnic plan). In
one, the model read "How about if I promise to share a secret with you after the picnic?"
as a threat to share intimate details. It learned the training scenario about
threatening to send private photos by its surface wording ("share", "secret"). Filed at
danger, this would have shown a safety notice over a harmless chat. That is the most
harmful single error in the test set.

### What would fix what

- **Danger hidden in a non-danger category** is the main safety gap. A second pass that
  asks "is any flag here actually one of the danger categories?" with the matching
  cards, or more training scenarios that sit between categories, are both aimed at it.
- **Surface-wording over-generalization** ("secret" means threat) needs more hard
  negatives that reuse the vocabulary of each danger scenario in an innocent way. The
  current 25 hard negatives imitate categories, not specific training lines.
- **Implicature** ("you know what that means") is what a 3B model is worst at. Of
  everything here, it is the strongest argument for keeping Claude as the default and the
  local model as the private option.

## Limits of these numbers

- **Small test set.** 33 flagged transcripts from 11 scenarios: one scenario is 3
  transcripts, about 9 points of recall. Confidence intervals would be wide; the ordering
  of fine-tuned vs not is robust, the gap between the two fine-tuned rows is not.
- **One generator, one family.** All transcripts are synthetic, from one Qwen2.5-7B, the
  same family as the student. The gold-set results are the check against
  "it learned the generator's style": 92% there is encouraging, but that is 9 cases.
- **Unaudited negatives.** Labels are exact for the planted lines. The owner's audit of
  50 sampled transcripts is still to do.
- **Single runs.** Each training configuration was run once, with seed 0.
