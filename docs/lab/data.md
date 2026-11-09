# Step 2 — Data

The lab uses two datasets: a **labelled** set of fictional chats for fine-tuning and
evaluating the red-flag check, and an **unlabelled** corpus for continued pre-training.
Neither contains a real person's private messages.

## Labelled transcripts (`lab/data/examples.jsonl`)

### How they are made

1. **Scenario grid** ([`lab/scenarios.toml`](../../lab/scenarios.toml)): 100 hand-written
   seeds. Each has a setting ("the owner did not reply for a day because of work") and,
   for most, the exact lines the match must say.
   - 55 **flag** scenarios: 5 for each of 11 categories (threat, coercion,
     sexual_pressure, financial_pressure, isolation, monitoring, boundary_pushing,
     demeaning, guilt_or_blame, inconsistency, evasion), each with a category and a tier.
   - 25 **hard negatives**: lines that look like a flag and are not ("text me when you
     get home safe?", "I will destroy you at scrabble"). Each records which category it
     imitates.
   - 20 **clean** conversations: nothing planted.
2. **Generation** (`python -m confidant.lab generate`): Qwen2.5-7B-Instruct (4-bit,
   temperature 0.8, top-p 0.95, seed 0) writes each scenario three times, with different
   names, lengths (8–32 messages), tones, and who speaks first. For scenarios with planted
   lines, the model writes the opening, the planted lines are appended **by code** as the
   match's next messages, and the model writes the continuation, including the owner's
   reaction.
   My first design asked the model to place the lines itself. Qwen2.5-7B repeated them,
   gave them to the wrong person, or paraphrased them: 1 transcript kept out of 11 tries.
   Placing them in code makes every label true by construction.
3. **Validation**: a transcript is rejected (and counted) if it has fewer than six
   messages, has only one speaker, repeats a planted line, or is too close to another
   variant of the same scenario (5-gram Jaccard ≥ 0.6). Markdown images and links are
   dropped.
4. **Labels** come from the scenario, never from the generator: the planted lines are the
   evidence, and the category, tier, behavior, and innocent reading are the ones written
   in the grid. Hard negatives and clean chats are labelled "no flags".
5. **Splits** are made by *scenario*, stratified by category (or kind), at 60/20/20:
   3 train / 1 val / 1 test scenario per category (15/5/5 hard negatives, 12/4/4 clean). All variants of a scenario land in the
   same split, so the model is never tested on a paraphrase of something it trained on.
6. **Contamination check**: no generated message may repeat a gold-set message of more
   than four words, and no planted line may appear in the gold set
   (`python -m confidant.lab check`).
7. **Preprocessing for training**: each example is redacted exactly as the backend
   redacts a chat at inference, rendered into the same prompt, and paired with its label
   as the assistant's answer. The prompt tokens are masked out of the loss.

### Counts

Generation kept **300 of 303 attempts**: 3 were rejected because the continuation repeated
a planted line, and none for length, one-sidedness, or near-duplication. Transcripts
average 18.7 messages. No generated message repeats a gold-set message. Raw record:
[`lab/data/generation.json`](../../lab/data/generation.json).

| | train | val | test |
|---|---:|---:|---:|
| flag (11 categories × 3 per scenario) | 99 | 33 | 33 |
| hard negative | 45 | 15 | 15 |
| clean | 36 | 12 | 12 |
| **total** | **180** | **60** | **60** |

Every category has 9 train, 3 val, and 3 test transcripts. Danger-tier transcripts are
18 of the 33 flagged ones in val and 15 of 33 in test (the five danger categories, plus any financial scenario filed as danger).

### Known weaknesses

- **One author.** One 7B model wrote every transcript, so the writing style is narrow and
  clearly synthetic. The generator is in the student's family (Qwen2.5), which may make
  the data look more familiar to the student than real chats would.
- **The owner's reactions are not always coherent.** A continuation sometimes has the
  owner answer a threat cheerfully. The *match's* planted lines, which are what is
  labelled, are always exact.
- **The filler is unaudited.** A clean chat's filler could contain something a careful
  reader would flag. The roadmap asks the owner to read 50 sampled transcripts and report
  a label-error rate. Until then, treat the labels as near-certain for flags and
  unverified for negatives.
- **English only.** The cultural-context work the synopsis mentions is out of scope here.

## Gold set (`examples/eval/`)

The 9 hand-written cases from the prompt hill-climb, never trained on, are scored with the
same checks as before (`confidant.evals.score_read`). There are too few for an F1, but they
are the only transcripts written by a person, so they are reported separately in every
comparison.

## Continued-pre-training corpus (`lab-cache/corpus/`, not committed)

`python -m confidant.lab corpus` downloads, cleans, and writes it. Counts are in
[`experiments/corpus/stats.json`](../../experiments/corpus/stats.json).

| Source | License | Documents |
|---|---|---:|
| DailyDialog | CC BY-NC-SA 4.0 | 2,000 |
| EmpatheticDialogues | CC BY-NC 4.0 | 2,000 |
| Synthetic-Persona-Chat | CC BY 4.0 | 2,000 |
| Wikipedia (13 articles on coercive control, intimate-partner violence, romance scams, stalking, consent) | CC BY-SA 4.0 | 107 chunks |
| **Total** | | 6,107 documents, ~947k words |

Cleaning:
- Dialogues with fewer than four turns are dropped (869 from DailyDialog, 113 from
  EmpatheticDialogues), and exact duplicates are removed (10).
- DailyDialog's tokenized spacing is undone.
- Every dialogue is redacted and rendered as `OWNER:` / `MATCH:` lines, exactly as the
  red-flag prompt shows a chat.
- Wikipedia articles are cut at "See also" and chunked into ~400-word pieces.
- 3% is held out for validation.

The non-commercial licenses (DailyDialog, EmpatheticDialogues) fit a course project. They
are also why the corpus is rebuilt from source rather than committed or redistributed.
