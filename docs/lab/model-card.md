# Model card — Confidant red-flag adapter (Qwen2.5-3B, LoRA, iter 300)

## What it is

A LoRA adapter (rank 8, 16 layers, 6.7M parameters, 0.2% of the base) for
`mlx-community/Qwen2.5-3B-Instruct-4bit`. It turns the model into Confidant's red-flag
check: given a redacted dating-app conversation and Confidant's red-flag prompt, it
answers with a JSON flag report (category, severity, behavior, verbatim quotes, innocent
reading).

The adapter is not published. It is rebuilt from
[`experiments/configs/sft.yaml`](../../experiments/configs/sft.yaml), using the iter-300
checkpoint (see [experiments.md](experiments.md) for why not the final one).

## Intended use

- Confidant's local backend (`MlxBackend` with the compact prompt and quote repair) for
  people who want the red-flag check without sending a transcript anywhere.
- **Always behind `analyze_flags`**: redaction before, quote grounding and danger-tier
  floors after. The adapter's raw output is not meant to reach anyone.

## Not intended for

- Any decision about a person. Like the rest of Confidant, it is a second opinion about
  messages, not a judgment about people (`docs/principles.md`).
- Being the only safety check. It misses about a quarter of the danger-tier cases in
  testing (below). Claude remains Confidant's default backend.
- Languages other than English, group chats, or anything but two-person dating chats.

## Training data

180 synthetic, fictional transcripts (99 with a planted flag across 11 categories, 45
hard negatives, 36 clean). They were written by Qwen2.5-7B-Instruct from a hand-written
scenario grid, with labels taken from the grid. No real messages were used. Full
datasheet: [data.md](data.md).

## Evaluation (test split: 60 unseen-scenario transcripts; and 9 hand-written gold cases)

| | Without RAG | With RAG |
|---|---:|---:|
| Detection F1 | 0.78 | 0.75 |
| Precision / recall | 0.88 / 0.70 | 0.91 / 0.64 |
| False alarms: hard negatives / clean chats | 7% / 17% | 0% / 17% |
| Category F1 | 0.34 | 0.46 |
| Danger-tier transcripts not flagged as danger | 4 of 15 | 4 of 15 |
| Gold-set checks passed | 92% | 89% |
| Diagnostic labels used | 0 | 0 |
| Seconds per read (M3 Pro) / peak memory | 3.9 s / 2.7 GB | 4.5 s / 2.7 GB |

The base model with the same prompt scored F1 0.00 and flagged none of the danger cases.
Details: [results.md](results.md).

## Known failure modes

- **Danger filed under a non-danger category.** For example, a threat filed as
  "evasion" at watch tier. The danger floor applies by category, so this is the one way
  a caught danger item reaches the user as a mild note.
- **Over-generalizing training wording.** "I promise to share a secret with you" was
  flagged as a threat at danger tier, because a training scenario threatened to share
  private photos.
- **Implicature.** "You know what that means" was missed every time.
- **Near-quotes.** It sometimes changes a word in a quote. Grounding then discards the
  flag, which costs the detection but never shows a misquote.
- **Stitching.** Without quote repair, it sometimes quotes two consecutive messages as
  one. `split_stitched` fixes the exact cases.

## Ethical considerations

- The training data includes fictional threats, coercion, and sexual pressure, written
  so the model can recognize them. Nothing in it describes real people.
- A false alarm can talk someone out of a good relationship. A miss can leave someone
  reassured about a dangerous one. Both happen at the rates above, which is why this is
  an option beside Claude, not a replacement for it.
- Synthetic training data from a single generator may make the model better at
  synthetic chats than real ones. Its 92% on the human-written gold set is a small,
  encouraging check, not a guarantee.

## Reproducing

```bash
pip install -e ".[lab]"
python -m confidant.lab generate          # or use the committed lab/data/examples.jsonl
python -m confidant.lab train experiments/configs/sft.yaml
python -m confidant.lab reproduce experiments/eval/test/sft
```

Training takes about 42 minutes on an M3 Pro. Generation and evaluation are greedy or
seeded. Rerunning the test evaluation from its record reproduced all 60 answers word for word
and the same scores (`experiments/eval/test/sft-rerun`).
