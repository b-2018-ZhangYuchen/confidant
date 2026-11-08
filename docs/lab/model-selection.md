# Step 1 — Model selection

Which open model should Confidant adapt for the red-flag check? I compared seven
instruction-tuned models that run locally on the project's laptop (Apple M3 Pro, 18 GB of
unified memory) through `mlx-lm`, against Claude as the commercial reference.

## Candidates

All candidates are 4-bit community conversions (`mlx-community/*`). This keeps every model
small enough to leave room for a LoRA training run beside it. They cover three families
(Llama 3.2, Qwen 2.5 / Qwen 3, Mistral) and sizes from 1B to 7B parameters:

| Model | Params | Architecture notes | Context |
|---|---:|---|---:|
| Llama-3.2-1B-Instruct | 1.2B | dense decoder, GQA | 128k |
| Qwen2.5-1.5B-Instruct | 1.5B | dense decoder, GQA, tied embeddings | 32k |
| Llama-3.2-3B-Instruct | 3.2B | dense decoder, GQA | 128k |
| Qwen2.5-3B-Instruct | 3.1B | dense decoder, GQA, tied embeddings | 32k |
| Qwen3-4B-Instruct-2507 | 4.0B | dense decoder, GQA, non-thinking instruct release | 256k |
| Mistral-7B-Instruct-v0.3 | 7.2B | dense decoder, GQA, extended vocabulary | 32k |
| Qwen2.5-7B-Instruct | 7.6B | dense decoder, GQA | 32k |

Confidant's longest prompt (system prompt plus schema plus a long chat) is about 3k
tokens, so every candidate's context window is more than enough. Context length did not
decide anything.

## Method

`python -m confidant.lab models` loads each model once. It then runs four fictional
transcripts from `examples/` (none of them in the gold set) through the real red-flag
path: `analyze_flags` on an `MlxBackend`, with the full JSON-Schema prompt, greedy
decoding, and redaction, grounding, and severity floors all applied. Columns:

- **Load (s)**: wall time to load, *including the first download*, so it mostly
  measures my connection. It is not a property of the model.
- **Peak memory**: as reported by MLX during generation.
- **Tokens/s**: generation speed.
- **s per read**: wall time for one complete red-flag check.
- **Valid JSON**: answers that parsed into the flag schema.
- **Sane**: whether the check flagged the one transcript with planted pressure
  (`pressure_chat`) and left the three ordinary ones alone. This is a smoke test, not an
  accuracy score. Accuracy is measured on labelled data in step 3.

## Results

Raw numbers: [`experiments/model-selection/models.json`](../../experiments/model-selection/models.json).

| Model | Load (s) | Peak memory (GB) | Tokens/s | s per read | Valid JSON | Sane |
|---|---:|---:|---:|---:|---:|---:|
| `Llama-3.2-1B-Instruct-4bit` | 99.7 | 1.54 | 131 | 8.3 | 0/4 | 0/4 |
| `Qwen2.5-1.5B-Instruct-4bit` | 0.7 | 1.63 | 121 | 2.2 | 0/4 | 0/4 |
| `Llama-3.2-3B-Instruct-4bit` | 207.8 | 2.81 | 59 | 6.4 | 3/4 | 0/4 |
| `Qwen2.5-3B-Instruct-4bit` | 207.0 | 2.81 | 64 | 3.8 | 4/4 | 3/4 |
| `Qwen3-4B-Instruct-2507-4bit` | 330.9 | 3.37 | 47 | 8.2 | 4/4 | 4/4 |
| `Mistral-7B-Instruct-v0.3-4bit` | 792.3 | 5.07 | 30 | 16.8 | 4/4 | 1/4 |
| `Qwen2.5-7B-Instruct-4bit` | 2.5 | 5.11 | 32 | 9.2 | 4/4 | 3/4 |

What the failures looked like:

- **Below 3B, neither model could hold the schema.** Qwen2.5-1.5B answered with a bare
  list of flags instead of the `{flags, summary, confidence}` object. Llama-3.2-1B ran out
  of tokens before closing its JSON. Both were rejected as unusable, never read as "no
  flags".
- **Llama-3.2-3B** wrote evidence as plain strings instead of `{quote, why_it_matters}`
  objects once. When it did answer in the schema, it flagged the ordinary chats.
- **Mistral-7B** answered in the schema every time but flagged three ordinary chats.

## Choice

- **Student (the model that gets adapted): Qwen2.5-3B-Instruct.** It is the smallest model
  that answered in the schema every time, and the fastest of those that did (3.8 s per
  read, 2.8 GB). Speed matters because steps 3–7 run hundreds of reads and a dozen
  training runs on one laptop: at Qwen3-4B's 8.2 s per read, every evaluation takes twice
  as long. It also has the most headroom: it already follows the format, so what
  fine-tuning has to teach is judgment, not syntax.
- **Off-the-shelf reference: Qwen3-4B-Instruct-2507.** It is the best model here
  without any adaptation. It appears in the step 6 comparison as the answer to "why not
  just use a bigger model?"
- **Data generator: Qwen2.5-7B-Instruct.** It is the largest candidate, and it writes the
  synthetic transcripts in step 2. The labels come from the scenario grid, not from the
  generator, so its judgment does not leak into the labels. Its writing style does, and
  it shares a family with the student. That is a limitation of this dataset, noted in
  the datasheet.
- **Commercial reference: Claude**, through the existing `ClaudeBackend`. This needs an
  API key, which the project's machine does not have, so that row waits (see the
  roadmap).
