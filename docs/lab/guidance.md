# RAG over relationship guidance

The first RAG design retrieved from 11 short pattern cards I wrote for the project. That
is barely retrieval: picking 2 of 11 is a classification, not a search. The cards were
also my own words, not anyone's published guidance. This page replaces them, and then
adds to them, with a real corpus: **relationship guidance written for people who are
dating**. It measures what that does on the synthetic splits and on both outside test
sets.

## The corpus (`python -m confidant.lab guidance`)

| Source | License | Pages |
|---|---|---:|
| womenshealth.gov (relationships and safety) | public domain (US government) | 9 |
| youth.gov (healthy and unhealthy relationship characteristics, dating violence) | public domain | 3 |
| CDC (teen dating violence, intimate partner violence) | public domain | 2 |
| HHS Office of Population Affairs (healthy relationships in adolescence) | public domain | 1 |
| Wikipedia: Intimate relationship, Dating, Online dating, Courtship, Personal boundaries, Sexual consent, Ghosting, Jealousy, Active listening, Interpersonal communication, Interpersonal relationship, Relationship maintenance, Conflict resolution, Trust, Breakup, Social rejection, Text messaging, Infatuation, Love bombing, Romance scam, Coercive control, Emotional abuse, Stalking, Dating violence | CC BY-SA 4.0 | 24 |

That is 39 pages and about 116k words, cut two ways for the chunking experiment:
1,937 **paragraphs**, or 737 **windows** of about 200 words that never split a paragraph.

- **Filtering:** 34 passages that used a diagnostic label ("narcissist", "attachment
  style") were dropped, because Confidant's principles rule them out and retrieved text is
  the easiest way for one to reach a report. Citation lines were dropped too.
- **Provenance:** every passage keeps its source, URL, and license.
- **Storage:** the corpus is rebuilt from the source list
  ([`guidance.py`](../../src/confidant/lab/guidance.py)) into `lab-cache/`, not committed.
- **wikiHow was planned and dropped.** It blocks Anthropic's crawler. I took that as the
  site owner's answer for any AI-driven collection, whatever user agent a fetch uses.
- **Skewed toward abuse.** Every government page is about abuse or violence; that is
  what US health agencies publish on relationships. So 884 of the 1,937 paragraphs
  (46%) are about abuse, and 1,053 about ordinary relationships.

## Retrieval

- **Query:** only the match's messages, as before. Lines under four words are skipped.
- **Scoring:** each passage keeps its best score against any single line.
- **Results:** the top k passages, from k different pages.
- **Prompt:** the passages go in after the "not a flag" card, under a note to use them
  only to judge what is ordinary, and to flag only what the match's own messages show.
- **Model:** every run uses the fine-tuned model (SFT iter 300, quote repair, greedy).

### Step 5 grid on val: chunking × retriever × k

| Chunking | Retriever | k | F1 | P | R | FA hard | Cat F1 | Danger missed |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| *no RAG* | | | 0.81 | 0.96 | 0.70 | 7% | 0.42 | 10/18 |
| *cards* | BM25, best line | 2 | 0.77 | 0.92 | 0.67 | 13% | **0.56** | 6/18 |
| window | BM25 | 2 | 0.74 | 0.95 | 0.61 | 7% | 0.48 | 10/18 |
| window | BM25 | 4 | 0.68 | 1.00 | 0.52 | 0% | 0.44 | 11/18 |
| window | MiniLM | 2 | 0.79 | 0.96 | 0.67 | 7% | 0.43 | 8/18 |
| window | MiniLM | 4 | 0.75 | 1.00 | 0.61 | 0% | 0.45 | 9/18 |
| paragraph | BM25 | 2 | 0.74 | 0.95 | 0.61 | 7% | 0.44 | 10/18 |
| paragraph | BM25 | 4 | 0.76 | 0.95 | 0.64 | 7% | 0.47 | 10/18 |
| paragraph | MiniLM | 2 | 0.76 | 0.95 | 0.64 | 7% | 0.44 | 8/18 |
| **paragraph** | **MiniLM** | **4** | **0.80** | **1.00** | 0.67 | **0%** | 0.44 | 8/18 |
| *cards + paragraph / MiniLM / 4* | | | 0.79 | 0.96 | 0.67 | 7% | 0.50 | **5/18** |

- **The retriever mattered most.** Embeddings beat BM25 in three of four pairings. BM25
  matches surface words, so a harmless chat can still pull domestic-violence pages; the
  spot check showed it doing exactly that for a clean transcript about books.
- **More guidance made the model more cautious.** k=4 raised precision (to 1.00 in two
  settings) and lowered recall. It works as a dial between catching more and
  false-alarming less.
- **Chunking mattered least.** Paragraphs and windows were within noise at k=2. At k=4,
  paragraphs were better: four windows is about 800 words of background, four paragraphs
  about 250.
- **Chosen on val F1:** paragraphs, MiniLM, k=4.

## Test split and the outside sets

| | Synthetic test | | | MentalManip | | | ReaMent | | |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| System | F1 | Danger missed | Gold | F1 | False alarms | Cat. agree | F1 | False alarms | at danger |
| Fine-tune, no RAG | 0.78 | 4/15 | **92%** | **0.74** | 58% | 0.39 | 0.67 | 87% | 12/60 |
| + cards | 0.75 | 4/15 | 89% | **0.74** | **52%** | **0.47** | 0.70 | 70% | **10/60** |
| + guidance | **0.79** | **2/15** | 86% | 0.72 | 58% | 0.39 | 0.69 | 78% | 15/60 |
| + cards + guidance | — | — | — | — | — | — | **0.71** | **68%** | 13/60 |
| + guidance, ordinary passages only | — | — | — | — | — | — | **0.71** | 77% | 18/60 |
| *Qwen3-4B as released, for reference* | 0.42 | 14/15 | 83% | 0.66 | 42% | 0.50 | 0.65 | 32% | 3/60 |

## What it shows

1. **On the synthetic test, guidance gives the best danger recall of any system.** Only
   2 of 15 danger cases were missed, against 4 without it, with no hard-negative false
   alarms and the best F1 (0.79). On val, cards plus guidance missed only 5 of 18 danger
   cases.
2. **On outside data, guidance did not fix false alarms, and the cards did more.** On
   ReaMent, guidance cut false alarms from 87% to 78%, where the cards reached 70% and the
   combination 68%. On MentalManip it changed nothing. Danger-tier false alarms on
   ReaMent *rose* with guidance (12 → 15).
3. **The obvious explanation was tested, and it was wrong.**
   - *The hypothesis:* nearly half the corpus is about abuse. For ordinary ReaMent
     conversations, 58% of the passages retrieved were abuse-focused. Reads that came back
     with a danger flag had retrieved more of them (70%) than reads that did not (55%).
     So I guessed the abuse material was priming danger flags.
   - *The test:* retrieve from the 1,053 ordinary-relationship passages only, with no
     abuse material at all.
   - *The result:* danger-tier false alarms went **up**, to 18.
   - *What it means:* the correlation was confounded. A tense exchange both retrieves
     abuse pages and draws a flag, and neither causes the other. What the data does
     support: **any** retrieved guidance pushes this fine-tuned model toward more
     danger-tier flags on real speech. A likely reason is that it was fine-tuned on
     prompts with no retrieved text, so a longer, unfamiliar prompt moves it off what it
     learned. Fine-tuning *with* retrieved passages in the training prompts is the
     experiment that would test this.
4. **Gold-set checks fell slightly with each layer of context** (92% → 89% → 86%, one
   check each). The checks lost were on two of the danger cases: the investment pitch
   (with cards and with guidance) and the "if you leave" case (with guidance). The
   gentle cases all still passed. The gold record keeps per-case counts, not which check
   failed, so the exact miss is not recorded. With 36 checks, one check is within noise.

## Recommendation

For the synthetic-test numbers and danger recall, guidance (or cards plus guidance) is the
best configuration. For real conversations, the measurements favor the cards. Either way,
no local configuration here matches Qwen3-4B's calibration on real speech (3 danger false
alarms). The next experiment is to fine-tune with retrieval in the loop, so the model
learns to read guidance rather than react to it. It needs training prompts that include
retrieved passages and is about a 45-minute run.
