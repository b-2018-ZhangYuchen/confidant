"""Retrieval for the ``rag`` prompt: hand-written pattern cards, picked per transcript.

The cards in ``lab/cards/`` describe each red-flag category: what it looks like, what
it is not, and how it is filed. They are written to ``docs/principles.md`` (behavior, not
labels). Retrieval looks only at the match's own lines, since those are what a flag
quotes, and returns the ``k`` most relevant category cards. The "what is not a flag"
card (``none.md``) is not retrieved: the prompt always carries it, so retrieval can only
add context about a pattern, never take away the reminder that most chats are fine.

Two retrievers are compared:

- ``bm25`` — word overlap between all of the match's lines and each card, no model.
- ``bm25max`` — the same, but each line is scored on its own and a card keeps its best
  line, as the embedding retriever does.
- ``embed`` — sentence embeddings (``mlx-embeddings``; MiniLM by default). Each card is
  embedded once; each match message is embedded, and a card scores by its best match
  with any one message, so a single worrying line in a long chat can still pull its card.

:func:`retrieval_recall` scores a retriever on labelled examples without running the
flag model: how often the gold category's card is in the top ``k`` on flagged examples.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

__all__ = ["CARDS_DIR", "Card", "CardIndex", "PassageIndex", "retrieval_recall"]

CARDS_DIR = Path("lab/cards")
DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"
_WORD = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class Card:
    key: str
    text: str


def load_cards(directory: Path = CARDS_DIR) -> list[Card]:
    return [
        Card(p.stem, p.read_text(encoding="utf-8").strip()) for p in sorted(directory.glob("*.md"))
    ]


def _words(text: str) -> list[str]:
    return _WORD.findall(text.casefold())


class _BM25:
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = [Counter(_words(d)) for d in docs]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / len(self.lengths)
        df = Counter(w for d in self.docs for w in d)
        n = len(docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        q = _words(query)
        out = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            s = 0.0
            for w in q:
                tf = doc.get(w, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg)
                    s += self.idf[w] * tf * (self.k1 + 1) / norm
            out.append(s)
        return out


class _Embedder:
    def __init__(self, model: str) -> None:
        from mlx_embeddings.utils import load

        self.model, self.tokenizer = load(model)

    def __call__(self, texts: Sequence[str], batch: int = 64):
        import mlx.core as mx

        parts = []
        for start in range(0, len(texts), batch):
            inputs = self.tokenizer.batch_encode_plus(
                list(texts[start : start + batch]),
                return_tensors="mlx",
                padding=True,
                truncation=True,
                max_length=256,
            )
            out = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
            e = out.text_embeds
            parts.append(e / mx.linalg.norm(e, axis=-1, keepdims=True))
        return mx.concatenate(parts, axis=0)


class CardIndex:
    def __init__(
        self, cards: list[Card], retriever: str = "bm25", embedder: str = DEFAULT_EMBEDDER
    ) -> None:
        self.cards = cards
        self.retriever = retriever
        if retriever in ("bm25", "bm25max"):
            self._bm25 = _BM25([c.text for c in cards])
        elif retriever == "embed" or retriever.startswith("embed:"):
            name = retriever.split(":", 1)[1] if ":" in retriever else embedder
            self._embed = _Embedder(name)
            self._card_vectors = self._embed([c.text for c in cards])
        else:
            raise ValueError(
                f"unknown retriever {retriever!r}: use bm25, bm25max, embed, or embed:<model>"
            )

    @classmethod
    def load(cls, directory: Path = CARDS_DIR, retriever: str = "bm25") -> CardIndex:
        return cls([c for c in load_cards(directory) if c.key != "none"], retriever)

    def search(self, match_lines: Sequence[str], k: int = 2) -> list[Card]:
        """The ``k`` cards most relevant to the match's messages."""
        if self.retriever == "bm25":
            scores = self._bm25.scores(" ".join(match_lines))
        elif self.retriever == "bm25max":
            per_line = [self._bm25.scores(line) for line in match_lines] or [
                [0.0] * len(self.cards)
            ]
            scores = [max(col) for col in zip(*per_line, strict=True)]
        else:
            vectors = self._embed(list(match_lines) or [""])
            sims = vectors @ self._card_vectors.T  # messages x cards
            scores = sims.max(axis=0).tolist()
        order = sorted(range(len(self.cards)), key=lambda i: -scores[i])
        return [self.cards[i] for i in order[:k]]


def retrieval_recall(index: CardIndex, examples: Sequence, k: int) -> dict[str, float]:
    """How often the gold category's card is among the top ``k``, on flagged examples."""
    flagged = [e for e in examples if e.kind == "flag"]
    hits = 0
    for e in flagged:
        lines = [m.text for m in e.conversation().match_messages]
        hits += e.category in {c.key for c in index.search(lines, k)}
    return {"k": k, "recall": hits / len(flagged) if flagged else 0.0, "n": len(flagged)}


class PassageIndex:
    """Retrieval over the relationship-guidance corpus (:mod:`confidant.lab.guidance`).

    The same two retrievers as the cards, over hundreds of passages instead of eleven:
    ``bm25max`` scores each of the match's lines against every passage and keeps a
    passage's best line; ``embed`` does the same with sentence embeddings. Lines under
    four words ("ok", "lol") are left out of the query, since against a large corpus
    they match nothing in particular and only add noise. Results are deduplicated by
    source page, so the ``k`` passages come from ``k`` different pages.
    """

    def __init__(self, passages: Sequence, retriever: str = "bm25max") -> None:
        self.passages = list(passages)
        self.retriever = retriever
        texts = [f"{p.title}. {p.text}" for p in self.passages]
        if retriever == "bm25max":
            self._bm25 = _BM25(texts)
        elif retriever == "embed" or retriever.startswith("embed:"):
            name = retriever.split(":", 1)[1] if ":" in retriever else DEFAULT_EMBEDDER
            self._embed = _Embedder(name)
            self._vectors = self._embed(texts)
        else:
            raise ValueError(f"unknown retriever {retriever!r} for passages: bm25max or embed")

    @classmethod
    def load(
        cls, chunking: str = "window", retriever: str = "bm25max", pool: str = "all"
    ) -> PassageIndex:
        from confidant.lab.guidance import load_passages

        return cls(load_passages(chunking, pool=pool), retriever)

    def search(self, match_lines: Sequence[str], k: int = 3) -> list:
        lines = [line for line in match_lines if len(line.split()) >= 4] or list(match_lines)
        if not lines:
            return []
        if self.retriever == "bm25max":
            per_line = [self._bm25.scores(line) for line in lines]
            scores = [max(col) for col in zip(*per_line, strict=True)]
        else:
            sims = self._embed(lines) @ self._vectors.T
            scores = sims.max(axis=0).tolist()
        found, pages = [], set()
        for i in sorted(range(len(self.passages)), key=lambda i: -scores[i]):
            page = self.passages[i].url
            if page in pages:
                continue
            pages.add(page)
            found.append(self.passages[i])
            if len(found) == k:
                break
        return found
