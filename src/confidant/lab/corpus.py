"""Step 2, the unlabelled half: a corpus for continued pre-training.

Continued pre-training asks whether a model that has read more everyday conversation, and
more about how abusive relationships work, reads a dating chat better. The corpus is two
kinds of text, all public, none of it a real person's private messages:

- **Dialogue**, redacted and rendered line by line exactly as the red-flag prompt renders
  a chat (``OWNER: ...`` / ``MATCH: ...``), so the model sees the shape it will be asked
  to read: DailyDialog (CC BY-NC-SA 4.0),
  EmpatheticDialogues (CC BY-NC 4.0), and Synthetic-Persona-Chat (CC BY 4.0).
- **Background**: Wikipedia articles on coercive control, intimate-partner abuse, and
  romance scams (CC BY-SA), as plain text.

Nothing downloaded is committed; :func:`build_corpus` writes ``train.jsonl`` and
``valid.jsonl`` (``{"text": ...}``, what ``mlx_lm lora`` reads) under a local directory
and returns counts for the datasheet. Dialogues are deduplicated, and every one is
passed through Confidant's redaction, which leaves the corpus as it would find a chat.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

__all__ = ["SOURCES", "WIKIPEDIA_TITLES", "build_corpus", "render_dialogue"]

SOURCES = {
    "dailydialog": ("li2017dailydialog/daily_dialog", "CC BY-NC-SA 4.0"),
    "empathetic": ("facebook/empathetic_dialogues", "CC BY-NC 4.0"),
    "persona": ("google/Synthetic-Persona-Chat", "CC BY 4.0"),
}
WIKIPEDIA_TITLES = (
    "Coercive control",
    "Intimate partner violence",
    "Domestic violence",
    "Romance scam",
    "Love bombing",
    "Emotional abuse",
    "Economic abuse",
    "Stalking",
    "Cyberstalking",
    "Sexual consent",
    "Sextortion",
    "Online dating",
    "Pig butchering scam",
)
# Wikimedia asks automated clients to say who they are and to go slowly.
USER_AGENT = "confidant-lab/0.1 (https://github.com/b-2018-ZhangYuchen/confidant)"
NAMES = ("Sam", "Robin", "Alex", "Casey", "Priya", "Theo", "Mei", "Kai", "Noor", "Jo")


def render_dialogue(turns: list[str], rng: random.Random) -> str | None:
    """A list of alternating turns as a two-person transcript, or None if too short."""
    turns = [re.sub(r"\s+", " ", t).strip() for t in turns if t and t.strip()]
    if len(turns) < 4:
        return None
    a, b = rng.sample(NAMES, 2)
    return "\n".join(f"{(a, b)[i % 2]}: {t}" for i, t in enumerate(turns))


def _fix_spacing(text: str) -> str:
    # DailyDialog is tokenized ("Hello , how are you ?"); undo the worst of it.
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    return re.sub(r"\s+'\s*", "'", text).replace(" n't", "n't")


def _parquet_rows(repo: str, split: str) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo, f"default/{split}/0000.parquet", repo_type="dataset", revision="refs/convert/parquet"
    )
    yield from pq.read_table(path).to_pylist()


def _dialogues(source: str, split: str) -> Iterator[list[str]]:
    repo, _ = SOURCES[source]
    if source == "dailydialog":
        for row in _parquet_rows(repo, split):
            yield [_fix_spacing(t) for t in row["dialog"]]
    elif source == "empathetic":
        convs: dict[str, list[tuple[int, str]]] = {}
        for row in _parquet_rows(repo, split):
            text = row["utterance"].replace("_comma_", ",")
            convs.setdefault(row["conv_id"], []).append((int(row["utterance_idx"]), text))
        for turns in convs.values():
            yield [t for _, t in sorted(turns)]
    elif source == "persona":
        for row in _parquet_rows(repo, split):
            lines = row["Best Generated Conversation"].splitlines()
            yield [re.sub(r"^User \d:\s*", "", line) for line in lines]


def _wikipedia(title: str) -> str | None:
    query = urllib.parse.urlencode(
        {
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
            "format": "json",
        }
    )
    request = urllib.request.Request(
        f"https://en.wikipedia.org/w/api.php?{query}",
        headers={"User-Agent": USER_AGENT},
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                pages = json.load(response)["query"]["pages"]
            break
        except urllib.error.HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            time.sleep(float(exc.headers.get("Retry-After") or 5 * 2**attempt))
    text = next(iter(pages.values())).get("extract")
    if not text:
        return None
    # Drop the reference-ish tail sections; keep the prose.
    text = re.split(r"\n==+ (See also|References|Further reading|External links) ==+", text)[0]
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _chunks(text: str, words: int = 400) -> Iterable[str]:
    paragraphs, current = text.split("\n\n"), []
    for p in paragraphs:
        current.append(p)
        if sum(len(c.split()) for c in current) >= words:
            yield "\n\n".join(current)
            current = []
    if current:
        yield "\n\n".join(current)


def build_corpus(
    out: Path,
    *,
    max_dialogues: int = 6000,
    seed: int = 0,
    valid_fraction: float = 0.03,
) -> dict[str, Any]:
    """Download, clean, and write the corpus; returns the counts."""
    from confidant.ingest.transcript import parse_transcript
    from confidant.prompts.common import render_line
    from confidant.redaction import redact

    rng = random.Random(seed)
    seen: set[str] = set()
    docs: list[tuple[str, str]] = []
    counts: Counter[str] = Counter()
    per_source = max_dialogues // len(SOURCES)
    for source in SOURCES:
        rendered = []
        for turns in _dialogues(source, "train"):
            text = render_dialogue(turns, rng)
            if text is None:
                counts[f"{source}:too_short"] += 1
                continue
            key = hashlib.sha1(text.split("\n", 1)[-1].casefold().encode()).hexdigest()
            if key in seen:
                counts[f"{source}:duplicate"] += 1
                continue
            seen.add(key)
            rendered.append(text)
        rng.shuffle(rendered)
        for text in rendered[:per_source]:
            a, b = (line.split(":", 1)[0] for line in text.splitlines()[:2])
            conversation = parse_transcript(f"# owner: {a}\n# match: {b}\n{text}\n")
            redacted = redact(conversation).conversation
            body = "\n".join(render_line(m) for m in redacted.messages)
            docs.append((source, body))
            counts[source] += 1

    for title in WIKIPEDIA_TITLES:
        time.sleep(1.5)
        text = _wikipedia(title)
        if text is None:
            counts["wikipedia:missing"] += 1
            continue
        for chunk in _chunks(text):
            docs.append(("wikipedia", chunk))
            counts["wikipedia"] += 1

    rng.shuffle(docs)
    n_valid = max(1, int(len(docs) * valid_fraction))
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("valid", docs[:n_valid]), ("train", docs[n_valid:])):
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for _, text in part:
                f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
    words = sum(len(t.split()) for _, t in docs)
    return {
        "documents": len(docs),
        "train": len(docs) - n_valid,
        "valid": n_valid,
        "words": words,
        "counts": dict(sorted(counts.items())),
        "licenses": {k: v[1] for k, v in SOURCES.items()} | {"wikipedia": "CC BY-SA 4.0"},
    }
