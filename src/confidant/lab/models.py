"""Step 1 of the lab: which local model to build on.

Each candidate is loaded once and given the same few fictional transcripts through the
real red-flag path (:func:`~confidant.analysis.flags.analyze_flags` on an
:class:`~confidant.lab.backend.MlxBackend`). What is recorded is what decides whether a
model is worth adapting on an 18 GB laptop: how much memory it takes, how fast it is, and
how often it answers in the schema at all. Whether its flags are *right* is the baseline's
job (``lab eval``), on labelled data; here two sanity checks stand in for it, so a model
that answers fast and in shape but sees nothing in an obvious case does not look good.
"""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from confidant.analysis.flags import analyze_flags
from confidant.ingest.transcript import read_transcript
from confidant.lab.backend import LocalModel, LocalOutputError, MlxBackend

__all__ = ["CANDIDATES", "TRANSCRIPTS", "ModelRow", "benchmark_model", "rows_to_markdown"]

# 4-bit community conversions, so every candidate fits in memory beside a training run.
CANDIDATES: tuple[str, ...] = (
    "mlx-community/Llama-3.2-1B-Instruct-4bit",
    "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
    "mlx-community/Llama-3.2-3B-Instruct-4bit",
    "mlx-community/Qwen2.5-3B-Instruct-4bit",
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
)

# Fictional chats outside the eval set. Expected: does a good check flag anything?
TRANSCRIPTS: dict[str, bool] = {
    "examples/pressure_chat.txt": True,
    "examples/sample_chat.txt": False,
    "examples/faded_chat.txt": False,
    "examples/details_chat.txt": False,
}


@dataclass
class ModelRow:
    model: str
    load_seconds: float = 0.0
    peak_memory_gb: float = 0.0
    generation_tps: float = 0.0
    seconds_per_read: float = 0.0
    valid: int = 0
    reads: int = 0
    sane: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def valid_rate(self) -> float:
        return self.valid / self.reads if self.reads else 0.0


def benchmark_model(
    model: str,
    transcripts: dict[str, bool] = TRANSCRIPTS,
    *,
    make_generate: Callable[[str], Callable] | None = None,
) -> ModelRow:
    """Load ``model`` and run every transcript through the red-flag check on it."""
    row = ModelRow(model=model)
    if make_generate is None:
        local = LocalModel(model, max_tokens=1536)
        started = time.perf_counter()
        local.load()
        row.load_seconds = time.perf_counter() - started
        generate = local
    else:
        generate = make_generate(model)
    backend = MlxBackend(model, generate=generate)

    seconds, speeds, peak = [], [], 0.0
    for path, expect_flags in transcripts.items():
        started = time.perf_counter()
        row.reads += 1
        try:
            report = analyze_flags(read_transcript(path), backend=backend)
        except LocalOutputError as exc:
            row.notes.append(f"{Path(path).name}: {exc}")
        else:
            row.valid += 1
            row.sane += bool(report.flags) == expect_flags
        seconds.append(time.perf_counter() - started)
        stats = backend.last_stats
        if stats is not None and stats.generation_tps:
            speeds.append(stats.generation_tps)
            peak = max(peak, stats.peak_memory_gb)

    row.seconds_per_read = sum(seconds) / len(seconds)
    row.generation_tps = sum(speeds) / len(speeds) if speeds else 0.0
    row.peak_memory_gb = peak
    del backend, generate
    gc.collect()
    return row


def rows_to_markdown(rows: Sequence[ModelRow]) -> str:
    lines = [
        "| Model | Load (s) | Peak memory (GB) | Tokens/s | s per read | Valid JSON | Sane |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r.model.split('/')[-1]}` | {r.load_seconds:.1f} | {r.peak_memory_gb:.2f} | "
            f"{r.generation_tps:.0f} | {r.seconds_per_read:.1f} | {r.valid}/{r.reads} | "
            f"{r.sane}/{r.reads} |"
        )
    return "\n".join(lines)


def save_rows(rows: Sequence[ModelRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(r) for r in rows], indent=2) + "\n")
