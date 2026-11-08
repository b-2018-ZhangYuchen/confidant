"""Step 4 of the lab: continued pre-training and LoRA fine-tuning, through ``mlx_lm lora``.

Both runs are LoRA on a 4-bit base (QLoRA), driven by a YAML config committed under
``experiments/configs/`` so any result can be rerun from its file:

- **Continued pre-training** (``kind: cpt``): causal-LM loss on the corpus from
  :mod:`confidant.lab.corpus`. The adapter is then fused into the base, so a fine-tune
  can start from it.
- **Fine-tuning** (``kind: sft``): the train split as chats (redacted prompt, label as the
  answer), with the prompt masked out of the loss so the model learns to answer, not to
  repeat transcripts; the val split for validation loss.

:func:`perplexity` measures a model, with or without an adapter, on plain texts, for the
before-and-after of continued pre-training.
"""

from __future__ import annotations

import json
import math
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

__all__ = ["fuse", "parse_log", "perplexity", "run_lora", "write_sft_data"]

# Keys that belong to the lab, not to mlx_lm lora.
_LAB_KEYS = {"kind", "prompt", "name", "notes", "fuse_to"}


def write_sft_data(out: Path, prompt: str = "compact") -> dict[str, int]:
    """The train and val splits as chat records for ``mlx_lm lora``."""
    from confidant.lab.data import load_examples, to_chat_records
    from confidant.lab.prompts import PROMPTS

    builder = PROMPTS[prompt](None)
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    source = Path("lab/data/examples.jsonl")
    for split, name in (("train", "train"), ("val", "valid")):
        records = to_chat_records(load_examples(source, split), builder)
        with (out / f"{name}.jsonl").open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[name] = len(records)
    return counts


_ANSI = re.compile(r"\x1b\[[0-9;]*m")
# mlx-lm prints a table: "  20    1.823 ▼    126   16.7k" for training and
# "   1    val 1.947    76.69s" for validation. Older versions printed
# "Iter 20: Train loss 1.823"; both are read.
_TRAIN_ROW = re.compile(r"^\s*(\d+)\s+([\d.]+)\s*[▼▲=]?\s+\d+", re.M)
_VAL_ROW = re.compile(r"^\s*(\d+)\s+val\s+([\d.]+)", re.M)
_OLD = re.compile(r"Iter (\d+): (Train|Val) loss ([\d.]+)")


def parse_log(text: str) -> dict[str, list[tuple[int, float]]]:
    """Training and validation losses from an ``mlx_lm lora`` log."""
    text = _ANSI.sub("", text)
    losses: dict[str, list[tuple[int, float]]] = {"train": [], "val": []}
    for it, loss in _TRAIN_ROW.findall(text):
        losses["train"].append((int(it), float(loss)))
    for it, loss in _VAL_ROW.findall(text):
        losses["val"].append((int(it), float(loss)))
    for it, which, loss in _OLD.findall(text):
        losses[which.lower()].append((int(it), float(loss)))
    return losses


def run_lora(config_path: Path, log_dir: Path) -> dict[str, Any]:
    """Run ``mlx_lm lora`` on a lab config; returns the losses it reported."""
    import yaml

    config = yaml.safe_load(config_path.read_text())
    mlx_config = {k: v for k, v in config.items() if k not in _LAB_KEYS}
    mlx_config["train"] = True
    log_dir.mkdir(parents=True, exist_ok=True)
    resolved = log_dir / "mlx_config.yaml"
    resolved.write_text(yaml.safe_dump(mlx_config, sort_keys=False))

    started = time.perf_counter()
    with (log_dir / "train.log").open("w") as log:
        proc = subprocess.run(
            [sys.executable, "-m", "mlx_lm", "lora", "-c", str(resolved)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    text = (log_dir / "train.log").read_text()
    losses = parse_log(text)
    peak = re.findall(r"Peak mem ([\d.]+) GB", text)
    result = {
        "config": str(config_path),
        "returncode": proc.returncode,
        "minutes": (time.perf_counter() - started) / 60,
        "losses": losses,
        "final_val_loss": losses["val"][-1][1] if losses["val"] else None,
        "best_val_loss": min((v for _, v in losses["val"]), default=None),
        "peak_memory_gb": max(map(float, peak), default=None),
    }
    (log_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if proc.returncode != 0:
        raise RuntimeError(f"mlx_lm lora failed; see {log_dir / 'train.log'}")
    if config.get("fuse_to"):
        fuse(mlx_config["model"], mlx_config["adapter_path"], Path(config["fuse_to"]))
    return result


def fuse(model: str, adapter: str, out: Path) -> None:
    """Fold an adapter into its base, so another run can start from the result."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "mlx_lm",
            "fuse",
            "--model",
            model,
            "--adapter-path",
            adapter,
            "--save-path",
            str(out),
        ],
        check=True,
    )


def perplexity(
    model: str, texts: list[str], *, adapter: str | None = None, max_tokens: int = 1024
) -> float:
    """Token-level perplexity of ``model`` on ``texts``."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load

    m, tokenizer = load(model, adapter_path=adapter)
    total, count = 0.0, 0
    for text in texts:
        ids = tokenizer.encode(text)[:max_tokens]
        if len(ids) < 2:
            continue
        x = mx.array(ids)[None]
        logits = m(x[:, :-1])
        loss = nn.losses.cross_entropy(logits, x[:, 1:], reduction="sum")
        total += loss.item()
        count += len(ids) - 1
    return math.exp(total / count) if count else float("nan")
