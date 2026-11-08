"""``python -m confidant.lab <command>``: the lab's experiments, one subcommand each."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

EXAMPLES = Path("lab/data/examples.jsonl")


def _say(text: str) -> None:
    print(text, file=sys.stderr, flush=True)


def _models(args: argparse.Namespace) -> int:
    from confidant.lab.models import CANDIDATES, benchmark_model, rows_to_markdown, save_rows

    rows = []
    for model in args.model or CANDIDATES:
        _say(f"benchmarking {model}")
        rows.append(benchmark_model(model))
        save_rows(rows, Path(args.out) / "models.json")
    print(rows_to_markdown(rows))
    return 0


def _check(args: argparse.Namespace) -> int:
    from confidant.lab.data import contamination, describe, load_examples, load_scenarios

    scenarios = load_scenarios()
    print(f"{len(scenarios)} scenarios, valid, none overlapping the gold set")
    if EXAMPLES.exists():
        examples = load_examples(EXAMPLES)
        hits = contamination(examples)
        print(describe(examples))
        if hits:
            print("contaminated:\n" + "\n".join(hits))
            return 1
        print(f"{len(examples)} examples, no message repeats the gold set")
    return 0


def _generate(args: argparse.Namespace) -> int:
    import mlx.core as mx

    from confidant.lab.backend import LocalModel
    from confidant.lab.data import (
        assign_splits,
        contamination,
        describe,
        generate_examples,
        load_scenarios,
        save_examples,
    )

    mx.random.seed(args.seed)  # sampling at temperature 0.8 is then repeatable
    scenarios = load_scenarios()
    if args.only:
        scenarios = [s for s in scenarios if s.id in args.only]
    model = LocalModel(args.model, max_tokens=900, temperature=0.8, top_p=0.95)

    def progress(example, report) -> None:
        name = example.id if example else "(dropped)"
        _say(f"{report.kept} kept / {report.attempts} attempts  {name}")

    examples, report = generate_examples(
        scenarios, model, per_scenario=args.per_scenario, seed=args.seed, on_example=progress
    )
    split_of = assign_splits(examples, load_scenarios(), seed=args.seed)
    save_examples(examples, EXAMPLES)
    summary = {
        "generator": args.model,
        "seed": args.seed,
        "per_scenario": args.per_scenario,
        "report": asdict(report),
        "splits": split_of,
        "contamination": contamination(examples),
    }
    Path("lab/data/generation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(describe(examples))
    print(json.dumps(asdict(report)))
    return 0


def _corpus(args: argparse.Namespace) -> int:
    from confidant.lab.corpus import build_corpus

    stats = build_corpus(Path(args.out), max_dialogues=args.max_dialogues, seed=args.seed)
    out = Path("experiments/corpus/stats.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0


def _backend(args: argparse.Namespace):
    from confidant.lab.backend import MlxBackend
    from confidant.lab.prompts import PROMPTS

    return MlxBackend(
        args.model,
        adapter=args.adapter,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        messages=PROMPTS[args.prompt](args),
    )


def _eval(args: argparse.Namespace) -> int:
    from confidant.lab.data import load_examples
    from confidant.lab.evaluate import evaluate, evaluate_gold, save_run, score, scores_table

    backend = _backend(args)
    examples = []
    for split in args.split:
        examples += load_examples(EXAMPLES, split)
    if args.limit:
        examples = examples[: args.limit]

    def progress(read) -> None:
        mark = "invalid" if not read.valid else (", ".join(c for c, _ in read.found) or "none")
        _say(f"{read.id:<22} {read.seconds:5.1f}s  {mark}")

    reads = evaluate(backend, examples, on_read=progress)
    scores = score(reads)
    gold = evaluate_gold(backend) if args.gold else None
    if gold is not None:
        scores.gold_pass = gold["pass_rate"]
    config = {k: v for k, v in vars(args).items() if k not in ("run", "argv")}
    config["argv"] = args.argv
    save_run(Path(args.out), config=config, reads=reads, scores=scores, gold=gold)
    print(scores_table([(Path(args.out).name, scores.row())]))
    return 0


def _train(args: argparse.Namespace) -> int:
    import yaml

    from confidant.lab.train import run_lora, write_sft_data

    config_path = Path(args.config)
    config = yaml.safe_load(config_path.read_text())
    if config.get("kind") == "sft":
        counts = write_sft_data(Path(config["data"]), config.get("prompt", "compact"))
        _say(f"wrote fine-tuning data: {counts}")
    out = Path("experiments") / "runs" / config.get("name", config_path.stem)
    result = run_lora(config_path, out)
    print(json.dumps({k: v for k, v in result.items() if k != "losses"}, indent=2))
    return 0


def _ppl(args: argparse.Namespace) -> int:
    from confidant.ingest.transcript import parse_transcript
    from confidant.lab.data import load_examples
    from confidant.lab.train import perplexity
    from confidant.prompts.common import render_line
    from confidant.redaction import redact

    texts = {}
    corpus = Path(args.corpus) / "valid.jsonl"
    texts["dialogue"] = [json.loads(line)["text"] for line in corpus.read_text().splitlines()]
    texts["dialogue"] = texts["dialogue"][: args.limit]
    val = load_examples(EXAMPLES, "val")[: args.limit]
    texts["transcripts"] = [
        "\n".join(render_line(m) for m in redact(parse_transcript(e.transcript)).conversation)
        for e in val
    ]
    result = {
        "model": args.model,
        "adapter": args.adapter,
        **{k: perplexity(args.model, v, adapter=args.adapter) for k, v in texts.items()},
    }
    print(json.dumps(result))
    if args.out:
        out = Path(args.out)
        rows = json.loads(out.read_text()) if out.exists() else []
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps([*rows, result], indent=2) + "\n")
    return 0


def _retrieval(args: argparse.Namespace) -> int:
    from confidant.lab.data import load_examples
    from confidant.lab.rag import CardIndex, retrieval_recall

    examples = load_examples(EXAMPLES, args.split)
    rows = []
    for retriever in args.retriever:
        index = CardIndex.load(retriever=retriever)
        for k in args.k:
            rows.append({"retriever": retriever, **retrieval_recall(index, examples, k)})
            _say(json.dumps(rows[-1]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2) + "\n")
    return 0


def _reproduce(args: argparse.Namespace) -> int:
    """Rerun a saved experiment from what it recorded about itself."""
    path = Path(args.run)
    if (path / "config.json").exists():
        argv = list(json.loads((path / "config.json").read_text())["argv"])
        # Write beside the original, so the two can be compared rather than overwritten.
        if "--out" in argv:
            argv[argv.index("--out") + 1] = str(path) + "-rerun"
    elif (path / "result.json").exists():
        argv = ["train", json.loads((path / "result.json").read_text())["config"]]
    else:
        _say(f"{path} has no config.json (an eval) or result.json (a training run)")
        return 2
    _say("rerunning: python -m confidant.lab " + " ".join(argv))
    return main(argv)


def _add_backend_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--prompt", default="schema", help="prompt variant (confidant.lab.prompts)")
    p.add_argument("--shots", type=int, default=2, help="examples in the few-shot prompt")
    p.add_argument("--cards", type=int, default=2, help="cards retrieved for the rag prompt")
    p.add_argument("--retriever", default="bm25", help="bm25 or embed, for the rag prompt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m confidant.lab")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("models", help="step 1: benchmark candidate local models")
    p.add_argument("--model", action="append", help="a model to try (repeatable)")
    p.add_argument("--out", default="experiments/model-selection")
    p.set_defaults(run=_models)

    p = sub.add_parser("check", help="validate the scenarios and the generated data")
    p.set_defaults(run=_check)

    p = sub.add_parser("generate", help="step 2: write the labelled transcripts")
    p.add_argument("--model", default="mlx-community/Qwen2.5-7B-Instruct-4bit")
    p.add_argument("--per-scenario", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", action="append", help="just this scenario id (repeatable)")
    p.set_defaults(run=_generate)

    p = sub.add_parser("corpus", help="step 2: build the continued-pre-training corpus")
    p.add_argument("--out", default="lab-cache/corpus")
    p.add_argument("--max-dialogues", type=int, default=6000)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(run=_corpus)

    p = sub.add_parser("eval", help="steps 3 and 6: score a backend on the splits")
    _add_backend_args(p)
    p.add_argument("--split", action="append", default=None, help="train, val, or test")
    p.add_argument("--gold", action="store_true", help="also score the gold set")
    p.add_argument("--limit", type=int)
    p.add_argument("--out", required=True)
    p.set_defaults(run=_eval)

    p = sub.add_parser("train", help="step 4: continued pre-training or a LoRA fine-tune")
    p.add_argument("config", help="a YAML file under experiments/configs/")
    p.set_defaults(run=_train)

    p = sub.add_parser("ppl", help="step 4: perplexity on dialogue and on val transcripts")
    p.add_argument("--model", required=True)
    p.add_argument("--adapter")
    p.add_argument("--corpus", default="lab-cache/corpus")
    p.add_argument("--limit", type=int, default=150)
    p.add_argument("--out")
    p.set_defaults(run=_ppl)

    p = sub.add_parser("retrieval", help="step 5: retrieval recall of the pattern cards")
    p.add_argument("--retriever", action="append", default=None)
    p.add_argument("--k", type=int, action="append", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--out", default="experiments/retrieval/recall.json")
    p.set_defaults(run=_retrieval)

    p = sub.add_parser("reproduce", help="step 8: rerun a saved experiment from its record")
    p.add_argument("run", help="an experiments/ directory")
    p.set_defaults(run=_reproduce)

    args = parser.parse_args(argv)
    args.argv = list(argv) if argv is not None else sys.argv[1:]
    if getattr(args, "retriever", "x") is None:
        args.retriever = ["bm25", "embed"]
    if getattr(args, "k", "x") is None:
        args.k = [1, 2, 3]
    if getattr(args, "split", "x") is None:
        args.split = ["val"]
    return args.run(args)


if __name__ == "__main__":
    raise SystemExit(main())
