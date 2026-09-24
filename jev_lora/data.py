"""Pinned PorTAL data import. Task labels and gold answers never enter routing inputs."""
import hashlib
import random
from collections import Counter, defaultdict
from pathlib import Path

from .core import (DATASET_ID, DATASET_REVISION, DESCRIPTIONS, TASKS, canonical_text,
                   digest, ensure_metadata, read_json, write_json, write_jsonl)


def routing_text(prompt, choices):
    return prompt + "\n\nCandidate continuations:\n" + "\n".join(
        f"[{i + 1}] {choice.lstrip()}" for i, choice in enumerate(choices))


def query_hash(row):
    return digest({"prompt": row["prompt"], "choices": row["choices"]})


def validate_row(row, prepared=False):
    if row.get("task") not in TASKS or not isinstance(row.get("prompt"), str):
        raise ValueError("Expected a PorTAL task and prompt; legacy LoraRetriever data are incompatible")
    choices = row.get("choices")
    if not isinstance(choices, list) or len(choices) < 2:
        raise ValueError("Each question needs at least two candidate continuations")
    if any(not isinstance(c, str) or not c.strip() or not c.startswith(" ") or c.startswith("  ") for c in choices):
        raise ValueError("PorTAL choices must have exactly one leading space and non-empty content")
    if row["prompt"] != row["prompt"].rstrip():
        raise ValueError("PorTAL prompts must not have trailing whitespace")
    gold = row.get("gold_idx")
    if isinstance(gold, bool) or not isinstance(gold, int) or not 0 <= gold < len(choices):
        raise ValueError("Invalid gold_idx")
    if prepared and (row.get("input_hash") != query_hash(row) or
                     row.get("inputs") != routing_text(row["prompt"], choices)):
        raise ValueError("Prepared input hash/routing text mismatch; rerun prepare")


def load_source(args):
    if args.source_json:
        path = Path(args.source_json)
        return read_json(path), {"kind": "local", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    from datasets import load_dataset
    raw_path = Path(args.raw_dir) / "portal_tasks.json"
    origin = {"kind": "huggingface", "dataset": DATASET_ID, "revision": DATASET_REVISION}
    if raw_path.exists():
        meta = read_json(str(raw_path) + ".meta.json")
        if meta["source"] != origin or meta["sha256"] != hashlib.sha256(raw_path.read_bytes()).hexdigest():
            raise ValueError("Raw PorTAL cache provenance mismatch")
        return read_json(raw_path), origin
    dataset = load_dataset(DATASET_ID, revision=DATASET_REVISION, cache_dir=args.hf_cache)
    source = {split: [dict(row) for row in dataset[split]] for split in ("train", "validation")}
    if (len(source["train"]), len(source["validation"])) != (129212, 19548):
        raise ValueError("Pinned PorTAL dataset has unexpected split sizes")
    write_json(raw_path, source)
    write_json(str(raw_path) + ".meta.json", {"source": origin, "sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()})
    return source, origin


def prepare(source, out_dir, seed=42, dev_per_task=5, shuffle_choices=True, provenance=None):
    if dev_per_task < 1:
        raise ValueError("Invalid development split size")
    out_dir = Path(out_dir)
    for split in ("train", "validation"):
        if not source.get(split):
            raise ValueError(f"Missing {split} data")
        for row in source[split]:
            validate_row(row)
        if {row["task"] for row in source[split]} != set(TASKS):
            raise ValueError(f"{split} must cover all 14 PorTAL tasks")
    settings = {"schema": "portal-paper-data-v1", "source": provenance or {"kind": "in-memory"},
                "source_hash": digest(source), "seed": seed, "dev_per_task": dev_per_task,
                "shuffle_choices": shuffle_choices}
    ensure_metadata(out_dir / "prepare.meta.json", settings)

    def convert(raw, split, index):
        choices = list(raw["choices"])
        order = list(range(len(choices)))
        if shuffle_choices:
            random.Random(f"{seed}:{split}:{index}").shuffle(order)
        row = {"id": f"portal-{split}-{index:06d}", "task": raw["task"], "prompt": raw["prompt"],
               "choices": [choices[j] for j in order], "gold_idx": order.index(raw["gold_idx"]),
               "source_split": split, "source_index": index}
        row["inputs"] = routing_text(row["prompt"], row["choices"])
        row["input_hash"] = query_hash(row)
        return row

    validation = [convert(r, "validation", i) for i, r in enumerate(source["validation"])]
    groups = defaultdict(list)
    for row in validation:
        groups[row["task"]].append(row)
    dev_prompts, smoke = set(), []
    for task in TASKS:
        ordered = list(groups[task])
        random.Random(f"{seed}:{task}").shuffle(ordered)
        if len(ordered) <= dev_per_task:
            raise ValueError(f"{task} has too few validation rows for a dev/eval split")
        smoke.append(ordered[0])
        dev_prompts.update(canonical_text(r["prompt"]) for r in ordered[:dev_per_task])
    dev = [r for r in validation if canonical_text(r["prompt"]) in dev_prompts]
    evaluation = [r for r in validation if canonical_text(r["prompt"]) not in dev_prompts]
    if {r["task"] for r in evaluation} != set(TASKS):
        raise ValueError("An eval task is empty after duplicate-prompt grouping")
    cards = [{"id": t, "description": DESCRIPTIONS[t], "examples": []} for t in TASKS]
    splits = {"validation": validation, "dev": dev, "eval": evaluation, "smoke": smoke}
    for name, rows in splits.items():
        write_jsonl(out_dir / f"{name}.jsonl", sorted(rows, key=lambda r: r["id"]))
    write_json(out_dir / "cards.json", cards)
    manifest = dict(settings, split_counts={s: len(rows) for s, rows in splits.items()},
                    task_counts={s: dict(Counter(r["task"] for r in rows)) for s, rows in splits.items()},
                    cards_hash=digest(cards),
                    warning="All evaluation subsets originate from upstream validation, used for upstream checkpoint selection; not an untouched test set.")
    write_json(out_dir / "manifest.json", manifest)
    return manifest


def prepare_command(args):
    source, provenance = load_source(args)
    manifest = prepare(source, args.output_dir, args.seed, args.dev_per_task,
                       not args.preserve_choice_order, provenance)
    print(f"Prepared 14 tasks: {manifest['split_counts']}")
    print(manifest["warning"])
