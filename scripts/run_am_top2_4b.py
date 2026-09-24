"""Resumable 4B full evaluation of a Top-2 Adaptive Minds adaptation.

This variant asks the same unadapted Qwen3-4B backbone for two ranked expert IDs in one generation, then
scores answers with an equal-weight mixture of their complete LoRA updates.
No projection is applied. This is an adaptation, not the official AM setting.
"""
import argparse
import fcntl
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jev_lora.baselines import (AM_SOURCE, am_keyword_fallback, card_text,
                                file_sha256, ordered_cards, routing_run)
from jev_lora.core import (TASKS, append_jsonl, digest, ensure_metadata,
                           read_json, read_jsonl, write_json)
from jev_lora.models import load_manifest, versions

METHOD = "am-top2-equal"
PROMPT = (
    "Select the two best distinct experts for this query, ranked from most to "
    "less relevant. Reply with exactly two lowercase IDs separated by a comma; "
    "do not explain.\n\nQuery:\n{query}\n\nExperts:\n{experts}\n\nSelected experts:"
)


def parse_two(text, names):
    """Read expert IDs in order from the first nonempty response line."""
    line = next((line for line in text.lower().splitlines() if line.strip()), "")
    allowed = set(names)
    found = []
    for token in re.findall(r"[a-z][a-z0-9_]*", line):
        if token in allowed and token not in found:
            found.append(token)
        if len(found) == 2:
            break
    return tuple(found)


def complete_two(parsed, query, cards, seed):
    """Keep parsed choices; fill only missing slots by seeded AM keywords."""
    if len(parsed) == 2:
        return parsed, None
    _, counts = am_keyword_fallback(query, cards, seed)
    order = [card["id"] for card in ordered_cards(cards, seed)]
    ranked = sorted(order, key=lambda name: -counts[name])
    choices = list(parsed)
    choices.extend(name for name in ranked if name not in choices)
    return tuple(choices[:2]), counts


def route(args):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    from jev_lora.models import load_base

    manifest = load_manifest(args.model_dir)
    if manifest["size"] != "4b":
        raise ValueError("AM Top-2 runner requires the 4B model manifest")
    settings = SimpleNamespace(data=str(args.data), cards=str(args.cards),
                               output=str(args.output), seed=args.seed)
    cards, names, out, old, todo = routing_run(settings, "am", {
        "base": manifest["models"]["base"], "dtype": "bfloat16",
        "versions": versions(), "max_new_tokens": 32, "max_input_tokens": 4096,
        "enable_thinking": False, "temperature": 0, "top_k": 2,
        "prompt": PROMPT, "source": AM_SOURCE,
        "adaptations": "One generated ranked pair; full shared cards; seeded card-keyword fallback",
        "invalid_output_policy": "retain parsed IDs; fill missing IDs by seeded keyword rank",
        "runner_sha256": file_sha256(__file__),
    })
    for rid, previous in old.items():
        choices = previous.get("choices")
        if (not isinstance(choices, list) or len(choices) != 2 or
                len(set(choices)) != 2 or set(choices) - set(names) or
                previous["scores"][choices[0]] != 2.0 or
                previous["scores"][choices[1]] != 1.0):
            raise ValueError(f"Invalid saved AM Top-2 route: {rid}")
    if not todo:
        print(f"Already complete: {out}", flush=True)
        return

    torch.manual_seed(args.seed)
    start = time.perf_counter()
    model, tokenizer = load_base(args.model_dir, manifest, "bfloat16")
    with torch.inference_mode():
        warm = tokenizer("Hello world", return_tensors="pt").to("cuda:0")
        model.generate(**warm, max_new_tokens=1, do_sample=False,
                       pad_token_id=tokenizer.pad_token_id)
    torch.cuda.synchronize()
    write_json(str(out) + ".runtime.json", {
        "gpu": torch.cuda.get_device_name(0), "versions": versions(),
        "load_and_warmup_s": time.perf_counter() - start,
    })

    class StopOnNewline(StoppingCriteria):
        def __init__(self, prompt_length):
            self.prompt_length = prompt_length

        def __call__(self, input_ids, scores, **kwargs):
            return "\n" in tokenizer.decode(
                input_ids[0, self.prompt_length:], skip_special_tokens=True)

    with torch.inference_mode(), out.open("a", encoding="utf-8") as handle:
        for index, row in enumerate(todo, 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            experts = "\n".join("- " + card_text(card)
                                for card in ordered_cards(cards, args.seed))
            messages = [{"role": "user", "content": PROMPT.format(
                query=row["inputs"], experts=experts)}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False)
            inputs = tokenizer(text, add_special_tokens=False,
                               return_tensors="pt").to("cuda:0")
            prompt_length = inputs.input_ids.shape[1]
            if prompt_length > 4096:
                raise ValueError("AM Top-2 prompt exceeds 4096 tokens")
            generated = model.generate(
                **inputs, max_new_tokens=32, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnNewline(prompt_length)]))
            answer = tokenizer.decode(
                generated[0, prompt_length:], skip_special_tokens=True)
            parsed = parse_two(answer, names)
            choices, fallback_scores = complete_two(
                parsed, row["inputs"], cards, args.seed)
            scores = {name: float(2 if name == choices[0] else
                                  1 if name == choices[1] else 0)
                      for name in names}
            torch.cuda.synchronize()
            append_jsonl(handle, {
                "id": row["id"], "input_hash": row["input_hash"], "kind": "am",
                "scores": scores, "choices": list(choices),
                "score_type": "ranked-pair-not-probability", "raw_response": answer,
                "fallback": len(parsed) < 2, "fallback_scores": fallback_scores,
                "latency_s": time.perf_counter() - start,
                "usage": {"input_tokens": prompt_length,
                          "output_tokens": generated.shape[1] - prompt_length},
                "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            })
            if index % 10 == 0 or index == len(todo):
                print(f"AM Top-2 routing: {len(old) + index}/{len(old) + len(todo)}",
                      flush=True)


def infer(args):
    from jev_lora.models import infer as score
    score(SimpleNamespace(method=METHOD, data=str(args.data),
                          routes=str(args.run_dir / "am-top2.routes.jsonl"),
                          cards=str(args.cards), model_dir=str(args.model_dir),
                          output=str(args.output), dtype="bfloat16",
                          max_prompt=768, choice_batch_size=1, seed=args.seed))


def evaluate(args):
    from jev_lora.evaluation import evaluate as summarize
    summarize(SimpleNamespace(data=str(args.data), predictions=[str(args.output)],
                              output_dir=str(args.run_dir / "report"),
                              bootstrap=1000, seed=args.seed,
                              input_price_per_million=None))


def all_stages(args):
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with (args.run_dir / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, cards = read_jsonl(args.data), read_json(args.cards)
        if len(rows) != 19477 or len(cards) != len(TASKS):
            raise ValueError("Expected the full 19,477-row PorTAL evaluation")
        manifest = load_manifest(args.model_dir)
        if manifest["size"] != "4b":
            raise ValueError("Expected Qwen3-4B")
        ensure_metadata(args.run_dir / "manifest.json", {
            "schema": "portal-am-top2-4b-v1", "method": METHOD,
            "data": str(args.data), "data_hash": digest(rows),
            "cards": str(args.cards), "cards_hash": digest(cards),
            "model_dir": str(args.model_dir), "model": manifest["models"]["base"],
            "seed": args.seed, "runner_sha256": file_sha256(__file__),
            "composition": "equal complete LoRA updates; no projection",
            "route": "one deterministic generation of two ranked IDs; seeded keyword fallback",
        })
        env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                   TOKENIZERS_PARALLELISM="false", PYTHONUNBUFFERED="1")
        for stage, output in (
            ("route", args.run_dir / "am-top2.routes.jsonl"),
            ("infer", args.run_dir / f"{METHOD}.predictions.jsonl"),
            ("evaluate", args.run_dir / f"{METHOD}.predictions.jsonl"),
        ):
            status = {"stage": stage, "status": "running",
                      "started_utc": datetime.now(timezone.utc).isoformat(),
                      "output": str(output)}
            write_json(args.run_dir / "status.json", status)
            command = [sys.executable, str(Path(__file__).resolve()), stage,
                       "--run-dir", str(args.run_dir), "--data", str(args.data),
                       "--cards", str(args.cards), "--model-dir", str(args.model_dir),
                       "--seed", str(args.seed), "--output", str(output)]
            profiled = [sys.executable, str(ROOT / "scripts/profile_stage.py"),
                        "--directory", str(args.run_dir / "telemetry"),
                        "--stage", "am-top2-" + stage, "--interval-ms", "500",
                        "--", *command]
            code = subprocess.call(profiled, cwd=ROOT, env=env)
            status.update(status="complete" if code == 0 else "failed",
                          exit_code=code,
                          finished_utc=datetime.now(timezone.utc).isoformat())
            write_json(args.run_dir / "status.json", status)
            if code:
                raise RuntimeError(f"AM Top-2 {stage} failed (exit {code}); rows retained")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "route", "infer", "evaluate"))
    parser.add_argument("--run-dir", type=Path,
                        default=Path("runs/am-top2-4b-full-v1"))
    parser.add_argument("--data", type=Path,
                        default=Path("artifacts/portal/data/eval.jsonl"))
    parser.add_argument("--cards", type=Path,
                        default=Path("artifacts/portal/data/cards.json"))
    parser.add_argument("--model-dir", type=Path,
                        default=Path("models/portal-qwen3-4b"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.output = args.output or (args.run_dir / (
        "am-top2.routes.jsonl" if args.stage == "route" else
        f"{METHOD}.predictions.jsonl"))
    {"all": all_stages, "route": route, "infer": infer,
     "evaluate": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
