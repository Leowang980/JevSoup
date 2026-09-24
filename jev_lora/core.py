"""PorTAL data contracts, deterministic evaluation subsets and shared I/O."""
import hashlib
import json
import math
import os
import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET_ID = "RampPublic/portallib-tasks"
DATASET_REVISION = "ffc3c0e44f529bf64a5ae62ed5db090952db97ea"
BASES = {
    "1.7b": ("Qwen/Qwen3-1.7B", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e", "be09be533b5c0418ad20269f19ebb63e9efbc330"),
    "4b": ("Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c", "1ff8d529c07082f9da067918da2aceafc65ebaf9"),
    "8b": ("Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218", "6593b49ae790ec6d601db5a081c81463cb8b3a5f"),
}
DESCRIPTIONS = {
    "truthfulqa": "Answer factual questions accurately while avoiding common misconceptions and false premises.",
    "rte": "Decide whether a hypothesis follows from a premise; binary textual entailment.",
    "cb": "Classify a hypothesis as entailed, contradicted, or neutral, including beliefs and reported speech.",
    "copa": "Choose the more plausible cause or effect of a described event.",
    "wic": "Determine whether a highlighted word has the same meaning in two contexts.",
    "wsc": "Resolve whether a pronoun refers to a named entity in a supplied passage.",
    "boolq": "Answer a yes/no question based on the supplied passage.",
    "arc_easy": "Answer elementary science multiple-choice questions using factual knowledge.",
    "arc_challenge": "Answer challenging science multiple-choice questions requiring reasoning across facts.",
    "hellaswag": "Choose the most plausible continuation of an everyday activity or situation.",
    "openbookqa": "Answer elementary science questions combining a core scientific fact with everyday knowledge.",
    "winogrande": "Fill a sentence blank with the entity that makes sense, resolving ambiguous references.",
    "commonsense_qa": "Answer multiple-choice questions about everyday concepts using common-sense relations.",
    "sciq": "Answer science multiple-choice questions about biology, chemistry, physics and related subjects.",
}
TASKS = tuple(DESCRIPTIONS)

def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def source_implementation_hash():
    """Exact package-source hash, including diagnostic code."""
    return digest({p.name: p.read_text(encoding="utf-8") for p in sorted(Path(__file__).parent.glob("*.py"))})

def implementation_hash():
    """Use the exact source identity; incompatible caches must not be mixed."""
    return source_implementation_hash()

def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(tmp, path)

def read_jsonl(path, repair_tail=False):
    """Only repair a crash-truncated final line; malformed interior records are fatal."""
    path = Path(path)
    if not path.exists():
        return []
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, offset = [], 0
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeDecodeError):
            if repair_tail and i == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as f:
                    f.truncate(offset)
                break
            raise ValueError(f"Invalid JSONL record at {path}:{i+1}")
        offset += len(line)
    if rows and repair_tail and raw and not raw.endswith(b"\n") and offset == len(raw):
        with path.open("ab") as f:
            f.write(b"\n")
    return rows

def append_jsonl(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()

def canonical_text(text):
    return " ".join(text.casefold().split())

def top_weights(scores, k, weighted):
    if k < 1 or k > len(scores):
        raise ValueError("Invalid top-k")
    if any(not math.isfinite(float(x)) for x in scores.values()):
        raise ValueError("Non-finite score")
    chosen = sorted(scores, key=lambda name: (-scores[name], name))[:k]
    if not weighted:
        return {name: 1.0/k for name in chosen}
    if any(x < 0 for x in scores.values()):
        raise ValueError("Probability mixture requires non-negative scores")
    total = sum(scores[name] for name in chosen)
    if total <= 0:
        raise ValueError("Selected probabilities have zero mass")
    return {name: scores[name]/total for name in chosen}

def validate_probabilities(response, candidates):
    answer = response["answers"]["route"]
    scores = answer["probabilities"]
    if answer.get("type") != "choice" or set(scores) != set(candidates):
        raise ValueError("Jev response must contain exactly the requested Choice options")
    if any(isinstance(p, bool) or not isinstance(p, (float, int)) or not math.isfinite(p)
           or p < 0 or p > 1 for p in scores.values()):
        raise ValueError("Invalid Jev probability")
    total = sum(scores.values())
    if not math.isclose(total, 1, abs_tol=0.01):
        raise ValueError(f"Jev probabilities sum to {total}, not 1")
    if answer.get("choice") not in scores:
        raise ValueError("Unknown Jev choice")
    return {k: v/total for k, v in scores.items()}

def unique_index(rows, label):
    result = {}
    for row in rows:
        if row["id"] in result:
            raise ValueError(f"Duplicate id in {label}: {row['id']}")
        result[row["id"]] = row
    return result

def ensure_metadata(path, metadata):
    path = Path(path)
    if path.exists() and read_json(path) != metadata:
        raise ValueError(f"Configuration changed; use a new output path: {path}")
    write_json(path, metadata)
