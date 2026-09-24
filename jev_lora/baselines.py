"""Shared AM Top-2/LoGo helpers and AdapterSoup embeddings on PorTAL.

Only download_embedding may access the Hub. Routing loads local files only.
These are adaptations to the shared PorTAL pool, not published-score reproductions.
"""
import hashlib
import math
import random
import re
import time
from pathlib import Path

from .core import (TASKS, append_jsonl, digest, ensure_metadata, implementation_hash,
                   read_json, read_jsonl, unique_index, write_json)
from .data import validate_row

EMBEDDING_REPO = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_INSTRUCTION = (
    "Given a user query, retrieve the LoRA expert whose capabilities best match the task."
)
AM_SOURCE = "https://github.com/qpiai/adaptive-minds/blob/main/adaptive_minds/router.py"
LOGO_SOURCE = "https://github.com/archon159/LoGo/blob/main/utils/model.py"
# Git blob hash of the upstream reference inspected for this adaptation.
LOGO_SOURCE_BLOB = "9aff237130bbaeb8ad394717a2a48f82d21a7b6b"
STOPWORDS = frozenset("a an the and or to of in on for with from as is are be by at this that whether "
                      "answer question questions choose supplied described using based expert task "
                      "input output determine decide classify more most same two one".split())


def ordered_cards(cards, seed):
    result = list(cards)
    random.Random(seed).shuffle(result)
    return result


def card_text(card):
    text = f"{card['id']}: {card['description']}"
    if card.get("examples"):
        text += "\nExample inputs:\n" + "\n".join(card["examples"])
    return text






def am_keyword_fallback(query, cards, seed=42):
    """AM-style keyword counts, with keywords derived only from the shared cards.

    PorTAL has no upstream AM keyword catalog. No training support is used. Ties/no hits pick the first expert in the seeded catalog order.
    """
    query = query.lower()
    scores = {}
    for card in ordered_cards(cards, seed):
        words = set(re.findall(r"[a-z0-9]+", card_text(card).lower())) - STOPWORDS
        scores[card["id"]] = sum(w in query for w in words if len(w) >= 3)
    return max(scores, key=scores.get), scores


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_embedding(args):
    """Explicit user-run download; resolve main once, then pin it on every retry."""
    from huggingface_hub import HfApi, snapshot_download
    root = Path(args.embedding_dir)
    lock = root / "download.meta.json"
    if lock.exists():
        settings = read_json(lock)
        if settings["repo"] != EMBEDDING_REPO or settings["requested_revision"] != args.revision:
            raise ValueError("Different embedding revision requested; use a new directory")
    else:
        revision = HfApi().model_info(EMBEDDING_REPO, revision=args.revision).sha
        settings = {"repo": EMBEDDING_REPO, "requested_revision": args.revision, "revision": revision}
    ensure_metadata(lock, settings)
    names = HfApi().list_repo_files(EMBEDDING_REPO, revision=settings["revision"])
    small = {"config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
             "added_tokens.json", "vocab.json", "merges.txt", "tokenizer.model", "model.safetensors.index.json"}
    files = sorted(n for n in names if "/" not in n and
                   (n in small or n == "model.safetensors" or re.fullmatch(r"model-\d+-of-\d+\.safetensors", n)))
    if not any(n.endswith(".safetensors") for n in files):
        raise ValueError("Embedding snapshot has no safetensors weights")
    write_json(root / "manifest.json", dict(settings, schema="embedding-model-v1", complete=False))
    snapshot_download(EMBEDDING_REPO, revision=settings["revision"], local_dir=str(root), allow_patterns=files)
    write_json(root / "manifest.json", dict(settings, schema="embedding-model-v1", complete=True,
                                             files={n: file_sha256(root / n) for n in files}))
    print(f"Embedding ready: {EMBEDDING_REPO}@{settings['revision']} in {root}")


def load_embedding_manifest(root):
    root = Path(root)
    info = read_json(root / "manifest.json")
    if info.get("schema") != "embedding-model-v1" or not info.get("complete") or info.get("repo") != EMBEDDING_REPO:
        raise ValueError("Incomplete embedding snapshot; run download-embedding on the GPU machine")
    if not re.fullmatch(r"[a-f0-9]{40}", info.get("revision", "")):
        raise ValueError("Embedding revision must be an immutable commit")
    files = info.get("files", {})
    if "config.json" not in files or "tokenizer_config.json" not in files or not any(n.endswith(".safetensors") for n in files):
        raise ValueError("Missing embedding config/tokenizer/weights")
    for name, expected in files.items():
        if Path(name).name != name or not (root / name).is_file() or file_sha256(root / name) != expected:
            raise ValueError(f"Changed or missing embedding file: {name}")
    return info


def routing_run(args, kind, extra):
    rows, cards = read_jsonl(args.data), read_json(args.cards)
    if not rows:
        raise ValueError("Empty routing data")
    by_id = unique_index(rows, "routing data")
    for row in rows:
        validate_row(row, prepared=True)
    names = [c["id"] for c in cards]
    if len(names) != len(TASKS) or set(names) != set(TASKS):
        raise ValueError("Routing requires exactly the 14 PorTAL expert cards")
    for card in cards:
        if not isinstance(card.get("description"), str) or not card["description"].strip():
            raise ValueError("Expert descriptions must be nonempty")
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"schema": "portal-local-router-v2", "kind": kind, "implementation_hash": implementation_hash(),
            "data_hash": digest(rows), "cards_hash": digest(cards), "seed": args.seed, **extra}
    ensure_metadata(str(out) + ".meta.json", meta)
    old = unique_index(read_jsonl(out, repair_tail=True), "routes")
    for rid, rr in old.items():
        if rid not in by_id or rr.get("input_hash") != by_id[rid]["input_hash"] or rr.get("kind") != kind:
            raise ValueError("Existing route ID/input/kind differs from this run")
        scores = rr.get("scores", {})
        if set(scores) != set(names) or any(not math.isfinite(v) for v in scores.values()):
            raise ValueError("Invalid existing route scores")
    todo = [r for r in rows if r["id"] not in old]
    return cards, names, out, old, todo


def last_token_pool(hidden, mask):
    """Last non-padding token, for either left or right padding."""
    import torch
    if mask.ndim != 2 or hidden.shape[:2] != mask.shape or not bool(mask.bool().any(dim=1).all()):
        raise ValueError("Invalid/all-padding input for last-token pooling")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    indices = positions.masked_fill(~mask.bool(), -1).max(dim=1).values
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), indices]


def encode_embeddings(model, tokenizer, texts, max_tokens, batch_size):
    import torch
    import torch.nn.functional as F
    if max_tokens < 1 or batch_size < 1 or not texts:
        raise ValueError("Positive embedding budgets and nonempty texts required")
    result, counts = [], []
    with torch.inference_mode():
        for begin in range(0, len(texts), batch_size):
            batch = tokenizer(texts[begin:begin+batch_size], padding=True, truncation=False, return_tensors="pt")
            if batch.input_ids.shape[1] > max_tokens:
                raise ValueError("Embedding input exceeds --max-input-tokens; no silent truncation")
            counts.extend(batch.attention_mask.sum(dim=1).tolist())
            batch = batch.to(next(model.parameters()).device)
            hidden = model(**batch, use_cache=False).last_hidden_state
            pooled = last_token_pool(hidden, batch.attention_mask).float()
            if not bool(torch.isfinite(pooled).all()) or bool((pooled.norm(dim=-1) == 0).any()):
                raise ValueError("Invalid or zero embedding")
            result.append(F.normalize(pooled, p=2, dim=-1))
    return torch.cat(result), counts






class LoGoProbe:
    """One all-adapter probe, last token of last block input, q-projection norm.

    The reference computes BA on outputs.hidden_states[layer_idx], BEFORE that
    block's input layer norm, not on the actual q_proj input. Capture the same
    tensor with a block pre-hook, without retaining all layers' hidden states.
    Every probe resets ALL adapters to original alpha/r, independent of history.
    """
    def __init__(self, model, mixer, names, layer_index=-1):
        from peft.tuners.lora.layer import LoraLayer
        self.model, self.mixer, self.names = model, mixer, list(names)
        layers = model.get_base_model().model.layers
        self.layer_index = len(layers)-1 if layer_index == -1 else layer_index
        if not 0 <= self.layer_index < len(layers):
            raise ValueError("LoGo layer index outside model")
        self.block = layers[self.layer_index]
        self.projection = self.block.self_attn.q_proj
        if not isinstance(self.projection, LoraLayer) or any(n not in self.projection.lora_A for n in names):
            raise ValueError("LoGo needs every expert attached to the selected q_proj")
        originals = next(scales for layer, scales in mixer.original if layer is self.projection)
        self.scales = {n: originals[n] for n in names}

    def scores(self, inputs):
        import torch
        self.mixer.activate_probe(self.names)
        captured = []

        def capture(module, positional, keyword):
            hidden = positional[0] if positional else keyword["hidden_states"]
            captured.append(last_token_pool(hidden, inputs["attention_mask"]).detach().clone())

        handle = self.block.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with torch.inference_mode():
                self.model(**inputs, use_cache=False, logits_to_keep=1)
        finally:
            handle.remove()
        if len(captured) != 1 or captured[0].shape[0] != 1:
            raise ValueError("Expected one unbatched LoGo probe")
        scores = {}
        with torch.inference_mode():
            for name in self.names:
                a, b = self.projection.lora_A[name], self.projection.lora_B[name]
                hidden = captured[0].to(a.weight.dtype)
                output = b(a(hidden)) * self.scales[name]
                scores[name] = float(output.float().norm(p=2, dim=-1).item())
        if any(not math.isfinite(s) or s < 0 for s in scores.values()) or sum(scores.values()) <= 0:
            raise ValueError("LoGo probe produced non-finite or all-zero projection norms")
        return scores


def logo_route(args):
    import torch
    from .models import ExactMixture, load_base, load_manifest, attach_adapters, versions
    manifest = load_manifest(args.model_dir, TASKS)
    cards, names, out, old, todo = routing_run(args, "logo", {
        "base": manifest["models"]["base"], "adapters": {n: manifest["models"][n] for n in TASKS},
        "dtype": args.dtype, "versions": versions(), "max_input_tokens": args.max_input_tokens,
        "layer_index": args.layer_index, "projection": "q_proj", "strategy": "norm",
        "hidden_state": "block-input-before-layernorm;last-nonpadding-token",
        "probe_adapters": "all; original unnormalized alpha/r restored before every query",
        "source": LOGO_SOURCE, "source_blob": LOGO_SOURCE_BLOB})
    if not todo:
        print(f"Already complete: {out}")
        return
    torch.manual_seed(args.seed)
    start = time.perf_counter()
    model, tokenizer = load_base(args.model_dir, manifest, args.dtype)
    model = attach_adapters(model, args.model_dir, manifest, names)
    probe = LoGoProbe(model, ExactMixture(model), names, args.layer_index)
    warm = tokenizer("Hello world", return_tensors="pt").to("cuda:0")
    probe.scores(warm)
    torch.cuda.synchronize()
    write_json(str(out) + ".runtime.json", {"gpu": torch.cuda.get_device_name(0), "versions": versions(),
        "load_and_warmup_s": time.perf_counter()-start, "resolved_layer_index": probe.layer_index})
    with out.open("a", encoding="utf-8") as handle:
        for i, row in enumerate(todo, 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            # Identical question + options used by every router, with no gold/task label.
            inputs = tokenizer(row["inputs"], return_tensors="pt").to("cuda:0")
            if inputs.input_ids.shape[1] > args.max_input_tokens:
                raise ValueError("LoGo input exceeds --max-input-tokens; no silent truncation")
            scores = probe.scores(inputs)
            torch.cuda.synchronize()
            append_jsonl(handle, {"id": row["id"], "input_hash": row["input_hash"], "kind": "logo",
                "scores": scores, "score_type": "q-projection-l2-norm", "latency_s": time.perf_counter()-start,
                "usage": {"input_tokens": inputs.input_ids.shape[1]},
                "peak_allocated_gib": torch.cuda.max_memory_allocated()/2**30})
            if i % 10 == 0 or i == len(todo):
                print(f"LoGo routing: {len(old)+i}/{len(old)+len(todo)}", flush=True)
