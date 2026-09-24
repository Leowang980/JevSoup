"""Pinned PorTAL exports and frozen request-level LoRA mixtures."""
import importlib.metadata
import fnmatch
import math
import time
from pathlib import Path, PurePosixPath

from .core import (BASES, TASKS, append_jsonl, digest, ensure_metadata, implementation_hash,
                   read_json, read_jsonl, top_weights, unique_index, write_json)
from .data import validate_row

METHODS = ("base", "am-top2-equal", "logo-top2-weighted", "jev-top1",
           "jev-top2-equal", "jev-top2-prob")
MAIN_METHODS = ("base", "am-top2-equal", "logo-top2-weighted", "adaptersoup-top2-equal",
                "arrow-top2-weighted", "jev-top2-orthogonal-equal")

def select_download_files(files):
    """Prefer safetensors per weight family and subdirectory; retain bin-only modules."""
    safe_families = set()
    for name in files:
        p = PurePosixPath(name)
        if p.name == "adapter_model.safetensors":
            safe_families.add((str(p.parent), "adapter_model"))
        elif p.name == "model.safetensors" or fnmatch.fnmatch(p.name, "model-*-of-*.safetensors"):
            safe_families.add((str(p.parent), "pytorch_model"))
    selected = []
    for name in files:
        if any(fnmatch.fnmatch(name, pattern) for pattern in ("*.h5", "*.msgpack", "*.ot", "consolidated.*", "original/*")):
            continue
        p = PurePosixPath(name)
        family = "adapter_model" if p.name.startswith("adapter_model") else "pytorch_model"
        bin_weight = p.name == f"{family}.bin" or p.name == f"{family}.bin.index.json" or fnmatch.fnmatch(p.name, f"{family}-*-of-*.bin")
        if bin_weight and (str(p.parent), family) in safe_families:
            continue
        selected.append(name)
    return sorted(selected)

class ExactMixture:
    """PEFT sums w_i * scale_i * B_i(A_i(x)); no factor averaging or cross terms.

    set_adapter(list) is the supported LoraModel API in PEFT.
    Every layer receives the same request-level weights. No learned token router.
    """
    def __init__(self, model):
        from peft.tuners.lora.layer import LoraLayer
        self.model = model
        self.layers = [m for m in model.modules() if isinstance(m, LoraLayer)]
        if not self.layers:
            raise ValueError("No LoRA layers loaded")
        self.original = [(layer, dict(layer.scaling)) for layer in self.layers]

    def activate(self, weights):
        if not weights or any(not math.isfinite(v) or v < 0 for v in weights.values()):
            raise ValueError("Mixture requires non-negative finite weights")
        if not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6):
            raise ValueError("Mixture weights must sum to one")
        if not set(weights) <= set(self.model.peft_config):
            raise ValueError("Unknown adapter")
        self.model.base_model.set_adapter(list(weights))
        for layer, initial in self.original:
            for name, scale in initial.items():
                # Reset from the captured original on every request: never compound weights.
                layer.scaling[name] = scale * weights.get(name, 1.0)
        # PEFT activation toggles requires_grad; this experiment always freezes everything.
        self.model.requires_grad_(False)
        self.model.eval()

    def activate_probe(self, names):
        """LoGo probe sums all original LoRA branches, without 1/N normalization."""
        if not names or len(set(names)) != len(names) or not set(names) <= set(self.model.peft_config):
            raise ValueError("Invalid probe adapter pool")
        self.model.base_model.set_adapter(list(names))
        for layer, initial in self.original:
            for name, scale in initial.items():
                layer.scaling[name] = scale
        self.model.requires_grad_(False)
        self.model.eval()

def check_adapter_config(config, expected_base):
    if config.get("peft_type") != "LORA":
        raise ValueError("Only LoRA adapters are supported")
    if config.get("bias", "none") != "none" or config.get("modules_to_save") or config.get("use_dora", False):
        raise ValueError("Bias, modules_to_save and DoRA require a different composition rule")
    recorded = config.get("base_model_name_or_path", "")
    # Require the exact published Qwen3 model identity.
    if recorded != expected_base:
        raise ValueError(f"Adapter base {recorded!r} differs from {expected_base!r}")



def versions():
    return {name: importlib.metadata.version(name) for name in
            ("torch", "transformers", "peft", "portallib", "huggingface-hub")}


def download(args):
    """Download only on the user's GPU machine; export frozen task weights on CPU."""
    from huggingface_hub import HfApi, snapshot_download
    from portallib import PortalModel
    base_id, base_rev, portal_rev = BASES[args.size]
    portal_id = f"RampPublic/portal-qwen3-{args.size}"
    root = Path(args.model_dir)
    settings = {"schema": "portal-models-v1", "size": args.size,
                "base_repo": base_id, "base_revision": base_rev,
                "portal_repo": portal_id, "portal_revision": portal_rev, "tasks": list(TASKS)}
    ensure_metadata(root / "download.meta.json", settings)
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else dict(settings, models={})
    api = HfApi()
    for name, repo, revision in (("portal", portal_id, portal_rev), ("base", base_id, base_rev)):
        if name == "base" and args.adapters_only:
            continue
        entry = {"repo": repo, "revision": revision, "path": name, "complete": False}
        manifest["models"][name] = entry
        write_json(manifest_path, manifest)
        files = select_download_files(api.list_repo_files(repo_id=repo, revision=revision))
        print(f"Downloading {repo}@{revision}", flush=True)
        snapshot_download(repo_id=repo, revision=revision, local_dir=str(root / name), allow_patterns=files)
        entry["complete"] = True
        write_json(manifest_path, manifest)
    portal = PortalModel.from_pretrained(str(root / "portal"), local_files_only=True)
    portal.validate_base_model(base_id, base_rev)
    if set(portal.config.tasks) != set(TASKS):
        raise ValueError("PorTAL artifact must contain exactly the expected 14 tasks")
    # Store hashes of small configs and actual adapter weight files for local provenance.
    import hashlib
    for task in TASKS:
        path = root / "adapters" / task
        portal.export_peft(task, str(path))
        cfg = read_json(path / "adapter_config.json")
        check_adapter_config(cfg, base_id)
        if cfg.get("revision") != base_rev:
            raise ValueError("Exported adapter revision does not match the pinned base")
        manifest["models"][task] = {"repo": portal_id, "revision": portal_rev,
                                   "base_revision": base_rev, "path": f"adapters/{task}", "complete": True,
                                   "config_hash": digest(cfg),
                                   "weights_sha256": hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest()}
        write_json(manifest_path, manifest)
    manifest["export_versions"] = versions()
    write_json(manifest_path, manifest)
    print(f"Exported all 14 adapters without training: {root}")


def select_weights(method, row, route_row, names):
    if method not in METHODS:
        raise ValueError(f"Unsupported request-level mixture: {method}")
    if method == "base":
        return {}
    expected_kind = method.split("-")[0]
    if route_row is None or route_row["kind"] != expected_kind:
        raise ValueError(f"{method} requires {expected_kind} routes")
    if route_row["input_hash"] != row["input_hash"]:
        raise ValueError("Route/query hash mismatch")
    if set(route_row["scores"]) != set(names):
        raise ValueError("Routes and loaded adapter pool differ")
    k = 2 if "top2" in method else 1
    return top_weights(route_row["scores"], k, method.endswith(("-prob", "-weighted")))


def load_manifest(root, names=()):
    import hashlib
    root = Path(root)
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema") != "portal-models-v1":
        raise ValueError("Expected PorTAL model manifest; run download in a new directory")
    base_repo, base_rev, portal_rev = BASES[manifest["size"]]
    if (manifest["base_repo"], manifest["base_revision"], manifest["portal_revision"]) != (base_repo, base_rev, portal_rev):
        raise ValueError("Pinned model revisions differ from the supported PorTAL release")
    for name in ["base", *names]:
        info = manifest["models"].get(name)
        if not info or not info["complete"] or not (root / info["path"]).is_dir():
            raise ValueError(f"Missing model/adapter {name}; run download")
    for name in names:
        info = manifest["models"][name]
        adapter_path = root / info["path"]
        cfg = read_json(adapter_path / "adapter_config.json")
        check_adapter_config(cfg, base_repo)
        if cfg.get("revision") != base_rev or digest(cfg) != manifest["models"][name]["config_hash"]:
            raise ValueError(f"Changed adapter config: {name}")
        if hashlib.sha256((adapter_path / "adapter_model.safetensors").read_bytes()).hexdigest() != info["weights_sha256"]:
            raise ValueError(f"Changed adapter weights: {name}")
    return manifest


def preflight(args):
    rows, cards = read_jsonl(args.data), read_json(args.cards)
    if not rows:
        raise ValueError("Empty evaluation data")
    unique_index(rows, "data")
    for row in rows:
        validate_row(row, prepared=True)
    names = [c["id"] for c in cards]
    if len(names) != len(TASKS) or set(names) != set(TASKS):
        raise ValueError("Expected the full 14-expert pool")
    manifest = load_manifest(args.model_dir, names)
    if getattr(args, "embedding_dir", None):
        from .baselines import load_embedding_manifest
        info = load_embedding_manifest(args.embedding_dir)
        print(f"Embedding verified: {info['repo']}@{info['revision']}")
    print(f"Ready: {len(rows)} questions, 14 adapters, {manifest['base_repo']}; local hashes verified")


def load_base(root, manifest, dtype_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires CUDA; run selfcheck for a tiny CPU integration test")
    if dtype_name == "bfloat16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU lacks BF16 support; use --dtype float16")
    path = Path(root) / manifest["models"]["base"]["path"]
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(path, dtype=getattr(torch, dtype_name),
                                               device_map={"": 0}, attn_implementation="sdpa",
                                               local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer


def attach_adapters(model, root, manifest, names):
    from peft import PeftModel
    for i, name in enumerate(names):
        path = str(Path(root) / manifest["models"][name]["path"])
        if i == 0:
            model = PeftModel.from_pretrained(model, path, adapter_name=name, is_trainable=False,
                                             autocast_adapter_dtype=False, local_files_only=True)
        else:
            model.load_adapter(path, adapter_name=name, is_trainable=False,
                               autocast_adapter_dtype=False, local_files_only=True)
    model.requires_grad_(False)
    model.eval()
    return model


def validate_router_models(route_meta, manifest):
    """AM and LoGo must use the exact downstream base/adapter snapshots."""
    if route_meta.get("kind") in ("am", "logo"):
        if route_meta.get("base") != manifest["models"]["base"]:
            raise ValueError("Routing base differs from inference base")
    if route_meta.get("kind") == "logo":
        if route_meta.get("adapters") != {n: manifest["models"][n] for n in TASKS}:
            raise ValueError("LoGo probe adapters differ from inference adapters")


def infer(args):
    import torch
    from .scoring import score_choices
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("Empty evaluation data")
    unique_index(rows, "data")
    for row in rows:
        validate_row(row, prepared=True)
    cards = read_json(args.cards)
    names = [c["id"] for c in cards]
    if len(names) != len(TASKS) or set(names) != set(TASKS):
        raise ValueError("Expected all 14 PorTAL adapter cards")
    if args.max_prompt < 1 or args.choice_batch_size < 1:
        raise ValueError("Positive max-prompt and choice-batch-size required")
    root = Path(args.model_dir)
    needed = [] if args.method == "base" else names
    manifest = load_manifest(root, needed)
    routes = unique_index(read_jsonl(args.routes), "routes") if args.routes else {}
    if args.method not in ("base",) and set(routes) != {r["id"] for r in rows}:
        raise ValueError("Provide complete routes for exactly this split")
    route_meta = None
    if routes:
        route_meta = read_json(str(args.routes) + ".meta.json")
        if route_meta["data_hash"] != digest(rows) or route_meta["cards_hash"] != digest(cards):
            raise ValueError("Route metadata uses different data/cards")
        if route_meta.get("kind") != args.method.split("-")[0]:
            raise ValueError("Route metadata kind does not match inference method")
        if route_meta.get("seed") != args.seed:
            raise ValueError("Router and inference seeds differ")
        if route_meta.get("dtype", args.dtype) != args.dtype:
            raise ValueError("Local router and inference dtypes differ")
        validate_router_models(route_meta, manifest)
    plans = {r["id"]: select_weights(args.method, r, routes.get(r["id"]), names) for r in rows}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"schema": "portal-inference-v1", "implementation_hash": implementation_hash(),
                "method": args.method, "data_hash": digest(rows), "cards_hash": digest(cards),
                "routes_hash": digest(sorted(routes.values(), key=lambda r: r["id"])) if routes else None,
                "route_metadata": route_meta,
                "models": {n: manifest["models"][n] for n in ["base", *needed]},
                "scoring": "portal-continuation-logprob-per-character-v1", "seed": args.seed,
                "max_prompt": args.max_prompt, "choice_batch_size": args.choice_batch_size,
                "dtype": args.dtype, "versions": versions()}
    ensure_metadata(str(out) + ".meta.json", metadata)
    old = unique_index(read_jsonl(out, repair_tail=True), "predictions")
    if not set(old) <= set(plans):
        raise ValueError("Unknown IDs in existing predictions")
    todo = [r for r in rows if r["id"] not in old]
    if not todo:
        print(f"Already complete: {out}")
        return
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    start = time.perf_counter()
    model, tokenizer = load_base(root, manifest, args.dtype)
    mixer = None
    if needed:
        model = attach_adapters(model, root, manifest, needed)
        mixer = ExactMixture(model)
    model.requires_grad_(False)
    model.eval()
    # A small teacher-forced pass initializes kernels without using evaluation labels.
    with torch.inference_mode():
        warm = tokenizer("Hello world", return_tensors="pt").to("cuda:0")
        model(**warm, use_cache=False)
    torch.cuda.synchronize()
    write_json(str(out) + ".runtime.json", {"gpu": torch.cuda.get_device_name(0), "versions": versions(),
                                           "load_and_warmup_s": time.perf_counter() - start})
    with out.open("a", encoding="utf-8") as handle:
        for i, row in enumerate(todo, 1):
            torch.cuda.synchronize()
            start = time.perf_counter()
            if mixer:
                mixer.activate(plans[row["id"]])
            selection_s = time.perf_counter() - start
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            scored = score_choices(model, tokenizer, row["prompt"], row["choices"],
                                   args.max_prompt, args.choice_batch_size)
            torch.cuda.synchronize()
            scoring_s = time.perf_counter() - start
            rr = routes.get(row["id"], {})
            result = dict(scored, id=row["id"], input_hash=row["input_hash"], method=args.method,
                          weights=plans[row["id"]], selection_s=selection_s, scoring_s=scoring_s,
                          route_s=rr.get("latency_s", 0), route_usage=rr.get("usage", {}),
                          route_payload_hash=rr.get("payload_hash"), route_cache_hit=rr.get("cache_hit"),
                          router_fallback=rr.get("fallback", False),
                          routing_peak_allocated_gib=rr.get("peak_allocated_gib", 0),
                          peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30)
            append_jsonl(handle, result)
            if i % 10 == 0 or i == len(todo):
                print(f"{args.method}: {len(old)+i}/{len(rows)}", flush=True)
    print(f"Predictions ready: {out}")


def selfcheck():
    """Real Qwen3 + PEFT + PorTAL CPU checks without any Hub downloads."""
    from .selfcheck import run
    run()
