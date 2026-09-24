"""PorTAL experiment CLI. Heavy dependencies are imported only when needed."""
import argparse


def main(argv=None):
    from .models import METHODS
    parser = argparse.ArgumentParser(description="Jev routing and LoRA mixtures on the PorTAL 14-task Qwen3 suite")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Download pinned PorTAL data or read local JSON; create reproducible subsets")
    p.add_argument("--source-json", help="Offline JSON with train/validation arrays in the official schema")
    p.add_argument("--raw-dir", default="data/raw/portal")
    p.add_argument("--hf-cache", default="cache/huggingface")
    p.add_argument("--output-dir", default="artifacts/portal/data")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dev-per-task", type=int, default=5)
    p.add_argument("--preserve-choice-order", action="store_true", help="Keep source order; default shuffles deterministically")

    p = sub.add_parser("download", help="Download pinned Qwen3/PorTAL snapshots and export all 14 PEFT adapters")
    p.add_argument("--size", choices=["1.7b", "4b", "8b"], default="4b")
    p.add_argument("--model-dir", help="Default: models/portal-qwen3-<size>")
    p.add_argument("--adapters-only", action="store_true", help="Export task adapters without downloading the base model")

    p = sub.add_parser("download-embedding", help="Explicitly download Qwen3-Embedding-0.6B, pin revision and hash files")
    p.add_argument("--embedding-dir", default="models/qwen3-embedding-0.6b")
    p.add_argument("--revision", default="main", help="Resolved to an immutable commit once, then reused on retries")

    p = sub.add_parser("check", help="Verify all local data/config/adapter hashes before paid API calls")
    p.add_argument("--data", required=True)
    p.add_argument("--cards", default="artifacts/portal/data/cards.json")
    p.add_argument("--model-dir", default="models/portal-qwen3-4b")
    p.add_argument("--embedding-dir", help="Also verify the local embedding snapshot before running the suite")

    p = sub.add_parser("route", help="Jev routes shared by the paper's composition variants")
    p.add_argument("--data", required=True)
    p.add_argument("--cards", default="artifacts/portal/data/cards.json")
    p.add_argument("--output", required=True)
    p.add_argument("--kind", choices=["jev"], default="jev")
    p.add_argument("--jev-model", default="jev-1.13.0")
    p.add_argument("--cache", default="cache/portal/jev")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--interval", type=float, default=.15)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("logo-route", help="LoGo all-adapter probe, last-block q-projection norm scores")
    p.add_argument("--data", required=True)
    p.add_argument("--cards", default="artifacts/portal/data/cards.json")
    p.add_argument("--model-dir", default="models/portal-qwen3-4b")
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--max-input-tokens", type=int, default=4096)
    p.add_argument("--layer-index", type=int, default=-1, help="Zero-based probe block; -1 means final block")
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("infer", help="Score candidate continuations with exact request-level LoRA mixtures")
    p.add_argument("--method", choices=METHODS, required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--routes")
    p.add_argument("--cards", default="artifacts/portal/data/cards.json")
    p.add_argument("--model-dir", default="models/portal-qwen3-4b")
    p.add_argument("--output", required=True)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--max-prompt", type=int, default=768)
    p.add_argument("--choice-batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("evaluate", help="Macro/micro MC accuracy, paired bootstrap, cost and timing")
    p.add_argument("--data", required=True)
    p.add_argument("--predictions", nargs="+", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--input-price-per-million", type=float, default=None,
                   help="Optional Jev input-token unit price; omit to report tokens without cost assumptions")
    sub.add_parser("selfcheck", help="Tiny random Qwen3 + real PEFT/PorTAL export and scoring checks; no downloads")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        from .data import prepare_command
        prepare_command(args)
    elif args.command == "download":
        from .models import download
        if args.model_dir is None:
            args.model_dir = f"models/portal-qwen3-{args.size}"
        download(args)
    elif args.command == "download-embedding":
        from .baselines import download_embedding
        download_embedding(args)
    elif args.command == "route":
        from .routing import route
        route(args)
    elif args.command == "check":
        from .models import preflight
        preflight(args)
    elif args.command == "logo-route":
        from .baselines import logo_route
        if args.max_input_tokens < 1 or args.layer_index < -1:
            parser.error("Invalid LoGo token budget or layer index")
        logo_route(args)
    elif args.command == "infer":
        from .models import infer
        infer(args)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        if args.bootstrap < 1 or (args.input_price_per_million is not None and args.input_price_per_million < 0):
            parser.error("Invalid bootstrap or unit price")
        evaluate(args)
    elif args.command == "selfcheck":
        from .models import selfcheck
        selfcheck()


if __name__ == "__main__":
    main()
