"""Public JevSoup inference and local AM routing commands."""
import argparse


METHODS = (
    'base', 'am-top2-equal', 'logo-top2-weighted', 'adaptersoup-top2-equal',
    'arrow-top2-weighted', 'jev-top1', 'jev-top2-equal', 'jev-top2-prob',
    'jev-top2-orthogonal-equal', 'jev-top2-orthogonal-prob', 'random-top2-equal',
)


def main(argv=None):
    parser = argparse.ArgumentParser(description='JevSoup: training-free routing and orthogonal LoRA composition')
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run', help='Resumable inference with JevSoup or a baseline')
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--cards', default='artifacts/portal/data/cards.json')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--routes', help='Locally generated JSONL routes and their .meta.json sidecar')
    p.add_argument('--output', required=True)
    p.add_argument('--arrow-prototypes', help='Prepared Arrow safetensors and metadata sidecars')
    p.add_argument('--strength', type=float, choices=[0, .25, .5, .75, 1], default=1)
    p.add_argument('--routing-seed', type=int, choices=[42, 43, 44], default=42)
    p.add_argument('--arrow-traces', action='store_true', help='Save large per-token/per-layer routing traces')
    p = sub.add_parser('am-route', help='Non-thinking AM Top-2 generation using the selected backbone')
    p.add_argument('--data', required=True)
    p.add_argument('--cards', default='artifacts/portal/data/cards.json')
    p.add_argument('--model-dir', required=True)
    p.add_argument('--output', required=True)
    p.set_defaults(seed=42)
    args = parser.parse_args(argv)
    if args.command == 'run':
        from .runner import run
        run(args)
    else:
        from .runner import am_route
        am_route(args)


if __name__ == '__main__':
    main()
