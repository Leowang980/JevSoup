"""Portable orchestration. Numerical engines in jev_lora/ and scripts/ are frozen.

This module never opens the original research workspace or pays for API calls.
New output metadata intentionally has its own identity: do not splice release
predictions into historical runs to bypass provenance checks.
"""
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from jev_lora.core import (TASKS, append_jsonl, digest, ensure_metadata,
    implementation_hash, read_json, read_jsonl, top_weights, unique_index, write_json)
from jev_lora.baselines import file_sha256
from jev_lora.data import validate_row
from jev_lora.evaluation import validate_prediction
from jev_lora.models import (ExactMixture, attach_adapters, load_base, load_manifest,
    select_weights, validate_router_models, versions)
from jev_lora.scoring import score_choices


def code_identity():
    files = [*sorted((ROOT / 'jevsoup').glob('*.py')),
             *sorted((ROOT / 'scripts').glob('*.py'))]
    return {str(p.relative_to(ROOT)): file_sha256(p) for p in files}


def plan(method, row, route, names, routing_seed=42):
    if method == 'random-top2-equal':
        from ablation_4b_impl import random_weights
        return random_weights(row['input_hash'], names, routing_seed)
    if method == 'arrow-top2-weighted':
        return {}
    if method == 'adaptersoup-top2-equal':
        from run_adapter_baselines import validate_soup_route
        validate_soup_route(row, route, names)
        return route['weights']
    selector = ('jev-top2-prob' if method.endswith('orthogonal-prob') else
                'jev-top2-equal' if method.endswith('orthogonal-equal') else method)
    return select_weights(selector, row, route, names)


def make_mixer(model, names, method, strength=1, prototypes=None):
    if method == 'base':
        return None
    if method == 'arrow-top2-weighted':
        from adapter_baseline_impl import FactorMixture
        return FactorMixture(model, names, 'arrow', prototypes, chunk_size=32)
    if method == 'adaptersoup-top2-equal':
        from soup_merge_cache import CachedFactorMixture
        return CachedFactorMixture(model, names, 'adaptersoup')
    if method == 'jev-top2-orthogonal-prob':
        from probability_orthogonal_impl import ProbabilityOrthogonalMixture
        return ProbabilityOrthogonalMixture(model, names)
    if method in ('jev-top2-orthogonal-equal', 'jev-top2-equal'):
        from orthogonal_lora import OrthogonalMixture
        if method.endswith('orthogonal-equal') and strength not in (0, 1):
            from sensitivity_4b_impl import PartialOrthogonalMixture
            return PartialOrthogonalMixture(model, names, strength)
        return OrthogonalMixture(model, names)
    return ExactMixture(model)


def activate(mixer, method, weights, strength=1):
    if mixer is None or method == 'arrow-top2-weighted':
        return
    if method == 'jev-top2-equal':
        mixer.activate(weights, strength=0)
    elif method.endswith('orthogonal-equal'):
        mixer.activate(weights, strength=0 if strength == 0 else 1)
    else:
        mixer.activate(weights)


def am_route(args):
    from run_adapter_baselines import locked
    size = load_manifest(args.model_dir)['size']
    if size == '4b':
        from run_am_top2_4b import route
    else:
        from run_am_top2_local import route
    with locked(str(args.output) + '.lock'):
        route(args)


def run(args):
    from run_adapter_baselines import locked
    with locked(str(args.output) + '.lock'):
        _run(args)


def _run(args):
    import torch
    from run_adapter_baselines import load_tensors

    if args.strength != 1 and args.method != 'jev-top2-orthogonal-equal':
        raise ValueError('--strength applies only to equal-weight JevSoup')
    if args.routing_seed != 42 and args.method != 'random-top2-equal':
        raise ValueError('--routing-seed applies only to the random-pair ablation')
    if args.arrow_traces and args.method != 'arrow-top2-weighted':
        raise ValueError('--arrow-traces applies only to Arrow')
    rows, cards = read_jsonl(args.data), read_json(args.cards)
    by_id = unique_index(rows, 'data')
    names = [c['id'] for c in cards]
    if not rows or names != list(TASKS):
        raise ValueError('Nonempty data and the original ordered 14-expert library are required')
    for row in rows:
        validate_row(row, prepared=True)
    needed = [] if args.method == 'base' else names
    manifest = load_manifest(args.model_dir, needed)
    routes, route_meta = {}, None
    needs_routes = args.method not in ('base', 'arrow-top2-weighted', 'random-top2-equal')
    if needs_routes != bool(args.routes):
        raise ValueError('Supply --routes exactly for methods that use external routing records')
    if needs_routes:
        routes = unique_index(read_jsonl(args.routes), 'routes')
        route_meta = read_json(str(args.routes) + '.meta.json')
        if (set(routes) != set(by_id) or route_meta['data_hash'] != digest(rows)
                or route_meta['cards_hash'] != digest(cards)
                or route_meta['kind'] != args.method.split('-')[0]
                or route_meta['seed'] != 42 or route_meta.get('dtype', 'bfloat16') != 'bfloat16'):
            raise ValueError('Routing split, cards, kind, seed or dtype differs')
        validate_router_models(route_meta, manifest)
    plans = {r['id']: plan(args.method, r, routes.get(r['id']), names, args.routing_seed) for r in rows}
    proto, proto_meta = None, None
    if args.method == 'arrow-top2-weighted':
        if not args.arrow_prototypes:
            raise ValueError('Arrow requires --arrow-prototypes')
        from run_adapter_baselines import code_identity as baseline_identity
        packed, proto_meta = load_tensors(args.arrow_prototypes)
        if (proto_meta['adapters'] != {n: manifest['models'][n] for n in names}
                or proto_meta['experts'] != names or proto_meta['code'] != baseline_identity()):
            raise ValueError('Arrow prototype order, adapter weights or implementation differs')
        proto = {k.removesuffix('.prototype'): v for k, v in packed.items() if k.endswith('.prototype')}
    out = Path(args.output)
    # Record settings before checking completion; stale caches must not silently pass.
    gpu = torch.cuda.get_device_name(0)
    metadata = dict(schema='portal-inference-v1', implementation_hash=implementation_hash(),
        method=args.method, data_hash=digest(rows), cards_hash=digest(cards),
        routes_hash=digest(sorted(routes.values(), key=lambda r: r['id'])) if routes else None,
        route_metadata=route_meta, models={n: manifest['models'][n] for n in ['base', *needed]},
        scoring='portal-continuation-logprob-per-character-v1', seed=42, max_prompt=768,
        choice_batch_size=1, dtype='bfloat16', versions=versions(),
        release_protocol=dict(code=code_identity(), gpu=gpu, cuda=torch.version.cuda,
            strength=args.strength, routing_seed=args.routing_seed,
            arrow_trace_capture=args.arrow_traces, prototypes=proto_meta,
            prototype_sha256=file_sha256(args.arrow_prototypes) if proto is not None else None))
    ensure_metadata(str(out) + '.meta.json', metadata)
    old = unique_index(read_jsonl(out, repair_tail=True), 'predictions')
    if not set(old) <= set(by_id):
        raise ValueError('Unknown saved prediction IDs')
    for rid, pred in old.items():
        validate_prediction(by_id[rid], pred)
        if pred['method'] != args.method or list(pred['weights'].items()) != list(plans[rid].items()):
            raise ValueError('Saved method or ordered expert weights changed')
        if args.arrow_traces:
            trace = Path(pred['dynamic_route_trace'])
            if not trace.is_file() or file_sha256(trace) != pred['dynamic_route_trace_sha256']:
                raise ValueError('Saved Arrow trace is missing or changed')
    todo = [r for r in rows if r['id'] not in old]
    if not todo:
        print(f'Already complete: {out} ({len(old)} rows; no inference)')
        return
    torch.set_num_threads(1)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    started = time.perf_counter()
    model, tokenizer = load_base(args.model_dir, manifest, 'bfloat16')
    if needed:
        model = attach_adapters(model, args.model_dir, manifest, needed)
    mixer = make_mixer(model, names, args.method, args.strength, proto)
    runtime = dict(gpu=gpu, versions=versions(), status='running', rows_before=len(old), new_rows=0)
    try:
        if args.method == 'adaptersoup-top2-equal':
            activate(mixer, args.method, plans[todo[0]['id']])
        with torch.inference_mode():
            model(**tokenizer('Hello world', return_tensors='pt').to('cuda:0'), use_cache=False)
        torch.cuda.synchronize()
        runtime['load_and_warmup_s'] = time.perf_counter() - started
        write_json(str(out) + '.runtime.json', runtime)
        with out.open('a', encoding='utf-8') as stream:
            for i, row in enumerate(todo, 1):
                torch.cuda.synchronize()
                started = time.perf_counter()
                activate(mixer, args.method, plans[row['id']], args.strength)
                if args.method == 'arrow-top2-weighted':
                    mixer.reset_trace(capture=args.arrow_traces)
                torch.cuda.synchronize()
                selection_s = time.perf_counter() - started
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                scored = score_choices(model, tokenizer, row['prompt'], row['choices'], 768, 1)
                torch.cuda.synchronize()
                rr = routes.get(row['id'], {})
                pred = dict(scored, id=row['id'], input_hash=row['input_hash'], method=args.method,
                    weights=plans[row['id']], selection_s=selection_s, scoring_s=time.perf_counter()-started,
                    route_s=rr.get('latency_s', 0), route_usage=rr.get('usage', {}),
                    route_payload_hash=rr.get('payload_hash'), router_fallback=rr.get('fallback', False),
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                if args.method == 'arrow-top2-weighted' and args.arrow_traces:
                    from safetensors.torch import save_file
                    trace, router_s = mixer.trace_cpu(len(row['choices']))
                    trace_path = out.parent / (out.stem + '.traces') / (row['id'] + '.safetensors')
                    trace_path.parent.mkdir(parents=True, exist_ok=True)
                    save_file(trace, str(trace_path))
                    pred.update(dynamic_route_trace=str(trace_path), dynamic_route_trace_sha256=file_sha256(trace_path),
                                dynamic_router_cuda_s=router_s)
                    mixer.reset_trace()
                validate_prediction(row, pred)
                append_jsonl(stream, pred)
                runtime['new_rows'] = i
                if i % 10 == 0 or i == len(todo):
                    write_json(str(out) + '.runtime.json', runtime)
                    print(f'{args.method}: {len(old)+i}/{len(rows)}', flush=True)
        runtime['status'] = 'completed'
    except BaseException:
        runtime['status'] = 'interrupted-or-failed'
        raise
    finally:
        if mixer is not None and hasattr(mixer, 'close'):
            mixer.close()
        write_json(str(out) + '.runtime.json', runtime)
