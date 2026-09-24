"""Reproducible Top-2 Arrow/AdapterSoup adapters for the frozen PorTAL benchmark.

No package-engine edits, no API calls, no training. All outputs are resumable;
Arrow saves full per-candidate, per-projection, per-token routing traces.
"""
import argparse
import fcntl
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from jev_lora.baselines import EMBEDDING_INSTRUCTION, encode_embeddings, file_sha256, load_embedding_manifest
from jev_lora.core import TASKS, append_jsonl, digest, ensure_metadata, implementation_hash, read_json, read_jsonl, top_weights, unique_index, write_json, write_jsonl
from jev_lora.data import validate_row
from jev_lora.models import load_manifest, versions
from adapter_baseline_impl import MTTL_COMMIT, MTTL_FILES, input_svd_prototype, mean_domain_vectors, sample_training_support

DEFAULT_RUN = Path('runs/shared-soup')


def code_identity():
    return {p.name: file_sha256(p) for p in (Path(__file__), Path(__file__).with_name('adapter_baseline_impl.py'))}


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def save_tensors(path, tensors, config):
    from safetensors.torch import save_file
    path = Path(path)
    ensure_metadata(str(path) + '.meta.json', config)
    if path.exists():
        raise ValueError(f'Refusing to overwrite tensor artifact: {path}')
    temporary = path.with_suffix('.tmp.safetensors')
    save_file({key: value.cpu().contiguous() for key, value in tensors.items()}, str(temporary))
    temporary.replace(path)
    write_json(str(path) + '.integrity.json', dict(sha256=file_sha256(path)))


def load_tensors(path, config=None):
    from safetensors.torch import load_file
    path = Path(path)
    meta = read_json(str(path) + '.meta.json')
    if config is not None and meta != config:
        raise ValueError(f'Tensor artifact configuration differs: {path}')
    if read_json(str(path) + '.integrity.json')['sha256'] != file_sha256(path):
        raise ValueError(f'Tensor artifact hash mismatch: {path}')
    return load_file(str(path)), meta


def checked_rows(args):
    rows, cards = read_jsonl(args.data), read_json(args.cards)
    if not rows:
        raise ValueError('Empty evaluation data')
    unique_index(rows, 'evaluation')
    for row in rows:
        validate_row(row, prepared=True)
    names = [card['id'] for card in cards]
    if set(names) != set(TASKS) or len(names) != len(TASKS):
        raise ValueError('Expected all 14 expert cards')
    return rows, cards, names


def prepare_support(args):
    raw = Path(args.source)
    source_meta = read_json(str(raw) + '.meta.json')
    if source_meta['sha256'] != file_sha256(raw):
        raise ValueError('Raw source changed')
    source = read_json(raw)
    data_manifest = read_json('artifacts/portal/data/manifest.json')
    if digest(source) != data_manifest['source_hash'] or source_meta['source'] != data_manifest['source']:
        raise ValueError('Support source does not match frozen benchmark')
    records, audit = sample_training_support(source, args.support_count, args.seed)
    root = Path(args.soup_index).parent
    path = root / 'support.jsonl'
    config = dict(schema='portal-adaptersoup-support-v1', source=source_meta, seed=args.seed,
        count_per_expert=args.support_count, support_hash=digest(records), audit=audit,
        text_policy='prompt+all shuffled candidates; no gold/task label in encoded text',
        exclusion='canonical prompt against ALL source validation; dedup within each expert', code=code_identity())
    ensure_metadata(str(path) + '.meta.json', config)
    if path.exists():
        if read_jsonl(path) != records:
            raise ValueError('Saved support inputs changed')
    else:
        write_jsonl(path, records)
    return records, config


def embedding_model(args):
    import torch
    from transformers import AutoModel, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError('Embedding execution requires CUDA')
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.embedding_dir, local_files_only=True, padding_side='left')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModel.from_pretrained(args.embedding_dir, local_files_only=True, dtype=getattr(torch, args.dtype),
                                    device_map={'': 0}, attn_implementation='sdpa')
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer


def soup_index(args):
    import torch
    path = Path(args.soup_index)
    with locked(str(path) + '.lock'):
        records, support_meta = prepare_support(args)
        info = load_embedding_manifest(args.embedding_dir)
        names = list(TASKS)
        config = dict(schema='portal-adaptersoup-index-v1', support=support_meta, experts=names,
            embedding=info, dtype=args.dtype, pooling='last-nonpadding-token', sample_normalization='L2',
            aggregation='mean of sample cosine similarities; mean vector NOT renormalized',
            query_instruction=EMBEDDING_INSTRUCTION, max_input_tokens=args.max_input_tokens,
            batch_size=args.embedding_batch_size, versions=versions(), code=code_identity())
        if path.exists():
            tensors, _ = load_tensors(path, config)
            expected = mean_domain_vectors(tensors['samples'], records, names)
            if not torch.equal(tensors['means'], expected):
                raise ValueError('Index domain means differ from saved samples')
            print(f'Already complete: {path}', flush=True)
            return
        start = time.perf_counter()
        model, tokenizer = embedding_model(args)
        vectors, counts = encode_embeddings(model, tokenizer, [r['inputs'] for r in records],
                                             args.max_input_tokens, args.embedding_batch_size)
        torch.cuda.synchronize()
        # Compute means on CPU consistently for validation and all later runs.
        vectors = vectors.cpu().float()
        means = mean_domain_vectors(vectors, records, names)
        save_tensors(path, dict(samples=vectors, means=means), config)
        write_json(str(path) + '.runtime.json', dict(build_s=time.perf_counter()-start, samples=len(records),
            input_tokens=sum(counts), peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
            gpu=torch.cuda.get_device_name(0)))
        print(f'AdapterSoup index ready: {len(records)} training inputs; {path}', flush=True)


def validate_soup_route(row, rr, names):
    scores = rr.get('scores', {})
    import math
    if rr.get('kind') != 'adaptersoup' or rr.get('input_hash') != row['input_hash'] or set(scores) != set(names):
        raise ValueError('AdapterSoup route/input mismatch')
    if any(not isinstance(v, (float, int)) or isinstance(v, bool) or not math.isfinite(v) or abs(v) > 1.00001 for v in scores.values()):
        raise ValueError('Invalid sample cosine similarity')
    if rr.get('weights') != top_weights(scores, 2, False):
        raise ValueError('AdapterSoup selected weights differ from Top-2 equal')


def soup_route(args):
    import torch
    rows, cards, names = checked_rows(args)
    out = Path(args.run_dir) / 'adaptersoup.routes.jsonl'
    if args.output is not None and Path(args.output).resolve() != out.resolve():
        raise ValueError('--output must match the canonical run-directory route path')
    with locked(str(out) + '.lock'):
        tensors, index_meta = load_tensors(args.soup_index)
        info = load_embedding_manifest(args.embedding_dir)
        if index_meta['embedding'] != info or index_meta['code'] != code_identity() or index_meta['dtype'] != args.dtype:
            raise ValueError('Index encoder/code/dtype changed')
        meta = dict(schema='portal-local-router-v2', kind='adaptersoup', implementation_hash=implementation_hash(),
            data_hash=digest(rows), cards_hash=digest(cards), seed=args.seed, dtype=args.dtype, versions=versions(),
            index_metadata=index_meta, index_sha256=file_sha256(args.soup_index), max_input_tokens=args.max_input_tokens,
            query_instruction=EMBEDDING_INSTRUCTION, top_k=2, weights='equal', similarity_threshold=None,
            scope='per question; no target-domain validation pool', code=code_identity())
        ensure_metadata(str(out) + '.meta.json', meta)
        old = unique_index(read_jsonl(out, repair_tail=True), 'routes')
        by_id = unique_index(rows, 'data')
        if not set(old) <= set(by_id):
            raise ValueError('Unknown route IDs')
        for rid, rr in old.items():
            validate_soup_route(by_id[rid], rr, names)
        todo = [row for row in rows if row['id'] not in old]
        if not todo:
            print(f'Already complete: {out}', flush=True)
            return
        start = time.perf_counter()
        model, tokenizer = embedding_model(args)
        means = tensors['means'].to('cuda:0')
        if index_meta['experts'] != names:
            means = means[[index_meta['experts'].index(n) for n in names]]
        warm, _ = encode_embeddings(model, tokenizer, ['Hello world'], args.max_input_tokens, 1)
        torch.cuda.synchronize()
        write_json(str(out) + '.runtime.json', dict(load_and_warmup_s=time.perf_counter()-start, gpu=torch.cuda.get_device_name(0)))
        with out.open('a', encoding='utf-8') as handle:
            for i, row in enumerate(todo, 1):
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                text = f'Instruct: {EMBEDDING_INSTRUCTION}\nQuery:{row["inputs"]}'
                vector, counts = encode_embeddings(model, tokenizer, [text], args.max_input_tokens, 1)
                scores = dict(zip(names, (vector @ means.T).squeeze(0).tolist()))
                torch.cuda.synchronize()
                record = dict(id=row['id'], input_hash=row['input_hash'], kind='adaptersoup', scores=scores,
                    weights=top_weights(scores, 2, False), latency_s=time.perf_counter()-start,
                    score_type='mean-sample-cosine-not-probability', usage={'input_tokens': counts[0]},
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                validate_soup_route(row, record, names)
                append_jsonl(handle, record)
                if i % 10 == 0 or i == len(todo):
                    print(f'AdapterSoup routing: {len(old)+i}/{len(rows)}', flush=True)


def arrow_prototypes(args):
    import torch
    from safetensors.torch import load_file
    torch.set_num_threads(4)
    path = Path(args.arrow_prototypes)
    with locked(str(path) + '.lock'):
        manifest = load_manifest(args.model_dir, TASKS)
        reference = ROOT / 'artifacts/references/mttl'
        commit = subprocess.check_output(['git', '-C', str(reference), 'rev-parse', 'HEAD'], text=True).strip()
        if commit != MTTL_COMMIT:
            raise ValueError('Arrow upstream checkout differs from audited commit')
        upstream = dict(repo='https://github.com/microsoft/mttl', commit=commit,
                        files={p: file_sha256(reference / p) for p in MTTL_FILES})
        config = dict(schema='portal-arrow-prototypes-v1', experts=list(TASKS),
            adapters={n: manifest['models'][n] for n in TASKS}, upstream=upstream,
            algorithm='exact low-rank SVD of B@A; top input-space right singular vector',
            ab_only=True, scale_by_eigenvalue=False, tied_modules=False, dtype='float32',
            sign='largest-absolute-coordinate positive', versions=versions(), code=code_identity())
        if path.exists():
            load_tensors(path, config)
            print(f'Already complete: {path}', flush=True)
            return
        start = time.perf_counter()
        matrices, eigenvalues, inventory = {}, {}, None
        for expert in TASKS:
            weights = load_file(str(Path(args.model_dir) / manifest['models'][expert]['path'] / 'adapter_model.safetensors'))
            keys = sorted(k for k in weights if k.endswith('.lora_A.weight'))
            if inventory is None:
                inventory = keys
            elif inventory != keys:
                raise ValueError('Expert target-module inventories differ')
            for key in keys:
                mate = key.replace('.lora_A.weight', '.lora_B.weight')
                module = key.removesuffix('.lora_A.weight').removeprefix('base_model.model.')
                proto, eigenvalue = input_svd_prototype(weights[key], weights[mate])
                matrices.setdefault(module, []).append(proto)
                eigenvalues.setdefault(module, []).append(eigenvalue)
            print(f'Arrow prototypes: {expert}, {len(keys)} projections', flush=True)
        packed = {key + '.prototype': torch.stack(values) for key, values in matrices.items()}
        packed.update({key + '.eigenvalue': torch.stack(values) for key, values in eigenvalues.items()})
        save_tensors(path, packed, config)
        write_json(str(path) + '.runtime.json', dict(build_s=time.perf_counter()-start, device='cpu', threads=4,
                                                    experts=len(TASKS), projections=len(matrices)))
        print(f'Arrow prototypes ready: {path}', flush=True)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('command', choices=('soup-index', 'soup-route', 'arrow-prototypes'))
    result.add_argument('--run-dir', type=Path, default=DEFAULT_RUN)
    result.add_argument('--data', default='artifacts/portal/data/eval.jsonl')
    result.add_argument('--cards', default='artifacts/portal/data/cards.json')
    result.add_argument('--model-dir', default='models/portal-qwen3-4b')
    result.add_argument('--embedding-dir', default='models/qwen3-embedding-0.6b')
    result.add_argument('--source', default='data/raw/portal/portal_tasks.json')
    result.add_argument('--soup-index', default='artifacts/portal/adaptersoup-qwen-n100-seed42/index.safetensors')
    result.add_argument('--arrow-prototypes', default='artifacts/portal/arrow/prototypes.safetensors')
    result.add_argument('--support-count', type=int, default=100)
    result.add_argument('--seed', type=int, default=42)
    result.add_argument('--dtype', choices=('bfloat16', 'float16'), default='bfloat16')
    result.add_argument('--embedding-batch-size', type=int, default=4)
    result.add_argument('--max-input-tokens', type=int, default=4096)
    result.add_argument('--output', help='Explicit canonical output for the existing stage profiler')
    return result


if __name__ == '__main__':
    args = parser().parse_args()
    os.chdir(ROOT)
    {'soup-index': soup_index, 'soup-route': soup_route, 'arrow-prototypes': arrow_prototypes}[args.command](args)
