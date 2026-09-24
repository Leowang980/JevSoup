"""Frozen Arrow and AdapterSoup kernels; independent of existing cache identity.

Arrow follows microsoft/mttl commit 169c9191be960e35a59e85c37af90f3f518fe125:
unit input-space SVD prototypes, abs-dot logits, top-k softmax (T=1), and
lora_merge_after=False (mix A and B BEFORE their product). No training.
AdapterSoup maps the paper's parameter averaging to LoRA factor averaging;
this is explicitly different from averaging the complete BA updates.
"""
import math
import random
from collections import Counter, defaultdict
from types import MethodType

import torch
from torch.nn import functional as F

from jev_lora.core import TASKS, canonical_text, digest
from jev_lora.data import routing_text

MTTL_COMMIT = '169c9191be960e35a59e85c37af90f3f518fe125'
MTTL_FILES = (
    'mttl/models/containers/selectors/arrow_selector.py',
    'mttl/models/containers/selectors/per_token_selector.py',
    'mttl/models/containers/selectors/base.py',
    'mttl/models/library/library_transforms.py',
    'mttl/models/modifiers/lora.py',
    'mttl/models/containers/lora_containers.py',
)


def sample_training_support(source, count=100, seed=42, names=TASKS):
    """Choose train-only inputs, blocking ALL validation prompts across tasks."""
    if count < 1 or not source.get('train') or not source.get('validation'):
        raise ValueError('Positive support count and train/validation splits required')
    blocked = {canonical_text(row['prompt']) for row in source['validation']}
    pools, seen = defaultdict(list), defaultdict(set)
    excluded = Counter()
    for index, row in enumerate(source['train']):
        task, key = row['task'], canonical_text(row['prompt'])
        if task not in names:
            continue
        if key in blocked:
            excluded['validation_overlap'] += 1
            continue
        if key in seen[task]:
            excluded['duplicate_within_task'] += 1
            continue
        seen[task].add(key)
        # Match the existing data builder's training-choice permutation, without
        # ever reading or serializing the gold answer.
        choices = list(row['choices'])
        order = list(range(len(choices)))
        random.Random(f'{seed}:train:{index}').shuffle(order)
        choices = [choices[j] for j in order]
        pools[task].append(dict(id=f'portal-train-{index:06d}', expert=task, source_index=index,
            prompt=row['prompt'], choices=choices, inputs=routing_text(row['prompt'], choices),
            input_hash=digest(dict(prompt=row['prompt'], choices=choices))))
    output = []
    for task in names:
        values = list(pools[task])
        random.Random(f'{seed}:adaptersoup:{task}').shuffle(values)
        if not values:
            raise ValueError(f'No disjoint training inputs for {task}')
        output.extend(values[:count])
    return output, dict(excluded= dict(excluded), eligible={t: len(pools[t]) for t in names},
                        selected=dict(Counter(row['expert'] for row in output)))


def mean_domain_vectors(vectors, records, names):
    """Do NOT renormalize the mean: q @ mean(e_i) = mean cosine(q, e_i)."""
    if vectors.ndim != 2 or len(records) != len(vectors) or not torch.isfinite(vectors).all():
        raise ValueError('Invalid support vectors')
    if not torch.allclose(vectors.float().norm(dim=-1), torch.ones(len(vectors), device=vectors.device), atol=2e-4):
        raise ValueError('Support vectors must be unit normalized')
    means = []
    for name in names:
        indices = [i for i, row in enumerate(records) if row['expert'] == name]
        if not indices:
            raise ValueError(f'Missing support for {name}')
        means.append(vectors[indices].float().mean(dim=0))
    return torch.stack(means)


@torch.no_grad()
def input_svd_prototype(a, b):
    """Top right singular vector of PEFT B@A, via MTTL's low-rank SVD.

MTTL stores A.T and B.T relative to PEFT. Work in float32 on CPU; no dense
out_features x in_features matrix or randomized approximation is necessary.
"""
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError('Expected PEFT A=[r,in], B=[out,r]')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError('Nonfinite adapter factors')
    aa, bb = a.float().T, b.float()
    ua, sa, vha = torch.linalg.svd(aa, full_matrices=False)
    ub, sb, vhb = torch.linalg.svd(bb, full_matrices=False)
    core = sa.diag() @ vha @ vhb.T @ sb.diag()
    uc, singular, _ = torch.linalg.svd(core, full_matrices=False)
    if not torch.isfinite(singular).all() or singular[0] <= 0:
        raise ValueError('Zero/invalid adapter has no unique principal direction')
    vector = (ua @ uc)[:, 0].contiguous()
    # SVD sign is arbitrary; fix it for repeatable artifacts. Routing uses abs.
    if vector[vector.abs().argmax()] < 0:
        vector = -vector
    return vector, singular[0].square()


def arrow_selection(x, prototypes, top_k=2, temperature=1.0):
    if not 1 <= top_k <= len(prototypes) or temperature <= 0:
        raise ValueError('Invalid Arrow top-k or temperature')
    scores = F.linear(x.float(), prototypes.float()).abs() / temperature
    top_scores, indices = torch.topk(scores, top_k, dim=-1)
    return scores, indices, F.softmax(top_scores, dim=-1)


def selected_factor_forward(x, a, b, indices, weights, scale=1.0, chunk_size=32):
    """Per-token B_bar(A_bar(x)), NOT sum_i w_i B_i A_i x.

Token chunking caps temporary memory without altering the selected experts.
a=[E,r,in], b=[E,out,r]; same rank and scaling required by this experiment.
"""
    if chunk_size < 1 or x.shape[-1] != a.shape[-1]:
        raise ValueError('Invalid factor routing input/chunk size')
    flat = x.reshape(-1, x.shape[-1]).to(a.dtype)
    ids = indices.reshape(len(flat), -1)
    ws = weights.reshape_as(ids).to(a.dtype)
    output = torch.empty((len(flat), b.shape[1]), dtype=a.dtype, device=x.device)
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        chosen, w = ids[start:stop], ws[start:stop]
        abar = torch.einsum('tk,tkrd->trd', w, a[chosen])
        bbar = torch.einsum('tk,tkor->tor', w, b[chosen])
        latent = torch.bmm(abar, flat[start:stop, :, None])
        output[start:stop] = torch.bmm(bbar, latent).squeeze(-1) * scale
    return output.reshape(*x.shape[:-1], b.shape[1]).to(x.dtype)


def layer_key(name):
    return name.removeprefix('base_model.model.')


class FactorMixture:
    """Reversible local forward override on frozen PEFT q/v projections."""
    def __init__(self, model, names, mode, prototypes=None, top_k=2, temperature=1.0, chunk_size=32):
        from peft.tuners.lora.layer import LoraLayer
        if mode not in ('arrow', 'adaptersoup'):
            raise ValueError('Unknown factor-mixture mode')
        self.model, self.names, self.mode = model, list(names), mode
        self.top_k, self.temperature, self.chunk_size = top_k, temperature, chunk_size
        self.layers = {}
        self.capture = False
        self.reset_trace()
        for name, layer in list(model.named_modules()):
            if not isinstance(layer, LoraLayer):
                continue
            key = layer_key(name)
            if set(layer.lora_A) != set(names) or set(layer.lora_B) != set(names):
                raise ValueError('Missing or unexpected adapter at ' + key)
            if layer.merged or any(layer.use_dora.values()):
                raise ValueError('Merged/DoRA adapters are not supported')
            if any(layer.lora_A[n].bias is not None or layer.lora_B[n].bias is not None for n in names):
                raise ValueError('LoRA bias is unsupported')
            scales = [layer.scaling[n] for n in names]
            if not all(math.isclose(s, scales[0]) for s in scales):
                raise ValueError('Factor averaging requires identical scaling')
            a = torch.stack([layer.lora_A[n].weight.detach() for n in names])
            b = torch.stack([layer.lora_B[n].weight.detach() for n in names])
            item = dict(layer=layer, old_forward=layer.forward, a=a, b=b, scale=scales[0])
            if mode == 'arrow':
                if prototypes is None or key not in prototypes:
                    raise ValueError('Missing Arrow prototype: ' + key)
                proto = prototypes[key]
                if tuple(proto.shape) != (len(names), a.shape[-1]) or not torch.isfinite(proto).all():
                    raise ValueError('Invalid Arrow prototype shape/value')
                if not torch.allclose(proto.float().norm(dim=-1), torch.ones(len(names), device=proto.device), atol=2e-4):
                    raise ValueError('Arrow prototypes must be unit vectors')
                item['prototypes'] = proto.to(a.device, dtype=torch.float32)
            self.layers[key] = item
        if not self.layers:
            raise ValueError('No LoRA layers found')
        if mode == 'arrow' and set(self.layers) != set(prototypes):
            raise ValueError('Prototype module inventory differs from model')
        # Finish all validation before replacing any forward.
        for key, item in self.layers.items():
            def patched(layer, x, *args, _key=key, **kwargs):
                return self.forward(_key, x, *args, **kwargs)
            item['layer'].forward = MethodType(patched, item['layer'])
        model.requires_grad_(False)
        model.eval()

    def activate(self, weights):
        if self.mode != 'adaptersoup' or not weights or not set(weights) <= set(self.names):
            raise ValueError('Invalid AdapterSoup selection')
        if any(not math.isfinite(v) or v < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-6):
            raise ValueError('Invalid mixture weights')
        ids = [self.names.index(name) for name in weights]
        for item in self.layers.values():
            ws = torch.tensor(list(weights.values()), device=item['a'].device, dtype=item['a'].dtype)
            item['abar'] = torch.einsum('k,krd->rd', ws, item['a'][ids])
            item['bbar'] = torch.einsum('k,kor->or', ws, item['b'][ids])

    def reset_trace(self, capture=False):
        self.capture = capture
        self.trace = {}
        self.events = []
        self.calls = Counter()

    def forward(self, key, x, *args, **kwargs):
        item = self.layers[key]
        if item['layer'].training:
            raise RuntimeError('These baselines are inference-only')
        if kwargs.get('adapter_names') is not None:
            raise ValueError('External mixed-batch adapter selection is unsupported')
        kwargs.pop('adapter_names', None)
        base = item['layer'].base_layer(x, *args, **kwargs)
        if self.mode == 'adaptersoup':
            if 'abar' not in item:
                raise RuntimeError('Activate AdapterSoup before inference')
            delta = F.linear(F.linear(x.to(item['a'].dtype), item['abar']), item['bbar']) * item['scale']
        else:
            timed = self.capture and x.is_cuda
            if timed:
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
            scores, indices, weights = arrow_selection(x, item['prototypes'], self.top_k, self.temperature)
            if timed:
                end.record()
                self.events.append((start, end))
            if self.capture:
                call = self.calls[key]
                self.calls[key] += 1
                prefix = f'choice{call:03d}.{key}'
                self.trace[prefix + '.scores'] = scores.detach()
                self.trace[prefix + '.experts'] = indices.detach().to(torch.int16)
                self.trace[prefix + '.weights'] = weights.detach()
            delta = selected_factor_forward(x, item['a'], item['b'], indices, weights,
                item['scale'], self.chunk_size)
        return base + delta.to(base.dtype)

    def trace_cpu(self, expected_choices):
        if any(self.calls[key] != expected_choices for key in self.layers):
            raise ValueError('Expected one forward per projection per candidate (batch size 1)')
        if self.events:
            torch.cuda.synchronize()
        router_cuda_s = sum(start.elapsed_time(end) for start, end in self.events) / 1000
        return {key: value.cpu().contiguous() for key, value in self.trace.items()}, router_cuda_s

    def close(self):
        for item in self.layers.values():
            item['layer'].forward = item['old_forward']
        self.reset_trace()
