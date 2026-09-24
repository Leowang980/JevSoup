"""Frozen, rank-ordered input-subspace orthogonalization of two LoRA updates.

For each adapted projection, Q spans the row space of B1 @ A1. Keep the first
expert unchanged and replace A2 with A2 - (A2 @ Q) @ Q.T. The complete second
update is therefore Delta2 @ (I - Q @ Q.T), not an average of LoRA factors.
Both branches retain weight 0.5 and their original LoRA scaling. No norm
restoration, training, label access, or clipping is used. This is an exploratory
post-training adaptation, not a reproduction of O-LoRA's training algorithm.
"""
import math
from collections import OrderedDict
from types import MethodType

import torch
import torch.nn.functional as F

from jev_lora.models import ExactMixture


def update_inner(a, b, c, d):
    """Frobenius inner product of BA and DC without dense updates."""
    return ((b.T @ d) * (a @ c.T)).sum()


def update_norm(a, b):
    return math.sqrt(max(0.0, float(update_inner(a, b, a, b))))


def update_row_basis(a, b, rtol=1e-10):
    """Exact complete-update row space via low-rank QR/SVD, CPU float64.

    Unlike blindly using row(A), this also handles rank-deficient or zero B.
    The relative tolerance is fixed before evaluation; it is not tuned.
    """
    a, b = a.detach().to(device='cpu', dtype=torch.float64), b.detach().to(device='cpu', dtype=torch.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
        raise ValueError('Invalid LoRA factor shapes')
    if not torch.isfinite(a).all() or not torch.isfinite(b).all() or not 0 < rtol < 1:
        raise ValueError('Invalid factors or SVD tolerance')
    _, rb = torch.linalg.qr(b, mode='reduced')
    _, s, vh = torch.linalg.svd(rb @ a, full_matrices=False)
    rank = int((s > s[0] * rtol).sum()) if len(s) and s[0] > 0 else 0
    return vh[:rank].T.contiguous()


def orthogonal_factor(a, basis):
    a = a.detach().to(device='cpu', dtype=torch.float64)
    return a - (a @ basis) @ basis.T


def projection_diagnostics(a1, b1, a2, b2, basis, residual, stored):
    n1, n2 = update_norm(a1, b1), update_norm(a2, b2)
    nr = update_norm(residual, b2)
    stored = stored.to(device='cpu', dtype=torch.float64)
    ns = update_norm(stored, b2)
    cosine = lambda a, n: float(update_inner(a1, b1, a, b2)) / (n1*n) if n1*n else 0.0
    removed = update_norm(a2-residual, b2)
    # Denominator is the original second update, so near-zero residuals are safe.
    overlap = update_norm(stored @ basis, b2) / n2 if n2 else 0.0
    return dict(anchor_rank=basis.shape[1], cosine_before=cosine(a2,n2),
                cosine_after_float64=cosine(residual,nr), cosine_after_stored=cosine(stored,ns),
                second_norm_retained=nr/n2 if n2 else 1.0,
                second_removed_energy_fraction=(removed/n2)**2 if n2 else 0.0,
                stored_overlap_relative_to_original=overlap,
                float64_overlap_relative_to_original=update_norm(residual @ basis,b2)/n2 if n2 else 0.0)


class OrthogonalMixture:
    """Reversible inference-only override; never modifies model parameters.

    Ordered-pair residuals are cached. Control strength=0 uses original factors
    in the same override, allowing exact comparison to the existing PEFT path.
    """
    def __init__(self, model, names, max_entries=128):
        from peft.tuners.lora.layer import LoraLayer
        if max_entries < 1:
            raise ValueError('Positive cache size required')
        self.model, self.names = model, list(names)
        self.reference = ExactMixture(model)
        self.initial_active = list(model.active_adapters)
        self.layers, self.bases = {}, {}
        self.cache, self.diagnostics = OrderedDict(), {}
        self.max_entries, self.hits, self.misses = max_entries, 0, 0
        self.enabled, self.weights, self.active = False, None, None
        for key, layer in model.named_modules():
            if not isinstance(layer, LoraLayer):
                continue
            if not key.endswith(('q_proj', 'v_proj')):
                raise ValueError('Only q/v projections are supported')
            if set(layer.lora_A) != set(names) or set(layer.lora_B) != set(names):
                raise ValueError('Adapter pool mismatch')
            if layer.merged or layer.disable_adapters or any(layer.use_dora.values()):
                raise ValueError('Merged, disabled and DoRA adapters are unsupported')
            if any(layer.lora_A[n].bias is not None or layer.lora_B[n].bias is not None for n in names):
                raise ValueError('LoRA bias unsupported')
            if any(not math.isfinite(layer.scaling[n]) or layer.scaling[n] <= 0 for n in names):
                raise ValueError('Positive original LoRA scaling required')
            for n in names:
                denominator = math.sqrt(layer.r[n]) if model.peft_config[n].use_rslora else layer.r[n]
                if not math.isclose(layer.scaling[n],layer.lora_alpha[n]/denominator):
                    raise ValueError('Construct the mixer before applying mixture weights')
            cpu = {n: (layer.lora_A[n].weight.detach().to(device='cpu',dtype=torch.float64),
                       layer.lora_B[n].weight.detach().to(device='cpu',dtype=torch.float64)) for n in names}
            self.layers[key] = dict(layer=layer, old_forward=layer.forward,
                                    scales=dict(layer.scaling), cpu=cpu)
        if not self.layers:
            raise ValueError('No LoRA projections')
        for key, item in self.layers.items():
            def patched(layer, x, *args, _key=key, **kwargs):
                return self.forward(_key, x, *args, **kwargs)
            item['layer'].forward = MethodType(patched, item['layer'])
        model.requires_grad_(False)
        model.eval()

    def prepare_pair(self, pair):
        pair = tuple(pair)
        if len(pair) != 2 or pair[0] == pair[1] or not set(pair) <= set(self.names):
            raise ValueError('Two distinct known experts required')
        if pair in self.cache:
            self.hits += 1
            self.cache.move_to_end(pair)
            return self.cache[pair]
        self.misses += 1
        tensors, diagnostics = {}, {}
        first, second = pair
        for key, item in self.layers.items():
            a1,b1 = item['cpu'][first]
            a2,b2 = item['cpu'][second]
            if (key,first) not in self.bases:
                self.bases[key,first] = update_row_basis(a1,b1)
            basis = self.bases[key,first]
            residual = orthogonal_factor(a2,basis)
            target = item['layer'].lora_A[second].weight
            tensors[key] = residual.to(device=target.device,dtype=target.dtype)
            stats = projection_diagnostics(a1,b1,a2,b2,basis,residual,tensors[key])
            if stats['float64_overlap_relative_to_original'] > 1e-9:
                raise ValueError('Projection failed float64 orthogonality audit')
            if stats['stored_overlap_relative_to_original'] > .01:
                raise ValueError('Projection rounding error exceeds 1% of original update')
            diagnostics[key] = stats
        if len(self.cache) >= self.max_entries:
            self.cache.popitem(last=False)
        self.cache[pair] = tensors
        self.diagnostics['|'.join(pair)] = diagnostics
        return tensors

    def activate(self, weights, strength=1):
        if len(weights) != 2 or any(v != .5 for v in weights.values()) or strength not in (0,1):
            raise ValueError('This fixed experiment requires Top-2 equal and strength 0 or 1')
        if not set(weights) <= set(self.names):
            raise ValueError('Unknown adapter')
        self.active = self.prepare_pair(tuple(weights)) if strength else None
        self.reference.activate(weights)
        self.weights, self.enabled = dict(weights), True

    def activate_reference(self, weights):
        self.reference.activate(weights)
        self.enabled = False

    def forward(self, key, x, *args, **kwargs):
        item = self.layers[key]
        if not self.enabled:
            return item['old_forward'](x,*args,**kwargs)
        layer = item['layer']
        if layer.training or layer.merged or layer.disable_adapters:
            raise RuntimeError('Orthogonal mixture is frozen, unmerged inference only')
        if kwargs.get('adapter_names') is not None:
            raise ValueError('External batch adapter selection unsupported')
        kwargs.pop('adapter_names',None)
        result = layer.base_layer(x,*args,**kwargs)
        dtype = result.dtype
        for i,(name,weight) in enumerate(self.weights.items()):
            a = self.active[key] if i == 1 and self.active is not None else layer.lora_A[name].weight
            b = layer.lora_B[name].weight
            branch = F.linear(F.linear(x.to(a.dtype),a),b)
            result = result + branch * (item['scales'][name]*weight)
        return result.to(dtype)

    def close(self):
        self.model.base_model.set_adapter(self.initial_active)
        for item in self.layers.values():
            item['layer'].forward = item['old_forward']
            item['layer'].scaling.update(item['scales'])
        self.model.requires_grad_(False)
        self.model.eval()
        self.cache.clear()
