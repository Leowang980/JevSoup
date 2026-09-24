"""Isolated partial orthogonalization; the existing endpoint engine stays frozen."""
import math
import random
import statistics
from collections import defaultdict

import torch

from jev_lora.core import canonical_text
from jev_lora.evaluation import percentile
from orthogonal_lora import OrthogonalMixture, update_norm


STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)


def partial_factor(a, basis, strength):
    if isinstance(strength, bool) or strength not in STRENGTHS:
        raise ValueError('Expected a preregistered strength: 0, .25, .5, .75, 1')
    a = a.detach().to(device='cpu', dtype=torch.float64)
    if not torch.isfinite(a).all() or not torch.isfinite(basis).all():
        raise ValueError('Nonfinite projection factors')
    # Compute from original float64 factors, NOT interpolation of BF16 endpoints.
    return a if strength == 0 else a - strength * ((a @ basis) @ basis.T)


class PartialOrthogonalMixture(OrthogonalMixture):
    """One fixed intermediate strength per worker; exact old endpoints for audits."""
    def __init__(self, model, names, strength, max_entries=128):
        if strength not in STRENGTHS[1:-1]:
            raise ValueError('Only the three intermediate strengths need new workers')
        self.strength = strength
        self.endpoint_cache = {}
        super().__init__(model, names, max_entries)

    def prepare_pair(self, pair):
        pair = tuple(pair)
        present = pair in self.cache
        tensors = super().prepare_pair(pair)
        if present:
            return tensors
        self.endpoint_cache[pair] = dict(tensors)
        first, second = pair
        for key, item in self.layers.items():
            a2, b2 = item['cpu'][second]
            partial = partial_factor(a2, self.bases[key, first], self.strength)
            target = item['layer'].lora_A[second].weight
            tensors[key] = partial.to(device=target.device, dtype=target.dtype)
            original_norm = update_norm(a2, b2)
            self.diagnostics['|'.join(pair)][key].update(
                strength=self.strength,
                partial_second_norm_retained=update_norm(partial, b2)/original_norm if original_norm else 1.0,
                partial_stored_second_norm_retained=update_norm(tensors[key].double().cpu(), b2)/original_norm if original_norm else 1.0)
        for old in list(self.endpoint_cache):
            if old not in self.cache:
                del self.endpoint_cache[old]
        return tensors

    def activate_endpoint(self, weights, strength):
        if strength not in (0, 1):
            raise ValueError('Endpoint audit requires 0 or 1')
        super().activate(weights, strength=strength)
        if strength:
            self.active = self.endpoint_cache[tuple(weights)]

    def close(self):
        super().close()
        self.endpoint_cache.clear()


def paired_metrics(rows, left, right, draws=1000, seed=42):
    """Task-stratified paired bootstrap; duplicate prompts are sampled as clusters.

    Matches the existing macro diagnostic's RNG/sampling, and adds Micro CIs.
    These are pointwise intervals, not a simultaneous band or a tuning criterion.
    """
    if not rows or draws < 1:
        raise ValueError('Nonempty rows and positive bootstrap draws required')
    groups = defaultdict(lambda: defaultdict(list))
    diffs = []
    for row in rows:
        delta = float(right[row['id']]) - float(left[row['id']])
        groups[row['task']][canonical_text(row['prompt'])].append(delta)
        diffs.append(delta)
    macro = statistics.mean(statistics.mean(d for values in task.values() for d in values)
                            for task in groups.values())
    compressed = [[(sum(values), len(values)) for values in task.values()] for task in groups.values()]
    rng = random.Random(seed)
    macro_samples, micro_samples = [], []
    for _ in range(draws):
        task_scores, total, count = [], 0, 0
        for clusters in compressed:
            selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
            task_sum, task_n = sum(s for s, n in selected), sum(n for s, n in selected)
            task_scores.append(task_sum/task_n)
            total += task_sum
            count += task_n
        macro_samples.append(100*statistics.mean(task_scores))
        micro_samples.append(100*total/count)
    return dict(delta_macro_pp=100*macro, delta_micro_pp=100*statistics.mean(diffs),
                macro_ci95_low=percentile(macro_samples, .025), macro_ci95_high=percentile(macro_samples, .975),
                micro_ci95_low=percentile(micro_samples, .025), micro_ci95_high=percentile(micro_samples, .975),
                wins=sum(d > 0 for d in diffs), losses=sum(d < 0 for d in diffs), ties=sum(d == 0 for d in diffs),
                task_prompt_clusters=sum(len(task) for task in groups.values()))
