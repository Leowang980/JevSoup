"""Probability-weighted composition with the existing frozen Orth projection."""
import math

from orthogonal_lora import OrthogonalMixture


class ProbabilityOrthogonalMixture(OrthogonalMixture):
    """Only the outer coefficients differ from equal-weight Orth.

    Pair order is supplied by descending cached routing probability. Projection
    uses the unweighted complete first update, retains the original LoRA scales,
    and never restores the second update's norm. Strength zero is audit-only.
    """
    def activate(self, weights, strength=1):
        if (len(weights) != 2 or strength not in (0, 1)
                or any(isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in weights.values())
                or not math.isclose(sum(weights.values()), 1.0, abs_tol=1e-6)
                or not set(weights) <= set(self.model.peft_config)):
            raise ValueError('Expected two known experts, normalized weights, and strength 0 or 1')
        self.active = self.prepare_pair(tuple(weights)) if strength else None
        self.reference.activate(weights)
        self.weights, self.enabled = dict(weights), True
