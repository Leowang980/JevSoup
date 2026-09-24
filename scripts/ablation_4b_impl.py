"""Input-hash-based Random Top-2 control for the paper ablation."""
import random


from jev_lora.core import digest


def random_weights(input_hash, names, seed):
    """Uniform unordered pair; depends only on input hash and routing seed.

    Duplicate inputs share a decision. Alphabetical execution order is fixed,
    so changing the routing seed does not introduce a second ordering factor.
    No gold label, task annotation, or Jev score enters this random router.
    """
    if seed not in (42, 43, 44) or len(names) != len(set(names)) or len(names) < 2:
        raise ValueError('Expected a supported seed and distinct expert names')
    rng = random.Random(digest(dict(input_hash=input_hash, routing_seed=seed)))
    return {name: .5 for name in sorted(rng.sample(sorted(names), 2))}
