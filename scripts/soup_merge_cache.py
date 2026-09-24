"""Bounded memoization of the unchanged AdapterSoup factor-averaging kernel.

Keys preserve expert order and exact weights. Cached tensors stay on the same
device/dtype as the frozen mixer. The public runner records source identity;
the numerical kernel is unchanged.
"""
import math
from collections import OrderedDict

from adapter_baseline_impl import FactorMixture

MAX_ENTRIES = 128
MAX_BYTES = 1024 ** 3
class CachedFactorMixture(FactorMixture):
    def __init__(self, *args, max_entries=MAX_ENTRIES, max_bytes=MAX_BYTES, **kwargs):
        if max_entries < 1 or max_bytes < 1:
            raise ValueError('Cache limits must be positive')
        super().__init__(*args, **kwargs)
        if self.mode != 'adaptersoup':
            self.close()
            raise ValueError('Merge cache is only for static AdapterSoup, never Arrow')
        self.cache = OrderedDict()
        self.max_entries, self.max_bytes = max_entries, max_bytes
        self.hits = self.misses = self.evictions = self.cache_bytes = self.peak_cache_bytes = 0
        self.last_cache_hit = False

    def activate(self, weights):
        # Validate even on hits. Do not canonicalize order, round weights, change
        # dtype, or replace the audited einsums with a different expression.
        if not weights or not set(weights) <= set(self.names):
            raise ValueError('Invalid AdapterSoup selection')
        if any(not math.isfinite(v) or v < 0 for v in weights.values()) or not math.isclose(sum(weights.values()), 1, abs_tol=1e-6):
            raise ValueError('Invalid mixture weights')
        key = tuple(weights.items())
        if key in self.cache:
            packed, size = self.cache.pop(key)
            for name, (a, b) in packed.items():
                self.layers[name]['abar'], self.layers[name]['bbar'] = a, b
            self.cache[key] = (packed, size)
            self.hits += 1
            self.last_cache_hit = True
            return
        self.last_cache_hit = False
        self.misses += 1
        super().activate(weights)
        packed = {name: (item['abar'], item['bbar']) for name, item in self.layers.items()}
        size = sum(t.numel() * t.element_size() for pair in packed.values() for t in pair)
        if size > self.max_bytes:
            return  # Oversized entries run correctly, but are not retained.
        while self.cache and (len(self.cache) >= self.max_entries or self.cache_bytes + size > self.max_bytes):
            _, (_, removed_bytes) = self.cache.popitem(last=False)
            self.cache_bytes -= removed_bytes
            self.evictions += 1
        self.cache[key] = (packed, size)
        self.cache_bytes += size
        self.peak_cache_bytes = max(self.peak_cache_bytes, self.cache_bytes)

    def cache_stats(self):
        return dict(hits=self.hits, misses=self.misses, evictions=self.evictions,
                    entries=len(self.cache), bytes=self.cache_bytes, peak_bytes=self.peak_cache_bytes)

    def close(self):
        if hasattr(self, 'cache'):
            self.cache.clear()
            self.cache_bytes = 0
        super().close()
