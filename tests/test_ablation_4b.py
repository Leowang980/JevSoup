import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from ablation_4b_impl import random_weights


class RandomRoutingTests(unittest.TestCase):
    def test_stable_input_based_seeded_uniform_pair(self):
        names = ['expert' + str(i) for i in range(14)]
        for seed in (42, 43, 44):
            result = random_weights('input', names, seed)
            self.assertEqual(result, random_weights('input', names[::-1], seed))
            self.assertEqual(len(result), 2)
            self.assertEqual(list(result), sorted(result))
            self.assertEqual(list(result.values()), [.5, .5])
        pairs = {tuple(random_weights(str(i), names, 42)) for i in range(3000)}
        self.assertEqual(len(pairs), 91)
        a = [random_weights(str(i), names, 42) for i in range(20)]
        b = [random_weights(str(i), names, 43) for i in range(20)]
        self.assertNotEqual(a, b)

    def test_invalid_seeds_and_inventory(self):
        with self.assertRaises(ValueError):
            random_weights('input', ['a', 'b'], 0)
        with self.assertRaises(ValueError):
            random_weights('input', ['a', 'a'], 42)


if __name__ == '__main__':
    unittest.main()
