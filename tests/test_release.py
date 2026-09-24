"""Offline tests for the public entry points and composition controls."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from jevsoup.__main__ import METHODS, main
from jevsoup.runner import activate, make_mixer, plan
import test_adapter_baselines as fixtures


class PublicNumericsTest(unittest.TestCase):
    def test_soup_cache_and_uncached_are_exact(self):
        model = fixtures.TinyQwenIntegrationTests().model()
        ids = torch.tensor([[1, 2, 3, 4]])
        results = []
        from adapter_baseline_impl import FactorMixture
        for cached in (False, True):
            mixer = (make_mixer(model, ['a', 'b'], 'adaptersoup-top2-equal') if cached
                     else FactorMixture(model, ['a', 'b'], 'adaptersoup'))
            with torch.inference_mode():
                for weights in ({'a': .5, 'b': .5}, {'b': .5, 'a': .5}, {'a': .5, 'b': .5}):
                    activate(mixer, 'adaptersoup-top2-equal', weights)
                    results.append(model(input_ids=ids, use_cache=False).logits.clone())
            mixer.close()
        for i in range(3):
            self.assertTrue(torch.equal(results[i], results[i+3]))

    def test_zero_strength_and_plain_mean_equal_original_peft(self):
        from jev_lora.models import ExactMixture
        model = fixtures.TinyQwenIntegrationTests().model()
        ids = torch.tensor([[1, 2, 3]])
        weights = {'b': .5, 'a': .5}
        original = ExactMixture(model)
        original.activate(weights)
        with torch.inference_mode():
            expected = model(input_ids=ids, use_cache=False).logits.clone()
        original.activate_probe(['a', 'b'])
        for method, strength in [('jev-top2-equal', 1), ('jev-top2-orthogonal-equal', 0)]:
            mixer = make_mixer(model, ['a', 'b'], method, strength)
            activate(mixer, method, weights, strength)
            with torch.inference_mode():
                self.assertTrue(torch.equal(expected, model(input_ids=ids, use_cache=False).logits))
            mixer.close()

    def test_partial_strength_and_probability_dispatch(self):
        from probability_orthogonal_impl import ProbabilityOrthogonalMixture
        from sensitivity_4b_impl import PartialOrthogonalMixture
        model = fixtures.TinyQwenIntegrationTests().model()
        for method, strength, cls in [('jev-top2-orthogonal-prob', 1, ProbabilityOrthogonalMixture),
                                     ('jev-top2-orthogonal-equal', .25, PartialOrthogonalMixture)]:
            mixer = make_mixer(model, ['a', 'b'], method, strength)
            self.assertIsInstance(mixer, cls)
            mixer.close()

    def test_order_and_random_seed_are_preserved(self):
        row = dict(id='one', input_hash='hash', task='a')
        route = dict(id='one', input_hash='hash', kind='jev', scores={'b': .8, 'a': .2})
        self.assertEqual(list(plan('jev-top2-orthogonal-equal', row, route, ['a','b'])), ['b','a'])
        self.assertEqual(plan('jev-top2-orthogonal-prob', row, route, ['a','b']), {'b':.8, 'a':.2})
        from ablation_4b_impl import random_weights
        for seed in (42, 43, 44):
            self.assertEqual(plan('random-top2-equal', row, None, ['a','b','c'], seed),
                             random_weights('hash', ['a','b','c'], seed))


class PublicCLITest(unittest.TestCase):
    def test_only_paper_methods_are_exposed(self):
        expected = {'base', 'am-top2-equal', 'logo-top2-weighted', 'adaptersoup-top2-equal',
            'arrow-top2-weighted', 'jev-top1', 'jev-top2-equal', 'jev-top2-prob',
            'jev-top2-orthogonal-equal', 'jev-top2-orthogonal-prob', 'random-top2-equal'}
        self.assertEqual(set(METHODS), expected)
        from jev_lora.models import METHODS as request_methods, select_weights
        self.assertTrue(set(request_methods) <= expected)
        with self.assertRaisesRegex(ValueError, 'Unsupported'):
            select_weights('unsupported-control', {}, None, ['a', 'b'])
        output = io.StringIO()
        with contextlib.redirect_stderr(output), self.assertRaises(SystemExit) as caught:
            main(['run', '--method', 'unsupported-control', '--data', 'data.jsonl',
                  '--model-dir', 'model', '--output', 'out.jsonl'])
        self.assertEqual(caught.exception.code, 2)

    def test_am_top2_public_dispatch(self):
        with patch('jevsoup.runner.am_route') as call:
            main(['am-route', '--data', 'data.jsonl', '--model-dir', 'models/4b',
                  '--output', 'routes.jsonl'])
        self.assertEqual(call.call_args.args[0].seed, 42)

    def test_run_dispatch_preserves_numerical_options(self):
        with patch('jevsoup.runner.run') as run:
            main(['run', '--method', 'jev-top2-orthogonal-equal', '--strength', '.25',
                  '--data', 'data.jsonl', '--model-dir', 'models/4b', '--routes', 'routes.jsonl',
                  '--output', 'predictions.jsonl'])
        args = run.call_args.args[0]
        self.assertEqual(args.strength, .25)
        self.assertEqual(args.routing_seed, 42)
        self.assertFalse(args.arrow_traces)

    def test_help_does_not_advertise_bundled_results(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SystemExit) as caught:
            main(['--help'])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn('am-route', output.getvalue())
        self.assertNotIn('restore-routes', output.getvalue())
        self.assertNotIn('snapshots', output.getvalue())


if __name__ == '__main__':
    unittest.main()
