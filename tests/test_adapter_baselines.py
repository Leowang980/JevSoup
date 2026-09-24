"""CPU tests, including direct comparisons with the pinned official Arrow code."""
import ast
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from adapter_baseline_impl import (FactorMixture, arrow_selection, input_svd_prototype,
    mean_domain_vectors, sample_training_support, selected_factor_forward)
from run_adapter_baselines import load_tensors, save_tensors, validate_soup_route

torch.set_num_threads(2)


def official_function(relative, class_name, method):
    """Extract only an inspected pure numerical method; no upstream imports/install."""
    path = ROOT / 'artifacts/references/mttl' / relative
    if not path.exists():
        raise unittest.SkipTest('Pinned official reference checkout is unavailable')
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
    fn = copy.deepcopy(next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == method))
    fn.decorator_list = []
    fn.returns = None
    for arg in [*fn.args.posonlyargs, *fn.args.args, *fn.args.kwonlyargs]:
        arg.annotation = None
    namespace = dict(torch=torch, F=F, np=np, EPS=1e-8, ALL_EXPERTS=object(),
        BatchSequenceExpertsAndWeightsSelectorOutput=SimpleNamespace,
        logger=SimpleNamespace(debug=lambda *args: None), debug_once=lambda *args: None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), str(path), 'exec'), namespace)
    return namespace[method]


class ArrowNumericsTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_low_rank_prototype_matches_dense_and_official(self):
        a, b = torch.randn(3, 13), torch.randn(9, 3)
        vector, eigenvalue = input_svd_prototype(a, b)
        _, singular, vh = torch.linalg.svd(b @ a, full_matrices=False)
        self.assertAlmostEqual(float(vector.norm()), 1, places=5)
        self.assertGreater(float((vector @ vh[0]).abs()), .99999)
        torch.testing.assert_close(eigenvalue, singular[0].square(), rtol=1e-5, atol=1e-5)
        reference = official_function('mttl/models/library/library_transforms.py', 'ArrowTransform', '_low_rank_svd')
        u, s, _ = reference(None, a.T, b.T)
        self.assertGreater(float((vector @ u[:, 0]).abs()), .99999)
        torch.testing.assert_close(eigenvalue, s[0].square(), rtol=1e-5, atol=1e-5)

    def test_zero_and_nonfinite_factors_rejected(self):
        with self.assertRaises(ValueError):
            input_svd_prototype(torch.zeros(2, 4), torch.randn(3, 2))
        with self.assertRaises(ValueError):
            input_svd_prototype(torch.full((2, 4), float('nan')), torch.randn(3, 2))

    def test_router_matches_official_topk_weights(self):
        x, proto = torch.randn(2, 5, 13), F.normalize(torch.randn(4, 13), dim=-1)
        reference = official_function('mttl/models/containers/selectors/per_token_selector.py', 'PerTokenSelector', 'forward')
        obj = SimpleNamespace(config=SimpleNamespace(router_temp=1., proto_init='arrow', top_k=2),
            prototypes=proto, expert_names=list('abcd'), input_norm=nn.Identity(), proto_norm=nn.Identity(),
            routing_infos=SimpleNamespace(task_names=None), _log_entropy=lambda *args: None)
        official = reference(obj, x)
        scores, indices, weights = arrow_selection(x, proto)
        torch.testing.assert_close(indices, official.experts)
        torch.testing.assert_close(weights, official.weights)
        torch.testing.assert_close(scores, (x @ proto.T).abs())
        torch.testing.assert_close(weights.sum(-1), torch.ones(2, 5))
        # Sign invariance and no hidden cosine normalization.
        scores2, indices2, weights2 = arrow_selection(x, -proto)
        torch.testing.assert_close(scores, scores2)
        torch.testing.assert_close(indices, indices2)
        torch.testing.assert_close(weights, weights2)
        torch.testing.assert_close(arrow_selection(2*x, proto)[0], 2*scores)

    def test_factor_composition_matches_official_and_dense(self):
        batch, sequence, experts, rank, d, out = 2, 5, 4, 3, 13, 9
        x, a, b = torch.randn(batch, sequence, d), torch.randn(experts, rank, d), torch.randn(experts, out, rank)
        _, indices, weights = arrow_selection(x, F.normalize(torch.randn(experts, d), dim=-1))
        dense_w = torch.zeros(batch, sequence, experts).scatter(-1, indices, weights)
        abar = torch.einsum('bte,erd->btrd', dense_w, a)
        bbar = torch.einsum('bte,eor->btor', dense_w, b)
        expected = torch.einsum('btor,btrd,btd->bto', bbar, abar, x) * 2
        for chunk in (1, 3, 32):
            actual = selected_factor_forward(x, a, b, indices, weights, scale=2, chunk_size=chunk)
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        reference = official_function('mttl/models/modifiers/lora.py', 'SkilledLoRA', 'parallel_linear_weighted_forward')
        base = nn.Linear(d, out)
        view = SimpleNamespace(layer=base, n_skills=experts,
            lora_a=a.transpose(1, 2).unsqueeze(1), lora_b=b.transpose(1, 2).unsqueeze(2),
            dropout_layer=nn.Identity(), scaling=2.)
        official = reference(None, x, [view], dense_w, ['batch', 'sequence', 'experts'], merge_after=False)
        torch.testing.assert_close(actual + base(x), official, rtol=2e-5, atol=2e-5)

    def test_factor_mean_is_not_mean_of_complete_deltas(self):
        x = torch.tensor([[[1.]]])
        a, b = torch.tensor([[[1.]], [[3.]]]), torch.tensor([[[2.]], [[4.]]])
        actual = selected_factor_forward(x, a, b, torch.tensor([[[0, 1]]]), torch.tensor([[[.5, .5]]]))
        self.assertEqual(float(actual.item()), 6.)
        self.assertNotEqual(float(actual.item()), .5*(1*2+3*4))


class SupportTests(unittest.TestCase):
    def test_train_only_dedup_no_gold_and_determinism(self):
        train = [dict(task=task, prompt=f'{task}-{i}', choices=[' a', ' b'], gold_idx=i%2)
                 for task in ('a', 'b') for i in range(15)]
        train += [dict(train[0]), dict(task='b', prompt=' EXCLUDED ', choices=[' a', ' b'], gold_idx=0)]
        source = dict(train=train, validation=[dict(prompt='excluded')])
        rows, audit = sample_training_support(source, 5, 42, names=['a', 'b'])
        self.assertEqual(len(rows), 10)
        self.assertEqual(audit['excluded'], dict(duplicate_within_task=1, validation_overlap=1))
        self.assertTrue(all('gold_idx' not in row and 'task' not in row for row in rows))
        self.assertEqual(rows, sample_training_support(source, 5, 42, names=['a', 'b'])[0])
        bigger = sample_training_support(source, 10, 42, names=['a', 'b'])[0]
        self.assertTrue({r['id'] for r in rows} < {r['id'] for r in bigger})
        changed_gold = copy.deepcopy(source)
        for row in changed_gold['train']:
            row['gold_idx'] = 1-row['gold_idx']
        self.assertEqual(rows, sample_training_support(changed_gold, 5, 42, names=['a', 'b'])[0])

    def test_mean_cosine_without_renormalizing_centroid(self):
        vectors = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [-1., 0.]])
        records = [dict(expert=e) for e in ['a', 'a', 'b', 'b']]
        means = mean_domain_vectors(vectors, records, ['a', 'b'])
        query = torch.tensor([1., 0.])
        torch.testing.assert_close(means @ query, torch.tensor([.5, -1.]))
        self.assertLess(float(means[0].norm()), 1)

    def test_index_integrity_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'index.safetensors'
            config = dict(seed=42)
            save_tensors(path, dict(samples=torch.eye(3)), config)
            tensors, meta = load_tensors(path, config)
            torch.testing.assert_close(tensors['samples'], torch.eye(3))
            with self.assertRaises(ValueError):
                load_tensors(path, dict(seed=43))
            with self.assertRaises(ValueError):
                save_tensors(path, tensors, config)
            with path.open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaises(ValueError):
                load_tensors(path, config)

    def test_negative_cosines_are_allowed_for_equal_soup(self):
        row = dict(input_hash='test')
        rr = dict(kind='adaptersoup', input_hash='test', scores={'a': -.1, 'b': -.2, 'c': -.5}, weights={'a': .5, 'b': .5})
        validate_soup_route(row, rr, ['a', 'b', 'c'])
        rr['weights'] = {'b': .5, 'c': .5}
        with self.assertRaises(ValueError):
            validate_soup_route(row, rr, ['a', 'b', 'c'])


class TinyQwenIntegrationTests(unittest.TestCase):
    def model(self):
        from transformers import Qwen3Config, Qwen3ForCausalLM
        from peft import LoraConfig, get_peft_model
        torch.manual_seed(42)
        base = Qwen3ForCausalLM(Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=64))
        config = LoraConfig(r=2, lora_alpha=4, target_modules=['q_proj', 'v_proj'], lora_dropout=0.)
        model = get_peft_model(base, config, adapter_name='a')
        model.add_adapter('b', copy.deepcopy(config))
        for name, parameter in model.named_parameters():
            if 'lora_B' in name:
                with torch.no_grad():
                    parameter.normal_(std=.04)
        model.requires_grad_(False)
        model.eval()
        return model

    def test_soup_switching_resets_and_does_not_mutate_parameters(self):
        model = self.model()
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        ids = torch.tensor([[1, 2, 3]])
        with torch.inference_mode():
            model.set_adapter('a')
            expected = model(input_ids=ids, use_cache=False).logits
            mixer = FactorMixture(model, ['a', 'b'], 'adaptersoup')
            mixer.activate({'a': 1.})
            torch.testing.assert_close(model(input_ids=ids, use_cache=False).logits, expected)
            mixer.activate({'a': .5, 'b': .5})
            first = model(input_ids=ids, use_cache=False).logits
            mixer.activate({'b': 1.})
            mixer.activate({'a': .5, 'b': .5})
            torch.testing.assert_close(first, model(input_ids=ids, use_cache=False).logits)
            mixer.close()
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(before[name], parameter)
            self.assertFalse(parameter.requires_grad)

    def test_arrow_causal_dynamic_trace_and_cleanup(self):
        from peft.tuners.lora.layer import LoraLayer
        from adapter_baseline_impl import layer_key
        model = self.model()
        protos = {layer_key(name): torch.stack([input_svd_prototype(layer.lora_A[n].weight, layer.lora_B[n].weight)[0]
                  for n in ['a', 'b']]) for name, layer in model.named_modules() if isinstance(layer, LoraLayer)}
        mixer = FactorMixture(model, ['a', 'b'], 'arrow', protos, chunk_size=2)
        mixer.reset_trace(capture=True)
        with torch.inference_mode():
            logits = model(input_ids=torch.tensor([[1, 2, 3], [1, 2, 4]]), use_cache=False).logits
        torch.testing.assert_close(logits[0, :2], logits[1, :2])
        trace, seconds = mixer.trace_cpu(expected_choices=1)
        self.assertEqual(len(trace), 4*3)
        self.assertEqual(seconds, 0)
        for name, value in trace.items():
            self.assertEqual(tuple(value.shape), (2, 3, 2))
            if name.endswith('.weights'):
                torch.testing.assert_close(value.sum(-1), torch.ones(2, 3))
        mixer.close()
        self.assertFalse(mixer.trace)


if __name__ == '__main__':
    unittest.main()
