import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from orthogonal_lora import OrthogonalMixture, orthogonal_factor, update_inner, update_row_basis
from jev_lora.models import ExactMixture
import test_adapter_baselines as reference_tests


class OrthogonalAlgebraTests(unittest.TestCase):
    def factors(self):
        torch.manual_seed(9)
        return [torch.randn(*shape,dtype=torch.float64) for shape in ((3,11),(8,3),(3,11),(8,3))]

    def test_complete_update_projection_and_inner_product(self):
        a,b,c,d=self.factors()
        q=update_row_basis(a,b)
        residual=orthogonal_factor(c,q)
        torch.testing.assert_close(update_inner(a,b,c,d),((b@a)*(d@c)).sum())
        torch.testing.assert_close(d@residual,(d@c)@(torch.eye(11,dtype=q.dtype)-q@q.T))
        self.assertLess(float(((b@a)@(d@residual).T).abs().max()),1e-10)
        self.assertLessEqual(float((d@residual).norm()),float((d@c).norm())+1e-10)

    def test_zero_and_identical_updates(self):
        a,b,c,d=self.factors()
        q=update_row_basis(a,b*0)
        self.assertEqual(q.shape,(11,0))
        torch.testing.assert_close(orthogonal_factor(c,q),c)
        q=update_row_basis(a,b)
        self.assertLess(float((b@orthogonal_factor(a,q)).abs().max()),1e-10)

    def test_rank_deficient_b_uses_complete_update_space(self):
        a,b,c,d=self.factors()
        b[:,1:]=0
        q=update_row_basis(a,b)
        self.assertEqual(q.shape[1],1)
        self.assertLess(float(((b@a)@(d@orthogonal_factor(c,q)).T).abs().max()),1e-10)

    def test_gauge_invariance(self):
        a,b,c,d=self.factors()
        transform=torch.tensor([[2.,1.,0.],[0.,3.,1.],[0.,0.,4.]],dtype=a.dtype)
        q=update_row_basis(a,b)
        other=update_row_basis(transform@a,b@torch.linalg.inv(transform))
        torch.testing.assert_close(q@q.T,other@other.T,rtol=1e-10,atol=1e-10)

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):update_row_basis(torch.zeros(3,4),torch.zeros(5,2))
        with self.assertRaises(ValueError):update_row_basis(torch.full((2,4),float('nan')),torch.zeros(5,2))


class OrthogonalIntegrationTests(unittest.TestCase):
    def test_control_exact_cache_order_and_immutable_weights(self):
        for dtype in (torch.float32,torch.bfloat16):
            model=reference_tests.TinyQwenIntegrationTests().model().to(dtype)
            before={n:p.detach().clone() for n,p in model.named_parameters()}
            ids=torch.tensor([[1,2,3]])
            reference=ExactMixture(model)
            mixer=OrthogonalMixture(model,['a','b'])
            reference.activate({'a':.5,'b':.5})
            with torch.inference_mode():
                expected=model(input_ids=ids,use_cache=False).logits
                mixer.activate({'a':.5,'b':.5},strength=0)
                self.assertTrue(torch.equal(expected,model(input_ids=ids,use_cache=False).logits))
                mixer.activate({'a':.5,'b':.5})
                first=model(input_ids=ids,use_cache=False).logits
                mixer.activate({'b':.5,'a':.5})
                mixer.activate({'a':.5,'b':.5})
                self.assertTrue(torch.equal(first,model(input_ids=ids,use_cache=False).logits))
                self.assertEqual(mixer.misses,2)
                self.assertEqual(mixer.hits,1)
                mixer.close()
                reference.activate({'a':.5,'b':.5})
                self.assertTrue(torch.equal(expected,model(input_ids=ids,use_cache=False).logits))
            for name,param in model.named_parameters():
                self.assertTrue(torch.equal(before[name],param))
                self.assertFalse(param.requires_grad)

    def test_reject_wrong_weights_and_strength(self):
        mixer=OrthogonalMixture(reference_tests.TinyQwenIntegrationTests().model(),['a','b'])
        try:
            for weights in ({'a':1.},{'a':.6,'b':.4},{'a':.5,'z':.5}):
                with self.assertRaises(ValueError):mixer.activate(weights)
            with self.assertRaises(ValueError):mixer.activate({'a':.5,'b':.5},strength=.5)
        finally:mixer.close()


if __name__=='__main__':
    unittest.main()
