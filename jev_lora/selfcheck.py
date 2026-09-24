"""Run after installation on the GPU machine; uses only tiny random CPU models."""
import copy
import tempfile
from pathlib import Path
from types import SimpleNamespace


class TinyTokenizer:
    bos_token_id, eos_token_id, pad_token_id = 1, 2, 0

    def __call__(self, text, add_special_tokens=True):
        return SimpleNamespace(input_ids=[3 + ord(c) % 25 for c in text])


def run():
    import torch
    from peft import PeftModel
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from portallib import ChoiceExample, PortalBase, PortalConfig, PortalEvaluator, PortalModel
    from .models import ExactMixture
    from .baselines import LoGoProbe, encode_embeddings, last_token_pool
    from .scoring import score_choices

    torch.manual_seed(42)
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=8, max_position_embeddings=128)
    base = Qwen3ForCausalLM(config).eval()
    untouched = copy.deepcopy(base)
    pcfg = PortalConfig.from_model(base, tasks=("rte", "boolq"), base_model_name_or_path="test/tiny-qwen3",
                                   rank=2, alpha=4, d_z=4, d_layer=4, hidden=8, d_core=8)
    portal = PortalModel(pcfg, torch.randn(2, 4))
    with torch.no_grad():
        for parameter in portal.parameters():
            parameter.normal_(0, .1)
    with tempfile.TemporaryDirectory() as tmp:
        for task in pcfg.tasks:
            portal.export_peft(task, str(Path(tmp) / task))
        model = PeftModel.from_pretrained(base, str(Path(tmp) / "rte"), adapter_name="rte")
        model.load_adapter(str(Path(tmp) / "boolq"), adapter_name="boolq")
        mixer = ExactMixture(model)
        mixer.activate({"rte": .7, "boolq": .3})
        layer, scales = mixer.original[0]
        x = torch.randn(2, 3, layer.in_features)
        expected = layer.base_layer(x) + sum(w * scales[n] * layer.lora_B[n](layer.lora_A[n](x))
                                            for n, w in {"rte": .7, "boolq": .3}.items())
        torch.testing.assert_close(layer(x), expected)
        mixer.activate({"rte": 1.0})
        model.add_weighted_adapter(["rte", "boolq"], [.7, .3], "reference", combination_type="cat")
        model.set_adapter("reference")
        ids = torch.tensor([[1, 4, 7, 2]])
        model.eval()
        with torch.inference_mode():
            reference = model(ids).logits
            mixer.activate({"rte": .7, "boolq": .3})
            torch.testing.assert_close(model(ids).logits, reference, rtol=1e-4, atol=1e-5)
            mixer.activate({"boolq": 1.0})
            mixer.activate({"rte": .7, "boolq": .3})
            torch.testing.assert_close(layer(x), expected)
            # Check the inference path before invoking this test-only PEFT context.
            assert not any(p.requires_grad for p in model.parameters())
            before_disable = {n: p.detach().clone() for n, p in model.named_parameters()}
            try:
                with model.disable_adapter():
                    torch.testing.assert_close(model(ids).logits, untouched(ids).logits)
            finally:
                # PEFT 0.21 re-enables gradients when restoring active adapters.
                # Restore our frozen-test invariant; this does not change weights.
                model.requires_grad_(False)
        assert not any(p.requires_grad for p in model.parameters())
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before_disable[n], rtol=0, atol=0)

        # LoGo must reproduce the reference hidden_states[layer_idx] calculation,
        # reset all experts before every probe, and never change learned weights.
        pool = ["rte", "boolq"]
        probe = LoGoProbe(model, mixer, pool)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        scores = probe.scores(inputs)
        with torch.inference_mode():
            outputs = model(**inputs, output_hidden_states=True, logits_to_keep=1, use_cache=False)
            hidden = outputs.hidden_states[probe.layer_index][:, -1, :]
            for n in pool:
                a, b = probe.projection.lora_A[n].weight, probe.projection.lora_B[n].weight
                reference_norm = ((hidden @ a.T) @ b.T * probe.scales[n]).float().norm().item()
                assert abs(scores[n]-reference_norm) < 1e-6
        mixer.activate({"rte": 1.0})
        repeated = probe.scores(inputs)
        for n in pool:
            assert abs(repeated[n]-scores[n]) < 1e-6
        for layer, scales in mixer.original:
            for n in pool:
                assert layer.scaling[n] == scales[n]
        assert set(model.active_adapters) == set(pool)
        for n, p in model.named_parameters():
            torch.testing.assert_close(p, before[n], rtol=0, atol=0)
        assert not any(p.requires_grad for p in model.parameters())
        # Hook cleanup and all-adapter reset must also survive an intervening mix.
        mixer.activate({"rte": .2, "boolq": .8})
        assert probe.scores(inputs) == scores
        assert not probe.block._forward_pre_hooks
        mixer.activate({"rte": .7, "boolq": .3})
        tokenizer = TinyTokenizer()
        examples = [ChoiceExample(task="rte", prompt="A premise?", choices=[" yes", " no"], gold_idx=0),
                    ChoiceExample(task="rte", prompt="", choices=[" an answer", " b"], gold_idx=1)]
        evaluator = PortalEvaluator(max_prompt=8, batch_size=2)
        wrapped = PortalBase(model_id="test/tiny-qwen3", model=model, tokenizer=tokenizer)
        with torch.inference_mode():
            expected_scores, expected_nll, expected_tokens = evaluator._score_rows(wrapped, examples)
            actual_nll, actual_tokens = 0, 0
            for i, row in enumerate(examples):
                result = score_choices(model, tokenizer, row.prompt, row.choices, max_prompt=8, batch_size=1)
                batched = score_choices(model, tokenizer, row.prompt, row.choices, max_prompt=8, batch_size=2)
                torch.testing.assert_close(torch.tensor(result["choice_scores"]), torch.tensor(expected_scores[i]), rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(torch.tensor(result["choice_scores"]), torch.tensor(batched["choice_scores"]), rtol=1e-5, atol=1e-6)
                actual_nll -= result["choice_logprobs"][row.gold_idx]
                actual_tokens += result["choice_token_counts"][row.gold_idx]
            assert actual_tokens == expected_tokens
            assert abs(actual_nll - expected_nll) < 1e-4
    from transformers import BatchEncoding, Qwen3Model

    class EmbeddingTokenizer:
        def __call__(self, texts, **kwargs):
            sequences = [[3+ord(c) % 25 for c in t] + [2] for t in texts]
            width = max(map(len, sequences))
            return BatchEncoding({"input_ids": torch.tensor([[0]*(width-len(s))+s for s in sequences]),
                                  "attention_mask": torch.tensor([[0]*(width-len(s))+[1]*len(s) for s in sequences])})

    encoder = Qwen3Model(config).eval().requires_grad_(False)
    enc_tokenizer = EmbeddingTokenizer()
    batch, counts = encode_embeddings(encoder, enc_tokenizer, ["a", "longer"], 32, 2)
    singles, _ = encode_embeddings(encoder, enc_tokenizer, ["a", "longer"], 32, 1)
    torch.testing.assert_close(batch, singles, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(batch.norm(dim=-1), torch.ones(2))
    assert counts == [2, 7]
    hidden = torch.arange(24).reshape(2, 4, 3)
    mask = torch.tensor([[0, 0, 1, 1], [1, 1, 0, 0]])
    torch.testing.assert_close(last_token_pool(hidden, mask), torch.stack([hidden[0, 3], hidden[1, 1]]))
    print("PASS: PorTAL exports → PEFT, weighted BA, switching/freezing, upstream scoring parity, LoGo projection/reset parity, embedding pooling/batching")
