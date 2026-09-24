"""Baseline contracts and frozen probe reset; no ML downloads."""
import copy
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from jev_lora.baselines import (EMBEDDING_REPO, am_keyword_fallback, card_text,
                               download_embedding, file_sha256, load_embedding_manifest,
                               ordered_cards, routing_run)
from jev_lora.cli import main
from scripts.run_am_top2_4b import PROMPT, parse_two
from jev_lora.core import DESCRIPTIONS, TASKS, read_json, read_jsonl, write_json, write_jsonl
from jev_lora.models import MAIN_METHODS, ExactMixture, select_weights, validate_router_models
from jev_lora.evaluation import evaluate
from test_experiment import make_row, prediction
from jev_lora.core import digest

CARDS = [{"id": n, "description": d, "examples": []} for n, d in DESCRIPTIONS.items()]


class BaselineTest(unittest.TestCase):
    def test_logo_weighted_composition(self):
        row = make_row()
        scores = {"rte": 6., "boolq": 3., "cb": 1.}
        rr = {"kind": "logo", "input_hash": row["input_hash"], "scores": scores}
        self.assertEqual(select_weights("logo-top2-weighted", row, rr, list(scores)), {"rte": 2/3, "boolq": 1/3})
        for bad in [math.nan, -1.]:
            rr["scores"]["rte"] = bad
            with self.assertRaises(ValueError): select_weights("logo-top2-weighted", row, rr, list(scores))
        rr["scores"] = {n: 0. for n in scores}
        with self.assertRaises(ValueError): select_weights("logo-top2-weighted", row, rr, list(scores))

    def test_am_top2_keeps_full_shared_cards_and_keyword_fallback(self):
        cards = [{"id": "science", "description": "z"*130 + " photosynthesis", "examples": []},
                 {"id": "grammar", "description": "Textual entailment", "examples": []}]
        message = {"content": PROMPT.format(query="Explain photosynthesis",
            experts="\n".join("- " + card_text(c) for c in ordered_cards(cards, 42)))}
        for card in cards:
            self.assertIn(card_text(card), message["content"])
        self.assertNotIn("gold_idx", message["content"])
        self.assertEqual(parse_two("`SCIENCE`, grammar\nextra", ["science", "grammar"]),
                         ("science", "grammar"))
        self.assertEqual(parse_two("", ["science", "grammar"]), ())
        chosen, scores = am_keyword_fallback("Explain photosynthesis", cards)
        self.assertEqual(chosen, "science")
        self.assertGreater(scores["science"], scores["grammar"])
        chosen, scores = am_keyword_fallback("unrelated", cards, seed=19)
        self.assertEqual(chosen, ordered_cards(cards, 19)[0]["id"])
        self.assertEqual(sum(scores.values()), 0)

    def test_probe_restores_all_original_scales_after_weighted_requests(self):
        model = SimpleNamespace(peft_config={"a": {}, "b": {}}, base_model=SimpleNamespace(set_adapter=Mock()),
                                requires_grad_=Mock(), eval=Mock())
        layer = SimpleNamespace(scaling={"a": 2., "b": 4.})
        mixer = ExactMixture.__new__(ExactMixture)
        mixer.model, mixer.original = model, [(layer, dict(layer.scaling))]
        mixer.activate({"a": .8, "b": .2})
        self.assertEqual(layer.scaling, {"a": 1.6, "b": .8})
        mixer.activate_probe(["a", "b"])
        self.assertEqual(layer.scaling, {"a": 2., "b": 4.})
        model.base_model.set_adapter.assert_called_with(["a", "b"])
        model.requires_grad_.assert_called_with(False)
        mixer.activate({"b": 1.})
        mixer.activate_probe(["a", "b"])
        self.assertEqual(layer.scaling, {"a": 2., "b": 4.})

    def test_router_model_identity_prevents_cross_base_or_expert_results(self):
        models = {"base": {"repo": "Qwen/test", "revision": "abc"}, **{n: {"hash": n} for n in TASKS}}
        manifest = {"models": models}
        meta = {"kind": "logo", "base": models["base"], "adapters": {n: models[n] for n in TASKS}}
        validate_router_models(meta, manifest)
        changed = copy.deepcopy(meta); changed["adapters"]["rte"]["hash"] = "new"
        with self.assertRaisesRegex(ValueError, "probe adapters"): validate_router_models(changed, manifest)
        changed = copy.deepcopy(meta); changed["base"]["revision"] = "other"
        with self.assertRaisesRegex(ValueError, "Routing base"): validate_router_models(changed, manifest)

    def test_routing_resume_locks_configuration_and_checks_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [make_row(i) for i in range(2)]
            write_jsonl(root/"data.jsonl", rows); write_json(root/"cards.json", CARDS)
            args = SimpleNamespace(data=root/"data.jsonl", cards=root/"cards.json", output=root/"routes.jsonl", seed=42)
            _, names, out, old, todo = routing_run(args, "logo", {"max_input_tokens": 4096})
            self.assertEqual(len(todo), 2)
            saved = {"id": "0", "kind": "logo", "input_hash": rows[0]["input_hash"], "scores": {n: 0. for n in names}}
            write_jsonl(out, [saved])
            self.assertEqual([r["id"] for r in routing_run(args, "logo", {"max_input_tokens": 4096})[-1]], ["1"])
            with self.assertRaisesRegex(ValueError, "Configuration changed"):
                routing_run(args, "logo", {"max_input_tokens": 2048})
            saved["input_hash"] = "corrupt"; write_jsonl(out, [saved])
            with self.assertRaisesRegex(ValueError, "ID/input/kind"):
                routing_run(args, "logo", {"max_input_tokens": 4096})

    def test_mock_embedding_download_pins_revision_and_detects_file_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            api = Mock()
            api.model_info.return_value = SimpleNamespace(sha="a"*40)
            api.list_repo_files.return_value = ["config.json", "tokenizer_config.json", "tokenizer.json",
                                               "model.safetensors", "onnx/model.onnx", "pytorch_model.bin"]
            revisions = []

            def snapshot(repo, revision, local_dir, allow_patterns):
                self.assertEqual(repo, EMBEDDING_REPO)
                revisions.append(revision)
                self.assertNotIn("pytorch_model.bin", allow_patterns)
                for name in allow_patterns:
                    (Path(local_dir)/name).write_bytes(b"synthetic, not model weights")

            fake = SimpleNamespace(HfApi=lambda: api, snapshot_download=snapshot)
            args = SimpleNamespace(embedding_dir=root, revision="main")
            with patch.dict(sys.modules, {"huggingface_hub": fake}):
                download_embedding(args)
                api.model_info.return_value = SimpleNamespace(sha="b"*40)
                download_embedding(args)
            self.assertEqual(revisions, ["a"*40, "a"*40])
            self.assertEqual(load_embedding_manifest(root)["revision"], "a"*40)
            (root/"model.safetensors").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "Changed or missing"):
                load_embedding_manifest(root)

    def test_logo_cli_dispatch_defaults(self):
        with patch("jev_lora.baselines.logo_route") as call:
            main(["logo-route", "--data", "data.jsonl", "--output", "routes.jsonl"])
        args = call.call_args.args[0]
        self.assertEqual(args.max_input_tokens, 4096)
        self.assertEqual(args.layer_index, -1)

    def test_full_matrix_report_keeps_local_tokens_out_of_jev_billing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = [make_row(i, task=task) for i, task in enumerate(TASKS)]
            write_jsonl(root/"data.jsonl", rows)
            files = []
            for method in MAIN_METHODS:
                path = root/f"{method}.jsonl"; files.append(path)
                preds = [prediction(r, method) for r in rows]
                for p in preds:
                    p["route_usage"] = {"input_tokens": 7, "output_tokens": 1}
                    p["routing_peak_allocated_gib"] = 1.5
                    if method.startswith("jev-"):
                        p["route_payload_hash"] = p["id"]
                write_jsonl(path, preds)
                write_json(str(path)+".meta.json", {"schema": "portal-inference-v1", "method": method,
                    "implementation_hash": "fixture", "data_hash": digest(rows), "cards_hash": "fixture",
                    "scoring": "portal", "seed": 42, "dtype": "fixture", "max_prompt": 768,
                    "choice_batch_size": 1, "versions": {}, "models": {"base": {"revision": "same"}}})
            evaluate(SimpleNamespace(data=root/"data.jsonl", predictions=files, output_dir=root/"report",
                                     bootstrap=8, seed=42, input_price_per_million=.5))
            report = read_json(root/"report/report.json")
            summaries = {r["method"]: r for r in report["summary"]}
            self.assertEqual(set(summaries), set(MAIN_METHODS))
            self.assertEqual(summaries["am-top2-equal"]["logical_unique_jev_calls"], 0)
            self.assertEqual(summaries["am-top2-equal"]["estimated_one_pass_jev_usd"], 0)
            self.assertEqual(summaries["am-top2-equal"]["routing_input_tokens"], 98)
            self.assertEqual(summaries["logo-top2-weighted"]["sequential_pipeline_peak_gib"], 1.5)
            self.assertEqual(summaries["jev-top2-orthogonal-equal"]["logical_unique_jev_calls"], 14)
            pairs = {(c["left"], c["right"]) for c in report["paired_accuracy"]}
            self.assertEqual(pairs, {(m, "jev-top2-orthogonal-equal")
                                     for m in MAIN_METHODS if m != "jev-top2-orthogonal-equal"})


if __name__ == "__main__":
    unittest.main()
