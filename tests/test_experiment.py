"""Offline tests: synthetic fixtures and mocked HTTP only, never Hub/API downloads."""
import concurrent.futures
import copy
import json
import math
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from jev_lora.core import (TASKS, canonical_text, digest, ensure_metadata, read_json,
                           read_jsonl, top_weights, validate_probabilities, write_json, write_jsonl)
from jev_lora.data import prepare, query_hash, routing_text, validate_row
from jev_lora.evaluation import evaluate, paired_diagnostic, validate_prediction
from jev_lora.models import check_adapter_config, select_download_files, select_weights
from jev_lora.routing import JevClient, make_payload, route
from jev_lora.baselines import card_text, ordered_cards
from scripts.run_am_top2_4b import PROMPT
from jev_lora.scoring import encode_choices, normalized_choice_scores

CARDS = [{"id": "rte", "description": "Textual entailment", "examples": []},
         {"id": "boolq", "description": "Passage yes no questions", "examples": []}]


def response(model="jev-1.13.0"):
    return {"model": model, "answers": {"route": {"type": "choice", "choice": "rte",
            "probabilities": {"rte": .7, "boolq": .3}, "confidence": .4}}, "usage": {"input_tokens": 100}}


def make_row(i=0, task="rte", gold=0):
    row = {"id": str(i), "task": task, "prompt": f"Question {i}?", "choices": [" yes", " no"], "gold_idx": gold}
    row["inputs"] = routing_text(row["prompt"], row["choices"])
    row["input_hash"] = query_hash(row)
    return row


def fixture():
    return {split: [{"task": t, "prompt": f"{split} prompt {t} {i}", "choices": [" yes", " no"], "gold_idx": i % 2}
                    for t in TASKS for i in range(n)] for split, n in [("train", 6), ("validation", 8)]}


def prediction(row, method="jev-top1", correct=True):
    winner = row["gold_idx"] if correct else 1-row["gold_idx"]
    logs = [-8.0, -8.0]
    logs[winner] = -.1
    chars = [len(c) for c in row["choices"]]
    scores, index = normalized_choice_scores(logs, chars)
    return {"id": row["id"], "input_hash": row["input_hash"], "method": method, "predicted_idx": index,
            "choice_scores": scores, "choice_logprobs": logs, "choice_characters": chars,
            "choice_token_counts": [2, 2], "weights": {row["task"]: 1.0}, "scoring_s": .1,
            "selection_s": .01, "route_s": .02, "peak_allocated_gib": 1, "truncated": False}


class DataTest(unittest.TestCase):
    def test_all_tasks_splits_choice_permutation_and_empty_cards(self):
        raw = fixture()
        raw["train"].append(dict(raw["validation"][0]))
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            m = prepare(raw, p, dev_per_task=1)
            self.assertEqual(m["split_counts"], dict(validation=112, dev=14, eval=98, smoke=14))
            full, dev, ev = [read_jsonl(p/f'{n}.jsonl') for n in ['validation', 'dev', 'eval']]
            self.assertEqual({r['id'] for r in dev+ev}, {r['id'] for r in full})
            self.assertFalse({canonical_text(r['prompt']) for r in dev} & {canonical_text(r['prompt']) for r in ev})
            for row in full:
                original = raw['validation'][row['source_index']]
                self.assertEqual(row['choices'][row['gold_idx']], original['choices'][original['gold_idx']])
                validate_row(row, prepared=True)
            cards = read_json(p/'cards.json')
            self.assertEqual({c['id'] for c in cards}, set(TASKS))
            self.assertNotIn('gold_idx', json.dumps(cards))
            self.assertTrue(all(c['examples'] == [] for c in cards))
            self.assertEqual({f.stem for f in p.glob('*.jsonl')}, {'validation', 'dev', 'eval', 'smoke'})
            old = (p/'eval.jsonl').read_bytes()
            prepare(raw, p, dev_per_task=1)
            self.assertEqual(old, (p/'eval.jsonl').read_bytes())
            with self.assertRaises(ValueError):
                prepare(raw, p, seed=43, dev_per_task=1)

    def test_bad_labels_and_legacy_inputs_rejected(self):
        for bad in [dict(make_row(), gold_idx=True), dict(make_row(), choices=['yes','no']),
                    {'inputs':'old', 'targets':'answer', 'task':'rte'}]:
            with self.assertRaises(ValueError):
                validate_row(bad)
        row = make_row()
        row['choices'].reverse()
        with self.assertRaises(ValueError):
            validate_row(row, prepared=True)

    def test_choice_order_changes_routing_hash_but_label_does_not(self):
        row = make_row()
        self.assertEqual(query_hash(row), query_hash(dict(row, gold_idx=1, task='boolq')))
        self.assertNotEqual(query_hash(row), query_hash(dict(row, choices=list(reversed(row['choices'])))))


class ContractsTest(unittest.TestCase):
    def test_download_excludes_duplicate_weights(self):
        files = ['config.json', 'model-00001-of-00002.safetensors', 'pytorch_model-00001-of-00002.bin',
                 'adapter_model.safetensors', 'adapter_model.bin', '2_Dense/pytorch_model.bin']
        self.assertEqual(set(select_download_files(files)), {files[0], files[1], files[3], files[5]})

    def test_probabilities_and_top2_mass(self):
        self.assertEqual(validate_probabilities(response(), ['rte','boolq']), {'rte':.7,'boolq':.3})
        for scores in ({'rte':.7}, {'rte':math.nan,'boolq':.3}, {'rte':-.1,'boolq':1.1}, {'rte':.1,'boolq':.1}):
            bad = response(); bad['answers']['route']['probabilities'] = scores
            with self.assertRaises(ValueError): validate_probabilities(bad, ['rte','boolq'])
        w = top_weights({'a':.6,'b':.3,'c':.1},2,True)
        self.assertAlmostEqual(w['a'], 2/3)
        self.assertAlmostEqual(w['b'], 1/3)
        with self.assertRaises(ValueError): top_weights({'a':0,'b':0},2,True)

    def test_only_prompt_and_options_are_routed(self):
        row = make_row()
        payload = make_payload(row['inputs'], CARDS, 'jev-1.13.0')
        self.assertEqual(payload['state'], {'query':row['inputs']})
        self.assertNotIn('gold_idx', json.dumps(payload))
        message = PROMPT.format(query=row['inputs'],
            experts="\n".join("- " + card_text(c) for c in ordered_cards(CARDS, 42)))
        self.assertNotIn('gold_idx', message)
        self.assertIn('Candidate continuations', message)

    def test_weight_plan_and_adapter_guard(self):
        row = make_row()
        rr = {'kind':'jev','input_hash':row['input_hash'],'scores':{'rte':.7,'boolq':.3}}
        self.assertEqual(select_weights('jev-top2-equal',row,rr,['rte','boolq']), {'rte':.5,'boolq':.5})
        self.assertEqual(select_weights('base',row,None,['rte','boolq']), {})
        rr['input_hash']='bad'
        with self.assertRaises(ValueError): select_weights('jev-top1',row,rr,['rte','boolq'])
        cfg={'peft_type':'LORA','base_model_name_or_path':'Qwen/Qwen3-4B'}
        check_adapter_config(cfg,'Qwen/Qwen3-4B')
        for change in ({'base_model_name_or_path':'Qwen/Qwen3-8B'}, {'use_dora':True}, {'modules_to_save':['lm_head']}):
            with self.assertRaises(ValueError): check_adapter_config(dict(cfg,**change),'Qwen/Qwen3-4B')

    def test_resume_repairs_only_partial_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'x.jsonl';p.write_bytes(b'{"id":1}\n{"id":')
            self.assertEqual(read_jsonl(p,repair_tail=True),[{'id':1}])
            p.write_bytes(b'broken\n{"id":2}\n')
            with self.assertRaises(ValueError): read_jsonl(p,repair_tail=True)
            ensure_metadata(Path(tmp)/'meta.json', {'seed':42})
            with self.assertRaises(ValueError): ensure_metadata(Path(tmp)/'meta.json', {'seed':43})


class ScoringTest(unittest.TestCase):
    class Tokenizer:
        bos_token_id, eos_token_id, pad_token_id = 1,2,0
        def __call__(self,text,add_special_tokens=True):
            return SimpleNamespace(input_ids=[ord(c) for c in text])

    def test_boundary_truncation_and_empty_prompt(self):
        encoded, info = encode_choices(self.Tokenizer(), 'abcdef ', [' yes',' no'], max_prompt=3)
        self.assertEqual(encoded[0]['ids'], [ord(c) for c in 'def  yes'])
        self.assertEqual(encoded[0]['prompt_length'], 3)
        self.assertEqual(encoded[0]['characters'], 5)
        self.assertTrue(info['truncated'])
        empty,_=encode_choices(self.Tokenizer(), '', [' a',' b'])
        self.assertEqual(empty[0]['ids'][0], 1)

    def test_character_normalization_can_change_winner(self):
        scores,winner=normalized_choice_scores([-2,-3],[2,6])
        self.assertEqual(scores,[-1,-.5]);self.assertEqual(winner,1)
        with self.assertRaises(ValueError): normalized_choice_scores([math.nan,-1],[1,2])


class MockAPITest(unittest.TestCase):
    def client(self,tmp,handler):
        import httpx
        c=JevClient(tmp,retries=1,interval=0);c.http.close()
        c.http=httpx.Client(transport=httpx.MockTransport(handler));return c

    @patch.dict('os.environ', {'TYPESAFE_API_KEY':'test-secret'})
    def test_singleflight_cache_order_and_secret_redaction(self):
        import httpx
        calls=[]
        def handler(request):
            calls.append(json.loads(request.content));return httpx.Response(200,json=response())
        with tempfile.TemporaryDirectory() as tmp:
            c=self.client(tmp,handler)
            try:
                payload=make_payload('query',CARDS,c.model)
                with concurrent.futures.ThreadPoolExecutor(4) as pool:
                    entries=list(pool.map(lambda _:c.request(payload),range(8)))
                self.assertEqual(len(calls),1)
                self.assertEqual(sum(not e['cache_hit'] for e in entries),1)
                self.assertNotIn('test-secret',next(Path(tmp).glob('*.json')).read_text())
                other=copy.deepcopy(payload)
                other['questions']['route']['criteria']=dict(reversed(list(other['questions']['route']['criteria'].items())))
                c.request(other);self.assertEqual(len(calls),2)
            finally: c.close()

    @patch.dict('os.environ', {'TYPESAFE_API_KEY':'test-secret'})
    def test_retry_and_error_redaction(self):
        import httpx
        calls=[]
        def handler(req):
            calls.append(1)
            return httpx.Response(429,headers={'retry-after':'0'}) if len(calls)==1 else httpx.Response(200,json=response())
        with tempfile.TemporaryDirectory() as tmp:
            c=self.client(tmp,handler)
            try: self.assertEqual(c.request(make_payload('q',CARDS,c.model))['attempts'],2)
            finally: c.close()
            c=self.client(tmp,lambda r:httpx.Response(401,text='test-secret'))
            try:
                with self.assertRaisesRegex(RuntimeError,'HTTP 401') as caught: c.request(make_payload('new',CARDS,c.model))
                self.assertNotIn('test-secret',str(caught.exception))
            finally: c.close()

    def test_jev_route_resume_avoids_repeated_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            rows = [make_row(i) for i in range(3)]
            write_jsonl(p/'data.jsonl', rows)
            write_json(p/'cards.json', CARDS)
            args = Namespace(data=p/'data.jsonl', cards=p/'cards.json', output=p/'jev.jsonl',
                kind='jev', seed=42, jev_model='jev-1.13.0', workers=2, interval=0, cache=p/'cache')
            with patch('jev_lora.routing.JevClient') as factory:
                factory.return_value.request.return_value = dict(response=response(),
                    latency_s=.1, cache_hit=False, payload_hash='fixture', attempts=1)
                route(args)
                content = args.output.read_bytes()
                route(args)
                self.assertEqual(factory.return_value.request.call_count, len(rows))
                self.assertEqual(content, args.output.read_bytes())
            self.assertTrue(all(r['scores'] == {'rte': .7, 'boolq': .3}
                                for r in read_jsonl(args.output)))


class EvaluationTest(unittest.TestCase):
    def test_report_missing_predictions_and_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);rows=[make_row(i,task='rte' if i<2 else 'boolq',gold=i%2) for i in range(3)]
            write_jsonl(p/'data.jsonl',rows);files=[]
            for method in ['jev-top1','jev-top2-orthogonal-equal']:
                f=p/f'{method}.jsonl';files.append(f)
                write_jsonl(f,[prediction(r,method,correct=method=='jev-top2-orthogonal-equal') for r in rows])
                write_json(str(f)+'.meta.json',{'schema':'portal-inference-v1','method':method,'implementation_hash':'test',
                    'data_hash':digest(rows),'cards_hash':'test','scoring':'portal','seed':42,'dtype':'test','max_prompt':768,
                    'choice_batch_size':1,'versions':{},'models':{'base':{'repo':'test'}}})
            args=Namespace(data=p/'data.jsonl',predictions=files,output_dir=p/'report',bootstrap=10,seed=42,input_price_per_million=None)
            evaluate(args)
            result=read_json(p/'report/report.json')
            self.assertEqual(result['summary'][0]['macro_accuracy_pct'],0)
            self.assertEqual(result['summary'][1]['macro_accuracy_pct'],100)
            self.assertEqual(result['paired_accuracy'][0]['delta_macro_pp'],100)
            meta=read_json(str(files[1])+'.meta.json');meta['max_prompt']=512;write_json(str(files[1])+'.meta.json',meta)
            with self.assertRaisesRegex(ValueError,'Incompatible scoring'): evaluate(args)
            write_jsonl(files[0],read_jsonl(files[0])[:1])
            with self.assertRaisesRegex(ValueError,'cover exactly'): evaluate(args)

    def test_prediction_score_integrity(self):
        row=make_row();p=prediction(row);validate_prediction(row,p)
        p['predicted_idx']=1
        with self.assertRaises(ValueError): validate_prediction(row,p)
        p=prediction(row);p['choice_scores'][0]=math.nan
        with self.assertRaises(ValueError): validate_prediction(row,p)

    def test_macro_bootstrap_does_not_overweight_large_tasks(self):
        rows=[make_row(i,task='rte' if i<9 else 'boolq') for i in range(10)]
        left={r['id']:0 for r in rows};right={r['id']:int(r['task']=='boolq') for r in rows}
        result=paired_diagnostic(rows,left,right,draws=10)
        self.assertEqual(result['delta_macro_pp'],50)
        self.assertEqual(result['ci95_low'],50)


if __name__ == '__main__':
    unittest.main()
