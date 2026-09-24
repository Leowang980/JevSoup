"""Complete, protocol-matched PorTAL multiple-choice evaluation and paired comparisons."""
import csv
import math
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .core import canonical_text, digest, read_json, read_jsonl, unique_index, write_json, write_jsonl
from .data import validate_row
from .scoring import normalized_choice_scores


def csv_write(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)


def percentile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * (len(values) - 1)))]


def paired_diagnostic(rows, left, right, draws=1000, seed=42):
    """Task-stratified paired bootstrap, clustering repeated prompts within each task."""
    groups = defaultdict(lambda: defaultdict(list))
    diffs = []
    for row in rows:
        delta = float(right[row["id"]]) - float(left[row["id"]])
        groups[row["task"]][canonical_text(row["prompt"])].append(delta)
        diffs.append(delta)
    macro = statistics.mean(statistics.mean(d for cluster in task.values() for d in cluster) for task in groups.values())
    compressed = [[(sum(v), len(v)) for v in task.values()] for task in groups.values()]
    rng, samples = random.Random(seed), []
    for _ in range(draws):
        task_scores = []
        for clusters in compressed:
            selected = [clusters[rng.randrange(len(clusters))] for _ in clusters]
            task_scores.append(sum(s for s, n in selected) / sum(n for s, n in selected))
        samples.append(100 * statistics.mean(task_scores))
    return {"delta_macro_pp": 100 * macro, "ci95_low": percentile(samples, .025),
            "ci95_high": percentile(samples, .975), "wins": sum(d > 0 for d in diffs),
            "losses": sum(d < 0 for d in diffs), "ties": sum(d == 0 for d in diffs),
            "task_prompt_clusters": sum(len(v) for v in groups.values())}


def validate_prediction(row, pred):
    if pred.get("input_hash") != row["input_hash"]:
        raise ValueError("Prediction/query hash mismatch")
    count = len(row["choices"])
    fields = ("choice_scores", "choice_logprobs", "choice_characters", "choice_token_counts")
    if any(not isinstance(pred.get(k), list) or len(pred[k]) != count for k in fields):
        raise ValueError("Prediction must contain one score, logprob and length per choice")
    if pred["choice_characters"] != [len(c) for c in row["choices"]]:
        raise ValueError("Choice character counts differ from the data")
    if any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in pred["choice_token_counts"]):
        raise ValueError("Invalid choice token count")
    expected_scores, expected_idx = normalized_choice_scores(pred["choice_logprobs"], pred["choice_characters"])
    if any(not math.isfinite(s) or not math.isclose(s, e, abs_tol=1e-6, rel_tol=1e-5)
           for s, e in zip(pred["choice_scores"], expected_scores)):
        raise ValueError("Stored choice scores differ from normalized log probabilities")
    if isinstance(pred.get("predicted_idx"), bool) or pred.get("predicted_idx") != expected_idx:
        raise ValueError("Predicted option is not the argmax of the recorded scores")
    for k in ("scoring_s", "selection_s", "route_s", "peak_allocated_gib"):
        if not math.isfinite(pred[k]) or pred[k] < 0:
            raise ValueError(f"Invalid timing/memory value: {k}")


def evaluate(args):
    rows = read_jsonl(args.data)
    if not rows:
        raise ValueError("Empty evaluation data")
    data_by_id = unique_index(rows, "data")
    for row in rows:
        validate_row(row, prepared=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summaries, task_table, example_table, all_scores, metadata = [], [], [], {}, {}
    for path in args.predictions:
        preds = unique_index(read_jsonl(path), "predictions")
        if set(preds) != set(data_by_id):
            raise ValueError("Predictions must cover exactly all evaluation IDs")
        methods = {p["method"] for p in preds.values()}
        if len(methods) != 1:
            raise ValueError("A prediction file must contain exactly one method")
        method = next(iter(methods))
        if method in all_scores:
            raise ValueError("Duplicate method; use a separate report")
        meta = read_json(str(path) + ".meta.json")
        if meta.get("schema") != "portal-inference-v1" or meta["data_hash"] != digest(rows) or meta["method"] != method:
            raise ValueError("Prediction metadata differs from this PorTAL evaluation")
        metadata[method] = meta
        by_task, scores = defaultdict(list), {}
        gold_nll, gold_tokens = 0, 0
        for row in rows:
            p = preds[row["id"]]
            validate_prediction(row, p)
            correct = int(p["predicted_idx"] == row["gold_idx"])
            scores[row["id"]] = correct
            nll = -p["choice_logprobs"][row["gold_idx"]]
            tokens = p["choice_token_counts"][row["gold_idx"]]
            gold_nll += nll
            gold_tokens += tokens
            by_task[row["task"]].append((correct, nll, tokens))
            example_table.append({"id": row["id"], "task": row["task"], "method": method,
                                  "correct": correct, "predicted_idx": p["predicted_idx"],
                                  "gold_idx": row["gold_idx"], "weights": p["weights"]})
        per_task = []
        for task, values in sorted(by_task.items()):
            record = {"method": method, "task": task, "n": len(values),
                      "accuracy_pct": 100 * statistics.mean(v[0] for v in values),
                      "gold_token_nll": sum(v[1] for v in values) / sum(v[2] for v in values)}
            per_task.append(record)
            task_table.append(record)
        all_scores[method] = scores
        values = list(preds.values())
        calls = {p["route_payload_hash"]: p.get("route_usage", {}) for p in values if p.get("route_payload_hash")}
        input_tokens = sum(u.get("input_tokens", 0) for u in calls.values())
        selected = Counter(n for p in values for n in p.get("weights", {}))
        unit_price = args.input_price_per_million
        summaries.append({"method": method, "n": len(rows), "tasks": len(by_task),
                          "macro_accuracy_pct": statistics.mean(t["accuracy_pct"] for t in per_task),
                          "micro_accuracy_pct": 100 * statistics.mean(scores.values()),
                          "gold_token_nll": gold_nll / gold_tokens,
                          "scoring_mean_s": statistics.mean(p["scoring_s"] for p in values),
                          "scoring_p95_s": percentile([p["scoring_s"] for p in values], .95),
                          "selection_mean_s": statistics.mean(p["selection_s"] for p in values),
                          "routing_cold_mean_s": statistics.mean(p["route_s"] for p in values),
                          "estimated_serial_cold_mean_s": statistics.mean(p["route_s"] + p["scoring_s"] + p["selection_s"] for p in values),
                          "peak_allocated_gib": max(p["peak_allocated_gib"] for p in values),
                          "routing_peak_allocated_gib": max(p.get("routing_peak_allocated_gib", 0) for p in values),
                          "sequential_pipeline_peak_gib": max(max(p["peak_allocated_gib"], p.get("routing_peak_allocated_gib", 0)) for p in values),
                          "routing_input_tokens": sum(p.get("route_usage", {}).get("input_tokens", 0) for p in values),
                          "routing_output_tokens": sum(p.get("route_usage", {}).get("output_tokens", 0) for p in values),
                          "truncated_rows": sum(p["truncated"] for p in values),
                          "task_adapter_in_selected_pct": 100 * statistics.mean(r["task"] in preds[r["id"]]["weights"] for r in rows),
                          "distinct_adapters_selected": len(selected),
                          "router_fallback_rows": sum(p.get("router_fallback", False) for p in values),
                          "logical_unique_jev_calls": len(calls), "logical_jev_input_tokens": input_tokens,
                          "estimated_one_pass_jev_usd": None if unit_price is None else input_tokens / 1e6 * unit_price})
    fair_keys = ("implementation_hash", "data_hash", "cards_hash", "scoring", "seed", "dtype", "max_prompt", "choice_batch_size", "versions")
    first = next(iter(metadata.values()))
    model_inventory = {}
    for method, meta in metadata.items():
        if any(meta[k] != first[k] for k in fair_keys) or meta["models"]["base"] != first["models"]["base"]:
            raise ValueError(f"Incompatible scoring protocols in {method}; use separate reports")
        for name, info in meta["models"].items():
            if name in model_inventory and info != model_inventory[name]:
                raise ValueError(f"Different adapter snapshots for {name}; use separate reports")
            model_inventory[name] = info
    comparisons = []
    reference = "jev-top2-orthogonal-equal"
    pairs = [(left, reference) for left in all_scores if left != reference and reference in all_scores]
    pairs += [(left, "jev-top2-equal") for left in ("jev-top1", "am-top2-equal", "jev-top2-prob")
              if left in all_scores and "jev-top2-equal" in all_scores]
    for left, right in pairs:
        comparisons.append(dict(left=left, right=right, **paired_diagnostic(rows, all_scores[left], all_scores[right], args.bootstrap, args.seed)))
    csv_write(out / "summary.csv", summaries)
    csv_write(out / "per_task.csv", task_table)
    csv_write(out / "paired_accuracy.csv", comparisons)
    write_jsonl(out / "per_example.jsonl", example_table)
    write_json(out / "report.json", {"summary": summaries, "tasks": task_table, "paired_accuracy": comparisons,
                                      "inference_metadata": metadata,
                                      "protocol": {"score": "sum continuation logprob / character length including leading space",
                                                   "metric": "macro average of per-task choice accuracy; 0-100",
                                                   "data": "subsets of upstream validation, not an untouched test set",
                                                   "bootstrap": "task-stratified, repeated prompts clustered within each task",
                                                   "input_price_per_million": args.input_price_per_million}})
    lines = ["# Jev–PorTAL 实验结果", "", f"样本 {len(rows)}；任务 {len(set(r['task'] for r in rows))}。",
             "", "| 方法 | 任务宏平均准确率 % | 样本准确率 % | 评分秒/题 | 路由秒/题 | 顺序流程峰值 GiB |",
             "|---|---:|---:|---:|---:|---:|"]
    for s in summaries:
        lines.append(f"| {s['method']} | " + " | ".join(f"{s[k]:.3f}" for k in
                     ("macro_accuracy_pct", "micro_accuracy_pct", "scoring_mean_s", "routing_cold_mean_s", "sequential_pipeline_peak_gib")) + " |")
    lines += ["", "配对比较：右侧 − 左侧，按任务分层、同任务重复 prompt 分组 bootstrap。", ""]
    for c in comparisons:
        lines.append(f"- {c['right']} vs {c['left']}: {c['delta_macro_pp']:+.2f} pp，95% CI [{c['ci95_low']:.2f}, {c['ci95_high']:.2f}]。")
    lines += ["", "解释边界：", "",
              "- 数据来自原项目用于选择 checkpoint 的 validation，不能称为未见测试集。",
              "- 任务专家命中率只作辅助分析，最终答案准确率为主指标。",
              "- 检查输入截断数量、LLM router 回退数量和每任务退化情况。",
              "- Jev 方法共用缓存；各行的逻辑调用数/估算费用不能相加。",
              "- 评分时间含分词及候选前向；路由时间为记录的单请求耗时。加载和吞吐量未计入。", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report ready: {out / 'report.md'}")
