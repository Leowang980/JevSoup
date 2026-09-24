# Experiments

## Main Comparison

Use the same full evaluation split, expert descriptions, fixed checkpoint
revisions, BF16 dtype and candidate batch size 1. Run each method separately
for Qwen3-1.7B, Qwen3-4B and Qwen3-8B:

| Method | `--method` | Routing |
| --- | --- | --- |
| Base | `base` | None |
| Adaptive Minds | `am-top2-equal` | Same-size local backbone |
| LoGo | `logo-top2-weighted` | Same-size local backbone and expert pool |
| AdapterSoup | `adaptersoup-top2-equal` | Shared training-sample embedding index |
| Arrow | `arrow-top2-weighted` | Dynamic routing with size-specific prototypes |
| JevSoup | `jev-top2-orthogonal-equal` | Shared Jev probabilities |

See [Getting Started](GETTING_STARTED.md) and [Baselines](BASELINES.md) for the
commands. Reuse one compatible Jev routing file across the backbone sizes.
Do not share AM, LoGo or Arrow artifacts across different backbones.

## Ablation: Qwen3-4B

Keep the evaluation set fixed and reuse the same Jev routing probabilities.
Give every configuration a distinct output filename.

| Configuration | `--method` | Additional settings |
| --- | --- | --- |
| Random Top-2 | `random-top2-equal` | No routes; repeat `--routing-seed 42`, `43`, `44` |
| Jev Top-1 | `jev-top1` | Jev routes |
| Probability weights, no projection | `jev-top2-prob` | Jev routes |
| Probability weights with projection | `jev-top2-orthogonal-prob` | Jev routes |
| AM Top-2 | `am-top2-equal` | 4B AM routes |
| Equal weights, no projection | `jev-top2-equal` | Jev routes |
| JevSoup | `jev-top2-orthogonal-equal` | Jev routes |

For Random Top-2, report the mean of the three seed-level accuracies, not the
accuracy of an ensemble of their predictions.

## Projection Sensitivity: Qwen3-4B

Vary only projection strength; keep the ordered expert pair and equal mixing
coefficients fixed:

```bash
for strength in 0 0.25 0.5 0.75 1; do
  python -m jevsoup run --method jev-top2-orthogonal-equal \
    --strength "$strength" --model-dir models/portal-qwen3-4b \
    --data artifacts/portal/data/eval.jsonl --routes runs/full/jev.routes.jsonl \
    --output "runs/sensitivity/lambda-${strength}.predictions.jsonl"
done
```

Evaluate each file in a separate report directory: the underlying method name
is the same, while the saved strength differs. Strength 0 is the unprojected
control; strength 1 is JevSoup.

## Timing and GPU Utilization

Per-example predictions include activation/scoring time, routing token counts
and peak GPU allocation. Wrap a command to collect overall wall time and GPU
utilization samples:

```bash
python scripts/profile_stage.py --directory runs/4b/telemetry --stage jevsoup -- \
  python -m jevsoup run --method jev-top2-orthogonal-equal \
  --model-dir models/portal-qwen3-4b --data artifacts/portal/data/eval.jsonl \
  --routes runs/full/jev.routes.jsonl --output runs/4b/jevsoup.predictions.jsonl
python scripts/summarize_telemetry.py runs/4b/telemetry
```

GPU utilization is measured for the whole device, including other processes.
Use measured memory headroom to choose concurrency, and use separate output
paths for concurrent jobs. No worker is launched automatically by installation.

## Outputs

Data, checkpoints, routes, predictions and telemetry stay in local ignored
directories. Existing results are validated and resumed rather than overwritten.
Keep these artifacts for your analysis, but do not commit them to the code repo.
