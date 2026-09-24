# Experiment protocol

## Data and scoring

The expert pool is the 14 task-specific q/v LoRA adapters distributed through
PorTAL, with Qwen3-1.7B, Qwen3-4B or Qwen3-8B as the frozen backbone. Immutable
model and dataset revisions are pinned in `jev_lora/core.py`. Downloads record
adapter configuration and weight hashes in local model manifests.

The pinned validation split has 19,548 examples. Seed-42 selection holds out
71 development examples, grouping repeated canonical prompts across tasks.
The remaining 19,477 examples form the evaluation set. `smoke.jsonl` contains
14 development examples and is intended only for pipeline checks.

This is upstream validation data, also used for upstream checkpoint selection;
it is not an untouched test set. The same deterministic choice permutation is
used across methods. Gold answers and task labels are never routing inputs.
Expert descriptions contain zero demonstrations.

Answer scoring is the original PorTAL-style continuation log probability,
summed over answer tokens and divided by the number of answer characters,
including the leading space. Each candidate is scored separately (batch 1),
with a left-truncated prompt of at most 768 tokens and no answer truncation.
There is no answer-generation/chat wrapper. Macro accuracy averages the 14
task accuracies equally; micro accuracy averages all questions.

## Methods

| CLI method | Selection | Composition |
| --- | --- | --- |
| `base` | None | Unadapted backbone |
| `am-top2-equal` | Same-size unadapted Qwen3 generates two ranked expert IDs; non-thinking, greedy, max 32 new tokens; seeded keyword completion if parsing is incomplete | Equal sum of complete LoRA updates |
| `logo-top2-weighted` | Original local LoGo adaptation: final-block q-projection activation norms | Selected-score-weighted complete updates |
| `adaptersoup-top2-equal` | Per-query average cosine to 100 deduplicated training samples per expert, using fixed Qwen3 embeddings | Average A and B separately, then multiply; memoization preserves ordered weights |
| `arrow-top2-weighted` | Top input singular vector of each adapter update; per-token/per-projection absolute dot product and top-2 softmax, temperature 1 | Weighted A and B separately, then multiply, matching the audited `merge_after=False` convention |
| `jev-top2-orthogonal-equal` | Jev ranks the expert descriptions; descending probability, alphabetic ties | JevSoup: preserve the first update, project the second out of the first update's row space, then combine equally |

JevSoup is **not** separate averaging of A and B, and none of these methods is
an ensemble of answer probabilities. Geometry is computed in CPU float64 with
relative SVD tolerance `1e-10`, separately at every q/v projection. Residual A
is cast back to BF16; the second update's norm is not restored. Original LoRA
scaling is retained. The ordered pair matters.

AdapterSoup support excludes every validation prompt and within-expert
duplicates. Sample vectors are individually normalized; their average is not
normalized again. The query instruction and support sampling are preserved in
the frozen implementation. This is a per-question, fixed-top-2 adaptation,
not the original paper's domain-level selection protocol. AM Top-2, fixed-top-2
LoGo and the common multiple-choice evaluation are likewise explicit adaptations.

## Ablation and sensitivity

The 4B ablation includes random Top-2 (mean of routing seeds 42/43/44), Jev Top-1,
probability-weighted Top-2 without and with projection, AM Top-2, equal-weight
Top-2 without projection, and JevSoup.

The sensitivity experiment keeps Jev's ordered pairs and equal coefficients
fixed and varies only projection strength over `0, .25, .5, .75, 1`. Intermediate
strengths are computed from the original float64 factors, not by interpolating
BF16 endpoints. Zero is the unprojected control and one is JevSoup.

## Timing and caching

Predictions are append-only with per-example scoring/activation time, routing
token usage and peak allocated GPU memory. Resume checks validate code,
versions, data, routes, model snapshots, settings and existing predictions.
The public runner adds a process lock and records its own identity and GPU
name; historical output files are not edited or reused as fresh measurements.

For device utilization use `scripts/profile_stage.py`. Device utilization is
whole-GPU, not a per-process attribution. Historical row times were collected
under varying hardware/concurrency and exclude some startup/pair-preparation
costs; they are not a controlled latency or monetary-cost comparison.
