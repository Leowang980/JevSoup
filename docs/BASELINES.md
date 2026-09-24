# Baselines

Prepare data and the selected Qwen3/PorTAL checkpoint as described in
[Getting Started](GETTING_STARTED.md). The commands below use 4B and the full
evaluation split. Change the model directory for 1.7B or 8B.

All main routed baselines use Top-2. They are adaptations to a shared expert
pool and scoring protocol; see [Protocol](PROTOCOL.md) for differences from
the original papers.

## Base

```bash
python -m jevsoup run --method base --model-dir models/portal-qwen3-4b \
  --data artifacts/portal/data/eval.jsonl --output runs/4b/base.predictions.jsonl
```

## Adaptive Minds

AM uses the same-size, unadapted Qwen3 backbone to generate two ranked expert
IDs in non-thinking mode. It records generated tokens and keyword fallbacks.

```bash
python -m jevsoup am-route --model-dir models/portal-qwen3-4b \
  --data artifacts/portal/data/eval.jsonl --output runs/4b/am.routes.jsonl
python -m jevsoup run --method am-top2-equal --model-dir models/portal-qwen3-4b \
  --data artifacts/portal/data/eval.jsonl --routes runs/4b/am.routes.jsonl \
  --output runs/4b/am.predictions.jsonl
```

## LoGo

LoGo selects experts using activation scores from an all-adapter probe.

```bash
python -m jev_lora logo-route --model-dir models/portal-qwen3-4b \
  --data artifacts/portal/data/eval.jsonl --output runs/4b/logo.routes.jsonl
python -m jevsoup run --method logo-top2-weighted --model-dir models/portal-qwen3-4b \
  --data artifacts/portal/data/eval.jsonl --routes runs/4b/logo.routes.jsonl \
  --output runs/4b/logo.predictions.jsonl
```

AM and LoGo are local GPU methods and require separate routing for each backbone.

## AdapterSoup

Download the fixed embedding checkpoint if it is not already present:

```bash
python -m jev_lora download-embedding --revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3
```

Build the training-sample index and route each query by average cosine similarity:

```bash
python scripts/run_adapter_baselines.py soup-index --data artifacts/portal/data/eval.jsonl
python scripts/run_adapter_baselines.py soup-route \
  --data artifacts/portal/data/eval.jsonl --run-dir runs/shared-soup
python -m jevsoup run --method adaptersoup-top2-equal --model-dir models/portal-qwen3-4b \
  --data artifacts/portal/data/eval.jsonl --routes runs/shared-soup/adaptersoup.routes.jsonl \
  --output runs/4b/adaptersoup.predictions.jsonl
```

Defaults use 100 training samples per expert, seed 42, embedding input budget
4096 and batch size 4. Training support excludes validation-prompt overlap and
within-expert duplicates. Use the default data locations for index preparation.

The selection index and routes are shared across backbones. A and B factors
are averaged separately; the result is not an average of complete updates.
Factor averages are cached in memory without changing the composition rule.
Offline regression tests check cached versus direct factor averaging.

## Arrow

The prototype builder checks the reference implementation's commit:

```bash
mkdir -p artifacts/references
git clone https://github.com/microsoft/mttl.git artifacts/references/mttl
git -C artifacts/references/mttl checkout --detach 169c9191be960e35a59e85c37af90f3f518fe125
python scripts/run_adapter_baselines.py arrow-prototypes \
  --model-dir models/portal-qwen3-4b \
  --arrow-prototypes artifacts/arrow-4b/prototypes.safetensors
python -m jevsoup run --method arrow-top2-weighted --model-dir models/portal-qwen3-4b \
  --data artifacts/portal/data/eval.jsonl \
  --arrow-prototypes artifacts/arrow-4b/prototypes.safetensors \
  --output runs/4b/arrow.predictions.jsonl
```

Prepare separate prototypes for each backbone. Arrow routes dynamically per
token and q/v projection; no per-question routing JSONL is needed. Add
`--arrow-traces` only if detailed traces are required: full traces can consume
substantial disk space. Disabling trace capture does not disable the router.

## Evaluate

```bash
python -m jev_lora evaluate --data artifacts/portal/data/eval.jsonl \
  --predictions runs/4b/base.predictions.jsonl runs/4b/am.predictions.jsonl \
  runs/4b/logo.predictions.jsonl runs/4b/adaptersoup.predictions.jsonl \
  runs/4b/arrow.predictions.jsonl --output-dir reports/4b-baselines
```

Use `python -m jevsoup` for inference and AM Top-2 routing. Lower-level Python
scripts retain compatibility helpers but are not alternative experiment recipes.
