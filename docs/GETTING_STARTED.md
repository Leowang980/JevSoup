# Getting Started

## Installation

Use Linux, Python 3.12 and an NVIDIA GPU with BF16 support. Start with one GPU
worker; required memory depends on the backbone and the enabled methods.
Run the following commands from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-gpu.txt
python -m pip install -e .
```

Use the source checkout or an editable install. The numerical engines use
repository-relative scripts; a standalone wheel is not the supported layout.

## Download Resources

For JevSoup, download the fixed PorTAL dataset, Qwen3 backbone and its 14 adapters:

```bash
python -m jev_lora prepare
python -m jev_lora download --size 4b
```

Supported sizes are `1.7b`, `4b` and `8b`. Data is prepared under
`artifacts/portal/data/`; model files are stored under `models/portal-qwen3-4b/`.
The code pins the dataset and checkpoint revisions and checks exported adapter
hashes. Compatible existing downloads are reused.

For all baselines, including the Qwen3 embedding model required by AdapterSoup:

```bash
bash scripts/download_resources.sh 4b
```

## Run a Small Experiment

### 1. Route with Jev

Set `TYPESAFE_API_KEY` in your environment or secret manager. Never commit the
key. New routing requests use the hosted TypeSafe service and may incur fees.

```bash
python -m jev_lora route --kind jev --jev-model jev-1.13.0 \
  --data artifacts/portal/data/smoke.jsonl \
  --output runs/smoke/jev.routes.jsonl --cache cache/jev
```

This routes the 14-question development smoke split. Query probabilities and
token usage are saved locally. Existing compatible cache entries are reused.

### 2. Compose and Evaluate

```bash
python -m jevsoup run --method jev-top2-orthogonal-equal \
  --model-dir models/portal-qwen3-4b \
  --data artifacts/portal/data/smoke.jsonl \
  --routes runs/smoke/jev.routes.jsonl \
  --output runs/smoke/jevsoup.predictions.jsonl

python -m jev_lora evaluate --data artifacts/portal/data/smoke.jsonl \
  --predictions runs/smoke/jevsoup.predictions.jsonl \
  --output-dir reports/smoke
```

The report includes task-macro and example-micro accuracy. Smoke results check
the pipeline; they are not the paper's full benchmark results.

## Full Evaluation

Replace `smoke.jsonl` with `eval.jsonl` in **both** routing and inference, and
use a new run directory.
The [experiment guide](EXPERIMENTS.md) covers the complete method matrix.

Jev routes can be reused across the three backbone sizes for the same inputs
and expert descriptions. AM and LoGo routes must match the execution backbone.

## Resume and Reproducibility

Repeat the same command to skip completed examples. Metadata checks reject
changed data, routes, settings, versions or model snapshots. Use a new output
path for a changed experiment; do not edit metadata to force reuse.

This code repository does not ship historical predictions or routing bundles.
Fresh API responses and changes in GPU/software can change numerical results.
Keep the same local routes and environment when comparing methods.

## Tests

```bash
python -m unittest discover -s tests -v
```

Tests use tiny random models and do not download checkpoints or call paid APIs.
Three additional comparisons against upstream Arrow code are skipped unless
the pinned MTTL checkout described in [Baselines](BASELINES.md#arrow) is present.
