"""Summarize persisted stage attempts without running any models again."""
import argparse
import csv
import json
from pathlib import Path


def summarize(directory):
    root = Path(directory)
    records = [json.loads(p.read_text()) for p in sorted(root.glob('*/stage.json'))]
    rows = []
    for record in records:
        for gpu in record.get('gpu') or [{}]:
            rows.append({**{key: record.get(key) for key in (
                'stage', 'status', 'started_utc', 'finished_utc', 'wall_s', 'exit_code',
                'rows_before', 'rows_after', 'new_rows', 'gpu_sample_interval_ms', 'suite_jobs', 'attempt_dir')},
                **{key: gpu.get(key) for key in ('uuid', 'gpu_index', 'samples', 'gpu_util_pct_mean',
                    'gpu_util_pct_max', 'memory_util_pct_mean', 'memory_util_pct_max',
                    'memory_used_mib_max', 'memory_total_mib_max', 'power_w_mean', 'power_w_max',
                    'temperature_c_max')}})
    if rows:
        with (root / 'stages.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    text = ['# Stage timing and GPU telemetry', '',
            'Each row describes one attempt on one physical GPU. All attempts, including failures and cache reuse, are retained.', '',
            'Wall time includes imports, validation, model load, warmup, computation and teardown. GPU utilization is the sample mean over that entire process, not steady-state inference utilization. GPU samples include other processes on the same device. The sampled memory peak may miss short spikes; prediction files also record PyTorch allocation peaks.', '',
            'For parallel runs, overlapping stages see shared device utilization and memory, not per-method GPU utilization. Do not sum overlapping GPU samples or stage wall times as suite elapsed time; use the separately recorded whole-suite attempt.', '',
            'Use per-example `latency_s`, `selection_s`, `scoring_s` and the saved `after.runtime.json` to separate routing, inference and initialization. Shared embedding/Jev routes must not be counted twice as actual suite work. Low whole-process utilization on a tiny smoke set is not evidence of proportional parallel speedup.', '',
            '| Stage | Status | New rows | Wall s | GPU | GPU mean % | GPU max % | Peak MiB |',
            '|---|---|---:|---:|---|---:|---:|---:|']

    def num(value):
        return '-' if value is None else f'{value:.2f}'

    for row in rows:
        text.append(f"| {row['stage']} | {row['status']} | {row['new_rows']} | {num(row['wall_s'])} | "
                    f"{row['gpu_index']} | {num(row['gpu_util_pct_mean'])} | {num(row['gpu_util_pct_max'])} | "
                    f"{num(row['memory_used_mib_max'])} |")
    (root / 'summary.md').write_text('\n'.join(text) + '\n')
    print(f'Telemetry summary: {root / "summary.md"}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory')
    summarize(parser.parse_args().directory)
