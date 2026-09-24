"""Run one suite stage, retaining every attempt and device-level GPU samples."""
import argparse
import csv
import fcntl
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


GPU_FIELDS = ['timestamp', 'index', 'uuid', 'utilization.gpu', 'utilization.memory',
              'memory.used', 'memory.total', 'power.draw', 'temperature.gpu']


def save_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def row_count(path):
    if path is None or not path.exists():
        return 0
    with path.open() as stream:
        return sum(bool(line.strip()) for line in stream)


def gpu_summary(path):
    groups = defaultdict(list)
    with path.open() as stream:
        for row in csv.DictReader(stream, skipinitialspace=True):
            if row.get('uuid'):
                groups[row['uuid']].append(row)
    summary = []
    for uuid, rows in groups.items():
        item = {'uuid': uuid, 'gpu_index': rows[0]['index'], 'samples': len(rows)}
        for field, label in [('utilization.gpu', 'gpu_util_pct'),
                             ('utilization.memory', 'memory_util_pct'),
                             ('memory.used', 'memory_used_mib'), ('memory.total', 'memory_total_mib'),
                             ('power.draw', 'power_w'), ('temperature.gpu', 'temperature_c')]:
            values = []
            for row in rows:
                try:
                    value = float(row[field])
                    if math.isfinite(value):
                        values.append(value)
                except (ValueError, TypeError, KeyError):
                    pass
            for suffix, value in [('mean', statistics.mean(values) if values else None),
                                  ('max', max(values) if values else None)]:
                item[label + '_' + suffix] = value
        summary.append(item)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--interval-ms', type=int, default=500)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.interval_ms < 100 or not re.fullmatch(r'[a-z0-9_-]+', args.stage):
        parser.error('Provide a command, a simple stage name, and interval >= 100 ms')
    root = Path(args.directory)
    root.mkdir(parents=True, exist_ok=True)
    start_utc = datetime.now(timezone.utc)
    attempt = Path(tempfile.mkdtemp(prefix=start_utc.strftime('%Y%m%dT%H%M%S_') + args.stage + '_', dir=root))
    output = Path(command[command.index('--output') + 1]) if '--output' in command else None
    record = {'stage': args.stage, 'attempt_dir': str(attempt), 'command': command,
              'cwd': os.getcwd(), 'started_utc': start_utc.isoformat(), 'status': 'running',
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'suite_jobs': int(os.environ.get('JOBS', '1')),
              'gpu_sample_interval_ms': args.interval_ms, 'rows_before': row_count(output),
              'gpu_scope': 'whole physical device, including other processes and startup/teardown',
              'wall_scope': 'child process including imports, validation, load, work and teardown',
              'driver_timestamp_timezone': list(time.tzname)}
    save_json(attempt / 'stage.json', record)
    for suffix in ('.meta.json', '.runtime.json'):
        if output and Path(str(output) + suffix).exists():
            shutil.copy2(str(output) + suffix, attempt / ('before' + suffix))
    print(f'[profile] {args.stage}: {attempt}', flush=True)
    child = monitor = None
    reused = False
    returncode = 1
    start = time.perf_counter()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    with (attempt / 'gpu.csv').open('w') as gpu, (attempt / 'monitor.log').open('w') as errors, \
            (attempt / 'console.log').open('w', buffering=1) as log:
        csv.writer(gpu).writerow(GPU_FIELDS)
        gpu.flush()
        try:
            try:
                monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=' + ','.join(GPU_FIELDS),
                    '--format=csv,noheader,nounits', '--loop-ms=' + str(args.interval_ms)],
                    stdout=gpu, stderr=errors, start_new_session=True)
            except OSError as error:
                record['monitor_error'] = str(error)
            child = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, bufsize=1, env=dict(os.environ, PYTHONUNBUFFERED='1'),
                                     start_new_session=True)
            record['pid'] = child.pid
            save_json(attempt / 'stage.json', record)
            for line in child.stdout:
                log.write(line)
                print(line, end='', flush=True)
                reused = reused or line.startswith('Already complete:')
            returncode = child.wait()
        except KeyboardInterrupt:
            record['interrupted'] = True
            returncode = 130
        except OSError as error:
            record['error'] = str(error)
            log.write(str(error) + '\n')
        finally:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            record['wall_s'] = time.perf_counter() - start
            record['finished_utc'] = datetime.now(timezone.utc).isoformat()
            if monitor is not None:
                record['monitor_exited_early'] = monitor.poll() is not None
                if monitor.poll() is None:
                    monitor.terminate()
                try:
                    monitor.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    monitor.kill()
                    monitor.wait()
    record.update(exit_code=returncode, rows_after=row_count(output),
                  status=('reused' if reused else 'completed') if returncode == 0 else 'failed',
                  gpu=gpu_summary(attempt / 'gpu.csv'))
    record['new_rows'] = record['rows_after'] - record['rows_before']
    for suffix in ('.meta.json', '.runtime.json'):
        if output and Path(str(output) + suffix).exists():
            shutil.copy2(str(output) + suffix, attempt / ('after' + suffix))
    save_json(attempt / 'stage.json', record)
    with (root / 'stages.jsonl').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        stream.write(json.dumps(record) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(f"[profile] {args.stage}: {record['status']}, {record['wall_s']:.2f}s, new rows={record['new_rows']}", flush=True)
    return returncode


if __name__ == '__main__':
    sys.exit(main())
