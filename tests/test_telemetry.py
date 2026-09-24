"""Failure retention and GPU aggregation must work without loading models."""
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('profile_stage', ROOT / 'scripts/profile_stage.py')
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


class TelemetryTest(unittest.TestCase):
    def test_failed_and_reused_attempts_are_both_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            command = [sys.executable, str(ROOT / 'scripts/profile_stage.py'),
                       '--directory', tmp, '--stage', 'base', '--', sys.executable, '-c']
            failed = subprocess.run(command + ["print('partial progress', flush=True); raise SystemExit(7)"],
                                    capture_output=True, text=True)
            self.assertEqual(failed.returncode, 7, failed.stderr)
            attempt = next(root.glob('*/stage.json'))
            original = attempt.read_bytes()
            self.assertIn('partial progress', (attempt.parent / 'console.log').read_text())
            reused = subprocess.run(command + ["print('Already complete: predictions.jsonl')"],
                                    capture_output=True, text=True)
            self.assertEqual(reused.returncode, 0, reused.stderr)
            records = [json.loads(line) for line in (root / 'stages.jsonl').read_text().splitlines()]
            self.assertEqual([r['status'] for r in records], ['failed', 'reused'])
            self.assertEqual([r['exit_code'] for r in records], [7, 0])
            self.assertEqual(attempt.read_bytes(), original)
            self.assertEqual(len(list(root.glob('*/console.log'))), 2)
            self.assertTrue(all(r['wall_s'] > 0 and r['new_rows'] == 0 for r in records))
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/summarize_telemetry.py'), tmp],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('failed', (root / 'summary.md').read_text())
            self.assertIn('reused', (root / 'stages.csv').read_text())

    def test_gpu_metrics_keep_devices_separate_and_missing_values_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'gpu.csv'
            path.write_text(','.join(PROFILE.GPU_FIELDS) + '\n' +
                            't, 0, GPU-a, 20, 10, 100, 1000, N/A, 30\n' +
                            't, 1, GPU-b, 80, 50, 800, 1000, 200, 50\n' +
                            't, 0, GPU-a, 40, 20, 200, 1000, N/A, 40\n')
            info = {r['uuid']: r for r in PROFILE.gpu_summary(path)}
            self.assertEqual(info['GPU-a']['gpu_util_pct_mean'], 30)
            self.assertEqual(info['GPU-a']['memory_used_mib_max'], 200)
            self.assertIsNone(info['GPU-a']['power_w_mean'])
            self.assertEqual(info['GPU-b']['gpu_util_pct_mean'], 80)


if __name__ == '__main__':
    unittest.main()
