"""Real offline selfcheck regression and strict source-based cache identity."""
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from jev_lora import core

ROOT = Path(__file__).resolve().parents[1]


class SelfcheckTest(unittest.TestCase):
    def test_full_selfcheck_offline(self):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
        env.pop('TYPESAFE_API_KEY', None)
        result = subprocess.run([sys.executable, '-m', 'jev_lora', 'selfcheck'], cwd=ROOT, env=env,
                                text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('LoGo projection/reset parity', result.stdout)

    def test_source_identity_is_not_aliased(self):
        self.assertEqual(core.implementation_hash(), core.source_implementation_hash())

    def test_unrecognized_source_changes_do_not_reuse_old_hash(self):
        with patch.object(core, 'source_implementation_hash', return_value='different-scoring-code'):
            self.assertEqual(core.implementation_hash(), 'different-scoring-code')


if __name__ == '__main__':
    unittest.main()
