# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ENV = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
       "PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTORCH_MPS_FAST_MATH": "0"}


class EvaluateFinetunedCliTests(unittest.TestCase):
    script = "evaluate_finetuned.py"
    error_prefix = "Evaluation error:"
    extra_args = ("--text-policy",)  # arg names unique to this script, beyond the shared set below

    def test_cli_missing_required_arguments_fails_clearly(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name(self.script))],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("required", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_missing_split_lock_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, str(Path(__file__).with_name(self.script)),
                       "--label", "unit-test",
                       "--model-dir", str(root / "model-dir"),
                       "--split-lock", str(root / "split-lock.json"),
                       "--data-audit", str(root / "data-audit")]
            for name in self.extra_args:
                command += [name, str(root / name.lstrip("-"))]
            for name in self._extra_required():
                command += [f"--{name}", str(root / name)]
            command += ["--output-dir", str(root / "output-dir")]
            result = subprocess.run(command, capture_output=True, text=True, env=ENV)
            self.assertEqual(result.returncode, 2)
            self.assertIn(self.error_prefix, result.stderr)
            self.assertIn("split-lock", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((root / "output-dir").exists())

    def test_cli_refuses_to_overwrite_an_existing_output_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "output-dir"
            output_dir.mkdir()
            command = [sys.executable, str(Path(__file__).with_name(self.script)),
                       "--label", "unit-test",
                       "--model-dir", str(root / "model-dir"),
                       "--split-lock", str(root / "split-lock.json"),
                       "--data-audit", str(root / "data-audit")]
            for name in self.extra_args:
                command += [name, str(root / name.lstrip("-"))]
            for name in self._extra_required():
                command += [f"--{name}", str(root / name)]
            command += ["--output-dir", str(output_dir)]
            result = subprocess.run(command, capture_output=True, text=True, env=ENV)
            self.assertEqual(result.returncode, 2)
            self.assertIn(self.error_prefix, result.stderr)
            self.assertIn("fresh output directory", result.stderr)
            self.assertNotIn("Traceback", result.stderr)

    def _extra_required(self):
        return ()


class EvaluateOfficialTestCliTests(EvaluateFinetunedCliTests):
    script = "evaluate_official_test.py"
    error_prefix = "Official test evaluation error:"
    extra_args = ()

    def _extra_required(self):
        return ("test-open",)


if __name__ == "__main__":
    unittest.main(verbosity=2)
