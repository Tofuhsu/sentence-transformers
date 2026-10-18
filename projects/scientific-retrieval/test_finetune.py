# SPDX-License-Identifier: Apache-2.0
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from finetune import load_dataset


class FinetuneTests(unittest.TestCase):
    def _write_jsonl(self, directory, rows):
        path = Path(directory) / "training-data.jsonl"
        with path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return path

    def test_load_dataset_builds_expected_columns_from_valid_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_jsonl(directory, [
                {"query_id": "1", "positive_document_id": "p1", "negative_document_id": "n1",
                 "anchor": "claim one", "positive": "doc one", "negative": "doc two"},
                {"query_id": "2", "positive_document_id": "p2", "negative_document_id": "n2",
                 "anchor": "claim two", "positive": "doc three", "negative": "doc four"},
            ])
            dataset, row_count = load_dataset(path)
            self.assertEqual(row_count, 2)
            self.assertEqual(set(dataset.column_names), {"anchor", "positive", "negative"})
            self.assertEqual(dataset["anchor"], ["claim one", "claim two"])
            self.assertEqual(dataset["positive"], ["doc one", "doc three"])
            self.assertEqual(dataset["negative"], ["doc two", "doc four"])

    def test_load_dataset_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = self._write_jsonl(directory, [])
            with self.assertRaisesRegex(ValueError, "empty"):
                load_dataset(path)

    def test_load_dataset_rejects_blank_text_in_any_column(self):
        base = {"query_id": "1", "positive_document_id": "p1", "negative_document_id": "n1",
                "anchor": "claim", "positive": "doc", "negative": "doc"}
        for column in ("anchor", "positive", "negative"):
            with self.subTest(column=column), tempfile.TemporaryDirectory() as directory:
                row = dict(base, **{column: "   "})
                path = self._write_jsonl(directory, [row])
                with self.assertRaisesRegex(ValueError, column):
                    load_dataset(path)

    def test_cli_missing_required_arguments_fails_clearly(self):
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("finetune.py"))],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("required", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_cli_missing_training_data_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, str(Path(__file__).with_name("finetune.py")),
                       "--variant", "B", "--seed", "42", "--epochs", "1",
                       "--training-data", str(root / "training-data"),
                       "--cache-dir", str(root / "cache-dir"),
                       "--token-audit", str(root / "token-audit"),
                       "--output-dir", str(root / "output-dir")]
            environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTORCH_MPS_FAST_MATH": "0"}
            result = subprocess.run(command, capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Fine-tuning error:", result.stderr)
            self.assertIn("training-data", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertFalse((root / "output-dir").exists())

    def test_cli_refuses_to_overwrite_an_existing_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output_dir = root / "output-dir"
            output_dir.mkdir()
            (output_dir / "train-result.json").write_text("{}")
            command = [sys.executable, str(Path(__file__).with_name("finetune.py")),
                       "--variant", "B", "--seed", "42", "--epochs", "1",
                       "--training-data", str(root / "does-not-exist.jsonl"),
                       "--cache-dir", str(root / "cache-dir"),
                       "--token-audit", str(root / "token-audit"),
                       "--output-dir", str(output_dir)]
            environment = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                           "PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTORCH_MPS_FAST_MATH": "0"}
            result = subprocess.run(command, capture_output=True, text=True, env=environment)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Fine-tuning error:", result.stderr)
            self.assertIn("Already has a result", result.stderr)
            self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
