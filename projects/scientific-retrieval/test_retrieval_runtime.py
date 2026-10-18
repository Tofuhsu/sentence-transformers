# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from retrieval_runtime import ExactCosineIndex, latency_summary, timed_call, validate_embeddings


class RetrievalRuntimeTests(unittest.TestCase):
    def test_exact_cosine_search_matches_known_vectors_and_negative_scores(self):
        matrix = np.array([[0, 1], [-1, 0], [0.6, 0.8], [1, 0]], dtype=np.float32)
        index = ExactCosineIndex(matrix, ["north", "west", "diagonal", "east"])
        top, scores, order = index.search(np.array([1, 0], dtype=np.float32), k=3)
        self.assertEqual([row["document_id"] for row in top], ["east", "diagonal", "north"])
        np.testing.assert_allclose(scores, [0, -1, 0.6, 1], rtol=0, atol=1e-7)
        self.assertEqual(order.tolist(), [3, 2, 0, 1])

    def test_ties_at_cutoff_are_order_invariant_over_the_entire_corpus(self):
        matrix = np.array([[1, 0], [1, 0], [1, 0]], dtype=np.float32)
        query = matrix[0]
        first = ExactCosineIndex(matrix, ["d2", "d10", "d1"]).search(query, k=2)[0]
        second = ExactCosineIndex(matrix, ["d1", "d2", "d10"]).search(query, k=2)[0]
        self.assertEqual(first, second)
        self.assertEqual([row["document_id"] for row in first], ["d1", "d10"])

    def test_invalid_embeddings_ids_and_search_dimensions_fail(self):
        for matrix in (np.array([[1, 0]], dtype=np.float64),
                       np.array([[float("nan"), 0]], dtype=np.float32),
                       np.array([[0, 0]], dtype=np.float32),
                       np.array([[2, 0]], dtype=np.float32)):
            with self.subTest(matrix=matrix), self.assertRaises(ValueError):
                validate_embeddings(matrix, 1, 2)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ExactCosineIndex(np.eye(2, dtype=np.float32), ["a", "a"])
        with self.assertRaisesRegex(ValueError, "dimensions"):
            ExactCosineIndex(np.eye(2, dtype=np.float32), ["a"])
        index = ExactCosineIndex(np.eye(2, dtype=np.float32), ["a", "b"])
        for query in (np.array([1, 0, 0], dtype=np.float32), np.eye(2, dtype=np.float32)):
            with self.subTest(shape=query.shape), self.assertRaises(ValueError):
                index.search(query, k=1)
        for k in (0, 3, True):
            with self.subTest(k=k), self.assertRaises(ValueError):
                index.search(np.array([1, 0], dtype=np.float32), k=k)

    def test_timer_waits_before_start_and_after_work(self):
        events = []
        times = iter([100.0, 100.25])
        def synchronize():
            events.append("synchronize")
        def clock():
            events.append("clock")
            return next(times)
        def operation():
            events.append("operation")
            return "result"
        result, elapsed = timed_call(operation, synchronize=synchronize, clock=clock)
        self.assertEqual(result, "result")
        self.assertEqual(elapsed, 0.25)
        self.assertEqual(events, ["synchronize", "clock", "operation", "synchronize", "clock"])

    def test_latency_summary_uses_known_order_statistics_and_rejects_bad_samples(self):
        result = latency_summary(list(range(20, 0, -1)))
        self.assertEqual((result["samples"], result["mean"], result["p50"], result["p95"]),
                         (20, 10.5, 10, 19))
        for samples in ([], [float("nan")], [-1]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                latency_summary(samples)

    def test_baseline_cli_missing_input_fails_before_creating_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            command = [sys.executable, str(Path(__file__).with_name("retrieval_baseline.py"))]
            for name in ("split-lock", "text-policy", "data-audit", "token-audit", "cache-dir", "output-dir", "artifact-dir"):
                command += [f"--{name}", str(root / name)]
            env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "PYTORCH_ENABLE_MPS_FALLBACK": "0", "PYTORCH_MPS_FAST_MATH": "0"}
            result = subprocess.run(command, capture_output=True, text=True, env=env)
            self.assertEqual(result.returncode, 2)
            self.assertIn("Retrieval baseline error:", result.stderr)
            self.assertIn("split-lock", result.stderr)
            self.assertFalse((root / "output-dir").exists())
            self.assertFalse((root / "artifact-dir").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
