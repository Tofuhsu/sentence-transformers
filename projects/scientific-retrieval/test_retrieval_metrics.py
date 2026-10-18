# SPDX-License-Identifier: Apache-2.0
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from retrieval_metrics import evaluate, load_json, score_query


class RetrievalMetricTests(unittest.TestCase):
    def test_known_single_and_multiple_positive_answers(self):
        for rels, ranked, expected_ndcg, expected_recall in (
            ({"a": 1}, ["a", "b"], 1.0, 1.0),
            ({"a": 1}, ["b", "a"], 0.6309297535714575, 1.0),
            ({"a": 1, "c": 1}, ["a", "b", "c"], 0.9197207891481876, 1.0),
            ({"a": 1, "c": 1}, ["a", "b"], 0.6131471927654584, 0.5),
            ({"a": 1, "b": 0}, ["b", "unjudged"], 0.0, 0.0),
        ):
            with self.subTest(rels=rels, ranked=ranked):
                result = score_query(rels, ranked)
                self.assertAlmostEqual(result["ndcg"], expected_ndcg, places=12)
                self.assertAlmostEqual(result["recall"], expected_recall, places=12)

    def test_cutoff_and_recall_denominator_include_unretrieved_positives(self):
        docs = [f"d{i:02d}" for i in range(1, 13)]
        self.assertEqual(score_query({docs[10]: 1}, docs)["ndcg"], 0.0)
        self.assertAlmostEqual(score_query({docs[9]: 1}, docs)["ndcg"], 0.2890648263178879)
        result = score_query(dict.fromkeys(docs, 1), docs)
        self.assertEqual(result["ndcg"], 1.0)
        self.assertAlmostEqual(result["recall"], 10 / 12)

    def test_explicit_empty_results_count_in_macro_average(self):
        result = evaluate({"q1": {"a": 1}, "q2": {"a": 1, "b": 1}},
                          {"q1": [{"document_id": "a", "score": 1.0}], "q2": []},
                          ["q1", "q2"], ["a", "b"])
        self.assertEqual(result["query_count"], 2)
        self.assertEqual(result["metrics"], {"nDCG@10": 0.5, "Recall@10": 0.5})
        self.assertEqual(result["per_query"][1]["retrieved_at_k"], 0)

    def test_missing_extra_and_duplicate_query_ids_are_rejected(self):
        for run, queries in (({}, ["q"]), ({"q": [], "extra": []}, ["q"]),
                             ({"q": []}, ["q", "q"]), ({"q": []}, [])):
            with self.subTest(run=run, queries=queries), self.assertRaises(ValueError):
                evaluate({"q": {"a": 1}}, run, queries, ["a"])
        with self.assertRaisesRegex(ValueError, "Qrels query IDs"):
            evaluate({"q": {"a": 1}, "extra": {"a": 1}}, {"q": []}, ["q"], ["a"])

    def test_unknown_and_duplicate_corpus_or_result_ids_are_rejected(self):
        invalid = (
            ({"q": {"a": 1}}, {"q": []}, ["a", "a"]),
            ({"q": {"missing": 1}}, {"q": []}, ["a"]),
            ({"q": {"a": 1}}, {"q": [{"document_id": "missing", "score": 1}]}, ["a"]),
            ({"q": {"a": 1}}, {"q": [{"document_id": "a", "score": 1}] * 11}, ["a"]),
        )
        for qrels, run, corpus in invalid:
            with self.subTest(run=run, corpus=corpus), self.assertRaises(ValueError):
                evaluate(qrels, run, ["q"], corpus)

    def test_nonbinary_labels_zero_positives_and_invalid_cutoffs_are_rejected(self):
        for relevance in ({}, {"a": 0}, {"a": 2}, {"a": -1}, {"a": "1"}, {"a": True}):
            with self.subTest(relevance=relevance), self.assertRaises(ValueError):
                score_query(relevance, ["a"])
        for k in (0, -1, True, 1.5):
            with self.subTest(k=k), self.assertRaises(ValueError):
                score_query({"a": 1}, ["a"], k)

    def test_bad_scores_and_malformed_rows_are_rejected(self):
        rows = [{"document_id": "a", "score": score}
                for score in (float("nan"), float("inf"), -float("inf"), True, "0.9")]
        rows += [{"document_id": "a"}, {"document_id": "a", "score": 1, "typo": 1}, None]
        for row in rows:
            with self.subTest(row=row), self.assertRaises(ValueError):
                evaluate({"q": {"a": 1}}, {"q": [row]}, ["q"], ["a"])

    def test_ties_are_order_invariant_inputs_unchanged_and_equal_namespaces_retained(self):
        qrels = {"a": {"a": 1}}
        run = {"a": [{"document_id": "b", "score": 0.5}, {"document_id": "a", "score": 0.5}]}
        before = copy.deepcopy((qrels, run))
        first = evaluate(qrels, run, ["a"], ["b", "a"], k=1)
        self.assertEqual((qrels, run), before)
        second = evaluate(qrels, {"a": list(reversed(run["a"]))}, ["a"], ["a", "b"], k=1)
        self.assertEqual(first, second)
        self.assertEqual(first["metrics"], {"nDCG@1": 1.0, "Recall@1": 1.0})

    def test_duplicate_json_keys_are_rejected_before_overwriting_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.json"
            path.write_text('{"q": {"a": 1, "a": 0}}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
                load_json(path)

    def test_cli_success_missing_input_bad_cutoff_and_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = {"qrels": {"q": {"a": 1}},
                       "run": {"q": [{"document_id": "a", "score": 1.0}]},
                       "query-ids": ["q"], "corpus-ids": ["a"]}
            cmd = [sys.executable, str(Path(__file__).with_name("retrieval_metrics.py"))]
            for name, data in fixture.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(data), encoding="utf-8")
                cmd += [f"--{name}", str(path)]
            output = root / "metrics.json"
            cmd += ["--output", str(output)]
            completed = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertEqual(load_json(output)["metrics"], {"nDCG@10": 1.0, "Recall@10": 1.0})
            saved = output.read_bytes()
            repeated = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(repeated.returncode, 2)
            self.assertIn("Retrieval metrics error:", repeated.stderr)
            self.assertEqual(output.read_bytes(), saved)
            output.unlink()
            invalid = subprocess.run(cmd + ["--k", "0"], capture_output=True, text=True)
            self.assertEqual(invalid.returncode, 2)
            self.assertIn("k must be a positive integer", invalid.stderr)
            self.assertFalse(output.exists())
            (root / "qrels.json").unlink()
            missing = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("Retrieval metrics error:", missing.stderr)
            self.assertIn("qrels.json", missing.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
