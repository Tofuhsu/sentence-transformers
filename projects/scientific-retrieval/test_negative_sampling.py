# SPDX-License-Identifier: Apache-2.0
"""Tests for the failure modes that could invalidate the B/C comparison."""
from __future__ import annotations

import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from negative_sampling import (
    candidate_pool, check_selection, mining_inputs, parse_mined_rows,
    require_unprompted_model, training_triplets,
)


class NegativeSamplingTests(unittest.TestCase):
    def setUp(self):
        self.documents = {"d1": "First positive", "d2": "Second positive", "d3": "Candidate", "d4": "Other"}
        self.queries = {"q1": "A claim", "q2": "a CLAIM"}
        self.positives = {"q1": ["d1"], "q2": ["d2"]}
        self.exclusions = {"q1": ["d1", "d2"], "q2": ["d1", "d2"]}

    def test_empty_named_prompts_are_accepted_without_mutation(self):
        for prompts in ({}, {"query": "", "document": ""}):
            with self.subTest(prompts=prompts):
                before = copy.deepcopy(prompts)
                require_unprompted_model(prompts, None)
                self.assertEqual(prompts, before)

    def test_actual_prompts_invalid_values_and_default_names_are_rejected(self):
        for prompts, default in [
            ({"query": "query: "}, None), ({"document": " "}, None),
            ({"query": None}, None), ({"query": False}, None),
            ({1: ""}, None), ([], None), (None, None),
            ({"query": ""}, "query"), ({}, ""),
        ]:
            with self.subTest(prompts=prompts, default=default):
                with self.assertRaises(ValueError):
                    require_unprompted_model(prompts, default)

    def test_exact_duplicate_union_is_exclusion_only_not_new_training_labels(self):
        original = copy.deepcopy(self.positives)
        pairs, anchors = mining_inputs(self.documents, self.queries, self.positives, self.exclusions)
        self.assertEqual(len(pairs), 4)
        rows = training_triplets(self.positives, {"q1": "d3", "q2": "d3"})
        self.assertEqual([(r["query_id"], r["positive_document_id"]) for r in rows], [("q1", "d1"), ("q2", "d2")])
        self.assertEqual(self.positives, original)
        self.assertEqual(len(anchors), 2)

    def test_invalid_whitelist_exclusions_and_ambiguous_document_text_fail(self):
        for docs, queries, pos, exclusions in [
            (self.documents, self.queries, self.positives, self.positives),
            (self.documents, {"q1": "A claim"}, self.positives, self.exclusions),
            ({**self.documents, "d4": "Candidate"}, self.queries, self.positives, self.exclusions),
            (self.documents, self.queries, {"q1": ["missing"], "q2": ["d2"]}, self.exclusions),
        ]:
            with self.subTest(docs=docs, queries=queries, pos=pos):
                with self.assertRaises(ValueError): mining_inputs(docs, queries, pos, exclusions)

    def test_miner_output_recovers_duplicate_query_ids_and_all_positives(self):
        queries = {"q1": "Same claim", "q2": "Same claim"}
        _, anchors = mining_inputs(self.documents, queries, self.positives, self.exclusions)
        rows = [{"anchor": "Same claim", "positive": self.documents[did], "negative": "Candidate", "scores": [.8, .3]}
                for did in ["d1", "d2"]]
        choices, scores = parse_mined_rows(rows, self.documents, anchors, self.exclusions)
        self.assertEqual(anchors["Same claim"], ["q1", "q2"])
        self.assertEqual(choices["Same claim"]["document_id"], "d3")
        self.assertEqual(set(scores["Same claim"]), {"d1", "d2"})
        for malformed in [rows[:1], rows + rows[:1], [{**rows[0], "negative": "Second positive"}, rows[1]],
                          [rows[0], {**rows[1], "negative": "Other"}]]:
            with self.subTest(rows=malformed):
                with self.assertRaises(ValueError): parse_mined_rows(malformed, self.documents, anchors, self.exclusions)

    def test_weakest_positive_sets_gap_and_all_known_positives_are_excluded(self):
        ids = ["d1", "d2", "d3", "d4"]
        scores = np.array([.9, .5, .6, .4], dtype=np.float32)
        mask, stats = candidate_pool(ids, scores, ["d1", "d2"], {"d1": .9, "d2": .5}, .05)
        self.assertEqual(mask.tolist(), [False, False, False, True])
        self.assertEqual(stats["eligible_documents"], 1)
        self.assertEqual(stats["score_gap_excluded_among_other_documents"], 1)
        with self.assertRaisesRegex(ValueError, "exclusions"):
            check_selection(ids, scores, mask, "d2", .5, hard=False)

    def test_gap_boundary_and_empty_pool_are_explicit(self):
        scores = np.array([.5, .25, .250001], dtype=np.float32)
        mask, _ = candidate_pool(["p", "equal", "closer"], scores, ["p"], {"p": .5}, .25)
        self.assertEqual(mask.tolist(), [False, True, False])
        mask, _ = candidate_pool(["p", "equal", "closer"], scores, ["p"], {"p": .5}, 1.)
        with self.assertRaisesRegex(ValueError, "No eligible"):
            check_selection(["p", "equal", "closer"], scores, mask, "equal", .25, hard=True)

    def test_hard_is_top_random_can_be_easier_and_ties_are_recorded(self):
        ids = ["p", "b", "a", "easy"]
        scores = np.array([.9, .5, .5, .1], dtype=np.float32)
        mask, _ = candidate_pool(ids, scores, ["p"], {"p": .9}, .05)
        result = check_selection(ids, scores, mask, "b", .5, hard=True)
        self.assertEqual(result["selected_score_tie_count"], 2)
        self.assertEqual(result["full_corpus_rank"], 3)
        check_selection(ids, scores, mask, "easy", .1, hard=False)
        with self.assertRaisesRegex(ValueError, "highest-scoring"):
            check_selection(ids, scores, mask, "easy", .1, hard=True)
        with self.assertRaisesRegex(ValueError, "score differs"):
            check_selection(ids, scores, mask, "b", .2, hard=True)

    def test_multiple_original_positives_and_shared_documents_keep_equal_pair_budgets(self):
        pos = {"q1": ["d1", "d2"], "q2": ["d1"]}
        b = training_triplets(pos, {"q1": "d3", "q2": "d4"})
        c = training_triplets(pos, {"q1": "d4", "q2": "d3"})
        self.assertEqual(len(b), 3)
        self.assertEqual([(r["query_id"], r["positive_document_id"]) for r in b],
                         [(r["query_id"], r["positive_document_id"]) for r in c])
        with self.assertRaises(ValueError): training_triplets(pos, {"q1": "d3"})

    def test_nonfinite_scores_and_missing_cli_input_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "finite float32"):
            candidate_pool(["p", "n"], np.array([.8, np.nan], dtype=np.float32), ["p"], {"p": .8}, .05)
        result = subprocess.run([sys.executable, str(Path(__file__).with_name("prepare_negatives.py"))],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("required", result.stderr)
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, str(Path(__file__).with_name("prepare_negatives.py"))]
            for name in ("split-lock", "text-policy", "negative-policy", "data-audit", "exclusions", "token-audit",
                         "cache-dir", "output-dir", "artifact-dir"):
                command += ["--"+name, str(Path(directory)/name)]
            environment = {**os.environ, "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
                           "PYTORCH_ENABLE_MPS_FALLBACK":"0", "PYTORCH_MPS_FAST_MATH":"0"}
            missing = subprocess.run(command, capture_output=True, text=True, env=environment)
            self.assertEqual(missing.returncode, 2)
            self.assertIn("Negative preparation error:", missing.stderr)
            self.assertIn("split-lock", missing.stderr)
            self.assertNotIn("Traceback", missing.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
