# SPDX-License-Identifier: Apache-2.0
"""Known-risk tests, using synthetic claims and document IDs."""

import copy
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from data_split import negative_exclusions, split_data, validate_grouping


def fixture():
    rows = []
    groups = []
    for i in range(10):
        qid, gid = str(i), f"group-{i}"
        rows.append({"query_id": qid, "text": f"distinct claim {i}",
                     "positive_document_ids": [f"doc-{i}"], "source_contexts": [],
                     "provisional_group": gid})
        groups.append({"group_id": gid, "query_ids": [qid]})
    rows.append({"query_id": "extra", "text": " DISTINCT   CLAIM 0 ",
                 "positive_document_ids": ["additional-positive"], "source_contexts": [],
                 "provisional_group": "group-0"})
    groups[0]["query_ids"].append("extra")
    return rows, groups, [row["query_id"] for row in rows]


class SplitTests(unittest.TestCase):
    def test_whole_groups_complete_coverage_and_order_invariance(self):
        rows, groups, ids = fixture()
        result, _ = split_data(rows, groups, ids)
        shuffled = copy.deepcopy(groups[::-1])
        for group in shuffled:
            group["query_ids"].reverse()
        reordered, _ = split_data(rows[::-1], shuffled, ids[::-1])
        self.assertEqual(result, reordered)
        train, val = map(set, (result["query_ids"]["train"], result["query_ids"]["validation"]))
        self.assertEqual(train | val, set(ids))
        self.assertFalse(train & val)
        self.assertEqual(result["counts"]["validation"]["groups"], 2)
        self.assertEqual("0" in val, "extra" in val)
        self.assertTrue(all(value == 0 for value in result["cross_split_overlap"].values()))

    def test_duplicate_missing_and_outside_whitelist_ids_rejected(self):
        rows, groups, ids = fixture()
        for changed in (rows + [rows[0]], rows[:-1], rows + [{**rows[0], "query_id": "outside"}]):
            with self.subTest(ids=[r["query_id"] for r in changed]):
                with self.assertRaises(ValueError):
                    validate_grouping(changed, groups, ids)

    def test_omitted_and_multiply_assigned_group_members_rejected(self):
        rows, groups, ids = fixture()
        missing = copy.deepcopy(groups)
        missing[0]["query_ids"].remove("extra")
        duplicate = copy.deepcopy(groups)
        duplicate[1]["query_ids"].append("extra")
        for changed in (missing, duplicate):
            with self.subTest(groups=changed), self.assertRaises(ValueError):
                validate_grouping(rows, changed, ids)

    def test_normalized_duplicate_crossing_groups_rejected(self):
        rows, groups, ids = fixture()
        rows[-1]["provisional_group"] = "new-group"
        groups[0]["query_ids"].remove("extra")
        groups.append({"group_id": "new-group", "query_ids": ["extra"]})
        with self.assertRaisesRegex(ValueError, "normalized_text crosses groups"):
            validate_grouping(rows, groups, ids)

    def test_shared_source_context_crossing_groups_rejected(self):
        rows, groups, ids = fixture()
        rows[1]["source_contexts"] = rows[2]["source_contexts"] = ["same-source"]
        with self.assertRaisesRegex(ValueError, "source_context crosses groups"):
            validate_grouping(rows, groups, ids)

    def test_second_known_positive_crossing_groups_rejected(self):
        rows, groups, ids = fixture()
        rows[1]["positive_document_ids"].append("doc-2")
        with self.assertRaisesRegex(ValueError, "known_positive crosses groups"):
            validate_grouping(rows, groups, ids)

    def test_positive_union_for_exact_duplicates_without_relabeling(self):
        rows, _, _ = fixture()
        rows[0]["positive_document_ids"].append("another-positive")
        original = copy.deepcopy(rows)
        union = negative_exclusions(rows)
        self.assertEqual(union["0"], ["additional-positive", "another-positive", "doc-0"])
        self.assertEqual(union["0"], union["extra"])
        self.assertEqual(union["1"], ["doc-1"])
        self.assertEqual(rows, original)

    def test_exclusions_only_export_training_ids(self):
        rows, groups, ids = fixture()
        result, exclusions = split_data(rows, groups, ids)
        self.assertEqual(set(exclusions), set(result["query_ids"]["train"]))
        self.assertFalse(set(exclusions) & set(result["query_ids"]["validation"]))

    def test_missing_input_cli_fails_clearly(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            proc = subprocess.run([sys.executable, str(Path(__file__).with_name("data_split.py")),
                                   "--audit-dir", str(path / "missing"),
                                   "--train-ids", str(path / "missing.json"),
                                   "--output-dir", str(path / "output")], capture_output=True, text=True)
            self.assertEqual(proc.returncode, 2)
            self.assertIn("Data split error:", proc.stderr)
            self.assertIn("SHA256SUMS", proc.stderr)
            self.assertNotIn("Traceback", proc.stderr)
            self.assertFalse((path / "output").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
