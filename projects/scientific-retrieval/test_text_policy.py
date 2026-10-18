# SPDX-License-Identifier: Apache-2.0
import unittest

from text_policy import abstract_record, document_text, query_text, summarize, token_record


class TextPolicyTests(unittest.TestCase):
    def test_format_preserves_internal_whitespace_case_and_unicode(self):
        self.assertEqual(document_text(" Title α ", " First  claim.\nSecond claim. "), "Title α\nFirst  claim.\nSecond claim.")
        self.assertEqual(document_text("  ", " Abstract. "), "Abstract.")
        self.assertEqual(query_text("  RUNX  is not expressed.  "), "RUNX  is not expressed.")

    def test_invalid_or_empty_text_fails_clearly(self):
        for title, abstract in ((None, "text"), ("title", None), ("title", "  ")):
            with self.subTest(title=title, abstract=abstract), self.assertRaises(ValueError):
                document_text(title, abstract)
        for text in (None, "", " \n "):
            with self.subTest(text=text), self.assertRaises(ValueError):
                query_text(text)

    def test_256_boundary_counts_special_tokens(self):
        exact = [101] + [7] * 254 + [102]
        over = [101] + [7] * 255 + [102]
        self.assertFalse(token_record(exact, exact, 101, 102)["truncated"])
        record = token_record(over, exact, 101, 102)
        self.assertEqual(record, {"input_tokens": 257, "used_tokens": 256, "removed_tokens": 1, "truncated": True})

    def test_actual_token_or_special_token_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "differs"):
            token_record([101, 7, 102], [101, 8, 102], 101, 102)
        with self.assertRaisesRegex(ValueError, "special-token"):
            token_record([7, 102], [7, 102], 101, 102)

    def test_title_budget_and_abstract_boundary_are_accounted_for(self):
        title, abstract = [3] * 3, [4] * 253
        record = abstract_record(title, abstract, [101] + title + abstract + [102], 101, 102)
        self.assertEqual(record["abstract_tokens_retained"], 251)
        self.assertTrue(record["abstract_truncated"])
        long_title = [3] * 260
        self.assertEqual(abstract_record(long_title, [4], [101] + long_title + [4, 102], 101, 102)["abstract_tokens_retained"], 0)
        with self.assertRaisesRegex(ValueError, "joined input"):
            abstract_record([3], [4], [101, 3, 5, 102], 101, 102)

    def test_summary_uses_item_denominator_and_known_percentiles(self):
        records = [token_record([101] + [7] * n + [102], [101] + [7] * min(n, 254) + [102], 101, 102)
                   for n in (1, 2, 3, 254, 255)]
        result = summarize(records)
        self.assertEqual(result["items"], 5)
        self.assertEqual(result["truncated_items"], 1)
        self.assertEqual(result["truncated_fraction"], 0.2)
        self.assertEqual(result["tokens_removed"], 1)
        self.assertEqual(result["input_length_percentiles"], {"p50": 5, "p90": 257, "p95": 257, "p99": 257})


if __name__ == "__main__":
    unittest.main(verbosity=2)
