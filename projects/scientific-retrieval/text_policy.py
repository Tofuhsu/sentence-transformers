# SPDX-License-Identifier: Apache-2.0
"""Shared input formatting and BERT token accounting for the first experiment."""

from __future__ import annotations

import math

MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
MAX_LENGTH = 256
POLICY = {
    "schema_version": 1,
    "model_id": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "document_format": "title.strip() + newline + abstract.strip(); omit empty title",
    "query_format": "text.strip()",
    "manual_case_or_unicode_normalization": False,
    "prompt": "",
    "chunking": False,
    "max_sequence_tokens_including_special_tokens": MAX_LENGTH,
    "truncation_side": "right",
    "applies_to": ["original_model", "ordinary_negative_finetuning", "hard_negative_finetuning"],
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def query_text(text):
    require(isinstance(text, str) and bool(text.strip()), "Query text must be a nonempty string")
    return text.strip()


def document_text(title, abstract):
    require(isinstance(title, str), "Document title must be a string")
    require(isinstance(abstract, str) and bool(abstract.strip()), "Abstract must be a nonempty string")
    return f"{title.strip()}\n{abstract.strip()}" if title.strip() else abstract.strip()


def token_record(full_ids, used_ids, cls_id, sep_id):
    """Check actual ST output against single-input BERT right truncation."""
    require(len(full_ids) >= 2 and full_ids[0] == cls_id and full_ids[-1] == sep_id,
            "Unexpected BERT special-token layout")
    expected = full_ids if len(full_ids) <= MAX_LENGTH else full_ids[: MAX_LENGTH - 1] + [sep_id]
    require(used_ids == expected, "SentenceTransformer tokenization differs from the recorded text policy")
    return {
        "input_tokens": len(full_ids),
        "used_tokens": len(used_ids),
        "removed_tokens": len(full_ids) - len(used_ids),
        "truncated": len(full_ids) > MAX_LENGTH,
    }


def abstract_record(title_ids, abstract_ids, full_ids, cls_id, sep_id):
    # Validate the title/body boundary instead of assuming separately tokenized
    # strings concatenate identically for every tokenizer.
    require(full_ids == [cls_id] + title_ids + abstract_ids + [sep_id],
            "Separate title/abstract tokenization differs from the joined input")
    retained = min(len(abstract_ids), max(0, MAX_LENGTH - 2 - len(title_ids)))
    return {
        "title_tokens": len(title_ids),
        "abstract_tokens": len(abstract_ids),
        "abstract_tokens_retained": retained,
        "abstract_truncated": retained < len(abstract_ids),
    }


def summarize(records):
    require(bool(records), "Cannot summarize an empty collection")
    lengths = sorted(row["input_tokens"] for row in records)
    n = len(lengths)
    truncated = sum(row["truncated"] for row in records)
    result = {
        "items": n,
        "truncated_items": truncated,
        "truncated_fraction": truncated / n,
        "truncated_percent": 100.0 * truncated / n,
        "tokens_before_truncation": sum(lengths),
        "tokens_retained": sum(row["used_tokens"] for row in records),
        "tokens_removed": sum(row["removed_tokens"] for row in records),
        "mean_input_tokens": sum(lengths) / n,
        "max_input_tokens": max(lengths),
        "percentile_method": "nearest rank",
        "input_length_percentiles": {f"p{p}": lengths[math.ceil(p / 100 * n) - 1] for p in (50, 90, 95, 99)},
    }
    if "abstract_truncated" in records[0]:
        result.update({
            "abstracts_truncated": sum(row["abstract_truncated"] for row in records),
            "abstract_truncation_fraction": sum(row["abstract_truncated"] for row in records) / n,
            "abstract_tokens_total": sum(row["abstract_tokens"] for row in records),
            "abstract_tokens_retained": sum(row["abstract_tokens_retained"] for row in records),
            "abstracts_with_no_tokens_retained": sum(row["abstract_tokens"] > 0 and row["abstract_tokens_retained"] == 0 for row in records),
        })
    return result
