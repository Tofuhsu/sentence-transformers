# SPDX-License-Identifier: Apache-2.0
"""Check synthetic known answers and independently compare with scikit-learn."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import sys

from retrieval_metrics import evaluate, require


def write_json(path, value):
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def verify(output):
    import numpy as np
    from sklearn.metrics import ndcg_score, recall_score

    require(not output.exists(), "Output directory exists; use a fresh directory")
    output.mkdir(parents=True)
    corpus = [f"d{i:02d}" for i in range(1, 13)]
    # Values are from the formulas documented in METRICS.md, not our evaluator.
    cases = [
        ("perfect", ["d01"], 1.0, 1.0),
        ("rank_two", ["d02"], 0.6309297535714575, 1.0),
        ("two_positives_at_one_and_three", ["d01", "d03"], 0.9197207891481876, 1.0),
        ("one_of_two_positives_in_top_ten", ["d01", "d12"], 0.6131471927654584, 0.5),
        ("no_hit_in_top_ten", ["d11", "d12"], 0.0, 0.0),
        ("positive_at_rank_ten", ["d10"], 0.2890648263178879, 1.0),
        ("twelve_positives", corpus, 1.0, 0.8333333333333334),
    ]
    qrels = {qid: dict.fromkeys(positives, 1) for qid, positives, _, _ in cases}
    run = {qid: [{"document_id": did, "score": float(len(corpus) - index)}
                 for index, did in enumerate(corpus)] for qid in qrels}
    query_ids = sorted(qrels)
    for name, value in (("qrels", qrels), ("run", run), ("query-ids", query_ids), ("corpus-ids", corpus)):
        write_json(output / f"fixture-{name}.json", value)
    actual = evaluate(qrels, run, query_ids, corpus, k=10)
    by_query = {row["query_id"]: row for row in actual["per_query"]}
    checked = []
    for qid, positives, expected_ndcg, expected_recall in cases:
        truth = np.array([int(did in positives) for did in corpus])
        scores = np.arange(len(corpus), 0, -1, dtype=float)
        # Scores are deliberately distinct. Raw tied scores use a different
        # averaging convention in sklearn, which is not our tie policy.
        reference_ndcg = float(ndcg_score(truth[None, :], scores[None, :], k=10))
        selected = np.array([int(index < 10) for index in range(len(corpus))])
        reference_recall = float(recall_score(truth, selected, average="binary", zero_division=0))
        measured = by_query[qid]
        for name, expected, reference in (("nDCG@10", expected_ndcg, reference_ndcg),
                                           ("Recall@10", expected_recall, reference_recall)):
            require(math.isclose(measured[name], expected, rel_tol=0, abs_tol=1e-12),
                    f"{qid}: {name} differs from the known answer")
            require(math.isclose(measured[name], reference, rel_tol=0, abs_tol=1e-12),
                    f"{qid}: {name} differs from scikit-learn")
        checked.append({"case": qid, "known_positive_ids": positives,
                        "expected_ndcg_at_10": expected_ndcg, "expected_recall_at_10": expected_recall,
                        "actual_ndcg_at_10": measured["nDCG@10"], "actual_recall_at_10": measured["Recall@10"],
                        "sklearn_ndcg_at_10": reference_ndcg, "sklearn_recall_at_10": reference_recall})
    summary = {
        "status": "passed", "scope": "synthetic metric correctness only; not model retrieval quality",
        "known_answer_cases": len(checked), "ndcg_reference_comparisons": len(checked),
        "recall_reference_comparisons": len(checked), "absolute_tolerance": 1e-12,
        "reference_tie_scope": "distinct scores only; deterministic ties tested separately",
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")},
        "model_loaded": False, "scifact_queries_or_labels_read": False,
        "cases": checked,
    }
    write_json(output / "metric-verification.json", summary)
    print(json.dumps({key: value for key, value in summary.items() if key != "cases"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        verify(args.output_dir)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        parser.exit(2, f"Metric verification error: {exc}\n")


if __name__ == "__main__":
    main()
