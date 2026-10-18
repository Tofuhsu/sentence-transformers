# SPDX-License-Identifier: Apache-2.0
"""Binary retrieval metrics with explicit query coverage and deterministic ties."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_ids(values, label, *, allow_empty=False):
    require(isinstance(values, list), f"{label} must be a list")
    require(all(isinstance(x, str) and x.strip() for x in values),
            f"{label} must contain nonempty string IDs")
    require(allow_empty or bool(values), f"{label} must not be empty")
    require(len(values) == len(set(values)), f"{label} contains duplicate IDs")
    return set(values)


def score_query(relevance, ranking, k=10):
    """Score an already ordered, unique document-ID list against binary qrels.

    Missing judgments contribute zero gain; this is not a nonrelevance label.
    A query with no known positives is rejected for this SciFact experiment.
    """
    require(type(k) is int and k > 0, "k must be a positive integer")
    require(isinstance(relevance, dict), "Query relevance must be a document-to-label object")
    checked_ids(list(relevance), "Qrel document IDs")
    require(all(type(label) is int and label in (0, 1) for label in relevance.values()),
            "Only integer binary relevance labels 0 and 1 are supported")
    positives = {did for did, label in relevance.items() if label == 1}
    require(bool(positives), "Query has no known positive documents")
    checked_ids(ranking, "Ranked document IDs", allow_empty=True)
    top = ranking[:k]
    hits = sum(did in positives for did in top)
    dcg = math.fsum(1.0 / math.log2(rank + 1)
                    for rank, did in enumerate(top, 1) if did in positives)
    idcg = math.fsum(1.0 / math.log2(rank + 1)
                     for rank in range(1, min(k, len(positives)) + 1))
    return {"ndcg": dcg / idcg, "recall": hits / len(positives),
            "relevant_retrieved": hits, "known_positives": len(positives),
            "retrieved_at_k": len(top)}


def rank_documents(rows, corpus_ids):
    require(isinstance(rows, list), "Query results must be a list")
    checked = []
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == {"document_id", "score"},
                f"Result item {index} must contain document_id and score only")
        did, score = row["document_id"], row["score"]
        require(isinstance(did, str) and did in corpus_ids,
                f"Result item {index} references an unknown corpus document")
        require(type(score) in (int, float) and math.isfinite(score),
                f"Result item {index} score must be a finite number")
        checked.append({"document_id": did, "score": float(score)})
    checked_ids([row["document_id"] for row in checked], "Result document IDs", allow_empty=True)
    return sorted(checked, key=lambda row: (-row["score"], row["document_id"]))


def evaluate(qrels, run, query_ids, corpus_ids, k=10):
    require(type(k) is int and k > 0, "k must be a positive integer")
    queries = checked_ids(query_ids, "Evaluation query IDs")
    corpus = checked_ids(corpus_ids, "Corpus document IDs")
    for name, value in (("Qrels", qrels), ("Run", run)):
        require(isinstance(value, dict), f"{name} must be an object keyed by query ID")
        require(set(value) == queries, f"{name} query IDs must exactly match evaluation query IDs")
    per_query = []
    for qid in sorted(queries):
        require(isinstance(qrels[qid], dict), f"Qrels for {qid} must be an object")
        require(set(qrels[qid]) <= corpus, f"Qrels for {qid} reference an unknown corpus document")
        ranked = rank_documents(run[qid], corpus)
        values = score_query(qrels[qid], [row["document_id"] for row in ranked], k)
        per_query.append({
            "query_id": qid, f"nDCG@{k}": values.pop("ndcg"),
            f"Recall@{k}": values.pop("recall"), **values,
            "top_documents": ranked[:k],
        })
    metrics = {name: math.fsum(row[name] for row in per_query) / len(per_query)
               for name in (f"nDCG@{k}", f"Recall@{k}")}
    return {
        "schema_version": 1, "cutoff": k, "query_count": len(per_query),
        "corpus_document_count": len(corpus), "metrics": metrics, "per_query": per_query,
        "policy": {
            "relevance": "binary labels 0/1; unjudged documents have zero gain",
            "dcg": "sum(relevance_at_rank / log2(rank + 1)) for ranks 1..k",
            "idcg": "ideal DCG from all known positives, limited to k",
            "recall_denominator": "all known positive documents for the query",
            "aggregation": "unweighted mean over exactly the supplied query whitelist",
            "ranking": "score descending; document ID string ascending on exact ties",
            "empty_result_list": "zero nDCG and Recall; query remains in denominator",
            "missing_query_result": "error",
            "no_known_positive_query": "error for this experiment",
            "query_document_id_equality": "retain; the IDs use separate namespaces",
        },
    }


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--query-ids", type=Path, required=True)
    parser.add_argument("--corpus-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    try:
        result = evaluate(load_json(args.qrels), load_json(args.run), load_json(args.query_ids),
                          load_json(args.corpus_ids), args.k)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (OSError, ValueError, TypeError, OverflowError) as exc:
        parser.exit(2, f"Retrieval metrics error: {exc}\n")
    print(json.dumps({"output": str(args.output), "query_count": result["query_count"],
                      "metrics": result["metrics"]}, indent=2))


if __name__ == "__main__":
    main()
