# SPDX-License-Identifier: Apache-2.0
"""Exact cosine search and wall-clock timing for the controlled experiment."""

from __future__ import annotations

import math
import time

import numpy as np

from retrieval_metrics import checked_ids, require


def validate_embeddings(values, rows, dimension):
    require(isinstance(values, np.ndarray) and values.dtype == np.float32,
            "Embeddings must be a float32 NumPy array")
    require(values.shape == (rows, dimension) and rows > 0 and dimension > 0,
            "Embedding dimensions do not match the expected input rows")
    require(bool(np.isfinite(values).all()), "Embeddings contain nonfinite values")
    norms = np.linalg.norm(values, axis=1)
    require(bool(np.allclose(norms, 1.0, rtol=0, atol=5e-4)),
            "Cosine search requires normalized, nonzero embeddings")


class ExactCosineIndex:
    """CPU float32 dot products, followed by a stable, documented full ranking."""

    def __init__(self, embeddings, document_ids):
        checked_ids(document_ids, "Index document IDs")
        require(isinstance(embeddings, np.ndarray) and embeddings.ndim == 2,
                "Corpus embeddings must have two dimensions")
        validate_embeddings(embeddings, len(document_ids), embeddings.shape[1])
        self.embeddings = np.ascontiguousarray(embeddings)
        self.document_ids = list(document_ids)
        self.tie_keys = np.asarray(document_ids)

    def search(self, query, k=10):
        require(type(k) is int and 1 <= k <= len(self.document_ids),
                "Search k must be between 1 and the corpus size")
        require(isinstance(query, np.ndarray) and query.ndim == 1,
                "Query embedding must have one dimension")
        validate_embeddings(query[None, :], 1, self.embeddings.shape[1])
        scores = self.embeddings @ query
        require(bool(np.isfinite(scores).all()), "Similarity scores contain nonfinite values")
        # Resolve ties over the FULL corpus before the top-k cutoff.
        order = np.lexsort((self.tie_keys, -scores))
        top = [{"document_id": self.document_ids[i], "score": float(scores[i])}
               for i in order[:k]]
        return top, scores, order


def timed_call(function, *, synchronize=None, clock=time.perf_counter):
    if synchronize is not None:
        synchronize()
    start = clock()
    value = function()
    if synchronize is not None:
        synchronize()
    duration = clock() - start
    require(math.isfinite(duration) and duration >= 0, "Invalid elapsed time")
    return value, duration


def latency_summary(milliseconds):
    require(bool(milliseconds) and all(math.isfinite(x) and x >= 0 for x in milliseconds),
            "Latency samples must be nonempty, finite and nonnegative")
    ordered = sorted(milliseconds)
    count = len(ordered)
    return {
        "samples": count, "unit": "ms", "mean": math.fsum(ordered) / count,
        "min": ordered[0], "max": ordered[-1],
        "p50": ordered[math.ceil(0.50 * count) - 1],
        "p95": ordered[math.ceil(0.95 * count) - 1],
        "percentile_method": "nearest rank",
    }
