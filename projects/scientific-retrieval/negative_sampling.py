# SPDX-License-Identifier: Apache-2.0
"""ID-safe adapters and independent checks around the upstream negative miner."""
from __future__ import annotations

import math

import numpy as np

from data_split import negative_exclusions
from retrieval_metrics import checked_ids, require


def require_unprompted_model(prompts, default_prompt_name):
    """Allow empty named prompts without changing the loaded model configuration."""
    require(default_prompt_name is None, "Unexpected default prompt name")
    require(isinstance(prompts, dict) and
            all(isinstance(key, str) and isinstance(value, str) and value == ""
                for key, value in prompts.items()),
            "Expected a prompt dictionary containing only empty strings")


def mining_inputs(documents, queries, positives, exclusions):
    """Keep original qrels separate from the larger mining-only exclusion set."""
    corpus_ids = checked_ids(list(documents), "Corpus IDs")
    query_ids = checked_ids(list(queries), "Training query IDs")
    require(set(positives) == set(exclusions) == query_ids,
            "Positives and exclusions must exactly match the training whitelist")
    require(all(isinstance(text, str) and text.strip() for text in documents.values()),
            "Corpus texts must be nonempty strings")
    require(len(set(documents.values())) == len(documents),
            "Formatted corpus text is not unique; the text-based miner cannot safely recover document IDs")
    rows = []
    for qid in sorted(queries):
        require(isinstance(queries[qid], str) and queries[qid].strip(), "Training query text is empty")
        known = checked_ids(positives[qid], f"Known positives for {qid}")
        excluded = checked_ids(exclusions[qid], f"Negative exclusions for {qid}")
        require(known <= excluded <= corpus_ids, f"Invalid positive/exclusion document IDs for {qid}")
        rows.append({"query_id": qid, "text": queries[qid], "positive_document_ids": sorted(known)})
    require(exclusions == negative_exclusions(rows),
            "Exclusions differ from the TRAIN-only exact-normalization positive union")
    anchor_map = {}
    for qid in sorted(queries):
        anchor_map.setdefault(queries[qid], []).append(qid)
    pairs = []
    for anchor, qids in anchor_map.items():
        reference = exclusions[qids[0]]
        require(all(exclusions[qid] == reference for qid in qids),
                "Identical query text has inconsistent exclusions")
        for did in reference:
            pairs.append({"anchor": anchor, "positive": documents[did]})
    return pairs, anchor_map


def parse_mined_rows(rows, documents, anchor_map, exclusions):
    """Recover IDs, reject dropped queries/positives, and require one negative per anchor."""
    by_text = {text: did for did, text in documents.items()}
    expected = {(anchor, did) for anchor, qids in anchor_map.items() for did in exclusions[qids[0]]}
    seen, selections, positive_scores = set(), {}, {}
    for row in rows:
        require(set(row) == {"anchor", "positive", "negative", "scores"}, "Unexpected miner output columns")
        anchor = row["anchor"]
        require(anchor in anchor_map, "Miner returned a query outside the training whitelist")
        require(row["positive"] in by_text and row["negative"] in by_text,
                "Miner returned a document outside the pinned corpus")
        positive, negative = by_text[row["positive"]], by_text[row["negative"]]
        key = (anchor, positive)
        require(key in expected and key not in seen, "Unexpected or duplicate miner anchor-positive pair")
        seen.add(key)
        values = row["scores"]
        require(isinstance(values, list) and len(values) == 2 and
                all(type(x) in (int, float) and math.isfinite(x) for x in values),
                "Miner must return finite [positive_score, negative_score]")
        for qid in anchor_map[anchor]:
            require(negative not in exclusions[qid], f"Known/exact-duplicate positive selected as negative for {qid}")
        if anchor in selections:
            require(selections[anchor]["document_id"] == negative and
                    abs(selections[anchor]["score"] - values[1]) <= 2e-6,
                    "Miner returned different negatives across positives of the same query")
        selections[anchor] = {"document_id": negative, "score": float(values[1])}
        positive_scores.setdefault(anchor, {})[positive] = float(values[0])
    require(seen == expected, "Miner dropped training anchors or their exclusion positives; inspect coverage.json")
    return selections, positive_scores


def candidate_pool(document_ids, scores, excluded_ids, positive_scores, margin):
    """Mirror the pinned float32 absolute-margin comparison on independently computed scores."""
    ids = checked_ids(document_ids, "Candidate document IDs")
    excluded = checked_ids(excluded_ids, "Excluded document IDs")
    require(excluded <= ids, "Excluded document is absent from the corpus")
    require(set(positive_scores) == excluded, "Positive scores must cover every exclusion document")
    require(isinstance(scores, np.ndarray) and scores.dtype == np.float32 and
            scores.shape == (len(document_ids),) and bool(np.isfinite(scores).all()),
            "Candidate scores must be a finite float32 vector aligned with corpus IDs")
    require(type(margin) in (int, float) and math.isfinite(margin) and margin >= 0,
            "Mining margin must be finite and nonnegative")
    require(all(math.isfinite(float(x)) for x in positive_scores.values()), "Positive scores must be finite")
    weakest = np.float32(min(positive_scores.values()))
    known = np.array([did in excluded for did in document_ids], dtype=bool)
    # Upstream uses score + absolute_margin <= min(positive_scores), in float32.
    too_close = scores + np.float32(margin) > weakest
    eligible = ~known & ~too_close
    return eligible, {
        "corpus_documents": len(document_ids), "known_or_duplicate_positives_excluded": int(known.sum()),
        "score_gap_excluded_among_other_documents": int((~known & too_close).sum()),
        "eligible_documents": int(eligible.sum()), "weakest_exclusion_positive_score": float(weakest),
    }


def check_selection(document_ids, scores, eligible, selected_id, reported_score, *, hard):
    require(bool(eligible.any()), "No eligible negative documents; no query may be silently dropped")
    require(selected_id in document_ids, "Selected negative document ID is absent")
    index = document_ids.index(selected_id)
    require(bool(eligible[index]), "Selected negative violates known-positive or score-gap exclusions")
    require(math.isfinite(reported_score) and abs(float(scores[index]) - reported_score) <= 2e-6,
            "Reported mining score differs from the independent cache-score check")
    best = float(scores[eligible].max())
    if hard:
        require(abs(float(scores[index]) - best) <= 2e-6, "Hard negative is not a highest-scoring eligible candidate")
    order = np.lexsort((np.asarray(document_ids), -scores))
    eligible_order = [i for i in order if eligible[i]]
    return {"negative_document_id": selected_id, "negative_score": float(scores[index]),
            "full_corpus_rank": int(np.flatnonzero(order == index)[0]) + 1,
            "eligible_rank": eligible_order.index(index) + 1,
            "best_eligible_score": best,
            "selected_score_tie_count": int((eligible & (scores == scores[index])).sum())}


def training_triplets(positives, negatives):
    """Expand ORIGINAL qrels only, so mining exclusions never become extra training labels."""
    require(set(negatives) == set(positives), "Negative IDs must cover exactly the training queries")
    rows = []
    for qid in sorted(positives):
        checked_ids(positives[qid], f"Training positives for {qid}")
        require(negatives[qid] not in positives[qid], "Training negative is a known positive")
        for did in sorted(positives[qid]):
            rows.append({"query_id": qid, "positive_document_id": did, "negative_document_id": negatives[qid]})
    return rows
