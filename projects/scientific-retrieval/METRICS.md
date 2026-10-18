# Retrieval evaluation contract

Personal project extension to Sentence Transformers. The implementation is
limited to the binary relevance labels used in this SciFact experiment.
This stage validates metric calculations; it does not measure model quality.

## Definitions

For one query, let R be its complete set of known positive document IDs.
Ranks start at 1. At cutoff k:

- DCG@k = sum of rel(rank) / log2(rank + 1) over the first k returned documents.
- IDCG@k = sum of 1 / log2(rank + 1) from rank 1 through min(k, |R|).
- nDCG@k = DCG@k / IDCG@k.
- Recall@k = number of distinct known positives retrieved in the first k / |R|.

Relevance is 0 or 1. For these labels, gain = rel and gain = 2^rel - 1 give
the same result. Graded labels are rejected instead of silently binarized.
An unjudged document contributes zero gain for benchmark scoring; this does
not establish that it is irrelevant, and is not a training negative label.

The final score is an unweighted mean of the per-query scores. Keep full
floating-point precision in JSON; round only for display. Preserve original
qrels, including all known positives. The duplicate-query positive union used
for negative exclusions must not replace evaluation qrels.

## Ranking and missing data

Results are sorted by descending score, then ascending document ID string on
exact ties. The retriever must use this rule before truncating to k; an
evaluator cannot recover tied documents already omitted from a candidate list.
This deterministic policy is explicit rather than an assertion that every
evaluation library resolves ties the same way.

Every whitelisted query must be present in both qrels and run. A missing or
extra query is an error. An explicitly empty result list receives zero and
remains in the averaging denominator. A query with no known positives is an
error for this experiment, where the audited splits all have positives.

Duplicate IDs, unknown documents, nonfinite scores, and malformed input are
errors. Query IDs and document IDs are separate namespaces: matching strings
are not automatically removed. The corpus remains shared across splits.

## Known-answer fixtures

All fixtures have 12 documents with strictly decreasing scores in d01..d12
order; k=10. These are synthetic calculation checks, not retrieval results.

| Known positive ranks | nDCG@10 | Recall@10 |
| --- | --- | --- |
| 1 | 1 | 1 |
| 2 | 1 / log2(3) | 1 |
| 1, 3 | (1 + 1/log2(4)) / (1 + 1/log2(3)) | 1 |
| 1, 12 | 1 / (1 + 1/log2(3)) | 1/2 |
| 11, 12 | 0 | 0 |
| 10 | 1 / log2(11) | 1 |
| 1 through 12 | 1 | 10/12 |

`verify_metrics.py` compares all seven with stored numeric known answers and
scikit-learn `ndcg_score` and `recall_score`. The latter is applied to binary
membership in the retrieved top ten. The reference check uses distinct scores;
tie ordering is covered separately in the unit tests. Agreement tolerance is
1e-12 on these small examples, not a claim of cross-hardware model determinism.

The CLI accepts four JSON inputs:

- qrels: `{ "query_id": { "document_id": 1 } }`
- run: `{ "query_id": [{ "document_id": "id", "score": 0.8 }] }`
- query-ids: the explicit evaluation query whitelist.
- corpus-ids: all candidate corpus IDs.

Run `retrieval_metrics.py --help` for flags. Output contains per-query metrics,
top documents, aggregate scores, and the evaluation policy. Existing output
files are not overwritten. Duplicate JSON object keys are rejected at load.

## Sources and limits

- [BEIR evaluation](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py)
  uses pytrec_eval ndcg_cut and recall. This project does not claim to have run
  BEIR or pytrec_eval; it checks binary metric definitions independently.
- [NIST nDCG cutoff implementation](https://github.com/usnistgov/trec_eval/blob/main/m_ndcg_cut.c)
- [NIST Recall implementation](https://github.com/usnistgov/trec_eval/blob/main/m_recall.c)
- [scikit-learn nDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- [scikit-learn Recall](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html)

The links explain definitions; executed package versions are recorded in each
run. All code and fixtures needed for this check are included locally. No
network access or additional package installation is needed for this stage.
