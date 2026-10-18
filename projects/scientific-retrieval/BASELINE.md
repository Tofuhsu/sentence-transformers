# Original MiniLM validation baseline

This personal project extension evaluates the unmodified, pinned MiniLM
checkpoint on the frozen custom SciFact validation split. It is not an upstream
Sentence Transformers bug fix or an official BEIR test result.

## Frozen evaluation choices

- Original model: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Shared corpus: 5,183 documents. Validation: 160 queries, 186 positive pairs.
- Query text comes only from the validation whitelist. Qrels come from the
  previously verified official TRAIN positive map, filtered to validation.
  The negative-exclusion union is never substituted for evaluation qrels.
- Text formatting and the 256-token limit follow `text-policy.json`.
- MPS float32 encoding with eager attention, normalized 384-dimensional output,
  batch size 32. No prompt, chunking, fine-tuning, or optimizer steps.
- CPU float32 dot products across the entire corpus, equivalent to cosine for
  normalized vectors. Sort by descending score, then ascending document ID
  string on exact ties before taking the top 10. No approximate index.
- nDCG@10 and Recall@10 follow `METRICS.md`. Retain per-query scores.

The source-level CONFIG is copied to each run before encoding. These choices
must stay matched in B/C evaluations; changes require a new comparison plan.
The earlier synthetic smoke-test checkpoint is never loaded.

## Checks before full encoding

The runner verifies the baseline commit, frozen inputs, prior evidence, and
source hashes. The model's cached files must match the successful token audit.
Eight documents and two validation queries are first encoded on CPU and MPS;
finite unit-normalized outputs must agree at rtol=1e-3 and atol=1e-4. The actual
maximum difference is recorded. An unsupported MPS operation or mismatch
stops execution; no automatic device or precision fallback occurs.

Model shape, dtype, finite outputs, normalization, query coverage, and original
file hashes are checked again during the run. Binary metric calculations are
cross-checked with scikit-learn on all validation rankings, using rank scores
to express the documented tie policy. This is an independent formula check
on the same retrieval output, not an independent retrieval system.

## Timing protocol

Corpus encoding and batched query encoding each have three warmup batches
of 32 texts, followed by one measured full pass. MPS is synchronized before
starting and after finishing each measured encoding call. Encoding includes
tokenization, forward computation, normalization, and CPU NumPy transfer;
model loading and saving files are excluded.

Single-query latency uses batch size 1, three warmup queries, then one request
per validation query in a fixed seed-42 shuffled order. Report encoding, exact
CPU search, and total wall time separately, with all 160 samples and nearest-
rank p50/p95. CPU search includes query validation, full dot products, sorting,
and top-10 materialization. Total latency includes the timing-wrapper overhead.
Corpus embeddings remain in CPU memory throughout this latency measurement.

Thread limits are requested through environment variables and threadpoolctl;
actual Torch thread counts and library threadpool information are recorded.
Keep power conditions consistent and pause other model workloads when comparing
variants. These are warmed, in-process measurements on one Mac, without network
requests, server queues, or concurrent users. A distribution across distinct
query texts is not repeated-trial uncertainty or a production latency claim.

Batch-1 embeddings are checked against batch-32 embeddings at the same CPU/MPS
tolerances. Any top-10 differences are recorded; tiny floating-point changes
near ties can change rankings. The quality scores use batch-32 query embeddings.

## Outputs

The evidence directory includes config, environment/packages, original model
file hashes, smoke-test measurements, validation qrels, top-10 results,
per-query JSON/CSV, full positive ranks, timing samples, and a summary.
`review-case-ids.json` identifies high/low scoring queries; qualitative error
analysis remains a subsequent task.

Larger artifacts are stored outside Git under
`~/scientific-retrieval-artifacts/retrieval-baselines/`: corpus/query embeddings,
the complete 160-by-5,183 score matrix, ordered IDs, and configuration. An
artifact manifest in the evidence records paths and SHA256 values. No model
weights, cache, credentials, or datasets are copied into the Git repository.

Use `retrieval_baseline.py --help` for the direct CLI. The stage runner supplies
the concrete local paths and records all exit codes. Preserve both failed and
successful runs rather than overwriting them.

## Limits

This is a custom grouped validation split. The official test remains held out.
Known positives are incomplete judgments; unjudged does not imply irrelevant.
3,681 of 5,183 document inputs were truncated in the prior audit. The pretrained
model card lists scientific training sources, so pretraining overlap is not
excluded. Fixed seeds do not establish bitwise reproducibility across hardware.
Do not claim a fine-tuning improvement or cost saving from this baseline alone.

References:

- [SentenceTransformer encode API](https://sbert.net/docs/package_reference/sentence_transformer/model.html)
- [PyTorch MPS synchronization](https://docs.pytorch.org/docs/2.14/generated/torch.mps.synchronize.html)
- [Pinned model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/blob/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/README.md)
