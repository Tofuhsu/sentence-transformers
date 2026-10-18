# Original-model validation baseline

Scientific Retrieval Fine-tuning — Personal Fork is an experiment extension in
Tofuhsu's Sentence Transformers fork. This checkpoint adds data checks, a frozen
grouped split, a tokenization audit, retrieval metrics, a baseline evaluator,
and execution evidence. It does not modify the upstream framework core.

## Identity and measured setup

- Upstream: https://github.com/huggingface/sentence-transformers
- Release v6.1.0, commit `b0bf39968ba461c34b29372144794eff1be06f6d`.
- Original MiniLM revision: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- macOS 26.6.2, Apple M4 Pro, 48 GiB unified memory, native arm64.
- Python 3.11.6; PyTorch 2.14.1; Sentence Transformers 6.1.0;
  Transformers 5.20.1; Datasets 5.1.0; NumPy 2.4.6; scikit-learn 1.9.1.
- Evaluation started 2026-10-18 00:00:44 UTC (October 17 in Los Angeles).
- MPS float32, eager attention, no MPS fallback, encoding batch size 32.
- Exact CPU cosine search over 5,183 normalized 384-dimensional embeddings.
- 649 training queries remain unused for this baseline's query evaluation.
  The custom validation contains 160 queries and 186 known positive pairs.
- Input limit: 256 tokens including two special tokens. No fine-tuning;
  optimizer steps = 0. Original cached model files were unchanged.

## Measured results

| Measurement | Result |
| --- | ---: |
| Macro nDCG@10 | 0.6134114432722721 |
| Macro Recall@10 | 0.7515625 |
| Queries retrieving all known positives in the top 10 | 117 / 160 |
| Queries retrieving some, but not all, known positives | 6 / 160 |
| Queries retrieving no known positives in the top 10 | 37 / 160 |
| Corpus encoding, one pass over 5,183 documents | 12.082816 seconds |
| Batched encoding of 160 validation queries | 0.123828 seconds |
| Single-query encoding p50 / p95 | 7.807583 / 21.931458 ms |
| Single-query CPU search p50 / p95 | 0.330458 / 0.399459 ms |
| Single-query total p50 / p95 | 8.161750 / 22.312500 ms |
| Single-query total mean / maximum | 17.509168 / 233.690291 ms |
| Parameters | 22,713,216 |
| Original model.safetensors file | 90,868,376 bytes (86.658836 MiB) |

Recall is averaged per query. Its 0.7515625 value is not the fraction of queries
with a hit, and is not a micro-average over the 186 positive pairs. This is a
starting point, not an improvement measurement.

The 160 online samples each use batch size 1, following three warmup queries.
Encoding includes tokenization, forward computation, normalization, and CPU
transfer. Search includes the full dot products, deterministic sorting, and
result materialization. MPS synchronization brackets measured encoding calls.
Corpus encoding is offline work; it is not included in online query latency.
The large maximum is retained in the evidence. One pass over different queries
does not estimate repeated-trial variance or production service latency.

## Verification and evidence

- 31 unit tests passed on the user's Mac; tests, baseline, and stage exit codes
  are all zero.
- CPU/MPS smoke comparison on 10 inputs passed (maximum absolute embedding
  difference 1.6763806343078613e-7; rtol 1e-3, atol 1e-4).
- Metric formulas were checked against scikit-learn for all 160 real rankings.
- Batch-1 and batch-32 encoding returned the same ordered top 10 for all 160
  queries in this run.
- The uploaded archive's 46 manifest entries were verified in the assistant
  workspace. Metrics and latency summaries were independently recomputed from
  its saved rankings and samples. This was an evidence review, not another
  model run. Full embedding arrays remain on the user's Mac; the checkpoint
  preparation script checks their recorded hashes before committing.

The complete original run archive is
[evidence/baseline-validation-20261018T000044Z.8Vyvp7.zip](evidence/baseline-validation-20261018T000044Z.8Vyvp7.zip).
It contains sources, environment versions, configuration, logs, exit codes,
metric JSON, per-query CSV, top-10 document IDs and scores, and latency samples.
Verify the archive with `cd evidence && shasum -a 256 -c SHA256SUMS`.
`baseline-lock.json` records its SHA256 and all model/data revision pins.

## Execution status and next steps

`BASELINE.md`, `METRICS.md`, and `TOKENIZATION.md` describe the existing CLIs and
policies. `requirements-macos-py311.txt` snapshots observed dependencies while
leaving Sentence Transformers as the editable local checkout. It is not a
hash-locked, cross-platform dependency file. Current audit CLIs still expect
the earlier pinned evidence/cache inputs. A complete clean-machine bootstrap
remains a later deliverable; no fresh-machine reproduction is claimed here.

The historical stage runner in the ZIP requires the original upstream HEAD.
After the project checkpoint commit, use the documented Python CLI for a new
run; do not reset the branch to bypass that historical runner's guard.
The checkpoint step itself does not rerun retrieval or start training.

Next: define a matched loss and negative-sampling protocol for B/C, conduct a
bounded validation pilot, then run three fixed seeds. Freeze selection before
opening official test queries or labels. No B/C result exists at this checkpoint.

## Limits and attribution

These scores use a custom grouped validation split, not the official BEIR test.
The audit records no official test qrels or test query text being loaded.
Unjudged documents may be relevant. Grouping reduces observed overlap but does
not prove zero leakage. 3,681 of 5,183 corpus inputs were truncated; that is a
document count, not a measured fraction of evidence lost. The model card lists
scientific pretraining sources, so pretraining overlap cannot be excluded.
Fixed seeds do not guarantee identical results across hardware.

Existing LICENSE, NOTICE.txt, copyright notices, and upstream Git history remain
in place. Code, model, dataset, and abstract rights are separate; this checkpoint
redistributes no model weights or corpus texts. Review source-specific licensing
before publishing any later weights or derived training data. Personal work here
is an experiment extension, not a merged upstream contribution. Tests were run
locally, not in GitHub Actions.
