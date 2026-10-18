# B/C controlled comparison: results

Research question: under a matched fine-tuning budget, do hard negatives
(C) improve scientific-claim retrieval over ordinary negatives (B)?

## Protocol actually run

- Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Loss: `TripletLoss`, cosine
  distance, margin 0.2. Optimizer: AdamW (`foreach=False`), weight decay
  0.01, linear decay, warmup ratio 0.1. Batch size 16, max sequence length
  256. Device: MPS, float32, eager attention.
- Negatives: B is `sampling_strategy="random"`; C is `sampling_strategy="top"`
  via the pinned upstream `mine_hard_negatives`. See [NEGATIVES.md](NEGATIVES.md)
  for the full mining design and policy.
- 3 queries (IDs 369, 760, 779) were dropped from **both** B and C training
  sets after a false-negative review flagged their C candidate as
  `possible_relevant` (see
  `~/scientific-retrieval-evidence/negative-false-negative-review-20261018T180000Z.claude01/`).
  733 -> 730 training rows per arm, counts stayed matched between arms.
- Learning rate was chosen from 4 pilot runs (B and C, 1 epoch, seed 42, at
  lr 2e-5 and 1e-5) using validation only: lr=1e-5 was picked because it was
  the only one where *neither* arm regressed on validation. No further
  tuning followed.
- Final runs: 3 epochs, seeds 13/42/2026, for both B and C (6 training runs).
  Official BEIR SciFact test (300 queries, 339 qrels; downloaded once, after
  the protocol above was frozen) was evaluated once per run, 7 runs total
  (A original + 3xB + 3xC). No tuning happened after seeing test numbers.

## Official test results (primary; 300 queries, 339 qrels)

| Variant | nDCG@10 (mean ± std) | Recall@10 (mean ± std) |
| --- | ---: | ---: |
| A: original, no fine-tuning | 0.6451 | 0.7833 |
| B: ordinary negatives | 0.6573 ± 0.0041 | 0.7989 ± 0.0044 |
| C: hard negatives | **0.6785 ± 0.0014** | **0.8154 ± 0.0034** |

Per-seed values, absolute/relative deltas, and the custom-validation numbers
are in
[`~/scientific-retrieval-evidence/final-results-20261018T220000Z.claude01/final-metrics.json`](../../../scientific-retrieval-evidence/final-results-20261018T220000Z.claude01/final-metrics.json)
and the sibling `.csv`.

- C beat B on **both metrics, on all 3 seeds**, with non-overlapping ranges
  (B's highest nDCG@10 seed, 0.6631, is below C's lowest, 0.6766).
- C vs B: nDCG@10 +0.0212 (+3.2% relative to B's mean), Recall@10 +0.0166
  (+2.1% relative).
- C vs A: nDCG@10 +0.0334 (+5.2%), Recall@10 +0.0322 (+4.1%).
- C's cross-seed std is smaller than B's on both metrics (more stable, not
  just higher on average).

## Validation results (used only to pick the shared pilot config, before test)

| Variant | nDCG@10 (mean ± std) | Recall@10 (mean ± std) |
| --- | ---: | ---: |
| A | 0.6134 | 0.7516 |
| B (3 epochs, 3 seeds) | 0.6084 ± 0.0043 | 0.7427 ± 0.0045 |
| C (3 epochs, 3 seeds) | 0.6151 ± 0.0005 | 0.7557 ± 0.0030 |

On this 160-query custom validation split, B (3 epochs) actually scored
*below* the un-fine-tuned baseline on both metrics; C scored slightly above
baseline. The validation and test query sets are disjoint samples of
different sizes (160 vs 300 queries) and are not directly comparable in
absolute terms -- only the repeated C > B direction across both splits is
evidence of a real, not incidental, effect.

## Error analysis (official test, seed 42; see `error-analysis.json` in the
evidence directory for the full data)

- 45 of 300 test queries get nDCG@10 = 0 under A, B, **and** C -- genuinely
  hard queries this setup does not solve.
- 11 queries go from 0 (both A and B fail) to a positive score under C.
- 4 queries go the other way: A and B both succeed, C scores 0. Net effect
  is clearly positive (11 fixed vs. 4 newly broken), consistent with the
  aggregate numbers, but C is not a strict improvement on every query.

## Limitations

- Custom validation is a grouped subset of the official TRAIN split, not an
  official BEIR split; see [BASELINE_RESULTS.md](BASELINE_RESULTS.md) and
  [README.md](README.md) for how it was built and its leakage controls.
- 3 queries were removed from training (not from evaluation) based on a
  nonexpert (AI) false-negative review; see [NEGATIVES.md](NEGATIVES.md).
  This is a documented, symmetric exclusion, not a certified relevance label.
- One fixed sampling seed (1729) was used for negative mining; negative-
  sampling variance itself was not measured, only training-seed variance.
- 3,681 of 5,183 corpus documents are truncated at 256 tokens (see
  [TOKENIZATION.md](TOKENIZATION.md)); this affects all three variants
  equally.
- The MiniLM model card lists scientific pretraining sources; pretraining
  overlap with SciFact cannot be excluded.
- Fixed seeds do not guarantee identical results across hardware/OS/library
  versions. Everything here ran once, on one Mac (Apple M4 Pro, MPS,
  float32); no cross-hardware reproduction was attempted.
- This measures retrieval quality (nDCG@10, Recall@10) only; it does not
  measure claim-verification accuracy, which is explicitly out of scope for
  this first version.
- No GitHub Actions run reproduced any of this; all runs are local, on the
  user's Mac, with the evidence in `~/scientific-retrieval-evidence/`.
