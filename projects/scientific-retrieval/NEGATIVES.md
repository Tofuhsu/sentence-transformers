# Paired negative preparation: B and C

This personal experiment uses the existing Sentence Transformers miner at
upstream commit `b0bf39968ba461c34b29372144794eff1be06f6d`. No framework core
files are changed. Hard-negative mining is an existing technique and tool,
not a method invented by this project.

## Controlled comparison

Both arms begin with the pinned original MiniLM revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. They use the frozen 649 training
queries, all 733 original positive pairs, and the shared 5,183-document corpus.
Validation/test queries are not embedded for mining. The saved official-train
positive map includes the custom validation portion; only frozen train IDs
are retained and passed to selection. No validation labels influence it.

| Setting | B: ordinary negatives | C: hard negatives |
| --- | --- | --- |
| Eligible pool | Entire shared corpus after identical filtering | Same |
| Choice | Upstream `sampling_strategy="random"`, uniform over eligible documents | Upstream `sampling_strategy="top"` |
| Negatives | One per unique exact query text | Same |
| Seed | 1729, fixed before sampling | 1729; top choice does not use random sampling |
| Positive pairs | All original training qrels | Same pairs, same order |
| Preprocessing | Existing title + newline + abstract; right truncate at 256 tokens | Same |

Both exclude every known positive for the query plus the previously frozen
union for exactly normalized duplicate queries. The miner receives these
exclusion documents as positive control pairs to prevent selecting them.
Those additional exclusion-only pairs are **never added to training labels**.
All original positive pairs are retained, including multi-positive queries.
The one chosen negative is reused for each positive of its query. Thus
queries with more positives contribute more rows, equally in B and C.

The miner keeps a candidate only when, in float32:

```text
candidate_score + 0.05 <= minimum_score_of_all_exclusion_positives
```

This is an absolute cosine-score gap, not a relative 5% threshold. It is also
different from the proposed training TripletLoss margin of 0.2. Using the
same safety filter makes B a **filtered uniform-random baseline**, rather
than uniform sampling from every nonpositive document without a score filter.

`range_min=0`, `range_max=5183`, `use_faiss=False`, and the full corpus are
explicit. In this pinned implementation the final rank window is applied
after filtering; using the full corpus avoids a hidden top-k restriction
on B. Selection ties follow upstream `torch.topk`; saved diagnostic ranks
use score descending then document-ID string ascending. Cross-hardware
bitwise selection equivalence is not promised.

## Checks and evidence

`prepare_negatives.py --help` lists explicit input/output paths. The supplied
stage runner uses this project's pinned Mac evidence locations and `.venv`.

1. Verify data, text, split, model, package and miner-source pins.
2. Run the actual upstream miner on a small synthetic multi-positive fixture
   on CPU, covering its full-window padding and weakest-positive gap.
3. Encode the original model on MPS and mine C. Store a fresh embedding cache.
4. Mine B using that exact same cache and check that its bytes did not change.
5. Independently recompute full-corpus scores using the public cosine
   functions on CPU. Check known-positive exclusions, score gap, full query
   coverage, score consistency, and highest-eligible-score selection for C.
6. Export ID-based training triplets, candidate coverage, scores, ranks,
   initial active-loss diagnostics, and paired review cases.

No query is silently dropped if an eligible negative cannot be found.
Exclusion unions do not modify qrels. The prior real CPU/MPS TripletLoss
contract check established row-wise loss behavior with no implicit in-batch
negatives; it did not certify the semantics of these real negative samples.

Outputs are saved under a new `negative-preparation-*` evidence directory.
`results/negative-summary.json`, `selected-negatives.jsonl`,
`B_random-triplets.jsonl`, `C_hard-triplets.jsonl`, and `coverage.json` contain
IDs and measured values. `review-cases.json` includes training texts for
inspection. Embedding arrays and the score matrix remain in
`~/scientific-retrieval-artifacts/negative-mining/`, outside ordinary Git.
Their SHA256 hashes and paths are recorded in `results/artifacts.json`.

## Review before training

The review sample is the union of 10 queries sampled with seed 2718, five
queries with the highest selected C score, and five with the smallest
eligible pools. Both B and C are shown for each selected query. The number
of distinct review queries can therefore be less than 20. No review is
claimed until the cases have actually been inspected.

Allowed annotations are `likely_unrelated`, `possible_relevant`, and
`uncertain`, with reasons. These are nonexpert judgments, not new gold
labels. If a screening rule needs changing, record its reason and apply it
equally to both arms before any pilot training; do not hand-pick only C's
bad cases away. Do not alter official qrels or consult test labels.

## Limitations

- Unjudged does not imply irrelevant; a score gap only reduces one source
  of risk and cannot prove that a negative is valid.
- This stage has no optimizer steps or retrieval-quality evaluation.
- Initial nonzero TripletLoss fractions describe available training signal
  for margin 0.2, not improvement. Do not increase B's training budget merely
  because random negatives often have zero loss.
- C's measured time includes fresh encoding; B reuses that cache. These
  timings cannot be presented as a mining-speed comparison.
- One fixed negative-sampling seed allows paired training-seed comparisons,
  but does not estimate variance from negative sampling itself.
- The original 256-token truncation and possible pretraining overlap remain.
- Results are not final training inputs until review is recorded. The stage
  reports `training_allowed=false` and performs no training or publication.
