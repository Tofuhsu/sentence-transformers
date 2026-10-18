# Model card: scientific-retrieval-hard-negatives (personal fork, experiment)

This is a **personal experiment fine-tune**, not an official or production
model. It is one of three variants compared in [RESULTS.md](RESULTS.md); see
that file for the full protocol, all-seed metrics, and limitations before
using any of these numbers out of context.

## Model details

- Base model: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
  revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` (unchanged architecture,
  22.7M parameters, 384-dim output).
- Fine-tuned by: Tofuhsu, as a personal extension in a fork of
  [huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers)
  (framework code itself unchanged; see [README.md](README.md)).
- Three seed checkpoints exist (13, 42, 2026); none is singled out as "the"
  release -- report the mean ± std across seeds, per [RESULTS.md](RESULTS.md).

## Intended use

Encode a scientific claim and a corpus of paper title+abstract texts into
384-dim vectors for cosine-similarity retrieval of the top-k most relevant
abstracts to a claim. This is a **retrieval-only** component: it does not
judge whether a claim is true, false, or unverifiable, and it is not
intended to be used for that without a separate verification stage.

## Training data and procedure

- Data: [`BeIR/scifact`](https://huggingface.co/datasets/BeIR/scifact)
  (revision `b3b5335604bf5ee3c4447671af975ea25143d4f5`) and
  [`BeIR/scifact-qrels`](https://huggingface.co/datasets/BeIR/scifact-qrels)
  (revision `2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029`). 5,183-document
  shared corpus; 730 (anchor claim, positive abstract, negative abstract)
  training triplets from a frozen 649-query TRAIN split (3 queries held out
  after a false-negative review; see [NEGATIVES.md](NEGATIVES.md)).
- Negatives for this variant ("C"): mined with the pinned upstream
  `mine_hard_negatives` (`sampling_strategy="top"`), one per unique training
  claim, excluding known positives and an exact-duplicate-query positive
  union, requiring the candidate's original-model score to be >=0.05 below
  the claim's weakest known positive.
- Loss: `TripletLoss`, cosine distance, margin 0.2. Optimizer: AdamW
  (`foreach=False`), weight decay 0.01, linear LR decay from 1e-5, 10%
  warmup. Batch size 16, 3 epochs, max sequence length 256 tokens. Trained
  on Apple Silicon MPS, float32, no mixed precision.

## Evaluation results

Official BEIR SciFact test split (300 queries, 339 qrels), mean over seeds
13/42/2026:

| Variant | nDCG@10 | Recall@10 |
| --- | ---: | ---: |
| Original (no fine-tuning) | 0.6451 | 0.7833 |
| Ordinary-negative fine-tune ("B") | 0.6573 ± 0.0041 | 0.7989 ± 0.0044 |
| **Hard-negative fine-tune ("C", this card)** | **0.6785 ± 0.0014** | **0.8154 ± 0.0034** |

Full per-seed numbers, the custom-validation numbers used to pick the shared
learning rate, and error analysis are in [RESULTS.md](RESULTS.md).

## Limitations

- Evaluated on one dataset (SciFact), one base model, one hardware target
  (Apple M4 Pro, MPS). Results may not transfer to other scientific
  retrieval datasets, models, or hardware.
- 3,681 of 5,183 corpus abstracts are truncated at 256 tokens.
- Pretraining overlap between the base model and scientific text cannot be
  excluded (see the base model's own model card).
- 45 of 300 official test queries score nDCG@10 = 0 under every variant
  tested, including this one -- this model does not solve every claim.
- Single-run, single-seed-set result on one machine; no cross-hardware or
  multi-run reproduction was attempted. See
  [RESULTS.md](RESULTS.md#limitations) for the complete list.

## License and provenance

Base model weights and architecture: see the base model's own license on
its Hugging Face page. SciFact claim annotations: CC-BY-4.0; abstracts:
ODC-By-1.0 (per the original SciFact release; the BeIR-hosted copy's
dataset card declares CC-BY-SA-4.0 -- both are recorded here since they
differ). This fork's own added code (training/evaluation scripts): Apache-2.0,
same as the upstream `sentence-transformers` repository. Model weights from
this fine-tune are not published anywhere as of this card; only the
scripts, configs, and metrics in this repository and in
`~/scientific-retrieval-evidence/` are shared.
