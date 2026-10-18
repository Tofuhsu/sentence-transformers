# Project brief for resume/portfolio writing (self-contained)

Audience: an AI agent helping Tofuhsu (hsuanfuh@usc.edu, USC M.S. Computer
Science, job search: SWE / AI Engineer / ML Engineer / AI infrastructure)
write resume bullets, portfolio copy, or interview talking points. This
file does not assume you have seen the conversation that produced it --
everything needed is below. Treat every number here as verified against
the evidence paths given; do not round up, generalize, or add claims not
present in this file.

## 1. One-line project identity

**Scientific Retrieval Fine-tuning -- Personal Fork**: a controlled
experiment, in a personal fork of
[huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers),
comparing hard-negative vs. ordinary-negative fine-tuning for scientific-
claim-to-evidence retrieval (SciFact dataset), run end-to-end on a personal
Mac (Apple M4 Pro, 48GB, PyTorch MPS, no cloud/GPU rented).

- Fork: `Tofuhsu/sentence-transformers`, branch `experiment/scifact-hard-negatives`
- Pull request (open on the fork, **not merged upstream**):
  https://github.com/Tofuhsu/sentence-transformers/pull/1
- Full write-up: `projects/scientific-retrieval/RESULTS.md` in that branch
- Model card: `projects/scientific-retrieval/MODEL_CARD.md`
- This file and `HANDOFF_REPORT.md` (longer, same repo path) cover
  attribution and what not to claim.

## 2. The research question, stated precisely

Under a **matched** fine-tuning budget (same loss function, batch size,
learning rate schedule, epoch count, same positive training pairs, 3 fixed
random seeds), does selecting **hard negatives** (candidates the original
model scores highly but that are not labeled relevant) produce better
scientific-claim retrieval than **ordinary (random) negatives**?

## 3. Verified, quantified result (the headline number)

Evaluated once on the official, held-out BEIR SciFact **test** split (300
claims, 339 relevance judgments), after the training protocol was frozen
using only a separate validation split:

| Variant | nDCG@10 (mean, 3 seeds) | Recall@10 (mean, 3 seeds) |
| --- | ---: | ---: |
| Original model, no fine-tuning | 0.6451 | 0.7833 |
| Fine-tuned with ordinary negatives | 0.6573 +/- 0.0041 | 0.7989 +/- 0.0044 |
| **Fine-tuned with hard negatives** | **0.6785 +/- 0.0014** | **0.8154 +/- 0.0034** |

- Hard negatives beat ordinary negatives on **both metrics, on all 3**
  training seeds (13, 42, 2026), with non-overlapping ranges: the ordinary-
  negative model's best seed (nDCG@10 0.6631) still scores below the hard-
  negative model's worst seed (0.6766).
- Hard negatives vs. ordinary negatives: nDCG@10 +0.0212 absolute (+3.2%
  relative to the ordinary-negative mean); Recall@10 +0.0166 absolute
  (+2.1% relative).
- Hard negatives vs. no fine-tuning: nDCG@10 +0.0334 absolute (+5.2%
  relative); Recall@10 +0.0322 absolute (+4.1% relative).
- Hard-negative fine-tuning was also more stable across seeds (std 0.0014
  vs. 0.0041 on nDCG@10).
- Evidence file with every per-seed number:
  `~/scientific-retrieval-evidence/final-results-20261018T220000Z.claude01/final-metrics.json`
  (+ sibling `.csv`).

Model: `sentence-transformers/all-MiniLM-L6-v2` (22.7M params, 384-dim
output), fine-tuned with `TripletLoss` (cosine distance, margin 0.2),
AdamW, lr 1e-5, 3 epochs, batch size 16, on Apple Silicon MPS, float32.

## 4. What is genuinely novel/hard about this (for interview framing)

- **Matched-budget comparison, not just "we tried hard negatives and it
  worked."** The learning rate was selected from 4 pilot runs using only a
  validation split, specifically to avoid the trap of tuning B and C
  separately and then crediting the whole gap to negative selection.
- **Official test set opened exactly once**, after the protocol was
  frozen, with the download itself gated behind an explicit evidence
  record (`~/scientific-retrieval-evidence/official-test-open-20261018T210000Z.claude01/`)
  -- a concrete anti-p-hacking discipline, not just a stated intention.
- **False-negative risk was treated as a real risk, not hand-waved.** A
  20-claim review sample was built (union of random, highest-scoring, and
  smallest-candidate-pool queries) and reviewed; 3 flagged candidates were
  excluded from training **symmetrically from both arms** (not just the
  arm it would have hurt). A later follow-up pass re-examined all 3 against
  their claims' full known-positive text and reclassified one, explicitly
  without retroactively changing the already-reported result.
- **Leakage-aware data split**: the 649/160 train/validation split groups
  claims by shared source context, near-duplicate text, and shared known-
  positive documents (490 groups total) before splitting, specifically to
  stop paraphrased claims from crossing the split.
- **Honest negative result along the way**: on the smaller validation
  split, the ordinary-negative model actually scored *below* the untrained
  baseline after 3 epochs. This was reported as-is, not hidden, and is part
  of why the official-test result (where both arms beat baseline) is
  credible rather than cherry-picked.

## 5. Quantifiable contributions (atomic, resume-bullet-ready)

Each line below is independently verifiable at the evidence path given.

1. Designed and ran a 3-arm (original / ordinary-negative / hard-negative),
   3-seed controlled fine-tuning comparison; hard negatives improved
   **nDCG@10 by 3.2%** and **Recall@10 by 2.1%** over matched ordinary-
   negative fine-tuning on the official BEIR SciFact test set (300 claims).
   -- `RESULTS.md`, `final-metrics.json`
2. Built a leakage-checked data split over 809 official training claims
   into 649 train / 160 validation queries across 490 dedup/near-duplicate
   groups, with zero query-ID, group, or known-positive-document overlap
   between splits and the 300-claim official test set.
   -- `split-lock.json`, `grouping-review.json`
3. Implemented and independently verified (against scikit-learn) nDCG@10
   and Recall@10 retrieval metrics, run across **7 full model evaluations**
   (1 baseline + 3 ordinary-negative seeds + 3 hard-negative seeds) on both
   a 160-query validation set and a 300-query official test set.
   -- `retrieval_metrics.py`, `test_retrieval_metrics.py`
4. Wrote and ran a false-negative review process over a 20-claim sample
   (random + highest-scoring + smallest-candidate-pool), flagging and
   symmetrically excluding 3 of 733 training pairs from both compared arms
   before training, with a documented follow-up re-review.
   -- `negative-false-negative-review-20261018T180000Z.claude01/`,
      `negative-false-negative-review-followup-20261018T230000Z.claude01/`
5. Authored 5 reusable CLIs (data split, baseline eval, negative mining,
   fine-tuning, validation/test evaluation) with pinned-input integrity
   checks (SHA256 hash verification of model weights, dataset snapshots,
   and configs before every run) and **53 passing unit tests** targeting
   real risks (ID/leakage correctness, metric correctness, save/reload
   consistency, CLI error handling) rather than coverage padding.
   -- `test_*.py` (53 tests total, all passing)
6. Verified model save/reload produces bit-identical embeddings
   (max absolute difference 0.0) before trusting any evaluation of a saved
   checkpoint.
   -- `final-results-20261018T220000Z.claude01/save-reload-consistency.json`

## 6. Tools and stack (for a skills section or "stack" line)

Python, PyTorch (MPS backend on Apple Silicon), Hugging Face
`sentence-transformers` (`SentenceTransformerTrainer`, `TripletLoss`,
`mine_hard_negatives`), `transformers`, `datasets`, `huggingface_hub`,
`pyarrow` (parquet I/O), `scikit-learn` (metric cross-validation), Git/
GitHub (fork, feature branch, pull request), bash orchestration scripts,
`unittest`.

## 7. Draft resume bullets (pick/adapt by target role)

For an ML/AI Engineer-leaning resume:

> Designed and ran a controlled fine-tuning comparison for scientific-claim
> retrieval, fine-tuning **MiniLM** with **PyTorch**/**Sentence
> Transformers** (`TripletLoss`, `mine_hard_negatives`) on Apple Silicon
> (**MPS**); across 3 training seeds, hard-negative fine-tuning improved
> held-out BEIR SciFact test **nDCG@10 by 3.2%** and **Recall@10 by 2.1%**
> over matched random-negative fine-tuning (0.6785 vs. 0.6573 nDCG@10,
> +/-0.004 std across seeds).

For an infra/pipeline-leaning resume:

> Built a leakage-aware data pipeline and reproducible evaluation harness
> (hash-pinned model/dataset versions, 53 unit tests, timestamped evidence
> directories) for a 3-arm ML experiment, with a documented false-negative
> review process and a one-time, protocol-frozen official test evaluation
> to avoid metric leakage/p-hacking.

Keep every number exactly as written here -- they come from the evidence
files cited, not from memory or estimation.

## 8. Do NOT claim (read before writing anything)

- **Not merged upstream.** This is an open PR on a personal fork
  (`Tofuhsu/sentence-transformers#1`), not a contribution accepted into
  `huggingface/sentence-transformers`. Never say "contributed to
  sentence-transformers" without that qualifier.
- **Not CI-verified / not reproduced from a clean environment.** Every run
  happened locally on one Mac. `requirements-macos-py311.txt` is
  explicitly a version *snapshot*, not a hash-locked, cross-platform
  lockfile, and the README states clean-environment reproduction is
  "future work," not something that was actually done. Do not say "CI
  passing" or "reproducible from a fresh install" -- neither is true yet.
- **No online (single-query) latency numbers exist for the fine-tuned
  models.** Only the original (untrained) baseline had warmed, per-query
  p50/p95 latency measured; the fine-tuned B/C evaluations only recorded
  batch corpus/query encoding time. Do not quote a "query latency"
  improvement number for the fine-tuned models -- it was never measured.
- **Not a universal claim about hard negatives.** This is one base model
  (MiniLM), one dataset (SciFact), one hardware target, one loss function,
  3 seeds. "Hard negatives improved retrieval in this controlled
  experiment" is accurate; "hard negatives are always better" is not
  supported by this work alone.
- **Not a fact-verification system.** This retrieves candidate evidence
  for a claim; it does not judge whether the claim is true or false.
- **Not proof that the remaining negatives are all correct.** A 20-claim
  sample was reviewed (not all 730 training pairs). 2 of 3 originally
  flagged candidates (QID 760, 779) remain genuinely unresolved and were
  deliberately left for a domain expert, not resolved by this project.
- **No model weights are published anywhere.** Only code, configs, and
  metrics are in the repository and in the evidence directories on the
  author's machine.
- Full, unabridged limitations list: `RESULTS.md` ("Limitations" section)
  and `HANDOFF_REPORT.md` ("What you cannot claim" section), same repo path.

## 9. Where to verify anything in this file

Everything above was produced across one extended session in this
project's own repository and `~/scientific-retrieval-evidence/` /
`~/scientific-retrieval-artifacts/` directories on the author's Mac --
every number has a cited JSON/CSV file backing it, and every stage has a
logged, exit-code-checked, SHA256-hashed evidence directory. If asked for a
number not in this file, say so rather than estimating one.
