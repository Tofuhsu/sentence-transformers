# Handoff report: Scientific Retrieval Fine-tuning — Personal Fork

Written for Tofuhsu (USC M.S. CS, job search: SWE / AI Engineer / ML
Engineer / AI infrastructure). Purpose: give an accurate account of what
happened, who/what did what, and what is and is not safe to claim in an
interview or on a resume.

## What was built

A personal fork of [huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers)
(`Tofuhsu/sentence-transformers`, branch `experiment/scifact-hard-negatives`,
PR [#1](https://github.com/Tofuhsu/sentence-transformers/pull/1)) extending
it with a self-contained experiment under `projects/scientific-retrieval/`
that:

1. Audits and freezes a leakage-checked TRAIN/VALIDATION split of the SciFact
   dataset (649 / 160 queries, grouped by claim-source/text/positive-document
   overlap to avoid near-duplicate claims crossing the split).
2. Measures an original-model (no fine-tuning) retrieval baseline.
3. Mines two matched sets of training negatives from the same frozen split:
   ordinary (random) negatives and hard negatives (framework's own
   `mine_hard_negatives`), with a documented, nonexpert false-negative review
   of a 20-query sample before either set is used for training.
4. Fine-tunes `all-MiniLM-L6-v2` on each negative set (`TripletLoss`, matched
   loss/batch/LR/epochs/seeds), via pilot runs to pick a shared config, then
   3 full training seeds per arm.
5. Evaluates all of it -- original model, both fine-tuned arms, all seeds --
   once on the held-out official BEIR SciFact test split, after the training
   protocol was already frozen.

Full protocol and numbers: [RESULTS.md](RESULTS.md). Model card:
[MODEL_CARD.md](MODEL_CARD.md).

## Headline, verified result

Under a matched training budget (same loss, batch size, LR schedule, epoch
count, and 3 fixed seeds), fine-tuning with **hard negatives** beat
fine-tuning with **ordinary negatives** on the official BEIR SciFact test
split (300 queries), on both metrics, on every one of the 3 seeds:

- nDCG@10: 0.6785 ± 0.0014 (hard) vs. 0.6573 ± 0.0041 (ordinary) vs. 0.6451
  (no fine-tuning).
- Recall@10: 0.8154 ± 0.0034 (hard) vs. 0.7989 ± 0.0044 (ordinary) vs.
  0.7833 (no fine-tuning).

This is a real, measured, reproducible-on-this-machine result, not an
expected or assumed outcome -- on the smaller custom validation split used
to pick hyperparameters, ordinary-negative fine-tuning actually scored
*below* the original model, which is part of why this is worth being
precise about in an interview rather than just quoting the test numbers.

## Who/what did what

This was done with **Claude Code** (Anthropic's AI coding agent) as a pair
programmer, in a long interactive session, with you running every command
yourself on your own Mac and pasting the output back.

**Your contribution** -- the parts that are genuinely yours to claim:
- Defined the research question, the overall experiment design (A/B/C
  comparison, matched-budget requirement, pilot-then-final-seeds structure,
  validation-before-test discipline), and the hardware/environment
  constraints (no CUDA, Mac-only, MPS-first).
- Ran and verified every command on your own machine -- the actual MPS
  training, the actual encoding, the actual numbers above are real outputs
  from your hardware, not generated or simulated by the assistant.
- Made every consequential decision at each checkpoint: approving the
  symmetric 3-query exclusion instead of a code change to the mining
  script, confirming the lr=1e-5 choice from the pilot comparison,
  explicitly confirming the one-time official-test download and evaluation.
- Caught and corrected process issues in real time (e.g., not opening a PR
  before the repair was actually verified; asking "what's left on
  HuggingFace permissions/status" before merging anything).

**What Claude Code did**: wrote the training/evaluation scripts
(`finetune.py`, `evaluate_finetuned.py`, `evaluate_official_test.py`,
`run_final.sh`, `run_official_test.sh`), read and reused the pre-existing
pinned-protocol scripts from earlier in this project (data split, negative
mining, metrics), diagnosed a packaging bug in an evidence bundle, drafted
the false-negative review labels (flagged explicitly as a *nonexpert*,
non-gold review), and compiled/wrote up the results and this report.

**What to say in an interview**: "I designed and ran a controlled
fine-tuning comparison on my own Mac, using Claude Code to help write the
training and evaluation code, and made the methodology calls myself" is
accurate. "I built an ML training pipeline from scratch" overstates the
authorship of the code; "I proved hard negatives always help" overstates
the result (see Limitations below and in RESULTS.md).

## Tools and stack actually used

**Python**, **PyTorch** (MPS backend), **Sentence Transformers**
(`SentenceTransformerTrainer`, `TripletLoss`, `mine_hard_negatives`),
**Hugging Face** `datasets`/`huggingface_hub`/`transformers`, **pyarrow**
(parquet I/O), **scikit-learn** (metric cross-check), **Git**/**GitHub**
(personal fork, feature branch, PR), bash scripting for run orchestration,
on **Apple Silicon (M4 Pro, 48GB, MPS)**.

## Evidence (on this machine)

- Repo: `~/scientific-retrieval-finetuning`, branch
  `experiment/scifact-hard-negatives`, PR
  [Tofuhsu/sentence-transformers#1](https://github.com/Tofuhsu/sentence-transformers/pull/1).
- Final metrics (all seeds, both splits):
  `~/scientific-retrieval-evidence/final-results-20261018T220000Z.claude01/`
- Official-test download/open record:
  `~/scientific-retrieval-evidence/official-test-open-20261018T210000Z.claude01/`
- False-negative review:
  `~/scientific-retrieval-evidence/negative-false-negative-review-20261018T180000Z.claude01/`
- Per-run training/eval artifacts (model weights, logs, per-query results):
  `~/scientific-retrieval-artifacts/final-{B,C}-seed{13,42,2026}/`,
  `~/scientific-retrieval-artifacts/official-test-*/`
- Every earlier stage (environment, baseline, data audit, split freeze,
  token audit, negative mining, training contracts) has its own timestamped
  evidence directory under `~/scientific-retrieval-evidence/`, each with
  logs, exit codes, and a `SHA256SUMS` manifest.

## What you cannot claim

- **Not a merged or accepted upstream contribution.** PR #1 is open on your
  own fork (`Tofuhsu/sentence-transformers`), not on `huggingface/sentence-transformers`.
  Do not say "contributed to sentence-transformers" without that
  qualification.
- **Not CI-verified.** Every run happened locally on one Mac. No GitHub
  Actions run reproduced any of it.
- **Not a universal claim about hard negatives.** This is one model, one
  dataset, one hardware target, one loss function, 3 seeds. "Hard negatives
  improved retrieval in this controlled experiment" is accurate; "hard
  negatives are better" as a general claim is not supported by this alone.
- **Not a fact-verification system.** This retrieves candidate evidence; it
  does not judge whether a claim is true.
- **Not proof of zero false negatives.** The mining used a score-gap filter
  and a 20-query nonexpert review that flagged (and excluded) 3 suspicious
  candidates; unreviewed documents are not certified irrelevant.
- **Not an officially published model.** No weights are hosted anywhere;
  only code, configs, and metrics are in the repo/evidence.

## Draft resume bullets

> Built a controlled **hard-negative vs. random-negative** fine-tuning
> comparison for scientific-claim retrieval, fine-tuning **MiniLM** with
> **PyTorch**/**Sentence Transformers** (`TripletLoss`, `mine_hard_negatives`)
> on Apple Silicon (**MPS**); across 3 training seeds, hard negatives
> improved held-out BEIR SciFact test **nDCG@10 by 3.2%** and **Recall@10 by
> 2.1%** over matched random-negative fine-tuning (0.6785 vs. 0.6573 nDCG@10;
> 0.8154 vs. 0.7989 Recall@10, both +/-0.004 std across seeds).

> Designed a leakage-aware train/validation split and false-negative review
> process for SciFact hard-negative mining, and ran a pilot-then-final-seed
> training protocol (frozen before touching the official test set) in a
> personal **sentence-transformers** fork, with full evidence/hash-pinned
> reproducibility for every stage.

Swap in different emphasis (infra/pipeline vs. ML-result) depending on the
role; keep the numbers as-is since they are the actually-measured ones.
