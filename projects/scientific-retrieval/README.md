# Scientific Retrieval Fine-tuning — Personal Fork

This directory contains Tofuhsu's experiment extension in a personal fork of
[Sentence Transformers](https://github.com/huggingface/sentence-transformers).
Upstream framework code is unchanged. The upstream LICENSE and NOTICE.txt remain
in the repository root. Added code uses Apache-2.0.

## Current stage: freeze the training/validation split

The research question is whether fixed hard negatives outperform ordinary
negatives under matched fine-tuning conditions for scientific evidence retrieval.
No retrieval result or model improvement is claimed at this stage.

- Framework baseline: `b0bf39968ba461c34b29372144794eff1be06f6d` (v6.1.0).
- Model: `sentence-transformers/all-MiniLM-L6-v2`, revision
  `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Data: `BeIR/scifact`, revision `b3b5335604bf5ee3c4447671af975ea25143d4f5`.
- Qrels: `BeIR/scifact-qrels`, revision `2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029`.

The reviewed audit partitions the 809 official training queries into 490
connected groups. Links use original claim-source contexts, normalized exact text,
shared known positive documents, or character TF-IDF cosine similarity >= 0.85.
Similarity >= 0.75 is a review threshold. Lexical links do not add components to
the structural grouping in this audited dataset. They are not semantic labels.
`grouping-review.json` records the AI-assisted review and its limitations.

The split seed is 42, fixed before retrieval evaluation. Groups are ordered by
SHA256 of compact JSON `[42, sorted_query_ids]`; the first ceil(490/5) hashes
select 98 validation groups. Remaining groups form training. No group is split
to force an exact query percentage. Input order and Python's random generator
implementation do not affect this assignment. There was no search over seeds.

Expected from the uploaded audit, executed in the assistant's Linux workspace:

| Partition | Queries | Groups | Original qrel rows | Known positive documents |
|---|---:|---:|---:|---:|
| Train | 649 | 392 | 733 | 450 |
| Validation | 160 | 98 | 186 | 115 |

The runner verifies these deterministic outputs on the user's Mac; Mac success
must be supported by that run's own logs and exit codes.

Training and validation have no overlap in IDs, groups, normalized exact text,
matched source contexts, or known positive document IDs. The full 5,183-document
corpus remains shared. The positive-document separation is a conservative design
choice, not an official BEIR split. Grouping can merge distinct claims and miss
paraphrases, so it is not proof of zero leakage. Official test qrels and held-out
query text are not read by this stage.

All original qrels remain unchanged. Queries 89 and 90 have identical normalized
text but different positive-document lists. `train-negative-exclusions.json`
unions known positives for exactly normalized duplicate TRAIN text only. It does
not propagate labels to every claim in a group, assert that other documents are
irrelevant, or implement a negative sampler. Later sampling must use this map.

Run tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s projects/scientific-retrieval -p test_data_split.py -v
```

The stage's `run-macos.sh` provides the full command with existing evidence paths,
checks pinned input hashes, writes a new evidence directory, and copies an
identical `split-lock.json` into this folder. It refuses to overwrite different
files. Each evidence directory includes logs, source snapshots, dependency
versions, exit codes, output hashes, and the training-only exclusion map.

The stage uses only Python's standard library and does not download data, change
dependencies, execute a model, commit, or push. Earlier audit scripts and raw
data stay in the recorded evidence/artifact directories for now. Integrating data
preparation into a clean-environment reproduction command remains future work.

Dataset attribution: SciFact (Wadden et al., *Fact or Fiction: Verifying
Scientific Claims*, EMNLP 2020); BEIR (Thakur et al., 2021). Preserve the pinned
Hugging Face data cards and original SciFact licensing statements. Their license
labels differ: HF cards declare CC-BY-SA-4.0; original SciFact distinguishes claim
annotations (CC-BY-4.0), abstracts (ODC-By-1.0), and code (Apache-2.0). This stage
does not redistribute abstracts or model weights. Final publication terms still
need to be documented per artifact.

## Original-model validation checkpoint

The original MiniLM baseline and supporting code are now recorded in this
personal fork. See [BASELINE_RESULTS.md](BASELINE_RESULTS.md) for measured
quality, timing, evidence, and the work still pending before controlled B/C
fine-tuning. [baseline-lock.json](baseline-lock.json) pins the upstream, model,
data, split, and evidence identities.
