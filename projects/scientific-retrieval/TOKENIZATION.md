# Input policy and token audit

Personal experiment extension; upstream framework code is unchanged.

The first comparison uses the same text processing for the original model,
ordinary-negative fine-tuning, and hard-negative fine-tuning:

- Document: stripped title, newline, stripped abstract; omit an empty title.
- Query: strip surrounding whitespace. Preserve internal whitespace, case,
  punctuation, and Unicode in the input string; the pinned tokenizer applies
  its own normalization. The grouping normalization is not reused as model input.
- No prompt prefix, chunking, or manual lowercasing.
- Keep the original Sentence Transformers limit of 256 tokens, including its
  two special tokens. With verified BERT single-input layout, content budget
  is 254 tokens. Truncate from the right.

The [official model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
describes a default 256-word-piece truncation limit. The actual audit checks the
cached model at revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` and verifies
special-token layout and truncation with the locally installed framework.
Online documentation does not replace those runtime checks.

`text_policy.py` contains the shared formatting/accounting functions.
`token_audit.py` reads only the pinned corpus and frozen train/validation query
whitelist. It loads the ORIGINAL model from cache on CPU, calls the actual
SentenceTransformer tokenizer, and performs no forward pass or optimizer step.
There are no model API calls or downloads. The synthetic smoke-test checkpoint
is not used.

The audit counts full token sequences without truncation, then compares the
actual retained token IDs with the expected BERT right-truncated sequence.
It separately tokenizes title and abstract and verifies their boundary against
the joined input before calculating how many abstract tokens remain. Three
real-tokenizer boundary cases have 255, 256, and 257 input tokens.

Results include per-document and per-query CSV files containing IDs, lengths,
truncation flags, and input hashes. Full corpus/query text and weights are not
copied into these reports. JSON summaries report numerator, denominator,
fractions, token totals, and nearest-rank length percentiles. Corpus and the two
query splits are reported separately. No official test query text or qrels are
used. Corpus sharing is the previously documented experiment setting.

`truncated_fraction` is the number of truncated inputs divided by the number of
inputs in that category. `abstract_truncation_fraction` is the fraction of corpus
abstracts losing at least one token after the title consumes its input budget.
Neither is a measure of evidence lost or retrieval quality. No timing result
from this CPU token audit should be reported as model inference latency.

Run tests from the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s projects/scientific-retrieval -p 'test_*.py' -v
```

The stage runner supplies pinned existing evidence/cache paths, preserves old
files, runs tests, and saves logs, environment, source snapshots, JSON/CSV results,
checksums, and exit codes outside the repo. It creates `text-policy.json` in this
directory only after the real audit succeeds. It does not commit or push.

Assistant-side checks cover pure Python tests and syntax. The actual pinned
MiniLM/SciFact token audit must be validated by the user's Mac evidence; no
truncation percentages have been prefilled or claimed.
