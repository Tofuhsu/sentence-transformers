# SPDX-License-Identifier: Apache-2.0
"""Validation retrieval evaluation for a locally fine-tuned model (B/C pilot or final runs).
Reuses the same frozen custom-validation inputs and metric formulas as the
original-model baseline (retrieval_baseline.py); no training happens here."""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_split import json_text  # noqa: E402
from retrieval_baseline import read_inputs  # noqa: E402
from retrieval_metrics import evaluate, require  # noqa: E402
from retrieval_runtime import ExactCosineIndex, timed_call, validate_embeddings  # noqa: E402


def save_json(root, name, value):
    (root / name).write_text(json_text(value), encoding="utf-8")


def run(args):
    import torch
    from sentence_transformers import SentenceTransformer

    require(not args.output_dir.exists(), f"Use a fresh output directory: {args.output_dir}")
    require(torch.backends.mps.is_built() and torch.backends.mps.is_available(), "MPS unavailable")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    doc_ids, query_ids, documents, queries, qrels = read_inputs(args)

    model = SentenceTransformer(str(args.model_dir), device="mps", local_files_only=True,
                                 trust_remote_code=False, similarity_fn_name="cosine",
                                 model_kwargs={"attn_implementation": "eager"})
    model.float().eval()
    require(model.max_seq_length == 256 and model.get_sentence_embedding_dimension() == 384,
            "Model length or embedding dimension differs")
    require(str(model.device).startswith("mps") and model.dtype == torch.float32, "Model device/dtype differs")

    def encode(texts, batch_size):
        return model.encode(texts, batch_size=batch_size, device="mps", show_progress_bar=False,
                             precision="float32", convert_to_numpy=True, normalize_embeddings=True)

    corpus_embeddings, corpus_seconds = timed_call(lambda: encode(documents, 32), synchronize=torch.mps.synchronize)
    validate_embeddings(corpus_embeddings, len(doc_ids), 384)
    index, _ = timed_call(lambda: ExactCosineIndex(corpus_embeddings, doc_ids))

    query_embeddings, query_seconds = timed_call(lambda: encode(queries, 32), synchronize=torch.mps.synchronize)
    validate_embeddings(query_embeddings, len(query_ids), 384)

    result_run = {}
    for i, qid in enumerate(query_ids):
        (top, _, _), _ = timed_call(lambda i=i: index.search(query_embeddings[i], k=10))
        result_run[qid] = top

    metrics = evaluate(qrels, result_run, query_ids, doc_ids, k=10)

    args.output_dir.mkdir(parents=True)
    save_json(args.output_dir, "metrics.json", metrics)
    save_json(args.output_dir, "top10.json", result_run)
    summary = {
        "label": args.label, "model_dir": str(args.model_dir),
        "corpus_documents": len(doc_ids), "validation_queries": len(query_ids), "validation_qrels": 186,
        "metrics": metrics["metrics"],
        "corpus_encoding_seconds": corpus_seconds, "query_batch_encoding_seconds": query_seconds,
        "optimizer_steps": 0, "scope": "validation-only retrieval evaluation; no training in this script",
    }
    save_json(args.output_dir, "summary.json", summary)
    print(json_text(summary), end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    for name in ("split-lock", "text-policy", "data-audit", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, AssertionError) as exc:
        parser.exit(2, f"Evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
