# SPDX-License-Identifier: Apache-2.0
"""One-time official BeIR/SciFact TEST retrieval evaluation. Opened only after
the B/C protocol and shared config were frozen from pilot + validation results
(see ~/scientific-retrieval-evidence/official-test-open-20261018T210000Z.claude01/).
No training and no further tuning happens here or after this script runs."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_split import digest, json_text  # noqa: E402
from retrieval_metrics import evaluate, load_json, require  # noqa: E402
from retrieval_runtime import ExactCosineIndex, timed_call, validate_embeddings  # noqa: E402
from text_policy import document_text, query_text  # noqa: E402

PROVENANCE_SHA = "3592b3489ae126cf3119326d66e22f98f2d4594f06c3530f9512bc3e037bfa11"


def save_json(root, name, value):
    (root / name).write_text(json_text(value), encoding="utf-8")


def read_inputs(args):
    require(digest(args.split_lock) == "4bb47f6e5c5da8d87960acbc0c48cfbbf390cbd54cd770e7e271b0d2aef5a624",
            "Frozen split checksum differs")
    provenance_path = args.data_audit / "data-provenance.json"
    require(digest(provenance_path) == PROVENANCE_SHA, "Data provenance checksum differs")
    provenance = load_json(provenance_path)["scifact"]
    lock = load_json(args.split_lock)
    require(provenance["revision"] == lock["scifact_revision"], "Data revision differs")

    test_open = load_json(args.test_open / "TEST-OPEN.json")
    qrels_raw = load_json(args.test_open / "data/official-test-qrels.json")
    test_ids = sorted(qrels_raw)
    require(len(test_ids) == test_open["test_queries"] == 300, "Official test query count differs")
    require(sum(map(len, qrels_raw.values())) == test_open["test_qrel_rows"] == 339,
            "Official test qrel row count differs")
    train_val_ids = set(lock["query_ids"]["train"]) | set(lock["query_ids"]["validation"])
    require(not (set(test_ids) & train_val_ids), "Official test query IDs overlap train/validation")

    paths = {}
    for kind in ("corpus", "queries"):
        relative = f"{kind}/{kind}-00000-of-00001.parquet"
        paths[kind] = Path(provenance["snapshot_path"]) / relative
        require(digest(paths[kind]) == provenance["files"][relative]["sha256"], f"Changed {kind} parquet")
    import pyarrow.parquet as pq
    corpus = sorted(pq.read_table(paths["corpus"], columns=["_id", "title", "text"]).to_pylist(), key=lambda r: r["_id"])
    queries = sorted(pq.read_table(paths["queries"], columns=["_id", "text"],
                                  filters=[("_id", "in", test_ids)]).to_pylist(), key=lambda r: r["_id"])
    doc_ids = [r["_id"] for r in corpus]
    require(len(doc_ids) == len(set(doc_ids)) == 5183, "Corpus count or IDs differ")
    require([r["_id"] for r in queries] == test_ids, "Query text did not match the official test whitelist")
    qrels = {qid: {did: 1 for did in qrels_raw[qid]} for qid in test_ids}
    require(all(rels and set(rels) <= set(doc_ids) for rels in qrels.values()), "Invalid official test positives")
    texts = [document_text(r["title"], r["text"]) for r in corpus]
    claims = [query_text(r["text"]) for r in queries]
    return doc_ids, test_ids, texts, claims, qrels


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
    require(model.max_seq_length == 256, "Model max sequence length differs")
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
        "label": args.label, "model_dir": str(args.model_dir), "split": "official_beir_scifact_test",
        "corpus_documents": len(doc_ids), "test_queries": len(query_ids), "test_qrels": 339,
        "metrics": metrics["metrics"],
        "corpus_encoding_seconds": corpus_seconds, "query_batch_encoding_seconds": query_seconds,
        "optimizer_steps": 0, "scope": "single official-test retrieval evaluation; no training in this script",
    }
    save_json(args.output_dir, "summary.json", summary)
    print(json_text(summary), end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    for name in ("split-lock", "data-audit", "test-open", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, AssertionError) as exc:
        parser.exit(2, f"Official test evaluation error: {exc}\n")


if __name__ == "__main__":
    main()
