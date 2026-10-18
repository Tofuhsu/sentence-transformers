# SPDX-License-Identifier: Apache-2.0
"""Original MiniLM validation retrieval on MPS, using pinned offline artifacts."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np

from data_split import digest, json_text
from retrieval_metrics import evaluate, load_json, require
from retrieval_runtime import ExactCosineIndex, latency_summary, timed_call, validate_embeddings
from text_policy import MAX_LENGTH, MODEL_ID, MODEL_REVISION, POLICY, document_text, query_text

CONFIG = {
    "schema_version": 1, "variant": "A_original", "evaluation_split": "custom_validation",
    "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "encoding_device": "mps", "search_device": "cpu", "dtype": "float32",
    "attention_implementation": "eager", "max_sequence_length": MAX_LENGTH,
    "embedding_dimension": 384, "normalize_embeddings": True, "prompt": None,
    "encoding_batch_size": 32, "latency_batch_size": 1,
    "corpus_warmup_batches": 3, "query_warmup_batches": 3,
    "latency_warmup_queries": 3, "latency_repetitions_per_query": 1,
    "seed": 42, "cutoff": 10, "cpu_threads": 1,
    "search": "exact float32 dot product; score descending, document ID ascending",
    "cpu_mps_rtol": 1e-3, "cpu_mps_atol": 1e-4,
    "optimizer_steps": 0, "official_test_access": False,
}
MODEL_FILES = ["config.json", "config_sentence_transformers.json", "modules.json",
               "sentence_bert_config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "vocab.txt",
               "1_Pooling/config.json"]
SPLIT_SHA = "4bb47f6e5c5da8d87960acbc0c48cfbbf390cbd54cd770e7e271b0d2aef5a624"
PROVENANCE_SHA = "3592b3489ae126cf3119326d66e22f98f2d4594f06c3530f9512bc3e037bfa11"
POSITIVES_SHA = "34b1b681cfdec81b451cacf4a4ca737623bb3b6224e751bf4a0ed5bb6bce7106"


def save_json(root, name, value):
    (root / name).write_text(json_text(value), encoding="utf-8")


def save_csv(root, name, rows):
    with (root / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_inputs(args):
    require(digest(args.split_lock) == SPLIT_SHA, "Frozen split checksum differs")
    require(load_json(args.text_policy) == POLICY, "Text policy differs from the audited policy")
    provenance_path = args.data_audit / "data-provenance.json"
    positives_path = args.data_audit / "train-positive-docs.json"
    require(digest(provenance_path) == PROVENANCE_SHA, "Data provenance checksum differs")
    require(digest(positives_path) == POSITIVES_SHA, "Original training positive map checksum differs")
    lock, provenance = load_json(args.split_lock), load_json(provenance_path)["scifact"]
    require(provenance["revision"] == lock["scifact_revision"], "Data revision differs")
    positives = load_json(positives_path)
    validation_ids = sorted(lock["query_ids"]["validation"])
    require(len(validation_ids) == len(set(validation_ids)) == 160, "Validation query count differs")
    require(set(positives) == set(lock["query_ids"]["train"] + validation_ids), "Positive-map query IDs differ")
    paths = {}
    for kind in ("corpus", "queries"):
        relative = f"{kind}/{kind}-00000-of-00001.parquet"
        paths[kind] = Path(provenance["snapshot_path"]) / relative
        require(digest(paths[kind]) == provenance["files"][relative]["sha256"], f"Changed {kind} parquet")
    import pyarrow.parquet as pq
    corpus = sorted(pq.read_table(paths["corpus"], columns=["_id", "title", "text"]).to_pylist(), key=lambda r: r["_id"])
    # Apply the frozen whitelist before materializing any query text.
    queries = sorted(pq.read_table(paths["queries"], columns=["_id", "text"],
                                  filters=[("_id", "in", validation_ids)]).to_pylist(), key=lambda r: r["_id"])
    doc_ids = [r["_id"] for r in corpus]
    require(len(doc_ids) == len(set(doc_ids)) == 5183, "Corpus count or IDs differ")
    require([r["_id"] for r in queries] == validation_ids, "Query text did not match the validation whitelist")
    qrels = {qid: {did: 1 for did in positives[qid]} for qid in validation_ids}
    require(sum(map(len, qrels.values())) == 186, "Validation qrel count differs")
    require(all(rels and set(rels) <= set(doc_ids) for rels in qrels.values()), "Invalid validation positives")
    texts = [document_text(r["title"], r["text"]) for r in corpus]
    claims = [query_text(r["text"]) for r in queries]
    return doc_ids, validation_ids, texts, claims, qrels


def run(args):
    require(not args.output_dir.exists() and not args.artifact_dir.exists(), "Output/artifact directory exists; use a fresh run")
    for variable, expected in (("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
                               ("PYTORCH_ENABLE_MPS_FALLBACK", "0"), ("PYTORCH_MPS_FAST_MATH", "0")):
        require(os.environ.get(variable) == expected, f"Set {variable}={expected} before running")
    print("1/6 Verify pinned inputs; load shared corpus and VALIDATION query text only", flush=True)
    doc_ids, query_ids, documents, queries, qrels = read_inputs(args)
    args.output_dir.mkdir(parents=True)
    args.artifact_dir.mkdir(parents=True)
    out, artifacts = args.output_dir, args.artifact_dir
    save_json(out, "config.json", CONFIG)
    save_json(out, "validation-qrels.json", qrels)
    save_json(out, "validation-query-ids.json", query_ids)
    save_json(out, "corpus-ids.json", doc_ids)
    save_json(out, "input-provenance.json", {
        "split_lock_sha256": digest(args.split_lock), "text_policy_sha256": digest(args.text_policy),
        "data_provenance_sha256": PROVENANCE_SHA, "original_positive_map_sha256": POSITIVES_SHA,
        "corpus_documents": len(doc_ids), "validation_queries": len(query_ids), "validation_qrels": 186,
        "query_text_whitelist": "frozen validation IDs only", "test_qrels_read": False,
        "official_test_query_text_loaded": False,
    })

    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from threadpoolctl import threadpool_info

    require(torch.backends.mps.is_built() and torch.backends.mps.is_available(), "MPS is unavailable; stopped without CPU fallback")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(CONFIG["seed"])
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    token_runtime = load_json(args.token_audit / "results/runtime.json")
    require(token_runtime["model_id"] == MODEL_ID and token_runtime["revision"] == MODEL_REVISION,
            "Token-audit model identity differs")
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(args.cache_dir),
                                     allow_patterns=MODEL_FILES, local_files_only=True))
    require(snapshot.name == MODEL_REVISION, "Model snapshot revision differs")
    for name, expected in token_runtime["model_files_sha256"].items():
        require(digest(snapshot / name) == expected, f"Cached model file changed since token audit: {name}")
    model = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True,
                                trust_remote_code=False, model_kwargs={"attn_implementation": "eager"})
    model.float().eval()
    require(model.max_seq_length == 256 and model.get_sentence_embedding_dimension() == 384,
            "Model length or embedding dimension differs")
    require(model.default_prompt_name is None, "Unexpected default prompt")

    def encode(texts, batch_size, device, progress=False):
        return model.encode(texts, batch_size=batch_size, device=device, show_progress_bar=progress,
                            precision="float32", convert_to_numpy=True, normalize_embeddings=True)

    print("2/6 Compare CPU and MPS embeddings on 8 documents and 2 validation queries", flush=True)
    smoke_texts = documents[:8] + queries[:2]
    cpu = encode(smoke_texts, 4, "cpu")
    validate_embeddings(cpu, 10, 384)
    model.to("mps")
    mps, _ = timed_call(lambda: encode(smoke_texts, 4, "mps"), synchronize=torch.mps.synchronize)
    validate_embeddings(mps, 10, 384)
    smoke = {"inputs": 10, "document_ids": doc_ids[:8], "query_ids": query_ids[:2],
             "rtol": CONFIG["cpu_mps_rtol"], "atol": CONFIG["cpu_mps_atol"],
             "max_abs_difference": float(np.max(np.abs(cpu - mps))),
             "passed": bool(np.allclose(cpu, mps, rtol=CONFIG["cpu_mps_rtol"], atol=CONFIG["cpu_mps_atol"]))}
    save_json(out, "cpu-mps-smoke.json", smoke)
    require(smoke["passed"], "CPU/MPS embedding check failed; inspect cpu-mps-smoke.json")
    require(str(model.device).startswith("mps") and model.dtype == torch.float32, "Unexpected model device or dtype")
    save_json(out, "runtime.json", {
        "python": sys.version, "platform": platform.platform(), "device": str(model.device),
        "dtype": str(model.dtype), "model_snapshot": str(snapshot),
        "requested_attention_implementation": "eager", "encode_signature": str(inspect.signature(model.encode)),
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "weights_bytes": (snapshot / "model.safetensors").stat().st_size,
        "model_files_sha256": token_runtime["model_files_sha256"],
        "packages": {name: importlib.metadata.version(name) for name in
                     ("sentence-transformers", "torch", "transformers", "numpy", "pyarrow", "scikit-learn", "huggingface-hub")},
        "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads(),
        "threadpools": threadpool_info(),
        "environment": {name: os.environ.get(name) for name in ("PYTORCH_ENABLE_MPS_FALLBACK", "PYTORCH_MPS_FAST_MATH",
                        "HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM", "OMP_NUM_THREADS",
                        "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
    })
    (out / "model-architecture.txt").write_text(str(model) + "\n", encoding="utf-8")

    batch = CONFIG["encoding_batch_size"]
    print("3/6 Warm up and encode all 5,183 corpus documents on MPS", flush=True)
    for _ in range(CONFIG["corpus_warmup_batches"]):
        encode(documents[:batch], batch, "mps")
    corpus_embeddings, corpus_seconds = timed_call(
        lambda: encode(documents, batch, "mps", True), synchronize=torch.mps.synchronize)
    validate_embeddings(corpus_embeddings, len(doc_ids), 384)
    np.save(artifacts / "corpus-embeddings.npy", corpus_embeddings, allow_pickle=False)
    index, index_seconds = timed_call(lambda: ExactCosineIndex(corpus_embeddings, doc_ids))

    print("4/6 Encode the 160 validation queries and perform exact full-corpus search", flush=True)
    for _ in range(CONFIG["query_warmup_batches"]):
        encode(queries[:batch], batch, "mps")
    query_embeddings, query_seconds = timed_call(
        lambda: encode(queries, batch, "mps", True), synchronize=torch.mps.synchronize)
    validate_embeddings(query_embeddings, len(query_ids), 384)
    np.save(artifacts / "validation-embeddings.npy", query_embeddings, allow_pickle=False)
    result_run, all_scores, all_orders, search_seconds = {}, [], [], []
    for i, qid in enumerate(query_ids):
        (top, scores, order), elapsed = timed_call(lambda i=i: index.search(query_embeddings[i], k=10))
        result_run[qid] = top
        all_scores.append(scores)
        all_orders.append(order)
        search_seconds.append(elapsed)
    scores_matrix = np.asarray(all_scores, dtype=np.float32)
    np.save(artifacts / "validation-scores.npy", scores_matrix, allow_pickle=False)
    save_json(artifacts, "corpus-ids.json", doc_ids)
    save_json(artifacts, "validation-query-ids.json", query_ids)
    save_json(artifacts, "config.json", CONFIG)
    metrics = evaluate(qrels, result_run, query_ids, doc_ids, k=10)
    save_json(out, "top10.json", result_run)
    save_json(out, "metrics.json", metrics)

    # Independent metric check on the real data. Rank scores encode our stated
    # tie rule; this is a formula check, not an independent retrieval model.
    from sklearn.metrics import ndcg_score, recall_score
    by_query = {row["query_id"]: row for row in metrics["per_query"]}
    references, per_query_rows = [], []
    for i, qid in enumerate(query_ids):
        labels = np.array([int(did in qrels[qid]) for did in doc_ids])
        rank_scores = np.empty(len(doc_ids), dtype=np.float64)
        rank_scores[all_orders[i]] = np.arange(len(doc_ids), 0, -1)
        selected = np.zeros(len(doc_ids), dtype=int)
        selected[all_orders[i][:10]] = 1
        ref_ndcg = float(ndcg_score(labels[None, :], rank_scores[None, :], k=10))
        ref_recall = float(recall_score(labels, selected, zero_division=0))
        own = by_query[qid]
        require(abs(ref_ndcg - own["nDCG@10"]) <= 1e-12 and abs(ref_recall - own["Recall@10"]) <= 1e-12,
                f"Reference metric mismatch for query {qid}")
        references.append({"query_id": qid, "reference_ndcg_at_10": ref_ndcg, "reference_recall_at_10": ref_recall})
        ranks = {doc_ids[j]: rank for rank, j in enumerate(all_orders[i], 1) if doc_ids[j] in qrels[qid]}
        per_query_rows.append({"query_id": qid, "ndcg_at_10": own["nDCG@10"], "recall_at_10": own["Recall@10"],
                               "known_positives": len(qrels[qid]), "relevant_retrieved": own["relevant_retrieved"],
                               "first_positive_rank": min(ranks.values()), "positive_ranks": json.dumps(ranks, sort_keys=True)})
    save_json(out, "reference-metrics.json", {"status": "passed", "queries": len(references),
              "scope": "metric formulas on the same real rankings, with deterministic tie resolution",
              "absolute_tolerance": 1e-12, "per_query": references})
    save_csv(out, "per-query.csv", per_query_rows)
    ordered_cases = sorted(per_query_rows, key=lambda row: (row["ndcg_at_10"], row["query_id"]))
    save_json(out, "review-case-ids.json", {"scope": "quantitative ranking cases; qualitative review pending",
              "lowest_ndcg_queries": ordered_cases[:5], "highest_ndcg_queries": ordered_cases[-5:]})

    print("5/6 Measure warmed single-query encoding and CPU search, once per validation query", flush=True)
    for text in queries[:CONFIG["latency_warmup_queries"]]:
        index.search(encode([text], 1, "mps")[0], k=10)
    order = list(range(len(query_ids)))
    random.Random(CONFIG["seed"]).shuffle(order)
    latency_rows = []
    for position, i in enumerate(order):
        torch.mps.synchronize()
        start = time.perf_counter()
        encoded, encode_seconds = timed_call(lambda i=i: encode([queries[i]], 1, "mps"), synchronize=torch.mps.synchronize)
        (online_top, _, _), retrieve_seconds = timed_call(lambda: index.search(encoded[0], k=10))
        total_seconds = time.perf_counter() - start
        difference = float(np.max(np.abs(encoded[0] - query_embeddings[i])))
        require(bool(np.allclose(encoded[0], query_embeddings[i], rtol=1e-3, atol=1e-4)),
                "Batch-1 and batch-32 embeddings differ beyond the recorded tolerance")
        latency_rows.append({"position": position, "query_id": query_ids[i], "query_encode_ms": encode_seconds * 1000,
                             "cpu_search_ms": retrieve_seconds * 1000, "end_to_end_ms": total_seconds * 1000,
                             "batch_embedding_max_abs_difference": difference,
                             "same_top10_as_batch_encoding": [r["document_id"] for r in online_top] == [r["document_id"] for r in result_run[query_ids[i]]]})
    save_csv(out, "latency-samples.csv", latency_rows)
    timings = {
        "corpus_encoding_seconds": corpus_seconds, "corpus_documents": len(doc_ids), "corpus_encoding_passes": 1,
        "index_build_seconds": index_seconds, "query_batch_encoding_seconds": query_seconds,
        "query_batch_count": len(query_ids), "encoding_batch_size": batch,
        "quality_evaluation_cpu_search_seconds": sum(search_seconds),
        "online_latency": {name: latency_summary([r[name] for r in latency_rows])
                           for name in ("query_encode_ms", "cpu_search_ms", "end_to_end_ms")},
        "online_batch_size": 1, "online_repetitions_per_query": 1,
        "online_top10_differences_from_batch_encoding": sum(not r["same_top10_as_batch_encoding"] for r in latency_rows),
        "encoding_scope": "tokenization, model forward, normalization, and CPU NumPy transfer; excludes model loading and file writes",
        "cpu_search_scope": "query validation, float32 dot product over entire corpus, full tie-resolved sorting, and top-10 materialization",
        "online_scope": "warmed in-process text-to-top10 pipeline; includes timing wrapper overhead; no network or server queue",
        "limitation": "One local run. Latency distribution is across different query texts, not repeated-trial uncertainty.",
    }
    save_json(out, "timings.json", timings)

    print("6/6 Verify original weights and save reproducible artifacts and reports", flush=True)
    for name, expected in token_runtime["model_files_sha256"].items():
        require(digest(snapshot / name) == expected, f"Cached model file changed during baseline: {name}")
    artifact_hashes = {p.name: digest(p) for p in sorted(artifacts.iterdir()) if p.is_file()}
    save_json(out, "artifacts.json", {"directory": str(artifacts), "sha256": artifact_hashes,
                                      "corpus_embedding_bytes": corpus_embeddings.nbytes})
    summary = {"status": "passed", "scope": "Original MiniLM on frozen custom validation; no fine-tuning",
               "variant": "A_original", "model_revision": MODEL_REVISION,
               "corpus_documents": len(doc_ids), "validation_queries": len(query_ids), "validation_qrels": 186,
               "metrics": metrics["metrics"], "cpu_mps_smoke_passed": True,
               "reference_metric_queries_passed": len(references), "optimizer_steps": 0,
               "original_cached_model_files_unchanged": True, "test_qrels_read": False,
               "official_test_query_text_loaded": False,
               "corpus_encoding_seconds": corpus_seconds, "query_batch_encoding_seconds": query_seconds,
               "online_latency_samples": len(latency_rows), "artifact_directory": str(artifacts),
               "limitations": ["Custom grouped validation, not official BEIR test scores.",
                               "Unjudged documents are not proven irrelevant.",
                               "256-token truncation affects 3681 of 5183 corpus documents.",
                               "Pretraining overlap is not excluded; the model card lists scientific sources.",
                               "Fixed seeds do not guarantee identical scores or timing across hardware."]}
    save_json(out, "baseline-summary.json", summary)
    print(json_text(summary), end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-lock", "text-policy", "data-audit", "token-audit", "cache-dir", "output-dir", "artifact-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1):
            run(args)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, AssertionError) as exc:
        parser.exit(2, f"Retrieval baseline error: {exc}\n")


if __name__ == "__main__":
    main()
