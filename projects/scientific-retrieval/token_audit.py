# SPDX-License-Identifier: Apache-2.0
"""Offline token audit using cached, pinned MiniLM and SciFact artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

from data_split import digest, json_text
from text_policy import MAX_LENGTH, MODEL_ID, MODEL_REVISION, POLICY, abstract_record, document_text, query_text, require, summarize, token_record

SPLIT_SHA256 = "4bb47f6e5c5da8d87960acbc0c48cfbbf390cbd54cd770e7e271b0d2aef5a624"
DATA_PROVENANCE_SHA256 = "3592b3489ae126cf3119326d66e22f98f2d4594f06c3530f9512bc3e037bfa11"
BATCH_SIZE = 64


def save_json(out, name, value):
    (out / name).write_text(json_text(value), encoding="utf-8")


def save_csv(out, name, rows):
    with (out / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def full_tokens(tokenizer, texts, special=True):
    return tokenizer(texts, add_special_tokens=special, truncation=False, padding=False,
                     return_attention_mask=False, return_token_type_ids=False, verbose=False)["input_ids"]


def used_tokens(model, texts):
    features = model.tokenize(texts)
    ids = features["input_ids"].tolist()
    masks = features["attention_mask"].tolist()
    require(len(ids) == len(masks) == len(texts), "Tokenized batch size mismatch")
    return [[token for token, mask in zip(row, masks[i]) if mask] for i, row in enumerate(ids)]


def audit(args):
    require(digest(args.split_lock) == SPLIT_SHA256, "Split lock differs from the verified Mac run")
    provenance_path = args.data_audit / "data-provenance.json"
    require(digest(provenance_path) == DATA_PROVENANCE_SHA256, "Data provenance checksum mismatch")
    lock = json.loads(args.split_lock.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))["scifact"]
    require(provenance["revision"] == lock["scifact_revision"], "Data revision mismatch")
    paths = {}
    for key in ("corpus", "queries"):
        relative = f"{key}/{key}-00000-of-00001.parquet"
        paths[key] = Path(provenance["snapshot_path"]) / relative
        require(digest(paths[key]) == provenance["files"][relative]["sha256"], f"Changed {key} parquet")
    require(not args.output_dir.exists(), "Output directory already exists; use a new run directory")
    args.output_dir.mkdir(parents=True)
    out = args.output_dir

    print("1/4 Read corpus and the frozen TRAIN/VALIDATION whitelist only", flush=True)
    import pyarrow.parquet as pq
    ids = sorted(lock["query_ids"]["train"] + lock["query_ids"]["validation"])
    require(len(ids) == len(set(ids)) == 809, "Unexpected query whitelist")
    corpus = sorted(pq.read_table(paths["corpus"], columns=["_id", "title", "text"]).to_pylist(), key=lambda row: row["_id"])
    # Filter before materializing query TEXT; the shared file also contains held-out rows.
    queries = sorted(pq.read_table(paths["queries"], columns=["_id", "text"], filters=[("_id", "in", ids)]).to_pylist(), key=lambda row: row["_id"])
    require(len(corpus) == len({row["_id"] for row in corpus}) == 5183, "Corpus ID/count mismatch")
    require(len(queries) == 809 and {row["_id"] for row in queries} == set(ids), "Query whitelist mismatch")

    print("2/4 Load the pinned ORIGINAL model from local cache on CPU; no forward pass", flush=True)
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    # The smoke test cached PyTorch files, not every ONNX/OpenVINO export.
    # Request only the files used by this pinned model and tokenizer.
    model_files = [
        "config.json",
        "config_sentence_transformers.json",
        "modules.json",
        "sentence_bert_config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "1_Pooling/config.json",
    ]
    snapshot = Path(snapshot_download(
        MODEL_ID, revision=MODEL_REVISION, cache_dir=str(args.cache_dir),
        allow_patterns=model_files, local_files_only=True,
    ))
    require(snapshot.name == MODEL_REVISION, "Unexpected model snapshot revision")
    missing = [name for name in model_files if not (snapshot / name).is_file()]
    require(not missing, f"Required cached PyTorch/tokenizer files are missing: {missing}")
    weights = snapshot / "model.safetensors"
    require(weights.is_file(), "Pinned model.safetensors is absent from the local cache")
    model_hashes = {str(p.relative_to(snapshot)): digest(p) for p in sorted(snapshot.rglob("*"))
                    if p.is_file() and (p.suffix in (".json", ".txt") or p == weights)}
    model = SentenceTransformer(str(snapshot), device="cpu", local_files_only=True,
                                trust_remote_code=False, model_kwargs={"attn_implementation": "eager"})
    model.eval()
    tokenizer = model.tokenizer
    require(model.max_seq_length == MAX_LENGTH, "Unexpected SentenceTransformer max_seq_length")
    require(tokenizer.model_max_length == MAX_LENGTH, "Unexpected tokenizer model_max_length")
    require(tokenizer.truncation_side == "right", "Expected right truncation")
    require(tokenizer.num_special_tokens_to_add(pair=False) == 2, "Expected two BERT special tokens")
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    require(cls_id is not None and sep_id is not None and cls_id != sep_id, "Unexpected special token IDs")
    require(getattr(model, "default_prompt_name", None) is None, "Unexpected default prompt")
    runtime = {
        "model_id": MODEL_ID, "revision": MODEL_REVISION, "snapshot_path": str(snapshot),
        "tokenizer_class": type(tokenizer).__name__, "tokenizer_max_length": tokenizer.model_max_length,
        "sentence_transformer_max_length": model.max_seq_length,
        "special_tokens": {"cls": cls_id, "sep": sep_id, "count": 2},
        "content_token_budget": MAX_LENGTH - 2,
        "tokenizer_do_lower_case": getattr(tokenizer, "do_lower_case", None),
        "tokenizer_is_fast": getattr(tokenizer, "is_fast", None),
        "model_device": str(model.device), "model_dtype": str(model.dtype),
        "tokenization_batch_size": BATCH_SIZE, "forward_passes": 0,
        "cache_allow_patterns": model_files, "local_files_only": True,
        "packages": {p: importlib.metadata.version(p) for p in ("sentence-transformers", "transformers", "tokenizers", "torch", "pyarrow", "huggingface-hub")},
        "model_files_sha256": model_hashes,
    }
    save_json(out, "runtime.json", runtime)
    try:
        source = inspect.getsource(type(model).tokenize) + "\n" + inspect.getsource(type(model[0]).tokenize)
    except (OSError, TypeError, AttributeError) as exc:
        source = f"Optional runtime source inspection unavailable: {exc}\n"
    (out / "runtime-tokenize-source.txt").write_text(
        "Upstream Sentence Transformers, pinned repository baseline; original LICENSE/NOTICE apply.\n\n" + source,
        encoding="utf-8")
    boundary = ["test " * n for n in (253, 254, 255)]
    full, used = full_tokens(tokenizer, boundary), used_tokens(model, boundary)
    boundary_records = [token_record(a, b, cls_id, sep_id) for a, b in zip(full, used)]
    require([row["input_tokens"] for row in boundary_records] == [255, 256, 257], "Boundary fixture token lengths differ")
    require([row["used_tokens"] for row in boundary_records] == [255, 256, 256], "Boundary truncation check failed")
    save_json(out, "real-tokenizer-boundary-check.json", boundary_records)

    print("3/4 Measure every corpus input and compare with SentenceTransformer.tokenize", flush=True)
    documents = []
    for start in range(0, len(corpus), BATCH_SIZE):
        batch = corpus[start : start + BATCH_SIZE]
        texts = [document_text(row["title"], row["text"]) for row in batch]
        full, used = full_tokens(tokenizer, texts), used_tokens(model, texts)
        title_ids = full_tokens(tokenizer, [row["title"].strip() for row in batch], special=False)
        abstract_ids = full_tokens(tokenizer, [row["text"].strip() for row in batch], special=False)
        for i, row in enumerate(batch):
            documents.append({"document_id": row["_id"],
                              **token_record(full[i], used[i], cls_id, sep_id),
                              **abstract_record(title_ids[i], abstract_ids[i], full[i], cls_id, sep_id),
                              "input_text_sha256": hashlib.sha256(texts[i].encode("utf-8")).hexdigest()})
    save_csv(out, "document-token-lengths.csv", documents)

    print("4/4 Measure only the frozen train/validation queries and write reports", flush=True)
    query_records = []
    train_ids = set(lock["query_ids"]["train"])
    for start in range(0, len(queries), BATCH_SIZE):
        batch = queries[start : start + BATCH_SIZE]
        texts = [query_text(row["text"]) for row in batch]
        full, used = full_tokens(tokenizer, texts), used_tokens(model, texts)
        for i, row in enumerate(batch):
            query_records.append({"query_id": row["_id"], "split": "train" if row["_id"] in train_ids else "validation",
                                  **token_record(full[i], used[i], cls_id, sep_id),
                                  "input_text_sha256": hashlib.sha256(texts[i].encode("utf-8")).hexdigest()})
    save_csv(out, "query-token-lengths.csv", query_records)
    summary = {"status": "passed", "scope": "Real tokenizer audit; no retrieval evaluation or training",
               "model_id": MODEL_ID, "model_revision": MODEL_REVISION, "max_sequence_length": MAX_LENGTH,
               "special_tokens_per_input": 2, "content_token_budget": MAX_LENGTH - 2,
               "corpus": summarize(documents),
               "train_queries": summarize([r for r in query_records if r["split"] == "train"]),
               "validation_queries": summarize([r for r in query_records if r["split"] == "validation"]),
               "tokenization_consistency_checks": len(documents) + len(query_records),
               "real_tokenizer_boundary_checks": len(boundary_records),
               "test_qrels_read": False, "official_test_query_text_loaded": False,
               "split_lock_sha256": digest(args.split_lock), "data_provenance_sha256": digest(provenance_path),
               "limitation": "Token truncation is not a measurement of evidence lost or retrieval quality."}
    require(digest(weights) == model_hashes["model.safetensors"], "Cached model weights changed")
    summary["cached_weights_unchanged"] = True
    save_json(out, "text-policy.json", POLICY)
    save_json(out, "token-audit.json", summary)
    save_json(out, "longest-input-ids.json", sorted(documents, key=lambda r: (-r["input_tokens"], r["document_id"]))[:10])
    print(json_text(summary), end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-lock", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit(args)
    except (OSError, ValueError, KeyError, TypeError, ImportError) as exc:
        parser.exit(2, f"Token audit error: {exc}\n")


if __name__ == "__main__":
    main()
