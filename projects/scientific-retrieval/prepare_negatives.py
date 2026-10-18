# SPDX-License-Identifier: Apache-2.0
"""Prepare paired B/C training negatives with the pinned upstream miner; no training."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import platform
import random
import sys

import numpy as np

from data_split import digest, json_text
from negative_sampling import (
    candidate_pool, check_selection, mining_inputs, parse_mined_rows,
    require_unprompted_model, training_triplets,
)
from retrieval_metrics import load_json, require
from retrieval_runtime import timed_call, validate_embeddings
from text_policy import MAX_LENGTH, MODEL_ID, MODEL_REVISION, POLICY, document_text, query_text

SPLIT_SHA = "4bb47f6e5c5da8d87960acbc0c48cfbbf390cbd54cd770e7e271b0d2aef5a624"
PROVENANCE_SHA = "3592b3489ae126cf3119326d66e22f98f2d4594f06c3530f9512bc3e037bfa11"
POSITIVES_SHA = "34b1b681cfdec81b451cacf4a4ca737623bb3b6224e751bf4a0ed5bb6bce7106"
EXCLUSIONS_SHA = "661e75fce6720b68c5892d3fbc0795a6d7b73ff33ad115b46e76fc1ac149ad65"
MINER_SHA = "e3d36d889491ebaba8087de7c5e9d424be61d07db2cebf6d3dcf589e1d3b93cb"
MODEL_FILES = ["config.json", "config_sentence_transformers.json", "modules.json",
               "sentence_bert_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
               "special_tokens_map.json", "vocab.txt", "1_Pooling/config.json"]
NEGATIVE_POLICY = {
    "schema_version": 1, "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
    "mining_function": "sentence_transformers.util.hard_negatives.mine_hard_negatives",
    "query_scope": "frozen_train_only", "candidate_corpus_documents": 5183,
    "range_min": 0, "range_max": 5183, "absolute_margin": 0.05, "relative_margin": None,
    "min_score": None, "max_score": None, "num_negatives_per_unique_query_text": 1,
    "B_sampling_strategy": "random", "C_sampling_strategy": "top", "sampling_seed": 1729,
    "encoding_batch_size": 32, "score_chunk_size": 16384, "encoding_device": "mps",
    "dtype": "float32", "attention_implementation": "eager", "max_sequence_length": 256,
    "query_prompt": "", "corpus_prompt": "", "use_faiss": False, "use_multi_process": False,
    "exclusions": "all known positives plus frozen exact-normalized duplicate-query positive union",
    "gap_reference": "minimum cosine score over the complete exclusion positive set",
    "training_rows": "one triplet per original train qrel; exclusion-only positives are not training labels",
    "review_seed": 2718, "review_random_queries": 10,
    "review_high_score_queries": 5, "review_small_pool_queries": 5,
    "training_executed": False, "review_required_before_training": True,
}


def save(root, name, value):
    (root / name).write_text(json_text(value), encoding="utf-8")


def save_jsonl(root, name, rows):
    with (root / name).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def read_inputs(args):
    for path, expected in [(args.split_lock, SPLIT_SHA),
                           (args.data_audit / "data-provenance.json", PROVENANCE_SHA),
                           (args.data_audit / "train-positive-docs.json", POSITIVES_SHA),
                           (args.exclusions, EXCLUSIONS_SHA)]:
        require(digest(path) == expected, f"Pinned input checksum differs: {path.name}")
    require(load_json(args.text_policy) == POLICY, "Text policy differs")
    require(load_json(args.negative_policy) == NEGATIVE_POLICY, "Negative sampling policy differs")
    lock = load_json(args.split_lock)
    train_ids = sorted(lock["query_ids"]["train"])
    require(len(train_ids) == len(set(train_ids)) == 649, "Frozen training query count differs")
    # The saved official-TRAIN map also contains the custom validation portion.
    # Whitelist immediately; those other labels do not enter mining or scoring.
    original_map = load_json(args.data_audit / "train-positive-docs.json")
    positives = {qid: original_map[qid] for qid in train_ids}
    del original_map
    require(sum(map(len, positives.values())) == 733, "Original train pair count differs")
    provenance = load_json(args.data_audit / "data-provenance.json")["scifact"]
    require(provenance["revision"] == lock["scifact_revision"], "Corpus/query revision differs")
    paths = {}
    for kind in ("corpus", "queries"):
        relative = f"{kind}/{kind}-00000-of-00001.parquet"
        paths[kind] = Path(provenance["snapshot_path"]) / relative
        require(digest(paths[kind]) == provenance["files"][relative]["sha256"], f"Changed {kind} data")
    import pyarrow.parquet as pq
    corpus = sorted(pq.read_table(paths["corpus"], columns=["_id", "title", "text"]).to_pylist(), key=lambda r: r["_id"])
    queries = sorted(pq.read_table(paths["queries"], columns=["_id", "text"],
                                  filters=[("_id", "in", train_ids)]).to_pylist(), key=lambda r: r["_id"])
    require(len(corpus) == len({r["_id"] for r in corpus}) == 5183, "Corpus IDs/count differ")
    require([r["_id"] for r in queries] == train_ids, "Query rows differ from TRAIN whitelist")
    documents = {r["_id"]: document_text(r["title"], r["text"]) for r in corpus}
    claims = {r["_id"]: query_text(r["text"]) for r in queries}
    exclusions = load_json(args.exclusions)
    pairs, anchors = mining_inputs(documents, claims, positives, exclusions)
    return documents, claims, positives, exclusions, pairs, anchors


def verify_framework_miner(mine, Dataset, torch, device):
    """Run the actual miner on known synthetic vectors before any model mining."""
    vectors = {"query": [1., 0.], "p1": [.9, (1-.9**2)**.5], "p2": [.5, (1-.5**2)**.5],
               "too_close": [.6, (1-.6**2)**.5], "eligible": [.4, (1-.4**2)**.5]}

    class KnownVectors:
        def __init__(self):
            self.device = torch.device(device)

        def encode_document(self, texts, **kwargs):
            return np.array([vectors[t] for t in texts], dtype=np.float32)

        encode_query = encode_document

        def similarity(self, a, b):
            return torch.nn.functional.normalize(torch.as_tensor(a), dim=1) @ torch.nn.functional.normalize(torch.as_tensor(b), dim=1).T

        def similarity_pairwise(self, a, b):
            return torch.nn.functional.cosine_similarity(torch.as_tensor(a), torch.as_tensor(b), dim=1)

    result = {}
    for strategy in ("random", "top"):
        random.seed(1729)
        mined = mine(Dataset.from_dict({"anchor": ["query", "query"], "positive": ["p1", "p2"]}),
                     KnownVectors(), anchor_column_name="anchor", positive_column_name="positive",
                     corpus=["p1", "p2", "too_close", "eligible"], range_min=0, range_max=4,
                     num_negatives=1, absolute_margin=.05, sampling_strategy=strategy,
                     output_format="triplet", output_scores=True, use_faiss=False, verbose=False)
        rows = list(mined)
        require(len(rows) == 2 and {r["positive"] for r in rows} == {"p1", "p2"} and
                all(r["negative"] == "eligible" for r in rows),
                "Actual framework miner failed its synthetic multi-positive/gap/full-window fixture")
        result[strategy] = {"status": "passed", "output_triplets": len(rows), "selected_negative": "eligible"}
    return {"status": "passed", "scope": "actual pinned miner, synthetic vectors",
            "selection_device": device, "strategies": result,
            "real_model_quality_measured": False}


def verify_prompt_routes(model, queries, documents, out, torch):
    """Compare baseline encode and the miner's two explicit empty-prompt routes."""
    observed = {"prompts": model.prompts, "default_prompt_name": model.default_prompt_name}
    report = {"status": "checking", "observed_configuration": observed,
              "requested_query_prompt": "", "requested_document_prompt": "",
              "scope": "Four TRAIN/corpus inputs; no retrieval evaluation or training"}
    # Preserve the actual configuration even if validation fails.
    save(out, "prompt-check.json", report)
    require_unprompted_model(model.prompts, model.default_prompt_name)
    original_prompts = dict(model.prompts)
    query_ids, document_ids = sorted(queries)[:2], sorted(documents)[:2]
    texts = [queries[qid] for qid in query_ids] + [documents[did] for did in document_ids]
    kwargs = dict(batch_size=4, show_progress_bar=False, convert_to_numpy=True,
                  normalize_embeddings=True)
    reference = model.encode(texts, prompt=None, **kwargs)
    validate_embeddings(reference, 4, 384)
    differences = {}
    for name, encoder in [("encode_query", model.encode_query), ("encode_document", model.encode_document)]:
        encoded = encoder(texts, prompt="", **kwargs)
        torch.mps.synchronize()
        validate_embeddings(encoded, 4, 384)
        differences[name] = float(np.max(np.abs(encoded - reference)))
        report.update({"query_ids": query_ids, "document_ids": document_ids,
                       "rtol": 1e-5, "atol": 1e-6, "max_abs_difference": differences})
        save(out, "prompt-check.json", report)
        require(bool(np.allclose(encoded, reference, rtol=1e-5, atol=1e-6)),
                f"Empty-prompt {name} differs from the unprompted baseline encode route")
    require(model.prompts == original_prompts and model.default_prompt_name is None,
            "Model prompt configuration changed during the check")
    report.update({"status": "passed", "configuration_unchanged": True,
                   "device": str(model.device), "dtype": str(model.dtype)})
    save(out, "prompt-check.json", report)


def run(args):
    require(not args.output_dir.exists() and not args.artifact_dir.exists(), "Use fresh output and artifact directories")
    for key, expected in [("HF_HUB_OFFLINE", "1"), ("TRANSFORMERS_OFFLINE", "1"),
                          ("PYTORCH_ENABLE_MPS_FALLBACK", "0"), ("PYTORCH_MPS_FAST_MATH", "0")]:
        require(os.environ.get(key) == expected, f"Set {key}={expected} before execution")
    print("1/6 Verify pins and select only the 649 frozen TRAIN queries", flush=True)
    documents, queries, positives, exclusions, pairs, anchors = read_inputs(args)
    out, artifacts = args.output_dir, args.artifact_dir
    out.mkdir(parents=True); artifacts.mkdir(parents=True)
    save(out, "negative-policy.json", NEGATIVE_POLICY)
    save(out, "train-query-ids.json", sorted(queries))
    save(out, "train-positive-docs.json", positives)
    save(out, "train-negative-exclusions.json", exclusions)
    save(out, "corpus-ids.json", list(documents))
    save(out, "input-provenance.json", {
        "split_lock_sha256": SPLIT_SHA, "data_provenance_sha256": PROVENANCE_SHA,
        "original_official_train_positive_map_sha256": POSITIVES_SHA, "exclusions_sha256": EXCLUSIONS_SHA,
        "official_train_map_contains_custom_validation_labels": True,
        "positive_map_selection": "Only frozen train IDs are retained and passed to mining",
        "validation_labels_used_for_mining": False, "validation_query_text_loaded": False,
        "official_test_query_text_loaded": False, "test_qrels_read": False,
        "unique_raw_query_texts": len(anchors), "miner_exclusion_pairs": len(pairs),
        "original_training_pairs": sum(map(len, positives.values())),
    })
    import torch
    from datasets import Dataset
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.util.hard_negatives import mine_hard_negatives

    require(digest(Path(inspect.getsourcefile(mine_hard_negatives))) == MINER_SHA, "Pinned miner source differs")
    require(torch.backends.mps.is_built() and torch.backends.mps.is_available(), "MPS unavailable; stopped")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.manual_seed(1729); np.random.seed(1729); random.seed(1729)
    print("2/6 Check the real framework miner on small synthetic vectors", flush=True)
    fixtures = {"status": "checking", "devices": {}, "real_model_quality_measured": False}
    for device in ("cpu", "mps"):
        fixtures["devices"][device] = verify_framework_miner(mine_hard_negatives, Dataset, torch, device)
        save(out, "framework-miner-fixture.json", fixtures)
    fixtures["status"] = "passed"
    save(out, "framework-miner-fixture.json", fixtures)
    token = load_json(args.token_audit / "results/runtime.json")
    require(token["model_id"] == MODEL_ID and token["revision"] == MODEL_REVISION, "Token-audit model pin differs")
    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(args.cache_dir),
                                     allow_patterns=MODEL_FILES, local_files_only=True))
    require(snapshot.name == MODEL_REVISION, "Model snapshot revision differs")
    for name, sha in token["model_files_sha256"].items():
        require(digest(snapshot/name) == sha, f"Model file differs from original: {name}")
    model = SentenceTransformer(str(snapshot), device="mps", local_files_only=True, trust_remote_code=False,
                                similarity_fn_name="cosine", model_kwargs={"attn_implementation": "eager"})
    model.float().eval()
    require(model.max_seq_length == MAX_LENGTH and model.get_sentence_embedding_dimension() == 384,
            "Unexpected model input length or embedding dimension")
    require(model.tokenizer.truncation_side == "right", "Tokenizer truncation side differs")
    require(str(model.device).startswith("mps") and model.dtype == torch.float32, "Model device/dtype differs")
    print("Check empty prompt configuration and equivalence of the three encoder routes", flush=True)
    verify_prompt_routes(model, queries, documents, out, torch)
    save(out, "runtime.json", {"python": sys.version, "platform": platform.platform(),
         "device": str(model.device), "dtype": str(model.dtype), "attention_implementation": "eager",
         "packages": {n: importlib.metadata.version(n) for n in ["sentence-transformers", "torch", "transformers",
                      "datasets", "numpy", "huggingface-hub"]}, "miner_source_sha256": MINER_SHA,
         "model_files_sha256": token["model_files_sha256"],
         "torch_num_threads": torch.get_num_threads(), "torch_num_interop_threads": torch.get_num_interop_threads()})
    corpus_ids, corpus_text = list(documents), list(documents.values())
    unique_queries = list(anchors)
    dataset = Dataset.from_list(pairs)
    cache = artifacts / "mining-embeddings"  # Fresh per-run cache; never reuse another model's cached vectors.
    common = dict(dataset=dataset, model=model, anchor_column_name="anchor", positive_column_name="positive",
                  corpus=corpus_text, range_min=0, range_max=len(documents), num_negatives=1,
                  absolute_margin=.05, relative_margin=None, max_score=None, min_score=None,
                  query_prompt="", corpus_prompt="", include_positives=False, output_format="triplet",
                  output_scores=True, batch_size=32, faiss_batch_size=16384, use_faiss=False,
                  use_multi_process=False, verbose=False, cache_folder=str(cache))
    print("3/6 Encode the original model and mine C from the full eligible corpus", flush=True)
    random.seed(1729)
    hard, hard_seconds = timed_call(lambda: mine_hard_negatives(**common, sampling_strategy="top"),
                                   synchronize=torch.mps.synchronize)
    query_files, corpus_files = list(cache.glob("query_embeddings_*.npy")), list(cache.glob("corpus_embeddings_*.npy"))
    require(len(query_files) == len(corpus_files) == 1, "Unexpected miner cache contents")
    qe, ce = np.load(query_files[0], allow_pickle=False), np.load(corpus_files[0], allow_pickle=False)
    validate_embeddings(qe, len(unique_queries), 384); validate_embeddings(ce, len(documents), 384)
    cache_hashes = {p.name: digest(p) for p in query_files + corpus_files}
    # Recompute using the same public cosine functions on CPU. This verifies
    # miner selections/filters, not the semantic correctness of unjudged labels.
    scores_tensor = model.similarity(qe, ce)
    require(scores_tensor.device.type == "cpu", "Unexpected audit score device")
    scores = scores_tensor.numpy().astype(np.float32, copy=False)
    require(scores.shape == (len(unique_queries), len(documents)) and bool(np.isfinite(scores).all()),
            "Invalid full-corpus score matrix")
    np.save(artifacts / "train-corpus-scores.npy", scores, allow_pickle=False)
    doc_index = {did: i for i, did in enumerate(corpus_ids)}
    pools, references, coverage = {}, {}, []
    for i, anchor in enumerate(unique_queries):
        excluded = exclusions[anchors[anchor][0]]
        p_scores = model.similarity_pairwise(np.repeat(qe[i:i+1], len(excluded), axis=0),
                                            ce[[doc_index[did] for did in excluded]]).numpy()
        references[anchor] = {did: float(value) for did, value in zip(excluded, p_scores)}
        pools[anchor], counts = candidate_pool(corpus_ids, scores[i], excluded, references[anchor], .05)
        coverage.extend({"query_id": qid, **counts} for qid in anchors[anchor])
    save(out, "coverage.json", {"per_query": sorted(coverage, key=lambda r: r["query_id"]),
         "training_queries": len(queries), "queries_with_no_eligible_negative": [r["query_id"] for r in coverage if r["eligible_documents"] == 0]})
    require(all(row["eligible_documents"] > 0 for row in coverage),
            "No eligible negatives for some queries; inspect coverage.json. No fallback or query dropping is allowed")
    hard_selected, hard_positive_scores = parse_mined_rows(list(hard), documents, anchors, exclusions)
    print("4/6 Sample B uniformly from the SAME eligible full-corpus pool and cached embeddings", flush=True)
    random.seed(1729)
    ordinary, ordinary_seconds = timed_call(lambda: mine_hard_negatives(**common, sampling_strategy="random"),
                                            synchronize=torch.mps.synchronize)
    random_selected, random_positive_scores = parse_mined_rows(list(ordinary), documents, anchors, exclusions)
    require({p.name: digest(p) for p in query_files + corpus_files} == cache_hashes, "Mining cache changed between arms")
    print("5/6 Audit both selections and expand only the 733 ORIGINAL training positive pairs", flush=True)
    selections = {"B_random": random_selected, "C_hard": hard_selected}
    reported_positive = {"B_random": random_positive_scores, "C_hard": hard_positive_scores}
    negative_ids = {arm: {} for arm in selections}
    selection_rows = []
    coverage_map = {row["query_id"]: row for row in coverage}
    for i, anchor in enumerate(unique_queries):
        for arm, chosen in selections.items():
            for did, value in references[anchor].items():
                require(abs(reported_positive[arm][anchor][did] - value) <= 2e-6, "Positive-score audit mismatch")
            checked = check_selection(corpus_ids, scores[i], pools[anchor], chosen[anchor]["document_id"],
                                      chosen[anchor]["score"], hard=arm == "C_hard")
            for qid in anchors[anchor]:
                negative_ids[arm][qid] = checked["negative_document_id"]
                selection_rows.append({"variant": arm, "query_id": qid, **checked, **{k:v for k,v in coverage_map[qid].items() if k != "query_id"}})
    selection_rows.sort(key=lambda r:(r["variant"],r["query_id"]))
    save_jsonl(out, "selected-negatives.jsonl", selection_rows)
    arm_rows, stats = {}, {}
    for arm in selections:
        rows = training_triplets(positives, negative_ids[arm])
        require(len(rows) == 733, "Training pair count changed")
        for row in rows:
            anchor = queries[row["query_id"]]
            row["positive_score"] = references[anchor][row["positive_document_id"]]
            row["negative_score"] = selections[arm][anchor]["score"]
            row["initial_triplet_loss"] = max(0., row["negative_score"] - row["positive_score"] + .2)
        arm_rows[arm] = rows
        save_jsonl(out, f"{arm}-triplets.jsonl", rows)
        active = sum(row["initial_triplet_loss"] > 0 for row in rows)
        stats[arm] = {"triplets": len(rows), "training_queries": len(negative_ids[arm]),
             "initial_active_triplets": active, "initial_active_triplet_fraction": active / len(rows),
             "initial_mean_triplet_loss": float(np.mean([row["initial_triplet_loss"] for row in rows])),
             "mean_selected_negative_score_per_query": float(np.mean([row["negative_score"] for row in selection_rows if row["variant"] == arm]))}
    require([(r["query_id"], r["positive_document_id"]) for r in arm_rows["B_random"]] ==
            [(r["query_id"], r["positive_document_id"]) for r in arm_rows["C_hard"]], "B/C positive pair budgets differ")
    print("6/6 Export paired false-negative review cases; do not start training", flush=True)
    random_ids = random.Random(2718).sample(sorted(queries), 10)
    hard_rows = [r for r in selection_rows if r["variant"] == "C_hard"]
    high_ids = [r["query_id"] for r in sorted(hard_rows, key=lambda r:(-r["negative_score"], r["query_id"]))[:5]]
    small_ids = [r["query_id"] for r in sorted(coverage, key=lambda r:(r["eligible_documents"],r["query_id"]))[:5]]
    review_ids = sorted(set(random_ids + high_ids + small_ids))
    indexed = {(r["variant"], r["query_id"]):r for r in selection_rows}
    cases = []
    for qid in review_ids:
        cases.append({"query_id": qid, "query": queries[qid],
            "selection_reasons": [name for name, ids in [("random",random_ids),("high_C_score",high_ids),("small_pool",small_ids)] if qid in ids],
            "known_positives": [{"document_id":did,"text":documents[did]} for did in positives[qid]],
            "candidates": [{**indexed[(arm,qid)],"text":documents[negative_ids[arm][qid]],
                             "review_label":None,"review_reason":None} for arm in selections]})
    save(out, "review-cases.json", {"status":"pending", "scope":"TRAIN samples only; full texts shown, model sees right-truncated 256 tokens",
         "allowed_review_labels":["likely_unrelated","possible_relevant","uncertain"], "cases":cases,
         "limitation":"Nonexpert review is not a gold relevance label. No candidate is certified negative."})
    save(artifacts, "query-text-order-sha256.json", [hashlib.sha256(q.encode()).hexdigest() for q in unique_queries])
    save(artifacts, "corpus-ids.json", corpus_ids)
    for name, sha in token["model_files_sha256"].items():
        require(digest(snapshot/name) == sha, f"Original model file changed: {name}")
    save(out, "artifacts.json", {"directory": str(artifacts), "sha256": {
         p.relative_to(artifacts).as_posix():digest(p) for p in sorted(artifacts.rglob("*")) if p.is_file()}})
    summary = {"status":"prepared_for_review", "scope":"fixed original MiniLM; TRAIN negative preparation only",
        "model_revision":MODEL_REVISION, "training_queries":len(queries), "unique_raw_query_texts":len(anchors),
        "original_positive_pairs":733, "miner_exclusion_control_pairs":len(pairs), "corpus_documents":len(documents),
        "arms":stats, "known_positive_violations":0, "queries_dropped":0, "review_queries":len(cases),
        "hard_top_candidate_check_passed":True, "same_positive_pair_order":True,
        "empty_prompt_routes_check_passed":True,
        "review_status":"pending", "training_allowed":False, "optimizer_steps":0,
        "original_cached_weights_unchanged":True, "validation_query_text_loaded":False,
        "validation_labels_used_for_mining":False, "official_test_query_text_loaded":False, "test_qrels_read":False,
        "timing_seconds":{"C_mining_including_fresh_encoding":hard_seconds,"B_mining_using_same_embedding_cache":ordinary_seconds},
        "limitations":["Unjudged candidates may be relevant; score gap is a heuristic.",
          "Same filtering in B/C means B is uniform over eligible documents, not the unfiltered corpus.",
          "One fixed sampling seed does not measure negative-sampling variance.",
          "Upstream torch.topk resolves selection ties; ranks in the audit use document-ID tie breaks.",
          "Initial active-loss fractions are diagnostics, not retrieval improvements.",
          "C includes fresh encoding while B reuses cache; these times are not a mining speed comparison."]}
    save(out, "negative-summary.json", summary)
    print(json_text(summary), end="", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("split-lock", "text-policy", "negative-policy", "data-audit", "exclusions", "token-audit",
                 "cache-dir", "output-dir", "artifact-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    try:
        from threadpoolctl import threadpool_limits
        with threadpool_limits(limits=1): run(args)
    except (OSError, ValueError, TypeError, KeyError, ImportError, RuntimeError, AssertionError) as exc:
        if args.output_dir.is_dir():
            save(args.output_dir,"failure.json",{"status":"failed","error":str(exc),"training_allowed":False})
        parser.exit(2, f"Negative preparation error: {exc}\n")


if __name__ == "__main__":
    main()
