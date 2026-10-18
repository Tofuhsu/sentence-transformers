# SPDX-License-Identifier: Apache-2.0
"""Freeze a group-disjoint split from the reviewed, training-only SciFact audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import unicodedata
from collections import defaultdict
from pathlib import Path

BASELINE = "b0bf39968ba461c34b29372144794eff1be06f6d"
AUDIT_MANIFEST_SHA256 = "5275556f2a22a9b5c920194f2b6ab26864a0120a60a5bee27eb0f25336d14c8a"
WHITELIST_SHA256 = "9cc41068c675ff93f821536db2a637d8d39c86c873adb66ed9aff1ba28f1321d"
SEED = 42


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def id_list(value, label, *, nonempty=True):
    require(isinstance(value, list), f"{label}: expected a list")
    require(all(isinstance(x, str) and x.strip() for x in value), f"{label}: invalid ID")
    require(len(value) == len(set(value)), f"{label}: duplicate IDs")
    require(bool(value) or not nonempty, f"{label}: empty list")


def validate_grouping(rows, groups, whitelist):
    """Validate coverage and constraints independently of the split assignment."""
    id_list(whitelist, "training whitelist")
    row_ids = [row["query_id"] for row in rows]
    id_list(row_ids, "query rows")
    require(set(row_ids) == set(whitelist), "Query IDs do not match the training whitelist")
    group_of, seen_groups = {}, set()
    for group in groups:
        gid = group["group_id"]
        require(isinstance(gid, str) and gid, "Invalid group ID")
        require(gid not in seen_groups, f"Duplicate group ID: {gid}")
        seen_groups.add(gid)
        id_list(group["query_ids"], f"Group {gid}")
        for qid in group["query_ids"]:
            require(qid not in group_of, f"Query {qid} belongs to multiple groups")
            group_of[qid] = gid
    require(set(group_of) == set(whitelist), "Group coverage does not match the training whitelist")

    owners = {}
    for row in rows:
        qid = row["query_id"]
        require(row["provisional_group"] == group_of[qid], f"Query {qid}: group mismatch")
        require(isinstance(row["text"], str) and row["text"].strip(), f"Query {qid}: empty text")
        id_list(row["positive_document_ids"], f"Query {qid} positives")
        id_list(row["source_contexts"], f"Query {qid} sources", nonempty=False)
        keys = [("normalized_text", normalize(row["text"]))]
        keys += [("known_positive", did) for did in row["positive_document_ids"]]
        keys += [("source_context", sid) for sid in row["source_contexts"]]
        for key in keys:
            prior = owners.setdefault(key, group_of[qid])
            require(prior == group_of[qid], f"{key[0]} crosses groups at query {qid}")
    return group_of


def negative_exclusions(rows):
    """Union known positives ONLY for exactly normalized duplicate TRAIN text."""
    by_text = defaultdict(set)
    for row in rows:
        by_text[normalize(row["text"])].update(row["positive_document_ids"])
    return {row["query_id"]: sorted(by_text[normalize(row["text"])]) for row in rows}


def split_data(rows, groups, whitelist, seed=SEED):
    group_of = validate_grouping(rows, groups, whitelist)
    require(type(seed) is int, "Seed must be an integer")
    require(len(groups) >= 2, "At least two groups are needed")

    def rank(group):
        encoded = json.dumps([seed, sorted(group["query_ids"])], ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    # Exactly ceil(number_of_groups / 5); never split a group to hit a query quota.
    validation_count = (len(groups) + 4) // 5
    ranked = sorted(groups, key=lambda group: (rank(group), sorted(group["query_ids"])))
    validation_groups = {group["group_id"] for group in ranked[:validation_count]}
    partitions = {"train": [], "validation": []}
    for row in sorted(rows, key=lambda row: row["query_id"]):
        name = "validation" if group_of[row["query_id"]] in validation_groups else "train"
        partitions[name].append(row)
    stats = {}
    ids = {}
    for name, part in partitions.items():
        ids[name] = [row["query_id"] for row in part]
        stats[name] = {
            "queries": len(part),
            "groups": len({group_of[qid] for qid in ids[name]}),
            "qrel_rows": sum(len(row["positive_document_ids"]) for row in part),
            "positive_documents": len({did for row in part for did in row["positive_document_ids"]}),
            "multi_positive_queries": sum(len(row["positive_document_ids"]) > 1 for row in part),
        }
    overlap = {}
    for label, extract in {
        "query_ids": lambda row: [row["query_id"]],
        "groups": lambda row: [group_of[row["query_id"]]],
        "normalized_text": lambda row: [normalize(row["text"])],
        "source_contexts": lambda row: row["source_contexts"],
        "known_positive_documents": lambda row: row["positive_document_ids"],
    }.items():
        left = {item for row in partitions["train"] for item in extract(row)}
        right = {item for row in partitions["validation"] for item in extract(row)}
        overlap[label] = len(left & right)
        require(overlap[label] == 0, f"Train/validation overlap: {label}")
    return {
        "seed": seed,
        "selection": "SHA256 of compact JSON [seed, sorted query IDs]; first ceil(groups/5) hashes are validation",
        "validation_fraction_unit": "groups",
        "validation_group_fraction": 0.2,
        "query_ids": ids,
        "group_ids": {name: sorted({group_of[qid] for qid in qs}) for name, qs in ids.items()},
        "counts": stats,
        "cross_split_overlap": overlap,
    }, negative_exclusions(partitions["train"])


def freeze(audit_dir, whitelist_path, output_dir):
    audit_dir, whitelist_path, output_dir = map(Path, (audit_dir, whitelist_path, output_dir))
    manifest = audit_dir / "SHA256SUMS"
    require(digest(manifest) == AUDIT_MANIFEST_SHA256, "Audit manifest differs from the reviewed run")
    input_hashes = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        wanted, name = line.split(maxsplit=1)
        name = name.removeprefix("*")
        path = (audit_dir / name).resolve()
        require(path.is_relative_to(audit_dir.resolve()), "Manifest path escapes audit directory")
        require(digest(path) == wanted, f"Audit checksum mismatch: {name}")
        input_hashes[name] = wanted
    require(digest(whitelist_path) == WHITELIST_SHA256, "Official training whitelist checksum mismatch")
    whitelist = json.loads(whitelist_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (audit_dir / "training-review.jsonl").read_text(encoding="utf-8").splitlines()]
    groups = json.loads((audit_dir / "provisional-groups.json").read_text(encoding="utf-8"))
    require(len(whitelist) == 809 and len(groups) == 490, "Unexpected reviewed input size")
    result, exclusions = split_data(rows, groups, whitelist)
    group_of = {qid: group["group_id"] for group in groups for qid in group["query_ids"]}
    with (audit_dir / "group-links.csv").open(encoding="utf-8", newline="") as handle:
        for link in csv.DictReader(handle):
            a, b = link["query_a"], link["query_b"]
            require(a in group_of and b in group_of, "Grouping edge uses a nontraining ID")
            require(group_of[a] == group_of[b], "Reviewed grouping edge was broken")
    result.update({
        "schema_version": 1,
        "status": "split_frozen",
        "baseline_commit": BASELINE,
        "scifact_revision": "b3b5335604bf5ee3c4447671af975ea25143d4f5",
        "qrels_revision": "2938d17dc3b09882fdb8c12bbbe2e2dc0e75a029",
        "reviewed_manifest_sha256": AUDIT_MANIFEST_SHA256,
        "training_whitelist_sha256": WHITELIST_SHA256,
        "input_sha256": input_hashes,
        "corpus_documents": 5183,
        "corpus_scope": "Shared retrieval corpus; known-positive documents are disjoint by grouping design",
        "test_qrels_read": False,
        "heldout_query_text_read": False,
        "original_qrels_modified": False,
        "negative_exclusions_scope": "TRAIN only; union of positives for exactly normalized duplicate text",
        "limitation": "Conservative heuristic grouping is not proof of zero semantic leakage; this is a custom validation split.",
    })
    # Refuse an existing directory, so a previous frozen run cannot be overwritten.
    output_dir.mkdir(parents=True, exist_ok=False)
    products = {
        "split-manifest.json": result,
        "train-query-ids.json": result["query_ids"]["train"],
        "validation-query-ids.json": result["query_ids"]["validation"],
        "train-negative-exclusions.json": exclusions,
    }
    for name, value in products.items():
        (output_dir / name).write_text(json_text(value), encoding="utf-8")
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{digest(output_dir / name)}  {name}\n" for name in sorted(products)), encoding="utf-8"
    )
    summary = {key: result[key] for key in ("status", "seed", "counts", "cross_split_overlap", "test_qrels_read", "original_qrels_modified")}
    summary["split_manifest_sha256"] = digest(output_dir / "split-manifest.json")
    print(json_text(summary), end="")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--train-ids", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        freeze(args.audit_dir, args.train_ids, args.output_dir)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Data split error: {exc}\n")


if __name__ == "__main__":
    main()
