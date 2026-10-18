# SPDX-License-Identifier: Apache-2.0
"""Fine-tune the pinned MiniLM checkpoint on a fixed B (random) or C (hard) negative
triplet set with TripletLoss. Pilot and final runs both use this script; no
retrieval evaluation happens here."""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrieval_metrics import require  # noqa: E402
from text_policy import MAX_LENGTH, MODEL_ID, MODEL_REVISION  # noqa: E402


def load_dataset(path):
    import json as _json

    from datasets import Dataset

    rows = [_json.loads(line) for line in Path(path).open(encoding="utf-8")]
    require(bool(rows), "Training data is empty")
    for key in ("anchor", "positive", "negative"):
        require(all(isinstance(r[key], str) and r[key].strip() for r in rows), f"Empty {key} text present")
    return Dataset.from_dict({
        "anchor": [r["anchor"] for r in rows],
        "positive": [r["positive"] for r in rows],
        "negative": [r["negative"] for r in rows],
    }), len(rows)


def load_model(cache_dir, token_audit):
    import torch
    from huggingface_hub import snapshot_download
    from sentence_transformers import SentenceTransformer

    from data_split import digest

    MODEL_FILES = ["config.json", "config_sentence_transformers.json", "modules.json",
                   "sentence_bert_config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                   "special_tokens_map.json", "vocab.txt", "1_Pooling/config.json"]

    require(torch.backends.mps.is_built() and torch.backends.mps.is_available(), "MPS unavailable; stopped")

    token = json.loads((token_audit / "results/runtime.json").read_text())
    require(token["model_id"] == MODEL_ID and token["revision"] == MODEL_REVISION, "Token-audit model pin differs")

    snapshot = Path(snapshot_download(MODEL_ID, revision=MODEL_REVISION, cache_dir=str(cache_dir),
                                       allow_patterns=MODEL_FILES, local_files_only=True))
    require(snapshot.name == MODEL_REVISION, "Model snapshot revision differs")
    for name, sha in token["model_files_sha256"].items():
        require(digest(snapshot / name) == sha, f"Model file differs from original: {name}")

    model = SentenceTransformer(str(snapshot), device="mps", local_files_only=True, trust_remote_code=False,
                                 similarity_fn_name="cosine", model_kwargs={"attn_implementation": "eager"})
    model.float()
    model.max_seq_length = MAX_LENGTH
    require(model.get_sentence_embedding_dimension() == 384, "Unexpected embedding dimension")
    require(str(model.device).startswith("mps") and model.dtype == torch.float32, "Model device/dtype differs")
    return model


def run(args):
    import torch
    from sentence_transformers import SentenceTransformerTrainer, SentenceTransformerTrainingArguments
    from sentence_transformers.sentence_transformer.losses import TripletLoss
    from sentence_transformers.sentence_transformer.losses.triplet import TripletDistanceMetric

    require(not (args.output_dir / "train-result.json").exists(), f"Already has a result: {args.output_dir}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    dataset, row_count = load_dataset(args.training_data)
    model = load_model(args.cache_dir, args.token_audit)

    loss = TripletLoss(model, distance_metric=TripletDistanceMetric.COSINE, triplet_margin=0.2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay,
                                   foreach=False)

    targs = SentenceTransformerTrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        weight_decay=args.weight_decay,
        seed=args.seed,
        data_seed=args.seed,
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        dataloader_num_workers=0,
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=targs, train_dataset=dataset, loss=loss,
        optimizers=(optimizer, None),
    )

    torch.mps.synchronize()
    start = time.time()
    result = trainer.train()
    torch.mps.synchronize()
    elapsed = time.time() - start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"
    model.save(str(model_dir))

    record = {
        "variant": args.variant,
        "training_data": str(args.training_data),
        "rows": row_count,
        "seed": args.seed,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "loss": "TripletLoss(cosine, margin=0.2)",
        "optimizer": "AdamW(foreach=False)",
        "global_step": result.global_step,
        "training_loss": result.training_loss,
        "log_history": trainer.state.log_history,
        "elapsed_seconds": elapsed,
        "device": str(model.device),
        "dtype": str(model.dtype),
    }
    (args.output_dir / "train-result.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"variant": args.variant, "rows": row_count, "global_step": result.global_step,
                       "training_loss": result.training_loss, "elapsed_seconds": elapsed}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=["B", "C"], required=True)
    parser.add_argument("--training-data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--token-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    args = parser.parse_args()
    try:
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"Fine-tuning error: {exc}\n")


if __name__ == "__main__":
    main()
