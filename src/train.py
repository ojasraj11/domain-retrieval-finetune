"""Fine-tune an embedding model on (query, relevant passage) pairs.

Loss: MultipleNegativesRankingLoss. For each query in a batch, its own passage is
the positive and every other passage in the batch acts as a negative. Bigger
batches therefore mean more negatives per step.

Example:
  python -m src.train --base-model sentence-transformers/all-MiniLM-L6-v2 \
      --output-dir models/minilm-ft --epochs 3
"""
import argparse
import json
from pathlib import Path

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
)
from sentence_transformers.training_args import BatchSamplers

from src.data import build_training_pairs, load_beir, make_splits


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--query-prefix", default="")
    p.add_argument("--hard-negatives", action="store_true",
                   help="Mine one hard negative per pair from ranks 10-50 of the base model")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def add_hard_negatives(ds, model, corpus, train_qrels, queries, query_prefix):
    """Mine hard negatives, then drop any that are actually relevant (false negatives)."""
    from sentence_transformers.util import mine_hard_negatives

    mined = mine_hard_negatives(
        ds,
        model,
        corpus=list(corpus.values()),
        range_min=10,       # skip the very top ranks, where unlabeled relevant docs often sit
        range_max=50,
        num_negatives=1,
        sampling_strategy="top",
        batch_size=64,
        use_faiss=False,
    )

    relevant_texts = {}
    for qid, doc_ids in train_qrels.items():
        if qid in queries:
            relevant_texts[query_prefix + queries[qid]] = {corpus[d] for d in doc_ids if d in corpus}

    keep = [i for i, row in enumerate(mined)
            if row["negative"] not in relevant_texts.get(row["anchor"], set())]
    print(f"Hard negatives: kept {len(keep)} of {len(mined)} rows after removing false negatives")
    return mined.select(keep)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    corpus, queries, qrels = load_beir(args.dataset)
    splits = make_splits(qrels)
    anchors, positives = build_training_pairs(splits["train"], corpus, queries, args.query_prefix)
    print(f"Training pairs: {len(anchors)} from {len(splits['train'])} queries")

    model = SentenceTransformer(args.base_model)
    train_ds = Dataset.from_dict({"anchor": anchors, "positive": positives})
    if args.hard_negatives:
        train_ds = add_hard_negatives(train_ds, model, corpus, splits["train"], queries, args.query_prefix)

    loss = losses.MultipleNegativesRankingLoss(model)

    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        fp16=torch.cuda.is_available(),
        # The same passage twice in a batch would become a false in-batch negative.
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        save_strategy="no",
        logging_steps=10,
        seed=args.seed,
        report_to="none",
    )

    trainer = SentenceTransformerTrainer(
        model=model, args=training_args, train_dataset=train_ds, loss=loss
    )
    trainer.train()

    final_dir = Path(args.output_dir) / "final"
    model.save(str(final_dir))
    (Path(args.output_dir) / "train_config.json").write_text(
        json.dumps({**vars(args), "n_pairs": len(train_ds)}, indent=2)
    )
    print(f"Saved fine-tuned model to {final_dir}")


if __name__ == "__main__":
    main()
