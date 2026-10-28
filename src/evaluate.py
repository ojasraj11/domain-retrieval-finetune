"""Evaluate an embedding model (and optionally a reranker) on dev or test.

Examples:
  python -m src.evaluate --model sentence-transformers/all-MiniLM-L6-v2 --name minilm-base --split dev
  python -m src.evaluate --model models/minilm-ft/final --name minilm-ft --split test \
      --reranker cross-encoder/ms-marco-MiniLM-L-6-v2
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.data import load_beir, make_splits
from src.metrics import evaluate_rankings
from src.retrieval import encode_corpus, rerank, search, to_rankings


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Hugging Face model id or local path")
    p.add_argument("--name", required=True, help="Label used in the results table")
    p.add_argument("--dataset", default="scifact")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--query-prefix", default="", help="e.g. the instruction bge models expect")
    p.add_argument("--reranker", default=None, help="Cross-encoder id, e.g. cross-encoder/ms-marco-MiniLM-L-6-v2")
    p.add_argument("--rerank-top-n", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default="results")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    corpus, queries, qrels = load_beir(args.dataset)
    split_qrels = make_splits(qrels)[args.split]
    split_queries = {q: queries[q] for q in split_qrels if q in queries}
    print(f"{args.dataset} / {args.split}: {len(corpus)} docs, {len(split_queries)} queries")

    model = SentenceTransformer(args.model, device=device)
    doc_ids, doc_emb = encode_corpus(model, corpus, args.batch_size)

    t0 = time.perf_counter()
    results = search(model, split_queries, doc_ids, doc_emb, top_k=100,
                     query_prefix=args.query_prefix, batch_size=args.batch_size)
    retrieval_ms = (time.perf_counter() - t0) * 1000 / max(1, len(split_queries))

    record = {
        "name": args.name,
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "n_queries": len(split_queries),
        "device": device,
        "metrics": evaluate_rankings(to_rankings(results), split_qrels),
        "retrieval_ms_per_query": round(retrieval_ms, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    if args.reranker:
        ce = CrossEncoder(args.reranker, device=device)
        t0 = time.perf_counter()
        reranked = rerank(ce, split_queries, corpus, results, top_n=args.rerank_top_n)
        rerank_ms = (time.perf_counter() - t0) * 1000 / max(1, len(split_queries))
        record["reranker"] = args.reranker
        record["rerank_top_n"] = args.rerank_top_n
        record["reranked_metrics"] = evaluate_rankings(to_rankings(reranked), split_qrels)
        record["rerank_ms_per_query"] = round(rerank_ms, 2)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.name}__{args.split}.json"
    out_path.write_text(json.dumps(record, indent=2))

    print(json.dumps(record["metrics"], indent=2))
    if "reranked_metrics" in record:
        print("After reranking:")
        print(json.dumps(record["reranked_metrics"], indent=2))
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
