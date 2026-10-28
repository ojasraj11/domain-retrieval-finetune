"""Collect results/*.json into a markdown table for the README.

  python -m src.make_table --split test
"""
import argparse
import json
from pathlib import Path

COLS = ["recall@10", "recall@100", "mrr@10", "ndcg@10"]


def rows_for(record):
    rows = [(record["name"], record["metrics"], record.get("retrieval_ms_per_query"))]
    if "reranked_metrics" in record:
        ms = None
        if record.get("retrieval_ms_per_query") is not None and record.get("rerank_ms_per_query") is not None:
            ms = round(record["retrieval_ms_per_query"] + record["rerank_ms_per_query"], 2)
        rows.append((record["name"] + " + rerank", record["reranked_metrics"], ms))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--results-dir", default="results")
    args = p.parse_args()

    records = [json.loads(f.read_text()) for f in sorted(Path(args.results_dir).glob("*.json"))]
    records = [r for r in records if r.get("split") == args.split]
    if not records:
        print(f"No results for split '{args.split}' in {args.results_dir}/")
        return

    print(f"Split: {args.split} ({records[0]['n_queries']} queries, {records[0]['dataset']})\n")
    print("| Model | " + " | ".join(COLS) + " | ms/query |")
    print("|---" * (len(COLS) + 2) + "|")
    for r in records:
        for name, m, ms in rows_for(r):
            cells = [f"{m.get(c, 0):.4f}" for c in COLS]
            print(f"| {name} | " + " | ".join(cells) + f" | {ms if ms is not None else '-'} |")


if __name__ == "__main__":
    main()
