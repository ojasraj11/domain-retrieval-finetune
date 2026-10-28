"""Retrieval metrics, written out by hand so you can explain every line.

rankings: {qid: [doc_id, doc_id, ...]}, best first
qrels:    {qid: set(relevant doc_ids)}

A query in qrels with no ranking scores zero, so missing results are penalised.
"""
import math


def recall_at_k(ranked, relevant, k):
    """Share of the relevant documents that appear in the top k."""
    return len(set(ranked[:k]) & relevant) / len(relevant)


def mrr_at_k(ranked, relevant, k=10):
    """1 / rank of the first relevant document, or 0 if none is in the top k."""
    for i, d in enumerate(ranked[:k]):
        if d in relevant:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked, relevant, k=10):
    """Binary-relevance nDCG: rewards relevant documents more the higher they rank."""
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in relevant)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / ideal if ideal > 0 else 0.0


def evaluate_rankings(rankings: dict, qrels: dict, recall_ks=(1, 5, 10, 100)):
    totals = {f"recall@{k}": 0.0 for k in recall_ks}
    totals.update({"mrr@10": 0.0, "ndcg@10": 0.0})
    n = 0
    for qid, relevant in qrels.items():
        if not relevant:
            continue
        ranked = rankings.get(qid, [])
        for k in recall_ks:
            totals[f"recall@{k}"] += recall_at_k(ranked, relevant, k)
        totals["mrr@10"] += mrr_at_k(ranked, relevant, 10)
        totals["ndcg@10"] += ndcg_at_k(ranked, relevant, 10)
        n += 1
    if n == 0:
        return totals
    return {m: round(v / n, 4) for m, v in totals.items()}
