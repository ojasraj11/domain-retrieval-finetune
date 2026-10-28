"""Dense retrieval with a bi-encoder, plus optional cross-encoder reranking."""
import numpy as np


def encode_corpus(model, corpus: dict, batch_size: int = 64):
    doc_ids = list(corpus)
    emb = model.encode(
        [corpus[d] for d in doc_ids],
        batch_size=batch_size,
        normalize_embeddings=True,  # unit vectors, so dot product equals cosine similarity
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return doc_ids, emb


def search(model, queries: dict, doc_ids, doc_emb, top_k=100, query_prefix="", batch_size=64):
    """Return {qid: [(doc_id, score), ...]}, sorted best first."""
    qids = list(queries)
    q_emb = model.encode(
        [query_prefix + queries[q] for q in qids],
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    scores = q_emb @ doc_emb.T  # shape: (n_queries, n_docs)
    top_k = min(top_k, len(doc_ids))
    top_idx = np.argpartition(-scores, top_k - 1, axis=1)[:, :top_k]

    results = {}
    for i, qid in enumerate(qids):
        idx = top_idx[i][np.argsort(-scores[i, top_idx[i]])]
        results[qid] = [(doc_ids[j], float(scores[i, j])) for j in idx]
    return results


def rerank(cross_encoder, queries: dict, corpus: dict, results: dict, top_n=100, batch_size=32):
    """Re-score the top_n candidates of each query with a cross-encoder.

    The cross-encoder only reorders existing candidates, so Recall@top_n cannot
    change. It can only move relevant documents higher (MRR, nDCG, Recall@small k).
    """
    out = {}
    for qid, hits in results.items():
        cands = hits[:top_n]
        pairs = [(queries[qid], corpus[d]) for d, _ in cands]
        s = np.asarray(cross_encoder.predict(pairs, batch_size=batch_size, show_progress_bar=False))
        order = np.argsort(-s)
        out[qid] = [(cands[i][0], float(s[i])) for i in order] + hits[top_n:]
    return out


def to_rankings(results: dict):
    return {qid: [d for d, _ in hits] for qid, hits in results.items()}
