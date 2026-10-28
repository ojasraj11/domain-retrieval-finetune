"""Load a BEIR retrieval dataset and build train / dev / test splits.

SciFact (the default) is small enough for a free Colab GPU:
about 5.2k documents, about 800 train queries and 300 test queries.
"""
import random


def _first_split(ds_dict):
    return ds_dict[list(ds_dict.keys())[0]]


def load_beir(name: str = "scifact"):
    """Return corpus {doc_id: text}, queries {qid: text}, qrels {split: {qid: set(doc_ids)}}."""
    from datasets import load_dataset

    corpus_ds = _first_split(load_dataset(f"BeIR/{name}", "corpus"))
    queries_ds = _first_split(load_dataset(f"BeIR/{name}", "queries"))
    qrels_ds = load_dataset(f"BeIR/{name}-qrels")

    corpus = {}
    for r in corpus_ds:
        title = (r.get("title") or "").strip()
        text = (r.get("text") or "").strip()
        corpus[str(r["_id"])] = f"{title}. {text}" if title else text

    queries = {str(r["_id"]): r["text"] for r in queries_ds}

    qrels = {}
    for split in qrels_ds:
        qrels[split] = {}
        for r in qrels_ds[split]:
            if int(r["score"]) > 0:
                qrels[split].setdefault(str(r["query-id"]), set()).add(str(r["corpus-id"]))
    return corpus, queries, qrels


def make_splits(qrels: dict, dev_fraction: float = 0.1, seed: int = 42):
    """Carve a dev set out of the official train queries.

    Tune hyperparameters on dev. Use test only for the final numbers:
    tuning on test is data leakage and inflates your results.
    """
    if "train" not in qrels or "test" not in qrels:
        raise ValueError(f"Expected train and test qrels, found: {list(qrels)}")

    train_qids = sorted(qrels["train"])
    random.Random(seed).shuffle(train_qids)
    n_dev = max(1, int(len(train_qids) * dev_fraction))
    dev_ids, train_ids = set(train_qids[:n_dev]), set(train_qids[n_dev:])

    splits = {
        "train": {q: qrels["train"][q] for q in train_ids},
        "dev": {q: qrels["train"][q] for q in dev_ids},
        "test": dict(qrels["test"]),
    }
    overlap = set(splits["train"]) & set(splits["test"])
    assert not overlap, f"Leakage: {len(overlap)} queries appear in both train and test"
    return splits


def build_training_pairs(split_qrels: dict, corpus: dict, queries: dict, query_prefix: str = ""):
    """One (query, relevant passage) pair per relevant document."""
    anchors, positives = [], []
    for qid, doc_ids in split_qrels.items():
        if qid not in queries:
            continue
        for d in sorted(doc_ids):
            if d in corpus:
                anchors.append(query_prefix + queries[qid])
                positives.append(corpus[d])
    return anchors, positives
