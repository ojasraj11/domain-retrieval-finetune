"""Run with:  python -m pytest -q   (or:  python tests/test_core.py)

These tests need no GPU and no model downloads.
"""
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import build_training_pairs, make_splits  # noqa: E402
from src.metrics import evaluate_rankings, mrr_at_k, ndcg_at_k, recall_at_k  # noqa: E402
from src.retrieval import rerank, search, to_rankings  # noqa: E402


def test_recall():
    assert recall_at_k(["a", "b", "c"], {"b", "z"}, 2) == 0.5
    assert recall_at_k(["a", "b", "c"], {"z"}, 3) == 0.0


def test_mrr():
    assert mrr_at_k(["a", "b", "c"], {"c"}) == 1 / 3
    assert mrr_at_k(["a"] * 20, {"z"}) == 0.0


def test_ndcg_perfect_and_partial():
    assert ndcg_at_k(["a", "b"], {"a", "b"}) == 1.0
    expected = (1 / math.log2(3)) / (1 / math.log2(2))  # relevant doc at rank 2, one relevant
    assert abs(ndcg_at_k(["x", "a"], {"a"}) - expected) < 1e-9


def test_missing_query_counts_as_zero():
    m = evaluate_rankings({"q1": ["a"]}, {"q1": {"a"}, "q2": {"b"}})
    assert m["recall@1"] == 0.5


def test_splits_no_leakage_and_deterministic():
    qrels = {"train": {f"t{i}": {"d"} for i in range(50)}, "test": {"x1": {"d"}}}
    s1, s2 = make_splits(qrels), make_splits(qrels)
    assert s1["dev"].keys() == s2["dev"].keys()
    assert not set(s1["train"]) & set(s1["dev"])
    assert len(s1["dev"]) == 5 and len(s1["train"]) == 45


def test_training_pairs_one_per_relevant_doc():
    a, p = build_training_pairs({"q": {"d1", "d2"}}, {"d1": "one", "d2": "two"}, {"q": "query"}, "pre: ")
    assert a == ["pre: query", "pre: query"] and sorted(p) == ["one", "two"]


class FakeEncoder:
    """Maps known texts to fixed vectors, so search results are predictable."""
    def __init__(self, table):
        self.table = table

    def encode(self, texts, **kwargs):
        v = np.array([self.table[t] for t in texts], dtype=float)
        return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_search_orders_by_cosine():
    table = {"q": [1, 0], "near": [0.9, 0.1], "mid": [0.5, 0.5], "far": [0, 1]}
    enc = FakeEncoder(table)
    doc_ids = ["far", "near", "mid"]
    doc_emb = enc.encode(doc_ids)
    res = search(enc, {"q1": "q"}, doc_ids, doc_emb, top_k=3)
    assert to_rankings(res)["q1"] == ["near", "mid", "far"]


def test_rerank_reorders_but_keeps_candidates():
    class FakeCE:
        def predict(self, pairs, **kwargs):
            return [len(doc) for _, doc in pairs]  # longer text scores higher

    corpus = {"a": "x", "b": "xxx", "c": "xx"}
    results = {"q1": [("a", 0.9), ("b", 0.8), ("c", 0.7)]}
    out = to_rankings(rerank(FakeCE(), {"q1": "q"}, corpus, results, top_n=3))
    assert out["q1"] == ["b", "c", "a"]
    assert set(out["q1"]) == {"a", "b", "c"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"PASS {name}")
