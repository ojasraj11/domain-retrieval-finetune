# Domain-Tuned Retrieval: Fine-Tuning Embeddings and Reranking for RAG

Off-the-shelf embedding models are trained on general web data, so they often miss domain-specific relevance. This project measures how much a small embedding model improves on a scientific-claim retrieval benchmark (BEIR SciFact) when it is fine-tuned on in-domain query–passage pairs, and how much a cross-encoder reranker adds on top, together with the latency cost of each stage.

## Results (SciFact test set, 300 queries)

> Fill this in from `python -m src.make_table --split test` once the runs finish.

| Model | Recall@10 | Recall@100 | MRR@10 | nDCG@10 | ms/query |
|---|---|---|---|---|---|
| all-MiniLM-L6-v2 (baseline) | | | | | |
| bge-small-en-v1.5 (stronger baseline) | | | | | |
| all-MiniLM-L6-v2 fine-tuned | | | | | |
| fine-tuned + cross-encoder rerank | | | | | |

## Method

1. **Data.** BEIR SciFact: about 5.2k abstracts. The official train queries are split 90/10 into train and dev with a fixed seed. Hyperparameters are chosen on dev; the test set is evaluated once.
2. **Baselines.** `all-MiniLM-L6-v2` and `bge-small-en-v1.5`, with cosine similarity over normalized embeddings.
3. **Fine-tuning.** `MultipleNegativesRankingLoss` (in-batch negatives) with a no-duplicates batch sampler. An optional variant mines one hard negative per pair from ranks 10–50 of the base model and removes any that are labelled relevant.
4. **Reranking.** `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores the top 100 candidates.
5. **Metrics.** Recall@k, MRR@10 and nDCG@10, implemented in `src/metrics.py` and unit-tested, plus per-query latency for retrieval and reranking.

## Run it

Open `notebooks/colab_run.ipynb` in Google Colab with a T4 GPU and run the cells in order. To run locally instead:

```bash
pip install -r requirements.txt
python -m src.evaluate --model sentence-transformers/all-MiniLM-L6-v2 --name minilm-base --split dev
python -m src.train --base-model sentence-transformers/all-MiniLM-L6-v2 --output-dir models/minilm-ft --epochs 3
python -m src.evaluate --model models/minilm-ft/final --name minilm-ft --split dev
python -m src.make_table --split dev
```

Unit tests need no GPU and no downloads: `python tests/test_core.py`

## Project layout

```
src/data.py        dataset loading and leakage-safe train/dev/test splits
src/train.py       fine-tuning with in-batch and optional hard negatives
src/retrieval.py   dense search and cross-encoder reranking
src/metrics.py     Recall@k, MRR@10, nDCG@10
src/evaluate.py    evaluation CLI that writes results/*.json
src/make_table.py  results table for this README
tests/             unit tests for metrics, splits, search and reranking
```
