"""Baseline retriever training loop.

Responsibility: train src/baseline/retrieval/model.py with in-batch
negatives (stage 1) and then modality-balanced hard negatives (stage 2),
matching the official FactMM-RAG train.py hyperparameters (AdamW,
epochs=15, early_stop=5, batch_size=32, lr=5e-6), with checkpointing,
embedding export, and MRR/factual retrieval metric tracking. No
implementation yet.
"""
