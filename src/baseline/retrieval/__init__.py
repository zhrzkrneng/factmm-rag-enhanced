"""Baseline MARVEL-style multimodal retriever.

Responsibility: implement the paper's fact-aware retriever — a CLIP
vision tower + T5-ANCE (MARVEL) text tower with patch-token-splicing
fusion, trained with an in-batch-negative contrastive loss followed by a
modality-balanced hard-negative stage — as a concrete implementation of
the src/retrieval/base.py interface.
"""
