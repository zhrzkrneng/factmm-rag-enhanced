"""MARVEL-style multimodal retriever architecture.

Responsibility: implement the CLIP-ViT-B/32 + T5-ANCE (MARVEL) encoder
with patch-token-splicing fusion, matching official FactMM-RAG
src/retriever/DPR/multi_model.py, as a concrete implementation of
src/retrieval/base.py's abstract retriever interface. Must expose the
paper-vs-code contrastive-temperature discrepancy (fixed tau=0.01 vs.
learned logit_scale — see docs/risk_register.md #1c) as a configurable
choice. No implementation yet.
"""
