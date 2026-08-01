"""Baseline LLaVA generator.

Responsibility: fine-tune and run inference with LLaVA-1.5
(vicuna-7b-v1.5 base) on the RAG dataset produced by
src/generation/rag_dataset.py, matching the paper's confirmed
hyperparameters (epochs=1, lr=2e-5, batch_size=128) and the official
FactMM-RAG train_llava.sh / inference_llava.sh scripts. This stage's
compute requirement (8x A6000, ~4h per the paper) is documented in
docs/compute_requirements.md — real training here is out of scope for
interactive/Colab sessions unless explicitly confirmed available. No
implementation yet.
"""
