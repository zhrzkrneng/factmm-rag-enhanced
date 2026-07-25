"""Baseline LLaVA-based retrieval-augmented generator.

Responsibility: implement the paper's generation stage — LLaVA-1.5
(vicuna-7b-v1.5) fine-tuned on image + single-retrieved-report prompts —
as a concrete implementation of the src/generation/base.py interface,
plus a lightweight mock generator for pipeline smoke tests where a real
multi-GPU LLaVA fine-tune is not available.
"""
