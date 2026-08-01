"""Lightweight mock generator for smoke tests.

Responsibility: provide a deterministic, template-filling stand-in for
the real LLaVA generator (src/baseline/generation/generator.py) so the
full pipeline (data -> annotation -> mining -> retrieval -> generation ->
evaluation) can be exercised end-to-end without GPU access or a
multi-GPU LLaVA fine-tune. Must implement the same interface as the real
generator (src/generation/base.py). No implementation yet.
"""
