"""RadGraph + CheXbert annotation.

Responsibility: wrap the RadGraph and CheXbert annotation process behind
a modular interface, matching the official FactMM-RAG data/label.py, with
a real backend and a deterministic mock backend for tests when GPU/model
weights are unavailable.
"""
