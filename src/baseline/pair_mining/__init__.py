"""Fact-aware report pair mining.

Responsibility: implement the paper's Equation 1/2 factual-similarity
mining procedure — restrict candidates by matching CheXbert labels, score
remaining candidates by RadGraph-based factual similarity, and keep the
top-k above threshold delta as positive pairs — matching (and exposing as
configurable) the official gen_similarity.py / gen_topk_pos.py logic.
"""
