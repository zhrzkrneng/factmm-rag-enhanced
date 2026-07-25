"""Generation quality metrics.

Responsibility: compute F1CheXbert (micro-averaged F1 over the 5-class
observation subset), F1RadGraph (partial/RG_ER reward), ROUGE-L, and
BERTScore between generated and reference reports, matching the exact
definitions used in the official FactMM-RAG evaluation script and the
paper (see docs/paper_analysis.md §15). No implementation yet.
"""
