"""Innovation D: Retrieval confidence gating.

Responsibility: decide whether retrieval should be used at all for a
given query, using signals such as top-1 similarity, top-1-vs-top-2
margin, candidate agreement, RadGraph consistency across candidates, or
retriever calibration/entropy. Compared against always-retrieve and
never-retrieve (the baseline's own non-RAG path) per
docs/innovation_proposals.md's Innovation D.
"""
