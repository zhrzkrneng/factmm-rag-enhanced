"""Real RadGraph + CheXbert annotator.

Responsibility: annotate a report's finding text with RadGraph entities
and relations, and CheXbert diagnostic labels (5-class subset), using the
`radgraph` (pinned 0.0.9) and `f1chexbert` packages, matching the
official FactMM-RAG data/label.py exactly. Caches annotations to avoid
redundant recomputation. No implementation yet.
"""
