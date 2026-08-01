"""Innovation B: Fact-aware multi-objective reranking.

Responsibility: rerank retrieved candidates using a configurable weighted
combination of image-report similarity, retriever score, RadGraph
factual similarity, CheXbert label agreement, negation consistency,
anatomical consistency, a redundancy penalty, and a diversity reward.
Weights are tuned only on the validation set; ground-truth test reports
are never used as a reranking signal, per docs/innovation_proposals.md.
"""
