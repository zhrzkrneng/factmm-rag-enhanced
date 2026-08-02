"""Retriever contrastive loss.

Responsibility: the single-direction (query -> candidate) cross-entropy
contrastive loss used by the official FactMM-RAG DPR/train.py and
ANCE/train.py identically -- `loss_function(score, target)` where
`loss_function = torch.nn.CrossEntropyLoss()`. This is deliberately NOT
a symmetric/bidirectional CLIP-style loss (which would also compute the
transposed direction and average): the official code only ever computes
it one way, and this module reproduces exactly that, not the more
commonly-assumed symmetric form.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(scores: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Single-direction cross-entropy contrastive loss.

    Args:
        scores: [num_queries, num_candidates] similarity scores (e.g.
            MultiModalRetriever.scaled_similarity's output).
        targets: [num_queries] long tensor giving each query's positive
            candidate's index within the candidate axis.

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: scores is not rank 2, targets is not rank 1, their
            lengths disagree, scores contains a non-finite value (NaN or
            Inf), or a target index falls outside [0, num_candidates).
    """
    if scores.ndim != 2:
        raise ValueError(
            f"scores must be rank 2 [num_queries, num_candidates], got shape {tuple(scores.shape)}"
        )
    if targets.ndim != 1:
        raise ValueError(f"targets must be rank 1 [num_queries], got shape {tuple(targets.shape)}")
    if targets.shape[0] != scores.shape[0]:
        raise ValueError(
            f"targets length ({targets.shape[0]}) must match scores.shape[0] ({scores.shape[0]})"
        )
    if not torch.isfinite(scores).all():
        raise ValueError("scores contains non-finite values (NaN or Inf)")
    if targets.numel() > 0:
        min_target = int(targets.min())
        max_target = int(targets.max())
        if min_target < 0 or max_target >= scores.shape[1]:
            raise ValueError(
                f"targets contains an index outside [0, {scores.shape[1]}) -- "
                f"observed range [{min_target}, {max_target}]"
            )

    return F.cross_entropy(scores, targets)
