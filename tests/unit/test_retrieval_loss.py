"""Unit tests for src/baseline/retrieval/loss.py."""

import pytest
import torch

from src.baseline.retrieval.loss import contrastive_loss


def test_contrastive_loss_matches_manual_cross_entropy():
    scores = torch.tensor([[2.0, 0.5, 0.1], [0.3, 3.0, 0.2]])
    targets = torch.tensor([0, 1], dtype=torch.long)

    expected = torch.nn.functional.cross_entropy(scores, targets)
    actual = contrastive_loss(scores, targets)

    assert torch.isclose(actual, expected)


def test_contrastive_loss_is_finite_on_valid_input():
    scores = torch.randn(4, 6)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    loss = contrastive_loss(scores, targets)

    assert torch.isfinite(loss)


def test_contrastive_loss_favors_correct_targets_with_lower_loss():
    # Scores strongly favoring the true diagonal target should produce a
    # much smaller loss than scores strongly favoring the wrong candidate.
    confident_correct = torch.tensor([[10.0, 0.0], [0.0, 10.0]])
    confident_wrong = torch.tensor([[0.0, 10.0], [10.0, 0.0]])
    targets = torch.tensor([0, 1], dtype=torch.long)

    loss_correct = contrastive_loss(confident_correct, targets)
    loss_wrong = contrastive_loss(confident_wrong, targets)

    assert loss_correct < loss_wrong


def test_contrastive_loss_rejects_rank1_scores():
    scores = torch.randn(4)
    targets = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    with pytest.raises(ValueError):
        contrastive_loss(scores, targets)


def test_contrastive_loss_rejects_rank2_targets():
    scores = torch.randn(4, 4)
    targets = torch.zeros(4, 1, dtype=torch.long)

    with pytest.raises(ValueError):
        contrastive_loss(scores, targets)


def test_contrastive_loss_rejects_length_mismatch():
    scores = torch.randn(4, 4)
    targets = torch.tensor([0, 1, 2], dtype=torch.long)

    with pytest.raises(ValueError):
        contrastive_loss(scores, targets)


def test_contrastive_loss_rejects_nan_scores():
    scores = torch.tensor([[1.0, float("nan")], [0.5, 0.2]])
    targets = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError):
        contrastive_loss(scores, targets)


def test_contrastive_loss_rejects_inf_scores():
    scores = torch.tensor([[1.0, float("inf")], [0.5, 0.2]])
    targets = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError):
        contrastive_loss(scores, targets)


def test_contrastive_loss_rejects_negative_target_index():
    scores = torch.randn(2, 3)
    targets = torch.tensor([0, -1], dtype=torch.long)

    with pytest.raises(ValueError):
        contrastive_loss(scores, targets)


def test_contrastive_loss_rejects_target_index_too_large():
    scores = torch.randn(2, 3)
    targets = torch.tensor([0, 3], dtype=torch.long)  # valid range is [0, 3)

    with pytest.raises(ValueError):
        contrastive_loss(scores, targets)


def test_contrastive_loss_accepts_empty_batch_without_indexing_targets():
    scores = torch.empty(0, 3)
    targets = torch.empty(0, dtype=torch.long)

    # Should not raise on the targets.min()/max() range check (skipped
    # for an empty batch) -- whatever F.cross_entropy itself does with
    # an empty batch is out of scope for this validation-focused test.
    contrastive_loss(scores, targets)
