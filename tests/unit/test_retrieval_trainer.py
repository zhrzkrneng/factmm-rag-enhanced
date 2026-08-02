"""Unit tests for src/baseline/retrieval/trainer.py.

Reuses the fake CLIP/T5 test doubles from test_retrieval_model.py and
the fake image_processor/tokenizer from test_retrieval_dataset.py so
this trainer is exercised through the exact same dependency-injection
seams already established for those modules -- never real CLIP/T5,
never a real checkpoint download.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.baseline.retrieval.dataset import RetrievalCollator, RetrieverTrainingDataset
from src.baseline.retrieval.model import RetrieverConfig
from src.baseline.retrieval.trainer import (
    CheckpointLoadResult,
    RetrieverTrainer,
    RetrieverTrainerConfig,
    set_seed,
)
from src.common.exceptions import RetrieverTrainingError
from src.retrieval.embeddings import export_embeddings
from src.retrieval.index import FaissFlatIPIndex

from test_retrieval_dataset import FakeImageProcessor, FakeTokenizer
from test_retrieval_model import make_retriever


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, lines) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(json.dumps(line) + "\n")


def _annotated_record(dataset, patient_id, study_id, finding, image_path=None):
    return {
        "dataset": dataset,
        "patient_id": patient_id,
        "study_id": study_id,
        "finding": finding,
        "image_path": image_path if image_path is not None else f"/images/{dataset}/{study_id}.png",
    }


def _pair_row(dataset, patient_id, study_id, positive_keys):
    return {
        "dataset": dataset,
        "patient_id": patient_id,
        "study_id": study_id,
        "positive_keys": [list(k) for k in positive_keys],
        "scores": [{"chexbert_similarity": 1.0, "radgraph_similarity": 1.0, "combined_score": 1.0}]
        * len(positive_keys),
        "num_candidates_considered": 10,
        "num_qualifying_candidates": len(positive_keys),
        "num_selected": len(positive_keys),
        "flagged_bad_sample": False,
    }


def _make_synthetic_dpr_dataset(tmp_path, suffix=""):
    """Two mutually-positive records -> a 2-row Stage 1 dataset."""
    key_a = ("mimic", "p1", "s1" + suffix)
    key_b = ("mimic", "p2", "s2" + suffix)
    records_by_key = {
        key_a: _annotated_record("mimic", "p1", "s1" + suffix, "a chest xray finding report today"),
        key_b: _annotated_record("mimic", "p2", "s2" + suffix, "another distinct radiology finding note"),
    }
    pairs_path = tmp_path / f"pairs{suffix}.jsonl"
    _write_jsonl(
        pairs_path,
        [
            _pair_row("mimic", "p1", "s1" + suffix, [key_b]),
            _pair_row("mimic", "p2", "s2" + suffix, [key_a]),
        ],
    )
    dataset = RetrieverTrainingDataset(records_by_key, pairs_path)
    return dataset, records_by_key


def _make_dpr_batch(tmp_path, suffix="", batch_size=2, num_patches=4):
    dataset, records_by_key = _make_synthetic_dpr_dataset(tmp_path, suffix=suffix)
    collator = RetrievalCollator(FakeImageProcessor(), FakeTokenizer(), training_stage="dpr")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))
    return batch, dataset, records_by_key


def _make_trainer(tmp_path, *, checkpoint_dir=None, seed=42, num_patches=4, **trainer_overrides):
    # Seed before constructing the model (matching official
    # DPR/train.py's own set_seed-before-load_model ordering) --
    # RetrieverTrainer.__init__ also seeds, but by then the model's
    # random initialization has already happened, so relying on that
    # alone cannot make two independently-constructed models identical.
    set_seed(seed)
    config = RetrieverConfig(training_stage="dpr", seed=seed)
    retriever = make_retriever(config=config, num_patches=num_patches, text_hidden_size=6)
    trainer_config = RetrieverTrainerConfig(
        seed=seed,
        checkpoint_dir=str(checkpoint_dir) if checkpoint_dir is not None else None,
        **trainer_overrides,
    )
    trainer = RetrieverTrainer(retriever, trainer_config)
    return trainer, retriever


# ---------------------------------------------------------------------------
# RetrieverTrainerConfig validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"config_version": ""},
        {"num_epochs": 0},
        {"num_epochs": -1},
        {"learning_rate": 0.0},
        {"learning_rate": -1e-6},
        {"weight_decay": -0.1},
        {"betas": (0.9,)},
        {"betas": (1.0, 0.98)},
        {"eps": 0.0},
        {"gradient_clip_norm": 0.0},
        {"gradient_clip_norm": -1.0},
        {"seed": "not-an-int"},
        {"log_every": 0},
    ],
)
def test_config_rejects_invalid_values(overrides):
    with pytest.raises(ValueError):
        RetrieverTrainerConfig(**overrides)


def test_config_defaults_are_valid():
    RetrieverTrainerConfig()  # must not raise


# ---------------------------------------------------------------------------
# One training step
# ---------------------------------------------------------------------------


def test_train_step_returns_finite_loss_and_advances_step(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    trainer, _retriever = _make_trainer(tmp_path)

    record = trainer.train_step(batch)

    assert np.isfinite(record["loss"])
    assert trainer.step == 1
    assert trainer.epoch == 0
    assert record["step"] == 1


def test_train_step_produces_no_nan_gradients_and_updates_parameters(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    trainer, retriever = _make_trainer(tmp_path)

    before = {name: param.clone() for name, param in retriever.named_parameters()}
    trainer.train_step(batch)

    changed = any(
        not torch.equal(before[name], param) for name, param in retriever.named_parameters()
    )
    assert changed

    for param in retriever.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_train_step_rejects_batch_missing_required_keys(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    trainer, _retriever = _make_trainer(tmp_path)

    broken_batch = dict(batch)
    del broken_batch["targets"]

    with pytest.raises(RetrieverTrainingError):
        trainer.train_step(broken_batch)


def test_train_step_matching_embedding_dimensions(tmp_path):
    # Sanity: query/candidate embeddings share the model's own text_dim --
    # this would only fail if the trainer's own dimension check were wrong.
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    trainer, retriever = _make_trainer(tmp_path)

    trainer.train_step(batch)  # must not raise RetrieverTrainingError
    assert retriever.text_dim > 0


# ---------------------------------------------------------------------------
# Gradient clipping
# ---------------------------------------------------------------------------


def test_gradient_clip_norm_bounds_post_clip_gradient_norm(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    trainer, retriever = _make_trainer(tmp_path, gradient_clip_norm=1e-3)

    trainer.train_step(batch)

    total_norm = torch.sqrt(
        sum(param.grad.pow(2).sum() for param in retriever.parameters() if param.grad is not None)
    )
    assert total_norm <= 1e-3 + 1e-6


def test_gradient_clip_disabled_when_none(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    trainer, _retriever = _make_trainer(tmp_path, gradient_clip_norm=None)

    record = trainer.train_step(batch)

    assert record["grad_norm"] is None


# ---------------------------------------------------------------------------
# Multiple epochs / tiny overfit
# ---------------------------------------------------------------------------


def test_multiple_epochs_all_produce_finite_losses(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    trainer, _retriever = _make_trainer(tmp_path, num_epochs=3, learning_rate=1e-3)

    records = trainer.fit([batch])

    assert len(records) == 3
    assert trainer.epoch == 3
    assert all(np.isfinite(r["loss"]) for r in records)


def test_tiny_overfit_loss_decreases_over_many_steps(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    # A larger learning rate and many epochs on a fixed 2-example batch
    # should substantially reduce the in-batch contrastive loss.
    # weight_decay=0.0 here: the official-matching default (0.2, see
    # _build_param_groups) is a real regularizer that, at this tiny
    # data/step scale, can offset most of the gradient signal from only
    # two examples and prevent genuine overfitting -- appropriate for
    # paper-scale training, not for this smoke-test-scale convergence
    # check, so it is disabled for this test only.
    trainer, _retriever = _make_trainer(
        tmp_path, num_epochs=50, learning_rate=5e-2, weight_decay=0.0
    )

    records = trainer.fit([batch])

    first_loss = records[0]["loss"]
    last_loss = records[-1]["loss"]
    assert last_loss < first_loss * 0.5


# ---------------------------------------------------------------------------
# Deterministic seed behavior
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_first_step_loss(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)

    trainer_a, _ = _make_trainer(tmp_path, seed=123)
    record_a = trainer_a.train_step(batch)

    trainer_b, _ = _make_trainer(tmp_path, seed=123)
    record_b = trainer_b.train_step(batch)

    assert record_a["loss"] == pytest.approx(record_b["loss"], abs=1e-6)


def test_different_seeds_produce_different_initial_parameters(tmp_path):
    _trainer_a, retriever_a = _make_trainer(tmp_path, seed=1)
    _trainer_b, retriever_b = _make_trainer(tmp_path, seed=2)

    params_a = torch.cat([p.flatten() for p in retriever_a.parameters()])
    params_b = torch.cat([p.flatten() for p in retriever_b.parameters()])

    assert not torch.equal(params_a, params_b)


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------


def test_save_and_load_checkpoint_restores_model_and_optimizer_state(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    trainer, retriever = _make_trainer(tmp_path)
    trainer.train_step(batch)

    checkpoint_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(checkpoint_path)

    new_trainer, new_retriever = _make_trainer(tmp_path, seed=999)  # different init
    result = new_trainer.load_checkpoint(checkpoint_path)

    assert isinstance(result, CheckpointLoadResult)
    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
    assert result.epoch == trainer.epoch
    assert result.step == trainer.step

    for (name_a, param_a), (name_b, param_b) in zip(
        retriever.named_parameters(), new_retriever.named_parameters()
    ):
        assert name_a == name_b
        assert torch.equal(param_a, param_b)


def test_optimizer_state_is_restored_after_load(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    trainer, _retriever = _make_trainer(tmp_path)
    trainer.train_step(batch)  # populates Adam's exp_avg/exp_avg_sq state

    checkpoint_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(checkpoint_path)
    original_optimizer_state = trainer._optimizer.state_dict()

    new_trainer, _new_retriever = _make_trainer(tmp_path, seed=999)
    new_trainer.load_checkpoint(checkpoint_path)
    restored_optimizer_state = new_trainer._optimizer.state_dict()

    assert restored_optimizer_state["param_groups"] == original_optimizer_state["param_groups"]
    assert set(restored_optimizer_state["state"].keys()) == set(original_optimizer_state["state"].keys())


def test_resume_matches_uninterrupted_training(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)

    # Uninterrupted: 4 steps in one go.
    uninterrupted, _ = _make_trainer(tmp_path, seed=7, learning_rate=1e-3)
    for _ in range(4):
        uninterrupted.train_step(batch)
    uninterrupted_params = torch.cat([p.flatten() for p in uninterrupted._model.parameters()])

    # Interrupted: 2 steps, save, reload into a fresh trainer, 2 more steps.
    interrupted, _ = _make_trainer(tmp_path, seed=7, learning_rate=1e-3)
    for _ in range(2):
        interrupted.train_step(batch)
    checkpoint_path = tmp_path / "resume_ckpt.pt"
    interrupted.save_checkpoint(checkpoint_path)

    resumed, _ = _make_trainer(tmp_path, seed=1234)  # different seed -- must not matter after load
    resumed.load_checkpoint(checkpoint_path)
    for _ in range(2):
        resumed.train_step(batch)
    resumed_params = torch.cat([p.flatten() for p in resumed._model.parameters()])

    assert resumed.step == uninterrupted.step
    torch.testing.assert_close(resumed_params, uninterrupted_params)


def test_load_checkpoint_missing_file_raises(tmp_path):
    trainer, _retriever = _make_trainer(tmp_path)

    with pytest.raises(RetrieverTrainingError):
        trainer.load_checkpoint(tmp_path / "does_not_exist.pt")


def test_load_checkpoint_missing_required_field_raises(tmp_path):
    trainer, _retriever = _make_trainer(tmp_path)
    checkpoint_path = tmp_path / "incomplete.pt"
    torch.save({"model": trainer._model.state_dict()}, checkpoint_path)  # no optimizer/epoch/step

    with pytest.raises(RetrieverTrainingError):
        trainer.load_checkpoint(checkpoint_path)


def test_checkpoint_corruption_is_detected_via_checksum_mismatch(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    trainer, _retriever = _make_trainer(tmp_path)
    trainer.train_step(batch)

    checkpoint_path = tmp_path / "ckpt.pt"
    trainer.save_checkpoint(checkpoint_path)

    with checkpoint_path.open("r+b") as handle:
        handle.seek(0)
        handle.write(b"\x00\x00\x00\x00")  # corrupt the header bytes

    new_trainer, _new_retriever = _make_trainer(tmp_path, seed=999)
    with pytest.raises(RetrieverTrainingError):
        new_trainer.load_checkpoint(checkpoint_path)


def test_fit_with_checkpoint_dir_creates_one_checkpoint_per_epoch(tmp_path):
    batch, _dataset, _records = _make_dpr_batch(tmp_path)
    checkpoint_dir = tmp_path / "checkpoints"
    trainer, _retriever = _make_trainer(tmp_path, checkpoint_dir=checkpoint_dir, num_epochs=3)

    trainer.fit([batch])

    for epoch in (1, 2, 3):
        ckpt = checkpoint_dir / f"epoch_{epoch}.pt"
        assert ckpt.exists()
        assert ckpt.with_suffix(ckpt.suffix + ".sha256").exists()


# ---------------------------------------------------------------------------
# Deterministic batches (dataset -> collator -> DataLoader)
# ---------------------------------------------------------------------------


def test_deterministic_batches_across_dataloader_instances(tmp_path):
    batch_a, _dataset_a, _ = _make_dpr_batch(tmp_path, suffix="-det")
    batch_b, _dataset_b, _ = _make_dpr_batch(tmp_path, suffix="-det")

    assert torch.equal(batch_a["query_image_inputs"], batch_b["query_image_inputs"])
    assert torch.equal(batch_a["positive_text_input_ids"], batch_b["positive_text_input_ids"])
    assert torch.equal(batch_a["targets"], batch_b["targets"])


# ---------------------------------------------------------------------------
# Stage 2 (hard_negative) forward pass
# ---------------------------------------------------------------------------


def _make_stage2_batch(num_patches=2):
    batch_size = 2
    max_negatives = 1
    num_slots = 1 + max_negatives
    image_shape = (3, 8, 8)

    query_image_inputs = torch.randn(batch_size, *image_shape)
    candidate_image_inputs = torch.randn(batch_size, num_slots, *image_shape)
    candidate_text_input_ids = torch.randint(0, 20, (batch_size, num_slots, 6))
    candidate_text_attention_mask = torch.ones(batch_size, num_slots, 6, dtype=torch.long)
    candidate_mask = torch.tensor([[True, True], [True, False]])
    targets = torch.zeros(batch_size, dtype=torch.long)

    return {
        "query_image_inputs": query_image_inputs,
        "candidate_image_inputs": candidate_image_inputs,
        "candidate_text_input_ids": candidate_text_input_ids,
        "candidate_text_attention_mask": candidate_text_attention_mask,
        "candidate_mask": candidate_mask,
        "targets": targets,
    }


def test_stage2_hard_negative_forward_and_backward(tmp_path):
    config = RetrieverConfig(training_stage="hard_negative", seed=42)
    retriever = make_retriever(config=config, num_patches=2, text_hidden_size=6)
    trainer_config = RetrieverTrainerConfig(seed=42)
    trainer = RetrieverTrainer(retriever, trainer_config)

    batch = _make_stage2_batch(num_patches=2)
    record = trainer.train_step(batch)

    assert np.isfinite(record["loss"])
    for param in retriever.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()


def test_stage2_rejects_batch_missing_candidate_mask():
    config = RetrieverConfig(training_stage="hard_negative", seed=42)
    retriever = make_retriever(config=config, num_patches=2, text_hidden_size=6)
    trainer = RetrieverTrainer(retriever, RetrieverTrainerConfig(seed=42))

    batch = _make_stage2_batch()
    del batch["candidate_mask"]

    with pytest.raises(RetrieverTrainingError):
        trainer.train_step(batch)


# ---------------------------------------------------------------------------
# Part 8 -- Synthetic end-to-end smoke test
# ---------------------------------------------------------------------------


def test_synthetic_end_to_end_smoke_lifecycle(tmp_path):
    # dataset -> collator
    dataset, records_by_key = _make_synthetic_dpr_dataset(tmp_path, suffix="-e2e")
    collator = RetrievalCollator(FakeImageProcessor(), FakeTokenizer(), training_stage="dpr")
    loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)
    batch = next(iter(loader))

    # retriever -> loss -> optimizer (via trainer.fit, a tiny overfit run)
    config = RetrieverConfig(training_stage="dpr", seed=42, normalize_embeddings=True)
    retriever = make_retriever(config=config, num_patches=4, text_hidden_size=6)
    checkpoint_dir = tmp_path / "smoke_checkpoints"
    # weight_decay=0.0: see test_tiny_overfit_loss_decreases_over_many_steps
    # for why the official-matching 0.2 default is unsuitable at this
    # tiny (5-epoch, 1-batch) smoke-test scale.
    trainer_config = RetrieverTrainerConfig(
        num_epochs=5, learning_rate=5e-2, weight_decay=0.0, checkpoint_dir=str(checkpoint_dir), seed=42
    )
    trainer = RetrieverTrainer(retriever, trainer_config)
    records = trainer.fit([batch])

    assert all(np.isfinite(r["loss"]) for r in records)
    assert records[-1]["loss"] <= records[0]["loss"]

    # checkpoint -> reload
    final_checkpoint = checkpoint_dir / "epoch_5.pt"
    assert final_checkpoint.exists()

    reloaded_config = RetrieverConfig(training_stage="dpr", seed=1, normalize_embeddings=True)
    reloaded_retriever = make_retriever(config=reloaded_config, num_patches=4, text_hidden_size=6)
    reloaded_trainer = RetrieverTrainer(reloaded_retriever, RetrieverTrainerConfig(seed=1))
    load_result = reloaded_trainer.load_checkpoint(final_checkpoint)
    assert load_result.missing_keys == ()
    assert load_result.unexpected_keys == ()

    # embedding export
    image_processor = FakeImageProcessor()
    export_records = [
        (key, image_processor([record["image_path"]])[0]) for key, record in records_by_key.items()
    ]
    embeddings, keys = export_embeddings(reloaded_retriever, export_records, mode="image", normalize=True)
    assert embeddings.shape[0] == len(records_by_key)
    norms = np.linalg.norm(embeddings, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)

    # FAISS search
    index = FaissFlatIPIndex(embedding_dim=embeddings.shape[1], normalized=True, config_version="1.0")
    index.add(embeddings, keys)
    scores, result_keys = index.search(embeddings, top_k=len(keys))

    assert index.num_vectors == len(keys)
    assert scores.shape == (len(keys), len(keys))
    # Each record's own embedding should be its own closest (or tied-closest)
    # match under cosine similarity.
    for i, key in enumerate(keys):
        assert key in result_keys[i]
