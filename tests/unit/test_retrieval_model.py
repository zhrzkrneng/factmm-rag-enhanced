"""Unit tests for src/baseline/retrieval/model.py.

Uses lightweight fake CLIP/T5 stand-ins (real nn.Module subclasses with
a handful of real trainable parameters, so gradient-flow and freezing
behavior are genuinely exercised) -- never the real `transformers`
package, never a real checkpoint download.
"""

import math
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from src.baseline.retrieval.loss import contrastive_loss
from src.baseline.retrieval.model import (
    MultiModalRetriever,
    RetrieverConfig,
    WarmStartLoadResult,
    load_warm_start_checkpoint,
)


class FakeCLIPVisionModel(nn.Module):
    """Minimal stand-in for transformers.CLIPVisionModel: same call
    contract (config.hidden_size; __call__(pixel_values,
    output_hidden_states=True) -> object with .last_hidden_state of
    shape [B, 1 + num_patches, hidden_size], position 0 == CLS).
    """

    def __init__(self, hidden_size=8, num_patches=4, in_channels=3):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.num_patches = num_patches
        self._proj = nn.Linear(in_channels, hidden_size)

    def forward(self, pixel_values, output_hidden_states=True):
        batch = pixel_values.shape[0]
        pooled_pixels = pixel_values.mean(dim=(2, 3))  # [B, channels]
        per_position = pooled_pixels.unsqueeze(1).expand(batch, self.num_patches + 1, -1)
        hidden = self._proj(per_position)  # [B, 1+num_patches, hidden_size]
        return SimpleNamespace(last_hidden_state=hidden)


class FakeCLIPModel(nn.Module):
    """Minimal stand-in for transformers.CLIPModel: only .logit_scale is used."""

    def __init__(self, init_value=math.log(1 / 0.07)):
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))


class FakeT5Model(nn.Module):
    """Minimal stand-in for transformers.T5Model: get_input_embeddings(),
    resize_token_embeddings(), and __call__(..., inputs_embeds=,
    decoder_input_ids=, return_dict=True) -> object with
    .last_hidden_state, matching the subset of the real T5Model API
    MultiModalRetriever actually uses.
    """

    def __init__(self, hidden_size=6, vocab_size=32):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self._embedding = nn.Embedding(vocab_size, hidden_size)
        self._decoder_proj = nn.Linear(hidden_size, hidden_size)

    def get_input_embeddings(self):
        return self._embedding

    def resize_token_embeddings(self, new_num_tokens):
        old = self._embedding
        new = nn.Embedding(new_num_tokens, old.embedding_dim)
        with torch.no_grad():
            n = min(old.num_embeddings, new_num_tokens)
            new.weight[:n] = old.weight[:n]
        self._embedding = new
        return new

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        decoder_input_ids=None,
        return_dict=True,
    ):
        pooled = inputs_embeds.mean(dim=1)  # [B, hidden_size]
        decoder_len = decoder_input_ids.shape[1]
        hidden = self._decoder_proj(pooled).unsqueeze(1).expand(-1, decoder_len, -1)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeTokenizer:
    """Minimal stand-in for transformers.T5Tokenizer: add_special_tokens()
    and __len__() only.
    """

    def __init__(self, vocab_size=32):
        self._vocab_size = vocab_size

    def add_special_tokens(self, mapping):
        self._vocab_size += len(mapping.get("additional_special_tokens", []))

    def __len__(self):
        return self._vocab_size


def make_retriever(
    *,
    config=None,
    clip_hidden_size=8,
    text_hidden_size=6,
    num_patches=4,
    vocab_size=32,
) -> MultiModalRetriever:
    config = config or RetrieverConfig()
    return MultiModalRetriever(
        config,
        clip_vision_factory=lambda: FakeCLIPVisionModel(
            hidden_size=clip_hidden_size, num_patches=num_patches
        ),
        clip_full_factory=lambda: FakeCLIPModel(),
        t5_factory=lambda: FakeT5Model(hidden_size=text_hidden_size, vocab_size=vocab_size),
        tokenizer_factory=lambda: FakeTokenizer(vocab_size=vocab_size),
    )


def make_caption_inputs(batch_size, seq_len, vocab_size=32):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


# --- RetrieverConfig validation ---


def test_config_defaults_are_valid():
    RetrieverConfig()  # must not raise


@pytest.mark.parametrize(
    "kwargs",
    [
        {"config_version": ""},
        {"training_stage": "ance"},
        {"seed": "not-an-int"},
        {"clip_model_name": ""},
        {"t5_model_name": ""},
        {"max_text_length": 0},
        {"max_text_length": -1},
        {"normalize_embeddings": "yes"},
        {"temperature_mode": "cosine_only"},
        {"fixed_temperature": 0.0},
        {"fixed_temperature": -0.01},
        {"initial_logit_scale": "warm"},
        {"freeze_image_encoder": 1},
        {"freeze_text_encoder": 1},
        {"warm_start_strict": "true"},
    ],
)
def test_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        RetrieverConfig(**kwargs)


# --- dependency injection ---


def test_factories_require_all_or_none():
    config = RetrieverConfig()
    with pytest.raises(ValueError):
        MultiModalRetriever(config, clip_vision_factory=lambda: FakeCLIPVisionModel())


# --- derived dimensions ---


def test_image_dim_and_text_dim_are_derived_not_hardcoded():
    retriever = make_retriever(clip_hidden_size=11, text_hidden_size=9)

    assert retriever.image_dim == 11
    assert retriever.text_dim == 9
    # Sanity: not accidentally the common "768" default some real CLIP/T5
    # checkpoints happen to use -- these dims come from the fakes only.
    assert retriever.image_dim != 768
    assert retriever.text_dim != 768


# --- encoding shapes ---


def test_encode_images_only_output_shape():
    retriever = make_retriever(text_hidden_size=6)
    pixel_values = torch.randn(3, 3, 16, 16)

    embeddings = retriever.encode_images_only(pixel_values)

    assert embeddings.shape == (3, 6)


def test_encode_text_only_output_shape():
    retriever = make_retriever(text_hidden_size=6, vocab_size=32)
    text_inputs = make_caption_inputs(batch_size=2, seq_len=5, vocab_size=32)

    embeddings = retriever.encode_text_only(text_inputs)

    assert embeddings.shape == (2, 6)


def test_encode_images_with_text_output_shape():
    # num_patches=4 -> caption must reserve >= 5 positions (1 leading + 4 patches)
    retriever = make_retriever(text_hidden_size=6, num_patches=4, vocab_size=32)
    pixel_values = torch.randn(2, 3, 16, 16)
    text_inputs = make_caption_inputs(batch_size=2, seq_len=8, vocab_size=32)

    embeddings = retriever.encode_images_with_text(pixel_values, text_inputs)

    assert embeddings.shape == (2, 6)


def test_splice_rejects_caption_shorter_than_patch_span():
    retriever = make_retriever(num_patches=4, vocab_size=32)
    pixel_values = torch.randn(1, 3, 16, 16)
    text_inputs = make_caption_inputs(batch_size=1, seq_len=3, vocab_size=32)  # too short

    with pytest.raises(ValueError):
        retriever.encode_images_with_text(pixel_values, text_inputs)


# --- normalization ---


def test_embeddings_are_normalized_when_configured():
    config = RetrieverConfig(normalize_embeddings=True)
    retriever = make_retriever(config=config)
    pixel_values = torch.randn(4, 3, 16, 16)

    embeddings = retriever.encode_images_only(pixel_values)
    norms = embeddings.norm(dim=-1)

    assert torch.allclose(norms, torch.ones(4), atol=1e-5)


def test_embeddings_are_not_normalized_when_disabled():
    config = RetrieverConfig(normalize_embeddings=False)
    retriever = make_retriever(config=config)
    pixel_values = torch.randn(4, 3, 16, 16)

    embeddings = retriever.encode_images_only(pixel_values)
    norms = embeddings.norm(dim=-1)

    # Extremely unlikely to be unit-norm by chance from an unnormalized
    # random linear projection -- proves normalization was actually skipped.
    assert not torch.allclose(norms, torch.ones(4), atol=1e-3)


# --- temperature modes ---


def test_scaled_similarity_is_finite():
    retriever = make_retriever()
    query = retriever.encode_images_only(torch.randn(2, 3, 16, 16))
    candidate = retriever.encode_images_only(torch.randn(2, 3, 16, 16))

    scores = retriever.scaled_similarity(query, candidate)

    assert torch.isfinite(scores).all()


def test_fixed_temperature_scaling_matches_manual_computation():
    config = RetrieverConfig(temperature_mode="fixed", fixed_temperature=0.05)
    retriever = make_retriever(config=config)
    query = retriever.encode_images_only(torch.randn(2, 3, 16, 16))
    candidate = retriever.encode_images_only(torch.randn(2, 3, 16, 16))

    scores = retriever.scaled_similarity(query, candidate)
    expected = (query @ candidate.t()) / 0.05

    assert torch.allclose(scores, expected, atol=1e-5)


def test_learned_logit_scale_scaling_matches_manual_computation():
    config = RetrieverConfig(temperature_mode="learned_logit_scale", initial_logit_scale=1.23)
    retriever = make_retriever(config=config)
    query = retriever.encode_images_only(torch.randn(2, 3, 16, 16))
    candidate = retriever.encode_images_only(torch.randn(2, 3, 16, 16))

    scores = retriever.scaled_similarity(query, candidate)
    expected = (query @ candidate.t()) * math.exp(1.23)

    assert torch.allclose(scores, expected, atol=1e-4)


def test_learned_logit_scale_defaults_to_clip_init_value_when_unset():
    config = RetrieverConfig(temperature_mode="learned_logit_scale", initial_logit_scale=None)
    retriever = make_retriever(config=config)

    assert torch.isclose(
        retriever.logit_scale, torch.tensor(math.log(1 / 0.07)), atol=1e-5
    )


def test_learned_logit_scale_is_trainable_parameter():
    config = RetrieverConfig(temperature_mode="learned_logit_scale")
    retriever = make_retriever(config=config)

    assert retriever.logit_scale.requires_grad


def test_fixed_temperature_mode_has_no_logit_scale_parameter():
    config = RetrieverConfig(temperature_mode="fixed")
    retriever = make_retriever(config=config)

    assert not hasattr(retriever, "logit_scale")


def test_temperature_modes_do_not_silently_produce_the_same_scores():
    fixed_config = RetrieverConfig(temperature_mode="fixed", fixed_temperature=0.01)
    learned_config = RetrieverConfig(
        temperature_mode="learned_logit_scale", initial_logit_scale=math.log(1 / 0.07)
    )
    torch.manual_seed(0)
    fixed_retriever = make_retriever(config=fixed_config)
    torch.manual_seed(0)
    learned_retriever = make_retriever(config=learned_config)

    pixel_values = torch.randn(2, 3, 16, 16)
    fixed_scores = fixed_retriever.scaled_similarity(
        fixed_retriever.encode_images_only(pixel_values),
        fixed_retriever.encode_images_only(pixel_values),
    )
    learned_scores = learned_retriever.scaled_similarity(
        learned_retriever.encode_images_only(pixel_values),
        learned_retriever.encode_images_only(pixel_values),
    )

    assert not torch.allclose(fixed_scores, learned_scores)


# --- encoder freezing ---


def test_freeze_image_encoder_disables_grad_on_vision_params_only():
    config = RetrieverConfig(freeze_image_encoder=True, freeze_text_encoder=False)
    retriever = make_retriever(config=config)

    assert all(not p.requires_grad for p in retriever._clip_vision.parameters())
    assert any(p.requires_grad for p in retriever._t5.parameters())


def test_freeze_text_encoder_disables_grad_on_text_params_only():
    config = RetrieverConfig(freeze_image_encoder=False, freeze_text_encoder=True)
    retriever = make_retriever(config=config)

    assert all(not p.requires_grad for p in retriever._t5.parameters())
    assert any(p.requires_grad for p in retriever._clip_vision.parameters())


def test_projector_remains_trainable_when_image_encoder_frozen():
    config = RetrieverConfig(freeze_image_encoder=True)
    retriever = make_retriever(config=config)

    assert all(p.requires_grad for p in retriever.projector.parameters())


# --- warm-start loading ---


def _save_checkpoint(tmp_path, state_dict, epoch=0):
    path = tmp_path / "checkpoint.pt"
    torch.save({"epoch": epoch, "model": state_dict}, path)
    return path


def test_warm_start_strict_true_succeeds_on_matching_state_dict(tmp_path):
    retriever = make_retriever()
    checkpoint_path = _save_checkpoint(tmp_path, retriever.state_dict())

    result = load_warm_start_checkpoint(retriever, checkpoint_path, strict=True)

    assert isinstance(result, WarmStartLoadResult)
    assert result.missing_keys == ()
    assert result.unexpected_keys == ()
    assert result.strict is True


def test_warm_start_strict_true_raises_on_mismatched_state_dict(tmp_path):
    source = make_retriever(text_hidden_size=6)
    target = make_retriever(text_hidden_size=6)
    incompatible_state = dict(source.state_dict())
    incompatible_state.pop("projector.weight")  # force a missing-key mismatch
    checkpoint_path = _save_checkpoint(tmp_path, incompatible_state)

    with pytest.raises(RuntimeError):
        load_warm_start_checkpoint(target, checkpoint_path, strict=True)


def test_warm_start_strict_false_reports_missing_and_unexpected_keys(tmp_path):
    source = make_retriever(text_hidden_size=6)
    target = make_retriever(text_hidden_size=6)
    incompatible_state = dict(source.state_dict())
    incompatible_state.pop("projector.weight")
    incompatible_state["totally_unexpected_param"] = torch.zeros(1)
    checkpoint_path = _save_checkpoint(tmp_path, incompatible_state)

    result = load_warm_start_checkpoint(target, checkpoint_path, strict=False)

    assert "projector.weight" in result.missing_keys
    assert "totally_unexpected_param" in result.unexpected_keys
    assert result.strict is False


def test_warm_start_never_silently_discards_keys_even_when_both_lists_nonempty(tmp_path):
    source = make_retriever(text_hidden_size=6)
    target = make_retriever(text_hidden_size=6)
    incompatible_state = dict(source.state_dict())
    incompatible_state.pop("projector.bias")
    incompatible_state["another_unexpected_param"] = torch.zeros(2)
    checkpoint_path = _save_checkpoint(tmp_path, incompatible_state)

    result = load_warm_start_checkpoint(target, checkpoint_path, strict=False)

    assert len(result.missing_keys) >= 1
    assert len(result.unexpected_keys) >= 1


# --- end-to-end gradient flow ---


def test_contrastive_loss_backward_updates_trainable_parameters_only():
    config = RetrieverConfig(freeze_image_encoder=True, freeze_text_encoder=False)
    retriever = make_retriever(config=config, text_hidden_size=6, num_patches=4, vocab_size=32)

    query_images = torch.randn(3, 3, 16, 16)
    candidate_images = torch.randn(3, 3, 16, 16)
    candidate_text = make_caption_inputs(batch_size=3, seq_len=8, vocab_size=32)

    query_emb = retriever.encode_images_only(query_images)
    candidate_emb = retriever.encode_images_with_text(candidate_images, candidate_text)
    scores = retriever.scaled_similarity(query_emb, candidate_emb)
    targets = torch.arange(3, dtype=torch.long)

    loss = contrastive_loss(scores, targets)
    assert torch.isfinite(loss)

    retriever.zero_grad()
    loss.backward()

    # Frozen image encoder: no gradient at all (frozen params don't
    # accumulate .grad since requires_grad is False).
    for p in retriever._clip_vision.parameters():
        assert p.grad is None

    # Trainable text encoder and projector: at least one parameter
    # actually received a non-zero gradient.
    text_grads = [p.grad for p in retriever._t5.parameters() if p.grad is not None]
    assert text_grads
    assert any(torch.any(g != 0) for g in text_grads)

    projector_grads = [p.grad for p in retriever.projector.parameters() if p.grad is not None]
    assert projector_grads
    assert any(torch.any(g != 0) for g in projector_grads)
