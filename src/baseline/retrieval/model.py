"""MARVEL-style multimodal retriever architecture.

Responsibility: reimplement the official FactMM-RAG
src/retriever/DPR/multi_model.py::MultiModal exactly -- a CLIP-ViT-B/32
vision tower whose per-patch hidden states (CLS token dropped) are
linearly projected into a T5 encoder-decoder's hidden space and spliced
into a tokenized caption's special-token span, with the pooled
representation always taken as the T5 decoder's first-position hidden
state (never a mean-pool, never CLIP's own pooled/projected output).

Three encoding paths are implemented, matching exactly what the official
forward() dispatch supports -- no additional fusion mechanism (e.g.
cross-attention, late fusion) is inferred or added:
  - encode_images_only: image-only, pooled -- used for QUERY encoding
    (paper Eq. 3).
  - encode_text_only: text-only, pooled -- matches what the official
    gen_embeddings.py actually computes for the retrieval corpus at
    inference time (a confirmed train/inference representation
    mismatch relative to training-time candidate encoding -- see
    docs/risk_register.md).
  - encode_images_with_text: joint image+text, pooled -- used for
    CANDIDATE encoding during training (paper Eq. 4).

The paper-vs-code contrastive-temperature discrepancy (fixed tau=0.01
vs. a learned CLIP-style logit_scale) is exposed as an explicit,
non-silent RetrieverConfig.temperature_mode choice -- never picked for
the caller. Warm-start checkpoint loading defaults to strict=True (the
safe choice); reproducing the official code's hardcoded strict=False
requires an explicit RetrieverConfig.warm_start_strict=False, and
missing/unexpected keys are always captured and returned, never
silently discarded.

image_dim/text_dim are never hardcoded -- both are read from the
constructed vision/text sub-models' own configs at __init__ time.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

_TRAINING_STAGES = ("dpr", "hard_negative")
_TEMPERATURE_MODES = ("learned_logit_scale", "fixed")


@dataclass(frozen=True)
class RetrieverConfig:
    """Architecture + reproducibility configuration for MultiModalRetriever.

    Deliberately does not include per-run optimizer hyperparameters
    (epochs, batch size, learning rate, early stopping) -- those belong
    to a separate, stage-specific training configuration, since the
    official DPR-stage and hard-negative-stage train.py scripts use
    materially different values for those (confirmed by reading both
    scripts, not assumed) while sharing this same architecture.
    """

    config_version: str = "1.0"
    clip_model_name: str = "openai/clip-vit-base-patch32"
    t5_model_name: str = "OpenMatch/t5-ance"
    training_stage: Literal["dpr", "hard_negative"] = "dpr"
    seed: int = 42
    max_text_length: Optional[int] = None
    normalize_embeddings: bool = True
    temperature_mode: Literal["learned_logit_scale", "fixed"] = "learned_logit_scale"
    fixed_temperature: float = 0.01
    initial_logit_scale: Optional[float] = None
    freeze_image_encoder: bool = False
    freeze_text_encoder: bool = False
    pretrained_checkpoint_path: Optional[str] = None
    warm_start_strict: bool = True

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if self.training_stage not in _TRAINING_STAGES:
            raise ValueError(
                f"training_stage must be one of {_TRAINING_STAGES}, got {self.training_stage!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if not self.clip_model_name:
            raise ValueError("clip_model_name must be a non-empty string")
        if not self.t5_model_name:
            raise ValueError("t5_model_name must be a non-empty string")
        if self.max_text_length is not None and self.max_text_length <= 0:
            raise ValueError(
                f"max_text_length must be a positive int or None, got {self.max_text_length!r}"
            )
        if not isinstance(self.normalize_embeddings, bool):
            raise ValueError("normalize_embeddings must be a bool")
        if self.temperature_mode not in _TEMPERATURE_MODES:
            raise ValueError(
                f"temperature_mode must be one of {_TEMPERATURE_MODES}, got {self.temperature_mode!r}"
            )
        if self.fixed_temperature <= 0:
            raise ValueError(
                f"fixed_temperature must be > 0, got {self.fixed_temperature!r}"
            )
        if self.initial_logit_scale is not None and not isinstance(
            self.initial_logit_scale, (int, float)
        ):
            raise ValueError(
                f"initial_logit_scale must be a number or None, got {self.initial_logit_scale!r}"
            )
        if not isinstance(self.freeze_image_encoder, bool):
            raise ValueError("freeze_image_encoder must be a bool")
        if not isinstance(self.freeze_text_encoder, bool):
            raise ValueError("freeze_text_encoder must be a bool")
        if not isinstance(self.warm_start_strict, bool):
            raise ValueError("warm_start_strict must be a bool")


def _default_clip_vision_factory(model_name: str):
    from transformers import CLIPVisionModel

    return CLIPVisionModel.from_pretrained(model_name)


def _default_clip_full_factory(model_name: str):
    from transformers import CLIPModel

    return CLIPModel.from_pretrained(model_name)


def _default_t5_factory(model_name: str):
    from transformers import T5Model

    return T5Model.from_pretrained(model_name)


def _default_tokenizer_factory(model_name: str):
    from transformers import T5Tokenizer

    return T5Tokenizer.from_pretrained(model_name)


class MultiModalRetriever(nn.Module):
    """MARVEL-style CLIP-vision + T5-multimodal retriever.

    Sub-model construction is fully dependency-injected (all four
    factories or none -- partial injection is a test-only escape
    hatch, not general DI, matching the pattern already established by
    RadGraphAnnotator). When all four are omitted, the real
    transformers.*.from_pretrained(...) calls are used; no
    CLIP/T5 import is even attempted otherwise, so unit tests never
    require the `transformers` package or a network connection.
    """

    def __init__(
        self,
        config: RetrieverConfig,
        *,
        clip_vision_factory: Optional[Callable[[], nn.Module]] = None,
        clip_full_factory: Optional[Callable[[], nn.Module]] = None,
        t5_factory: Optional[Callable[[], nn.Module]] = None,
        tokenizer_factory: Optional[Callable[[], object]] = None,
    ) -> None:
        super().__init__()

        factories = (clip_vision_factory, clip_full_factory, t5_factory, tokenizer_factory)
        provided = [f is not None for f in factories]
        if any(provided) and not all(provided):
            raise ValueError(
                "clip_vision_factory, clip_full_factory, t5_factory, and "
                "tokenizer_factory must all be provided together, or all "
                "omitted -- partial injection is a test-only escape hatch, "
                "not general DI."
            )

        if clip_vision_factory is None:
            clip_vision_factory = lambda: _default_clip_vision_factory(config.clip_model_name)
            clip_full_factory = lambda: _default_clip_full_factory(config.clip_model_name)
            t5_factory = lambda: _default_t5_factory(config.t5_model_name)
            tokenizer_factory = lambda: _default_tokenizer_factory(config.t5_model_name)

        self.config = config

        self._clip_vision = clip_vision_factory()
        self._t5 = t5_factory()
        self._tokenizer = tokenizer_factory()

        # Only the logit_scale scalar is ever read from the full CLIP
        # model -- matching official code's own (wasteful, but exactly
        # replicated) pattern of loading a whole separate CLIPModel just
        # for this one parameter. Not retained as a submodule afterward.
        clip_full = clip_full_factory()
        clip_logit_scale_init = float(clip_full.logit_scale.detach())

        self._tokenizer.add_special_tokens(
            {
                "additional_special_tokens": [
                    DEFAULT_IM_START_TOKEN,
                    DEFAULT_IMAGE_PATCH_TOKEN,
                    DEFAULT_IM_END_TOKEN,
                ]
            }
        )
        self._t5.resize_token_embeddings(len(self._tokenizer))

        self._image_dim = int(self._clip_vision.config.hidden_size)
        self._text_dim = int(self._t5.config.hidden_size)

        self.projector = nn.Linear(self._image_dim, self._text_dim)

        if config.temperature_mode == "learned_logit_scale":
            init_value = (
                config.initial_logit_scale
                if config.initial_logit_scale is not None
                else clip_logit_scale_init
            )
            self.logit_scale = nn.Parameter(torch.tensor(float(init_value), dtype=torch.float32))

        if config.freeze_image_encoder:
            for p in self._clip_vision.parameters():
                p.requires_grad_(False)
        if config.freeze_text_encoder:
            for p in self._t5.parameters():
                p.requires_grad_(False)

    @property
    def image_dim(self) -> int:
        """Resolved from self._clip_vision.config.hidden_size at construction -- never hardcoded."""
        return self._image_dim

    @property
    def text_dim(self) -> int:
        """Resolved from self._t5.config.hidden_size at construction -- never hardcoded."""
        return self._text_dim

    def _project_image_patches(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Matches official encode_images_only(): CLIPVisionModel ->
        last_hidden_state[:, 1:, :] (CLS token dropped) -> self.projector.
        Returns the raw per-patch sequence, shape [B, N_patches, text_dim]
        -- an intermediate, not itself a pooled representation.
        """
        outputs = self._clip_vision(pixel_values, output_hidden_states=True)
        patches = outputs.last_hidden_state[:, 1:, :]
        return self.projector(patches)

    def _embed_text_tokens(self, text_inputs: dict) -> torch.Tensor:
        """Matches official get_text_inputs_embeds(): T5 input-embedding lookup."""
        embedding_layer = self._t5.get_input_embeddings()
        return embedding_layer(text_inputs["input_ids"])

    def _splice_image_into_caption(
        self,
        pixel_values: torch.Tensor,
        text_inputs: dict,
    ) -> torch.Tensor:
        """Matches official get_images_with_caption_inputs_embeds(): the
        caption's tokenized embeddings are assumed to already reserve an
        N_patches-token placeholder span (via <im_patch> * N_patches
        between <im_start> and <im_end>) -- the real projected patch
        embeddings are spliced into that span by position, preserving
        the caption's total sequence length.
        """
        image_patch_embeds = self._project_image_patches(pixel_values)
        caption_embeds = self._embed_text_tokens(text_inputs)
        n_patches = image_patch_embeds.size(1)

        if caption_embeds.size(1) < n_patches + 1:
            raise ValueError(
                f"caption sequence length ({caption_embeds.size(1)}) is too "
                f"short to hold the reserved {n_patches}-patch placeholder "
                f"span plus at least one leading token."
            )

        return torch.cat(
            (
                caption_embeds[:, 0:1, :],
                image_patch_embeds,
                caption_embeds[:, n_patches + 1 :, :],
            ),
            dim=1,
        )

    def _pool(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Matches official get_rep(): runs the T5 encoder-decoder with a
        single zero decoder_input_ids token; the pooled representation is
        the decoder's first-position hidden state. This is the only
        pooling mechanism used by any of the three encode_* methods.
        """
        batch_size = inputs_embeds.shape[0]
        decoder_input_ids = torch.zeros(
            (batch_size, 1), dtype=torch.long, device=inputs_embeds.device
        )
        outputs = self._t5(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            decoder_input_ids=decoder_input_ids,
            return_dict=True,
        )
        return outputs.last_hidden_state[:, 0, :]

    def _maybe_normalize(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.config.normalize_embeddings:
            return F.normalize(embeddings, dim=-1)
        return embeddings

    def encode_images_only(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Pooled image-only representation -- QUERY encoding (paper Eq. 3).
        L2-normalized when config.normalize_embeddings. Shape [B, text_dim].
        """
        pooled = self._pool(self._project_image_patches(pixel_values), attention_mask=None)
        return self._maybe_normalize(pooled)

    def encode_text_only(self, text_inputs: dict) -> torch.Tensor:
        """Pooled text-only representation. Matches what the official
        gen_embeddings.py actually computes for the retrieval corpus at
        inference time -- NOT what training uses for candidates (see
        encode_images_with_text). Named explicitly "text_only" (never
        bare "text") so callers cannot mistake it for the joint
        candidate representation. L2-normalized when
        config.normalize_embeddings. Shape [B, text_dim].
        """
        pooled = self._pool(
            self._embed_text_tokens(text_inputs), attention_mask=text_inputs["attention_mask"]
        )
        return self._maybe_normalize(pooled)

    def encode_images_with_text(
        self,
        pixel_values: torch.Tensor,
        text_inputs: dict,
    ) -> torch.Tensor:
        """Pooled joint image+text representation -- CANDIDATE encoding
        during TRAINING (paper Eq. 4). This special-token patch-splicing
        fusion is exactly and only what the official code supports; no
        alternative fusion mechanism is implemented. L2-normalized when
        config.normalize_embeddings. Shape [B, text_dim].
        """
        merged = self._splice_image_into_caption(pixel_values, text_inputs)
        pooled = self._pool(merged, attention_mask=text_inputs["attention_mask"])
        return self._maybe_normalize(pooled)

    def scaled_similarity(
        self,
        query_emb: torch.Tensor,
        candidate_emb: torch.Tensor,
    ) -> torch.Tensor:
        """cosine(query, candidate) scaled by the configured temperature
        mechanism -- exp(logit_scale) when temperature_mode is
        "learned_logit_scale", or division by fixed_temperature when
        "fixed". The two modes never mix within a single call: exactly
        one branch is taken based on self.config.temperature_mode.
        query_emb/candidate_emb are assumed already normalized (an
        invariant every encode_* method guarantees on its own output
        when normalize_embeddings=True); this method does not
        re-normalize.
        """
        cosine = query_emb @ candidate_emb.t()
        if self.config.temperature_mode == "learned_logit_scale":
            return cosine * self.logit_scale.exp()
        return cosine / self.config.fixed_temperature


@dataclass(frozen=True)
class WarmStartLoadResult:
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]
    strict: bool


def load_warm_start_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    *,
    strict: bool,
) -> WarmStartLoadResult:
    """Loads a warm-start checkpoint's 'model' state dict onto `model`,
    matching the official code's checkpoint structure
    (torch.save({'epoch': epoch, 'model': model.state_dict()}, path)).

    strict=True lets torch's own strict loading raise RuntimeError on any
    key/shape mismatch -- never caught or wrapped here. strict=False must
    be an explicit caller choice (reproducing official code's hardcoded
    strict=False); missing_keys/unexpected_keys are always captured from
    load_state_dict's return value and returned here, in both modes --
    never silently discarded.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model"]
    result = model.load_state_dict(state_dict, strict=strict)
    return WarmStartLoadResult(
        missing_keys=tuple(result.missing_keys),
        unexpected_keys=tuple(result.unexpected_keys),
        strict=strict,
    )
