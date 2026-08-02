"""Lightweight Retriever training loop -- smoke-test-oriented, not the
paper-scale reproduction.

Responsibility: exercise the completed Retriever stack (model.py,
loss.py, dataset.py, hard_negatives.py) end-to-end with a minimal,
testable training loop, using fake encoders and tiny synthetic data
only. This is deliberately NOT a faithful reproduction of the official
DPR/train.py's full training regime (no cosine LR scheduler, no
wandb logging, no MRR-based checkpoint selection / early stopping,
no paper-scale epoch counts) -- those are out of scope for this
milestone's smoke test and are explicitly deferred.

What IS reproduced from the official DPR/train.py, verbatim:
  - the AdamW parameter grouping that excludes 1-D params (bias/
    LayerNorm-style gain) and `logit_scale` from weight decay:
    `exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or
    "bias" in n or "logit_scale" in n` (see _build_param_groups);
  - betas=(0.9, 0.98), eps=1e-6;
  - the checkpoint's 'model' key holding model.state_dict() (so
    src.baseline.retrieval.model.load_warm_start_checkpoint can read a
    checkpoint saved by this trainer unchanged).

Deliberate additions beyond the official script (disclosed, not
hidden):
  - gradient clipping (torch.nn.utils.clip_grad_norm_) -- the official
    script does not clip gradients at all;
  - the checkpoint also persists 'optimizer' state and 'epoch'/'step'
    counters (the official script's checkpoint is warm-start-only:
    {'epoch', 'model'}) -- needed for genuine train-resume, not just
    weight warm-starting;
  - a sha256 sidecar file (`<checkpoint>.sha256`) written alongside
    every checkpoint and verified on load, so a corrupted checkpoint
    is rejected loudly rather than silently loaded or crashing deep
    inside torch's unpickling.

Stage 2 (hard_negative) loss computation is this trainer's own
addition -- the official train.py only ever implements the Stage 1
(in-batch-negative) forward pass shown above; ANCE/train.py's hard-
negative-stage forward pass was not directly available to audit line-
by-line for this cell, so the Stage 2 per-query masked-candidate
similarity computation here (_compute_stage2_loss) is a
REASONABLE_INFERENCE built directly on top of RetrievalCollator's own
approved padded/masked batch schema (see src.baseline.retrieval.dataset),
not a verified reproduction of official ANCE training code.
"""

from __future__ import annotations

import hashlib
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

from src.baseline.retrieval.loss import contrastive_loss
from src.baseline.retrieval.model import MultiModalRetriever
from src.common.exceptions import RetrieverTrainingError

_STAGE1_REQUIRED_KEYS: Tuple[str, ...] = (
    "query_image_inputs",
    "positive_image_inputs",
    "positive_text_input_ids",
    "positive_text_attention_mask",
    "targets",
)
_STAGE2_REQUIRED_KEYS: Tuple[str, ...] = (
    "query_image_inputs",
    "candidate_image_inputs",
    "candidate_text_input_ids",
    "candidate_text_attention_mask",
    "candidate_mask",
    "targets",
)

# Large-magnitude finite stand-in for a hard exclusion mask. contrastive_loss
# (see src.baseline.retrieval.loss) deliberately rejects non-finite scores,
# so a literal -inf mask is unusable here; this value drives the masked
# candidate's softmax weight to ~0 without violating that finiteness
# contract.
_MASK_FILL_VALUE = -1.0e9


def set_seed(seed: int) -> None:
    """Matches official DPR/train.py::set_seed (random, numpy, torch), plus
    one deliberate addition: torch.set_num_threads(1).

    On CPU, torch's multi-threaded reduction ops (matmul, sum, etc.) do
    not guarantee a fixed floating-point summation order across runs,
    so torch.manual_seed alone does not make two runs bit-identical --
    only statistically similar. Pinning to a single thread here is what
    actually makes this trainer's "deterministic seed behavior"
    contract hold in practice; the official script does not do this
    (it runs on GPU, where this project's CPU-only reproducibility
    concern does not apply the same way).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _sha256_of_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _build_param_groups(model: nn.Module, weight_decay: float) -> List[dict]:
    """Matches official DPR/train.py's AdamW param-group split exactly."""

    def _excluded(name: str, param: torch.Tensor) -> bool:
        return param.ndim < 2 or "bn" in name or "ln" in name or "bias" in name or "logit_scale" in name

    gain_or_bias_params = []
    rest_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if _excluded(name, param):
            gain_or_bias_params.append(param)
        else:
            rest_params.append(param)

    return [
        {"params": gain_or_bias_params, "weight_decay": 0.0},
        {"params": rest_params, "weight_decay": weight_decay},
    ]


@dataclass(frozen=True)
class RetrieverTrainerConfig:
    config_version: str = "1.0"
    num_epochs: int = 1
    learning_rate: float = 5e-6
    weight_decay: float = 0.2
    betas: Tuple[float, float] = (0.9, 0.98)
    eps: float = 1e-6
    gradient_clip_norm: Optional[float] = 1.0
    seed: int = 42
    checkpoint_dir: Optional[str] = None
    log_every: int = 1

    def __post_init__(self) -> None:
        if not self.config_version:
            raise ValueError("config_version must be a non-empty string")
        if not isinstance(self.num_epochs, int) or isinstance(self.num_epochs, bool) or self.num_epochs < 1:
            raise ValueError(f"num_epochs must be a positive integer, got {self.num_epochs!r}")
        if self.learning_rate <= 0:
            raise ValueError(f"learning_rate must be > 0, got {self.learning_rate!r}")
        if self.weight_decay < 0:
            raise ValueError(f"weight_decay must be >= 0, got {self.weight_decay!r}")
        if len(self.betas) != 2 or not all(0.0 <= b < 1.0 for b in self.betas):
            raise ValueError(f"betas must be a 2-tuple in [0, 1), got {self.betas!r}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps!r}")
        if self.gradient_clip_norm is not None and self.gradient_clip_norm <= 0:
            raise ValueError(
                f"gradient_clip_norm must be a positive number or None, got {self.gradient_clip_norm!r}"
            )
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError(f"seed must be an int, got {self.seed!r}")
        if not isinstance(self.log_every, int) or isinstance(self.log_every, bool) or self.log_every < 1:
            raise ValueError(f"log_every must be a positive integer, got {self.log_every!r}")


@dataclass(frozen=True)
class CheckpointLoadResult:
    missing_keys: Tuple[str, ...]
    unexpected_keys: Tuple[str, ...]
    epoch: int
    step: int


def _validate_batch(batch: dict, required_keys: Sequence[str], stage: str) -> None:
    missing = [key for key in required_keys if key not in batch]
    if missing:
        raise RetrieverTrainingError(
            f"Batch is missing required keys for training_stage={stage!r}: {missing}"
        )


def _scale_by_temperature(model: MultiModalRetriever, cosine: torch.Tensor) -> torch.Tensor:
    if model.config.temperature_mode == "learned_logit_scale":
        return cosine * model.logit_scale.exp()
    return cosine / model.config.fixed_temperature


class RetrieverTrainer:
    """Minimal, fully-testable training loop over a MultiModalRetriever.

    training_stage is read from model.config.training_stage (the same
    single source of truth RetrievalCollator's own training_stage
    should be constructed to match) -- never a separately-tracked
    duplicate value that could silently drift out of sync.
    """

    def __init__(
        self,
        model: MultiModalRetriever,
        config: RetrieverTrainerConfig,
        *,
        optimizer_factory: Optional[Callable[[nn.Module, RetrieverTrainerConfig], torch.optim.Optimizer]] = None,
    ) -> None:
        set_seed(config.seed)

        self._model = model
        self._config = config
        if optimizer_factory is None:
            param_groups = _build_param_groups(model, config.weight_decay)
            self._optimizer = torch.optim.AdamW(
                param_groups, lr=config.learning_rate, betas=config.betas, eps=config.eps
            )
        else:
            self._optimizer = optimizer_factory(model, config)

        self._epoch = 0
        self._step = 0
        self._log: List[dict] = []

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def step(self) -> int:
        return self._step

    @property
    def log(self) -> List[dict]:
        return list(self._log)

    def _compute_stage1_loss(self, batch: dict) -> torch.Tensor:
        _validate_batch(batch, _STAGE1_REQUIRED_KEYS, "dpr")

        query_emb = self._model.encode_images_only(batch["query_image_inputs"])
        text_inputs = {
            "input_ids": batch["positive_text_input_ids"],
            "attention_mask": batch["positive_text_attention_mask"],
        }
        candidate_emb = self._model.encode_images_with_text(batch["positive_image_inputs"], text_inputs)

        if query_emb.shape[-1] != candidate_emb.shape[-1]:
            raise RetrieverTrainingError(
                f"Query embedding dim ({query_emb.shape[-1]}) does not match "
                f"candidate embedding dim ({candidate_emb.shape[-1]})"
            )

        scores = self._model.scaled_similarity(query_emb, candidate_emb)
        return contrastive_loss(scores, batch["targets"])

    def _compute_stage2_loss(self, batch: dict) -> torch.Tensor:
        _validate_batch(batch, _STAGE2_REQUIRED_KEYS, "hard_negative")

        query_emb = self._model.encode_images_only(batch["query_image_inputs"])

        batch_size, num_slots = batch["candidate_mask"].shape
        candidate_images = batch["candidate_image_inputs"]
        flat_images = candidate_images.reshape(batch_size * num_slots, *candidate_images.shape[2:])
        flat_text_inputs = {
            "input_ids": batch["candidate_text_input_ids"].reshape(batch_size * num_slots, -1),
            "attention_mask": batch["candidate_text_attention_mask"].reshape(batch_size * num_slots, -1),
        }
        flat_candidate_emb = self._model.encode_images_with_text(flat_images, flat_text_inputs)

        if query_emb.shape[-1] != flat_candidate_emb.shape[-1]:
            raise RetrieverTrainingError(
                f"Query embedding dim ({query_emb.shape[-1]}) does not match "
                f"candidate embedding dim ({flat_candidate_emb.shape[-1]})"
            )

        candidate_emb = flat_candidate_emb.reshape(batch_size, num_slots, -1)
        cosine = torch.einsum("bd,bnd->bn", query_emb, candidate_emb)
        scores = _scale_by_temperature(self._model, cosine)
        scores = scores.masked_fill(~batch["candidate_mask"], _MASK_FILL_VALUE)

        return contrastive_loss(scores, batch["targets"])

    def _compute_batch_loss(self, batch: dict) -> torch.Tensor:
        stage = self._model.config.training_stage
        if stage == "dpr":
            return self._compute_stage1_loss(batch)
        return self._compute_stage2_loss(batch)

    def train_step(self, batch: dict) -> dict:
        """Runs one forward + backward + optimizer step on `batch`.

        Raises:
            RetrieverTrainingError: batch is missing required keys for
                the model's configured training_stage, the computed
                loss is non-finite, or any parameter's gradient is
                non-finite after backward().
        """
        self._model.train()
        self._optimizer.zero_grad()

        loss = self._compute_batch_loss(batch)
        if not torch.isfinite(loss):
            raise RetrieverTrainingError(f"Non-finite loss encountered: {loss.item()!r}")

        loss.backward()

        for name, param in self._model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                raise RetrieverTrainingError(f"Non-finite gradient in parameter {name!r}")

        grad_norm = None
        if self._config.gradient_clip_norm is not None:
            grad_norm = float(
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._config.gradient_clip_norm)
            )

        self._optimizer.step()
        self._step += 1

        record = {
            "step": self._step,
            "epoch": self._epoch,
            "loss": float(loss.item()),
            "grad_norm": grad_norm,
        }
        self._log.append(record)
        return record

    def train_epoch(self, batches: Sequence[dict]) -> List[dict]:
        records = [self.train_step(batch) for batch in batches]
        self._epoch += 1
        return records

    def fit(self, batches_per_epoch: Sequence[dict], *, num_epochs: Optional[int] = None) -> List[dict]:
        epochs = num_epochs if num_epochs is not None else self._config.num_epochs
        all_records: List[dict] = []
        for _ in range(epochs):
            all_records.extend(self.train_epoch(batches_per_epoch))
            if self._config.checkpoint_dir is not None:
                checkpoint_path = Path(self._config.checkpoint_dir) / f"epoch_{self._epoch}.pt"
                self.save_checkpoint(checkpoint_path)
        return all_records

    def save_checkpoint(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "config_version": self._config.config_version,
            "epoch": self._epoch,
            "step": self._step,
            "seed": self._config.seed,
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
        }

        tmp_path = path.with_name(path.name + f".tmp{os.getpid()}")
        torch.save(payload, tmp_path)
        os.replace(str(tmp_path), str(path))

        checksum = _sha256_of_file(path)
        path.with_suffix(path.suffix + ".sha256").write_text(checksum, encoding="utf-8")

    def load_checkpoint(self, path: Path, *, strict: bool = True) -> CheckpointLoadResult:
        """Loads a checkpoint saved by save_checkpoint(), restoring model
        weights, optimizer state, and epoch/step counters.

        Raises:
            RetrieverTrainingError: path does not exist, its sidecar
                `.sha256` checksum (if present) does not match the
                file's actual contents, or the loaded payload is
                missing a required field.
        """
        path = Path(path)
        if not path.exists():
            raise RetrieverTrainingError(f"Checkpoint not found at {path}")

        checksum_path = path.with_suffix(path.suffix + ".sha256")
        if checksum_path.exists():
            expected = checksum_path.read_text(encoding="utf-8").strip()
            actual = _sha256_of_file(path)
            if actual != expected:
                raise RetrieverTrainingError(
                    f"Checkpoint at {path} failed integrity verification: expected "
                    f"sha256 {expected}, got {actual} -- refusing to load a possibly "
                    f"corrupted checkpoint"
                )

        checkpoint = torch.load(path, map_location="cpu")
        required_fields = ("model", "optimizer", "epoch", "step")
        missing_fields = [field for field in required_fields if field not in checkpoint]
        if missing_fields:
            raise RetrieverTrainingError(f"Checkpoint at {path} missing required fields: {missing_fields}")

        result = self._model.load_state_dict(checkpoint["model"], strict=strict)
        self._optimizer.load_state_dict(checkpoint["optimizer"])
        self._epoch = checkpoint["epoch"]
        self._step = checkpoint["step"]

        return CheckpointLoadResult(
            missing_keys=tuple(result.missing_keys),
            unexpected_keys=tuple(result.unexpected_keys),
            epoch=self._epoch,
            step=self._step,
        )
