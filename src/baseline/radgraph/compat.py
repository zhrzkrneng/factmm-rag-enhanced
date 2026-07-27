"""RadGraph / CheXbert compatibility layer.

Responsibility: isolate every workaround needed to run
``radgraph==0.0.9`` and ``f1chexbert==0.0.2`` (both vendoring ~2022-era
AllenNLP code) against a modern Python/``transformers``/
``huggingface_hub`` stack, as empirically discovered and verified in
Colab Cell 14 (see ``docs/colab_execution_log.md`` for the full
root-cause/fix/scientific-impact writeup of each shim below).

This module contains **no scientific or annotation logic whatsoever** —
it only makes third-party import/construction/inference calls succeed.
Nothing here decides what an entity, relation, or label *means*; it only
ensures the packages that compute those things can run at all in this
environment.

Design principles:

- Every patch is version-gated: :func:`check_environment` refuses to
  proceed (raises ``CompatibilityError``) if the installed
  ``radgraph``/``f1chexbert``/``transformers`` versions don't match
  what these specific shims were validated against. A future package
  upgrade must not silently inherit patches that were never verified
  against it.
- Every patch is individually self-guarding (checks whether the thing
  it targets is actually missing/broken before touching it) and
  idempotent (safe to call more than once in the same process). No
  patch wraps "whatever the current value is" — an earlier prototype
  that did caused a ``RecursionError`` across repeated Colab cell
  executions in the same kernel (see Cell 14's log entry); every patch
  here is a fully self-contained replacement instead.
- Every patch function accepts its target module/class as an optional
  parameter, defaulting to a real import only when not supplied. This
  makes the patching logic unit-testable with lightweight fake
  stand-ins, without requiring the actual (heavy) ML dependencies to be
  installed just to test this module.
- Shim 6 (see :func:`patch_preprocess_reports_dataset_field`) is
  categorically different from the other five: it supplies a data
  value the shipped code never set, rather than restoring documented
  legacy behavior. It requires an explicit ``allow_experimental=True``
  to apply, specifically so it can never be silently invoked as part of
  routine "apply everything" calls.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from importlib import metadata
from typing import Callable, Dict, List, Optional

from src.common.exceptions import CompatibilityError

# --- Versions these shims were empirically validated against ----------
# (Colab Cell 14, docs/colab_execution_log.md). Exact match required for
# radgraph/f1chexbert (the packages these shims specifically target).
# transformers is checked against a small allowlist rather than a single
# exact version, since it is not itself the target of these shims, but
# still deliberately restrictive -- add a new entry only after
# re-validating all shims against it.

TESTED_RADGRAPH_VERSION = "0.0.9"
TESTED_F1CHEXBERT_VERSION = "0.0.2"
TESTED_TRANSFORMERS_VERSIONS = ("4.57.6",)
MINIMUM_PYTHON_VERSION = (3, 11)  # shim 1 targets a bug that requires this


def _installed_version(package: str) -> str:
    """Return the installed version of `package`, or "not installed"."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not installed"


@dataclass(frozen=True)
class EnvironmentReport:
    """Snapshot of the versions relevant to these compatibility shims."""

    python_version: tuple
    transformers_version: str
    radgraph_version: str
    f1chexbert_version: str


def detect_environment() -> EnvironmentReport:
    """Return the currently installed versions, no validation performed."""
    return EnvironmentReport(
        python_version=sys.version_info[:2],
        transformers_version=_installed_version("transformers"),
        radgraph_version=_installed_version("radgraph"),
        f1chexbert_version=_installed_version("f1chexbert"),
    )


def check_environment(
    report: Optional[EnvironmentReport] = None,
) -> EnvironmentReport:
    """Verify the environment matches what these shims were validated against.

    Args:
        report: an already-computed :class:`EnvironmentReport`, or None
            to detect the live environment. Injectable for testing.

    Returns:
        The validated report.

    Raises:
        CompatibilityError: if ``radgraph``/``f1chexbert`` aren't at the
            exact tested versions, ``transformers`` isn't in the tested
            allowlist, or Python predates the version shim 1 requires.
            This is deliberate: these shims target specific,
            empirically-confirmed bugs in specific package versions.
            Silently applying them to an unvalidated future version
            could reintroduce a bug that version already fixed, or
            patch something that no longer needs patching in a way that
            masks a different real problem.
    """
    report = report if report is not None else detect_environment()
    problems: List[str] = []

    if report.python_version < MINIMUM_PYTHON_VERSION:
        problems.append(
            f"Python {'.'.join(map(str, report.python_version))} is "
            f"older than any version these shims were validated against "
            f"({'.'.join(map(str, MINIMUM_PYTHON_VERSION))}+). Shim 1 "
            f"targets a bytecode-encoding bug specific to Python "
            f"{'.'.join(map(str, MINIMUM_PYTHON_VERSION))}+; applying "
            f"it below that version has not been validated."
        )
    if report.radgraph_version != TESTED_RADGRAPH_VERSION:
        problems.append(
            f"radgraph=={report.radgraph_version!r} installed, but "
            f"these shims were only validated against "
            f"radgraph=={TESTED_RADGRAPH_VERSION!r}."
        )
    if report.f1chexbert_version != TESTED_F1CHEXBERT_VERSION:
        problems.append(
            f"f1chexbert=={report.f1chexbert_version!r} installed, but "
            f"these shims were only validated against "
            f"f1chexbert=={TESTED_F1CHEXBERT_VERSION!r}."
        )
    if report.transformers_version not in TESTED_TRANSFORMERS_VERSIONS:
        problems.append(
            f"transformers=={report.transformers_version!r} installed, "
            f"but these shims were only validated against one of "
            f"{TESTED_TRANSFORMERS_VERSIONS!r}. Do not assume the same "
            f"shims are correct for a different transformers version "
            f"without re-validating via the Cell 14 audit process."
        )

    if problems:
        raise CompatibilityError(
            "Unsupported environment for RadGraph/CheXbert compatibility "
            "shims:\n  - " + "\n  - ".join(problems)
        )
    return report


# --- Shim 1: overrides_.overrides bytecode-introspection bug -----------

def patch_overrides_bytecode_bug(overrides_module=None) -> bool:
    """Shim 1 (PERMANENT). Neutralize a Python 3.11+ bytecode-introspection
    bug in radgraph's vendored ``overrides_`` package.

    Original error: ``IndexError: tuple index out of range``, raised
    inside ``overrides_/overrides.py``'s ``_get_base_class_names`` while
    defining ``radgraph.allennlp.common.params.Params`` (via
    ``@overrides`` on ``Params.pop``).

    Root cause: ``overrides_.overrides`` determines which base class a
    decorated method overrides by disassembling the *calling* frame's
    bytecode (``dis.opname``, ``co_names[oparg]``). This assumes a
    pre-3.11 CPython bytecode encoding; Python 3.11+ changed how
    ``LOAD_GLOBAL``/``LOAD_ATTR`` operands are encoded, so the index
    lookup goes out of range.

    Why this is behavior-preserving: the decorator's only behavioral
    effect used elsewhere in the codebase is setting
    ``method.__override__ = True`` (checked by radgraph's vendored
    ``EnforceOverridesMeta``). The bytecode-walking loop this patch
    skips is a static assertion with no runtime effect on annotation
    results, model architecture, or numeric output.

    Must be called before ``import radgraph`` (or any of its
    submodules) — the decorator runs at radgraph's own module-import
    time.

    Args:
        overrides_module: the ``overrides_`` module object, or None to
            import it directly. Injectable for testing.

    Returns:
        True if the patch was applied, False if skipped (already
        patched in this process).
    """
    if overrides_module is None:
        import overrides_ as overrides_module

    if getattr(overrides_module.overrides, "_is_compat_shim", False):
        return False  # already patched; idempotent no-op

    def _safe_overrides(method):
        method.__override__ = True
        return method

    _safe_overrides._is_compat_shim = True
    overrides_module.overrides = _safe_overrides
    return True


# --- Shim 2: transformers.AdamW removed ---------------------------------

def patch_transformers_adamw(transformers_module=None, torch_optim_module=None) -> bool:
    """Shim 2 (PERMANENT). Restore ``transformers.AdamW`` if missing.

    Original error: ``AttributeError: module transformers has no
    attribute AdamW``.

    Root cause: ``radgraph.allennlp.training.optimizers`` defines
    ``class HuggingfaceAdamWOptimizer(Optimizer, transformers.AdamW)``
    at import time, unconditionally — even though this optimizer class
    is never instantiated for inference-only use (RadGraph is never
    trained by this project). ``transformers.AdamW`` was a deprecated
    re-export of ``torch.optim.AdamW``, removed in some releases.

    Why this is behavior-preserving: the class is defined but never
    instantiated by anything this project calls; the patch only lets
    the class *definition* succeed at import time. ``torch.optim.AdamW``
    is exactly what ``transformers.AdamW`` used to just be — a
    re-export, not a different implementation.

    Must be called before ``import radgraph`` — referenced at
    radgraph's own module-import time.

    Args:
        transformers_module: the ``transformers`` module, or None to
            import it directly. Injectable for testing.
        torch_optim_module: the ``torch.optim`` module, or None to
            import it directly. Injectable for testing.

    Returns:
        True if the patch was applied, False if ``transformers.AdamW``
        is already present natively (e.g. at ``transformers==4.57.6``,
        the version these shims were validated against).
    """
    if transformers_module is None:
        import transformers as transformers_module
    if torch_optim_module is None:
        import torch.optim as torch_optim_module

    if hasattr(transformers_module, "AdamW"):
        return False

    transformers_module.AdamW = torch_optim_module.AdamW
    return True


# --- Shim 3: cached_transformers.get_tokenizer add_special_tokens conflict --

def patch_cached_transformers_get_tokenizer(
    cached_transformers_module=None, transformers_module=None
) -> bool:
    """Shim 3 (PERMANENT). Drop the conflicting ``add_special_tokens``
    constructor kwarg from radgraph's cached tokenizer loader.

    Original error: ``AttributeError: add_special_tokens conflicts with
    the method add_special_tokens in BertTokenizer``.

    Root cause: ``PretrainedTransformerTokenizer.__init__`` calls
    ``cached_transformers.get_tokenizer(model_name,
    add_special_tokens=False, **kwargs)``, forwarding to
    ``AutoTokenizer.from_pretrained(..., add_special_tokens=False)``.
    Modern ``transformers`` added a defensive check in
    ``PreTrainedTokenizerBase.__init__`` that rejects any init kwarg
    whose name collides with an existing callable method —
    ``add_special_tokens`` is both a legacy init kwarg and a real method
    (adds new vocabulary tokens) on ``BertTokenizer``.

    Why this is behavior-preserving: modern tokenizers only honor
    ``add_special_tokens`` as a per-call encode-time argument, never as
    a stored constructor default — that is precisely why the
    conflict-check exists (the kwarg no longer did anything meaningful
    at construction time in this transformers generation). Dropping it
    here changes nothing about how tokens are actually produced;
    RadGraph's real encode-time calls (via ``encode_plus``, shim 4)
    still pass ``add_special_tokens`` explicitly.

    Must be called after ``import radgraph`` (needs the
    ``radgraph.allennlp.common.cached_transformers`` submodule to
    exist). Self-contained — does not wrap "the current value" — so
    it is safe to call repeatedly in the same process; an earlier
    prototype that wrapped prior state caused a ``RecursionError``
    across repeated notebook executions (see Cell 14's log entry).

    Args:
        cached_transformers_module: the
            ``radgraph.allennlp.common.cached_transformers`` module, or
            None to import it directly. Injectable for testing.
        transformers_module: the ``transformers`` module, or None to
            import it directly. Injectable for testing.

    Returns:
        True if the patch was applied, False if already applied
        (idempotent no-op).
    """
    if cached_transformers_module is None:
        import radgraph.allennlp.common.cached_transformers as cached_transformers_module
    if transformers_module is None:
        import transformers as transformers_module

    if getattr(cached_transformers_module.get_tokenizer, "_is_compat_shim", False):
        return False

    tokenizer_cache: Dict = {}

    def _patched_get_tokenizer(model_name, **kwargs):
        kwargs.pop("add_special_tokens", None)
        cache_key = (model_name, frozenset(kwargs.items()))
        if cache_key not in tokenizer_cache:
            tokenizer_cache[cache_key] = (
                transformers_module.AutoTokenizer.from_pretrained(
                    model_name, **kwargs
                )
            )
        return tokenizer_cache[cache_key]

    _patched_get_tokenizer._is_compat_shim = True
    cached_transformers_module.get_tokenizer = _patched_get_tokenizer
    return True


# --- Shim 4: PreTrainedTokenizerBase.encode_plus removed ----------------

def patch_tokenizer_encode_plus(tokenizer_base_class=None) -> bool:
    """Shim 4 (PERMANENT). Restore ``PreTrainedTokenizerBase.encode_plus``.

    Original error: ``AttributeError: BertTokenizer has no attribute
    encode_plus``.

    Root cause: ``encode_plus``/``batch_encode_plus`` (long-deprecated
    single/batch encoding methods) have been fully removed from this
    transformers generation's tokenizer class. AllenNLP's
    ``PretrainedTransformerTokenizer`` calls ``.encode_plus(...)`` at 4
    call sites (confirmed by direct source inspection, all within one
    file), all using parameter names (``add_special_tokens``,
    ``max_length``, ``stride``, ``truncation``, ``return_tensors``,
    ``return_offsets_mapping``, ``return_attention_mask``,
    ``return_token_type_ids``) that the modern ``__call__`` method still
    accepts — ``encode_plus`` was historically just a thin wrapper
    around the same internal logic ``__call__`` uses.

    Also handles ``f1chexbert``'s own use of ``encode_plus`` with a
    ``List[str]`` (pre-split word tokens) as ``text`` — the historic
    ``encode_plus`` implicitly treated a bare ``List[str]`` as one
    pre-tokenized example, whereas modern ``__call__`` treats it as a
    *batch* of separate single-word examples unless
    ``is_split_into_words=True`` is set. This function restores that
    implicit behavior.

    Why this is behavior-preserving: this is a direct reimplementation
    of ``encode_plus``'s exact, unchanged-for-years wrapper behavior; it
    does not alter what token IDs are produced for either RadGraph's or
    CheXbert's actual encoding calls, only how the removed method name
    is dispatched to the modern equivalent API.

    Args:
        tokenizer_base_class: the ``transformers.PreTrainedTokenizerBase``
            class, or None to import ``transformers`` and use its
            attribute directly. Injectable for testing.

    Returns:
        True if the patch was applied, False if ``encode_plus`` is
        already present (native method, or already patched by us).
    """
    if tokenizer_base_class is None:
        import transformers

        tokenizer_base_class = transformers.PreTrainedTokenizerBase

    if getattr(tokenizer_base_class, "encode_plus", None) is not None:
        return False

    def _encode_plus_compat(self, text, text_pair=None, **kwargs):
        if isinstance(text, list) and "is_split_into_words" not in kwargs:
            kwargs["is_split_into_words"] = True
        return self(text, text_pair=text_pair, **kwargs)

    _encode_plus_compat._is_compat_shim = True
    tokenizer_base_class.encode_plus = _encode_plus_compat
    return True


# --- Shim 5: PreTrainedTokenizerBase.build_inputs_with_special_tokens removed --

def patch_tokenizer_build_inputs_with_special_tokens(tokenizer_base_class=None) -> bool:
    """Shim 5 (PERMANENT). Restore
    ``PreTrainedTokenizerBase.build_inputs_with_special_tokens``.

    Original error: ``AttributeError: BertTokenizer has no attribute
    build_inputs_with_special_tokens``.

    Root cause: another BERT-tokenizer method removed from this
    tokenizer backend. AllenNLP's ``PretrainedTransformerIndexer.
    _postprocess_output`` calls it directly to wrap token-ID segments
    with CLS/SEP markers.

    Why this is behavior-preserving: this is the standard,
    unchanged-for-years BERT convention (``[CLS] A [SEP]`` for a single
    sequence, ``[CLS] A [SEP] B [SEP]`` for a pair), using
    ``cls_token_id``/``sep_token_id`` (still-present, non-deprecated
    attributes) — every historical ``transformers`` version implemented
    this identically, so restoring it directly cannot produce output
    different from what the original (working) package version would
    have.

    Args:
        tokenizer_base_class: the ``transformers.PreTrainedTokenizerBase``
            class, or None to import ``transformers`` and use its
            attribute directly. Injectable for testing.

    Returns:
        True if the patch was applied, False if the method is already
        present (native, or already patched by us).
    """
    if tokenizer_base_class is None:
        import transformers

        tokenizer_base_class = transformers.PreTrainedTokenizerBase

    if (
        getattr(tokenizer_base_class, "build_inputs_with_special_tokens", None)
        is not None
    ):
        return False

    def _build_inputs_with_special_tokens_compat(self, token_ids_0, token_ids_1=None):
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    _build_inputs_with_special_tokens_compat._is_compat_shim = True
    tokenizer_base_class.build_inputs_with_special_tokens = (
        _build_inputs_with_special_tokens_compat
    )
    return True


# --- Shim 6 (TEMPORARY, pending validation): preprocess_reports "dataset" field --

def patch_preprocess_reports_dataset_field(
    *, allow_experimental: bool = False, radgraph_module=None
) -> bool:
    """Shim 6 (TEMPORARY — pending validation in Milestone 2.3).

    Unlike shims 1-5, this is **not** a restoration of documented,
    unchanged legacy behavior — it supplies a data value the shipped
    code never set. Requires ``allow_experimental=True`` to apply,
    specifically so it can never be silently invoked as part of a
    routine "apply everything" call.

    Original error: ``KeyError: 'None__ner_labels'``, deep in the
    DyGIE++ model's NER-scorer ``ModuleDict`` lookup
    (``radgraph/dygie/models/ner.py``).

    Root cause: confirmed via direct inspection of the loaded model's
    vocabulary (``model._ner._ner_scorers.keys()``,  ``model.vocab``) —
    the only registered namespaces are
    ``radgraph__ner_labels``/``radgraph__relation_labels``, meaning
    every training instance for this model was tagged
    ``"dataset": "radgraph"``. The package's own
    ``preprocess_reports()`` (used by the public ``RadGraph.__call__``
    inference path) never sets a ``"dataset"`` field on the instances
    it builds, so ``Document.from_json``'s ``js.get("dataset")``
    silently returns ``None``, which stringifies to the literal
    ``"None"`` in the constructed label-namespace name — matching
    nothing in the model's actual vocabulary.

    Why this stays temporary: ``"radgraph"`` is the *only* namespace
    present in the archived model's vocabulary, so there is no other
    plausible value it could need — but this has not been independently
    verified against a reference run from the original paper-era
    environment, only that it is the value this specific downloaded
    model archive's vocabulary requires. Must be re-examined during
    Milestone 2.3 (pair mining) validation, by checking whether
    annotation output on real or synthetic data looks sensible and
    internally consistent.

    Must be called after ``import radgraph`` (needs the
    ``radgraph.radgraph`` submodule).

    Args:
        allow_experimental: must be explicitly True to apply the patch.
            Defaults to False, so this is never silently invoked as
            part of routine "apply everything" calls.
        radgraph_module: the ``radgraph.radgraph`` submodule object, or
            None to import it directly. Injectable for testing.

    Returns:
        True if the patch was applied, False if skipped (already
        applied, or ``allow_experimental`` was not set).
    """
    if not allow_experimental:
        return False

    if radgraph_module is None:
        import radgraph.radgraph as radgraph_module

    if getattr(radgraph_module.preprocess_reports, "_is_compat_shim", False):
        return False

    def _patched_preprocess_reports(report_list):
        final_list = []
        for idx, report in enumerate(report_list):
            sen = re.sub(
                "(?<! )(?=[/,-,:,.,!?()])|(?<=[/,-,:,.,!?()])(?! )", r" ", report
            ).split()
            final_list.append({
                "doc_key": str(idx),
                "dataset": "radgraph",
                "sentences": [sen],
            })
        return [json.dumps(item) for item in final_list]

    _patched_preprocess_reports._is_compat_shim = True
    radgraph_module.preprocess_reports = _patched_preprocess_reports
    return True


# --- Orchestration -------------------------------------------------------

def patch_pre_import() -> List[str]:
    """Apply every shim that must run before ``import radgraph``.

    Checks the environment first (raises ``CompatibilityError`` on an
    unvalidated version combination).

    Returns:
        Names of the shims actually applied (empty if none were needed).
    """
    check_environment()
    applied = []
    if patch_overrides_bytecode_bug():
        applied.append("overrides_bytecode_bug")
    if patch_transformers_adamw():
        applied.append("transformers_adamw")
    if patch_tokenizer_encode_plus():
        applied.append("tokenizer_encode_plus")
    if patch_tokenizer_build_inputs_with_special_tokens():
        applied.append("tokenizer_build_inputs_with_special_tokens")
    return applied


def patch_post_import(*, allow_experimental_dataset_field: bool = False) -> List[str]:
    """Apply every shim that requires radgraph's submodules to already
    be importable.

    Args:
        allow_experimental_dataset_field: must be explicitly True to
            apply shim 6 (see
            :func:`patch_preprocess_reports_dataset_field`).

    Returns:
        Names of the shims actually applied.
    """
    check_environment()
    applied = []
    if patch_cached_transformers_get_tokenizer():
        applied.append("cached_transformers_get_tokenizer")
    if patch_preprocess_reports_dataset_field(
        allow_experimental=allow_experimental_dataset_field
    ):
        applied.append("preprocess_reports_dataset_field [EXPERIMENTAL]")
    return applied


def patch_all(*, allow_experimental_dataset_field: bool = False) -> List[str]:
    """Convenience: patch_pre_import() -> import radgraph -> patch_post_import().

    For callers who don't need finer control over exactly when
    ``import radgraph`` happens. Use :func:`patch_pre_import`/
    :func:`patch_post_import` directly for more control (e.g. testing).
    """
    applied = patch_pre_import()
    import radgraph  # noqa: F401

    applied += patch_post_import(
        allow_experimental_dataset_field=allow_experimental_dataset_field
    )
    return applied
