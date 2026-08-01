"""Unit tests for src/baseline/radgraph/compat.py.

These tests exercise only the compatibility-shim logic itself, using
lightweight fake stand-ins for `overrides_`/`transformers`/`torch.optim`/
`radgraph` submodules (injected via each function's optional module
parameters). They deliberately do NOT install or import the real heavy
ML packages, and do NOT run any part of the actual RadGraph annotation
pipeline -- that is out of scope until Milestone 2.3.
"""

import json
import types
from pathlib import Path

import pytest

from src.baseline.radgraph import compat
from src.common.exceptions import CompatibilityError


# --- detect_environment / check_environment -----------------------------


def test_installed_version_reports_not_installed_for_missing_package():
    assert compat._installed_version("definitely-not-a-real-package-xyz") == "not installed"


def test_detect_environment_returns_report_with_python_version():
    report = compat.detect_environment()
    assert report.python_version == compat.EnvironmentReport(
        python_version=report.python_version,
        transformers_version=report.transformers_version,
        radgraph_version=report.radgraph_version,
        f1chexbert_version=report.f1chexbert_version,
    ).python_version
    assert isinstance(report.python_version, tuple)


def test_check_environment_passes_on_exactly_tested_versions():
    report = compat.EnvironmentReport(
        python_version=(3, 12),
        transformers_version=compat.TESTED_TRANSFORMERS_VERSIONS[0],
        radgraph_version=compat.TESTED_RADGRAPH_VERSION,
        f1chexbert_version=compat.TESTED_F1CHEXBERT_VERSION,
    )
    assert compat.check_environment(report) is report


def test_check_environment_raises_on_wrong_radgraph_version():
    report = compat.EnvironmentReport(
        python_version=(3, 12),
        transformers_version=compat.TESTED_TRANSFORMERS_VERSIONS[0],
        radgraph_version="0.0.8",
        f1chexbert_version=compat.TESTED_F1CHEXBERT_VERSION,
    )
    with pytest.raises(CompatibilityError, match="radgraph"):
        compat.check_environment(report)


def test_check_environment_raises_on_wrong_f1chexbert_version():
    report = compat.EnvironmentReport(
        python_version=(3, 12),
        transformers_version=compat.TESTED_TRANSFORMERS_VERSIONS[0],
        radgraph_version=compat.TESTED_RADGRAPH_VERSION,
        f1chexbert_version="0.0.1",
    )
    with pytest.raises(CompatibilityError, match="f1chexbert"):
        compat.check_environment(report)


def test_check_environment_raises_on_untested_transformers_version():
    report = compat.EnvironmentReport(
        python_version=(3, 12),
        transformers_version="4.23.1",
        radgraph_version=compat.TESTED_RADGRAPH_VERSION,
        f1chexbert_version=compat.TESTED_F1CHEXBERT_VERSION,
    )
    with pytest.raises(CompatibilityError, match="transformers"):
        compat.check_environment(report)


def test_check_environment_raises_on_python_below_minimum():
    report = compat.EnvironmentReport(
        python_version=(3, 10),
        transformers_version=compat.TESTED_TRANSFORMERS_VERSIONS[0],
        radgraph_version=compat.TESTED_RADGRAPH_VERSION,
        f1chexbert_version=compat.TESTED_F1CHEXBERT_VERSION,
    )
    with pytest.raises(CompatibilityError, match="Python"):
        compat.check_environment(report)


def test_check_environment_reports_every_problem_at_once():
    report = compat.EnvironmentReport(
        python_version=(3, 10),
        transformers_version="4.23.1",
        radgraph_version="0.0.8",
        f1chexbert_version="0.0.1",
    )
    with pytest.raises(CompatibilityError) as exc_info:
        compat.check_environment(report)
    message = str(exc_info.value)
    for keyword in ("Python", "radgraph", "f1chexbert", "transformers"):
        assert keyword in message


# --- Shim 1: overrides_ bytecode bug -------------------------------------


def test_patch_overrides_bytecode_bug_applies_and_is_idempotent():
    def _broken_overrides(method):
        raise IndexError("tuple index out of range")

    fake_overrides_module = types.SimpleNamespace(overrides=_broken_overrides)

    applied = compat.patch_overrides_bytecode_bug(fake_overrides_module)
    assert applied is True
    assert fake_overrides_module.overrides is not _broken_overrides

    def dummy():
        pass

    result = fake_overrides_module.overrides(dummy)
    assert result is dummy
    assert dummy.__override__ is True

    applied_again = compat.patch_overrides_bytecode_bug(fake_overrides_module)
    assert applied_again is False


# --- Shim 2: transformers.AdamW ------------------------------------------


def test_patch_transformers_adamw_applies_when_missing():
    fake_transformers_module = types.SimpleNamespace()
    sentinel_adamw = object()
    fake_torch_optim_module = types.SimpleNamespace(AdamW=sentinel_adamw)

    applied = compat.patch_transformers_adamw(
        fake_transformers_module, fake_torch_optim_module
    )
    assert applied is True
    assert fake_transformers_module.AdamW is sentinel_adamw


def test_patch_transformers_adamw_skips_when_already_present():
    existing_adamw = object()
    fake_transformers_module = types.SimpleNamespace(AdamW=existing_adamw)
    fake_torch_optim_module = types.SimpleNamespace(AdamW=object())

    applied = compat.patch_transformers_adamw(
        fake_transformers_module, fake_torch_optim_module
    )
    assert applied is False
    assert fake_transformers_module.AdamW is existing_adamw


# --- Shim 3: cached_transformers.get_tokenizer ---------------------------


def test_patch_cached_transformers_get_tokenizer_strips_conflicting_kwarg():
    calls = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return object()

    fake_transformers_module = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    fake_cached_transformers_module = types.SimpleNamespace(
        get_tokenizer=lambda model_name, **kwargs: None
    )

    applied = compat.patch_cached_transformers_get_tokenizer(
        fake_cached_transformers_module, fake_transformers_module
    )
    assert applied is True

    fake_cached_transformers_module.get_tokenizer(
        "bert-base-uncased", add_special_tokens=False, do_lower_case=True
    )
    assert len(calls) == 1
    model_name, kwargs = calls[0]
    assert model_name == "bert-base-uncased"
    assert "add_special_tokens" not in kwargs
    assert kwargs == {"do_lower_case": True}


def test_patch_cached_transformers_get_tokenizer_caches_by_args():
    calls = []

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(model_name, **kwargs):
            calls.append((model_name, kwargs))
            return object()

    fake_transformers_module = types.SimpleNamespace(AutoTokenizer=FakeAutoTokenizer)
    fake_cached_transformers_module = types.SimpleNamespace(
        get_tokenizer=lambda model_name, **kwargs: None
    )

    compat.patch_cached_transformers_get_tokenizer(
        fake_cached_transformers_module, fake_transformers_module
    )

    first = fake_cached_transformers_module.get_tokenizer("bert-base-uncased")
    second = fake_cached_transformers_module.get_tokenizer("bert-base-uncased")
    assert first is second
    assert len(calls) == 1


def test_patch_cached_transformers_get_tokenizer_is_idempotent():
    fake_transformers_module = types.SimpleNamespace(AutoTokenizer=object())
    fake_cached_transformers_module = types.SimpleNamespace(
        get_tokenizer=lambda model_name, **kwargs: None
    )

    first_applied = compat.patch_cached_transformers_get_tokenizer(
        fake_cached_transformers_module, fake_transformers_module
    )
    second_applied = compat.patch_cached_transformers_get_tokenizer(
        fake_cached_transformers_module, fake_transformers_module
    )
    assert first_applied is True
    assert second_applied is False


# --- Shim 4: encode_plus --------------------------------------------------


class _FakeTokenizerBase:
    def __call__(self, text, text_pair=None, **kwargs):
        return {"text": text, "text_pair": text_pair, "kwargs": kwargs}


def test_patch_tokenizer_encode_plus_applies_when_missing():
    applied = compat.patch_tokenizer_encode_plus(_FakeTokenizerBase)
    assert applied is True
    assert _FakeTokenizerBase.encode_plus is not None


def test_patch_tokenizer_encode_plus_skips_when_already_present():
    class TokenizerWithEncodePlus:
        def encode_plus(self, text, **kwargs):
            return "native"

    applied = compat.patch_tokenizer_encode_plus(TokenizerWithEncodePlus)
    assert applied is False


def test_encode_plus_sets_is_split_into_words_for_list_input():
    class FakeTokenizer(_FakeTokenizerBase):
        pass

    compat.patch_tokenizer_encode_plus(FakeTokenizer)
    instance = FakeTokenizer()
    result = instance.encode_plus(["already", "tokenized", "words"])
    assert result["kwargs"]["is_split_into_words"] is True


def test_encode_plus_leaves_string_input_untouched():
    class FakeTokenizer(_FakeTokenizerBase):
        pass

    compat.patch_tokenizer_encode_plus(FakeTokenizer)
    instance = FakeTokenizer()
    result = instance.encode_plus("a plain string")
    assert "is_split_into_words" not in result["kwargs"]


def test_encode_plus_respects_explicit_is_split_into_words():
    class FakeTokenizer(_FakeTokenizerBase):
        pass

    compat.patch_tokenizer_encode_plus(FakeTokenizer)
    instance = FakeTokenizer()
    result = instance.encode_plus(["a", "list"], is_split_into_words=False)
    assert result["kwargs"]["is_split_into_words"] is False


# --- Shim 5: build_inputs_with_special_tokens ----------------------------


class _FakeTokenizerForBuildInputs:
    cls_token_id = 101
    sep_token_id = 102


def test_patch_build_inputs_with_special_tokens_applies_when_missing():
    applied = compat.patch_tokenizer_build_inputs_with_special_tokens(
        _FakeTokenizerForBuildInputs
    )
    assert applied is True


def test_patch_build_inputs_with_special_tokens_skips_when_present():
    class TokenizerWithMethod:
        def build_inputs_with_special_tokens(self, ids0, ids1=None):
            return "native"

    applied = compat.patch_tokenizer_build_inputs_with_special_tokens(
        TokenizerWithMethod
    )
    assert applied is False


def test_build_inputs_with_special_tokens_single_sequence():
    class FakeTokenizer(_FakeTokenizerForBuildInputs):
        pass

    compat.patch_tokenizer_build_inputs_with_special_tokens(FakeTokenizer)
    instance = FakeTokenizer()
    result = instance.build_inputs_with_special_tokens([5, 6, 7])
    assert result == [101, 5, 6, 7, 102]


def test_build_inputs_with_special_tokens_pair():
    class FakeTokenizer(_FakeTokenizerForBuildInputs):
        pass

    compat.patch_tokenizer_build_inputs_with_special_tokens(FakeTokenizer)
    instance = FakeTokenizer()
    result = instance.build_inputs_with_special_tokens([5, 6], [8, 9])
    assert result == [101, 5, 6, 102, 8, 9, 102]


# --- Shim 6: preprocess_reports dataset field (experimental) -------------


def test_patch_preprocess_reports_skipped_without_allow_experimental():
    fake_radgraph_module = types.SimpleNamespace(
        preprocess_reports=lambda report_list: []
    )
    original = fake_radgraph_module.preprocess_reports

    applied = compat.patch_preprocess_reports_dataset_field(
        allow_experimental=False, radgraph_module=fake_radgraph_module
    )
    assert applied is False
    assert fake_radgraph_module.preprocess_reports is original


def test_patch_preprocess_reports_applies_with_allow_experimental():
    fake_radgraph_module = types.SimpleNamespace(
        preprocess_reports=lambda report_list: []
    )

    applied = compat.patch_preprocess_reports_dataset_field(
        allow_experimental=True, radgraph_module=fake_radgraph_module
    )
    assert applied is True

    output = fake_radgraph_module.preprocess_reports(["No acute findings."])
    assert len(output) == 1
    instance = json.loads(output[0])
    assert instance["dataset"] == "radgraph"
    assert instance["doc_key"] == "0"
    assert instance["sentences"] == [["No", "acute", "findings", "."]]


def test_patch_preprocess_reports_is_idempotent():
    fake_radgraph_module = types.SimpleNamespace(
        preprocess_reports=lambda report_list: []
    )

    first_applied = compat.patch_preprocess_reports_dataset_field(
        allow_experimental=True, radgraph_module=fake_radgraph_module
    )
    second_applied = compat.patch_preprocess_reports_dataset_field(
        allow_experimental=True, radgraph_module=fake_radgraph_module
    )
    assert first_applied is True
    assert second_applied is False


# --- Orchestration: fail-safe behavior on an unvalidated environment ----
#
# These force check_environment() to reject the environment via
# monkeypatch, rather than relying on radgraph/f1chexbert/transformers
# actually being absent from whatever machine runs this suite. The
# latter is true in the author's own sandbox but false once these
# packages are installed and validated (e.g. after Cell 14 in Colab) --
# a hermetic test must not depend on that incidental fact.


def _force_incompatible_environment(monkeypatch):
    def _raise(report=None):
        raise CompatibilityError("forced by test: unsupported environment")

    monkeypatch.setattr(compat, "check_environment", _raise)


def test_patch_pre_import_fails_safe_on_unsupported_environment(monkeypatch):
    _force_incompatible_environment(monkeypatch)
    with pytest.raises(CompatibilityError):
        compat.patch_pre_import()


def test_patch_post_import_fails_safe_on_unsupported_environment(monkeypatch):
    _force_incompatible_environment(monkeypatch)
    with pytest.raises(CompatibilityError):
        compat.patch_post_import()


def test_patch_all_fails_safe_on_unsupported_environment_before_import(monkeypatch):
    _force_incompatible_environment(monkeypatch)
    # patch_all() must reject the environment in patch_pre_import() and
    # never reach `import radgraph` at all -- simulate radgraph being
    # unimportable so the test would fail loudly (ImportError, not the
    # expected CompatibilityError) if that ordering were ever broken.
    import builtins

    real_import = builtins.__import__

    def _fail_if_radgraph_imported(name, *args, **kwargs):
        if name == "radgraph" or name.startswith("radgraph."):
            raise AssertionError(
                "patch_all() attempted `import radgraph` despite "
                "check_environment() having already rejected the "
                "environment -- the pre-import safety check is not "
                "actually gating the import."
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fail_if_radgraph_imported)
    with pytest.raises(CompatibilityError):
        compat.patch_all()


# --- Checkpoint cache preparation (NOT a compatibility shim) -------------


def test_preplace_checkpoint_creates_cache_directory_when_missing(tmp_path):
    cache_dir = tmp_path / "nested" / "cache_dir"
    spec = compat.CheckpointSpec(filename="model.bin", cache_dir=str(cache_dir))

    def fake_download(*, repo_id, filename, cache_dir):
        return str(Path(cache_dir) / "models--fake--repo" / "snapshots" / "abc123" / filename)

    def fake_copy(src, dst):
        Path(dst).write_bytes(b"fake-bytes")

    assert not cache_dir.exists()
    result = compat.preplace_checkpoint(spec, hf_hub_download_fn=fake_download, copy_fn=fake_copy)
    assert cache_dir.is_dir()
    assert result.newly_placed is True
    assert Path(result.destination_path).is_file()


def test_preplace_checkpoint_downloads_and_extracts_revision_via_download_path(tmp_path):
    cache_dir = tmp_path / "cache"
    spec = compat.CheckpointSpec(filename="model.bin", cache_dir=str(cache_dir))

    def fake_download(*, repo_id, filename, cache_dir):
        return str(
            Path(cache_dir) / "models--StanfordAIMI--RRG_scorers" / "snapshots"
            / "deadbeef1234" / filename
        )

    def fake_copy(src, dst):
        Path(dst).write_bytes(b"fake-bytes")

    result = compat.preplace_checkpoint(spec, hf_hub_download_fn=fake_download, copy_fn=fake_copy)
    assert result.newly_placed is True
    assert result.resolved_revision == "deadbeef1234"
    assert result.revision_source == "download_path"


def test_preplace_checkpoint_skips_download_and_reports_preexisting_file_when_already_present(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    destination = cache_dir / "model.bin"
    destination.write_bytes(b"already-here")
    spec = compat.CheckpointSpec(filename="model.bin", cache_dir=str(cache_dir))

    calls = []

    def fake_download(**kwargs):
        calls.append(kwargs)
        raise AssertionError("hf_hub_download_fn should not be called when destination already exists")

    result = compat.preplace_checkpoint(spec, hf_hub_download_fn=fake_download)
    assert result.newly_placed is False
    assert result.resolved_revision is None
    assert result.revision_source == "preexisting_file"
    assert calls == []


def test_preplace_checkpoint_uses_temp_destination_then_atomic_replace(tmp_path):
    cache_dir = tmp_path / "cache"
    spec = compat.CheckpointSpec(filename="model.bin", cache_dir=str(cache_dir))
    final_destination = cache_dir / "model.bin"

    copy_calls = []

    def fake_download(*, repo_id, filename, cache_dir):
        return str(Path(cache_dir) / "snapshots" / "rev1" / filename)

    def fake_copy(src, dst):
        copy_calls.append((src, dst))
        Path(dst).write_bytes(b"fake-bytes")

    compat.preplace_checkpoint(spec, hf_hub_download_fn=fake_download, copy_fn=fake_copy)

    assert len(copy_calls) == 1
    _, copy_dst = copy_calls[0]
    assert copy_dst != str(final_destination)
    assert ".tmp" in copy_dst
    assert final_destination.is_file()
    assert not Path(copy_dst).exists()  # renamed into place by os.replace, not left behind


def test_preplace_checkpoint_failed_copy_never_leaves_file_at_final_destination(tmp_path):
    cache_dir = tmp_path / "cache"
    spec = compat.CheckpointSpec(filename="model.bin", cache_dir=str(cache_dir))
    final_destination = cache_dir / "model.bin"

    def fake_download(*, repo_id, filename, cache_dir):
        return str(Path(cache_dir) / "snapshots" / "rev1" / filename)

    def failing_copy(src, dst):
        raise OSError("simulated disk failure mid-copy")

    with pytest.raises(OSError):
        compat.preplace_checkpoint(spec, hf_hub_download_fn=fake_download, copy_fn=failing_copy)
    assert not final_destination.exists()


def test_preplace_checkpoint_unknown_revision_is_none_and_unavailable_not_unresolved_string(tmp_path):
    cache_dir = tmp_path / "cache"
    spec = compat.CheckpointSpec(filename="model.bin", cache_dir=str(cache_dir))

    def fake_download(*, repo_id, filename, cache_dir):
        # No "snapshots/<hash>/" segment -- simulates a future cache-layout change.
        return str(Path(cache_dir) / "flat" / filename)

    def fake_copy(src, dst):
        Path(dst).write_bytes(b"fake-bytes")

    result = compat.preplace_checkpoint(spec, hf_hub_download_fn=fake_download, copy_fn=fake_copy)
    assert result.resolved_revision is None
    assert result.revision_source == "unavailable"
    assert result.resolved_revision != "unresolved"


def test_preplace_all_checkpoints_returns_result_for_both_files(tmp_path, monkeypatch):
    monkeypatch.setattr(
        compat, "RADGRAPH_CHECKPOINT",
        compat.CheckpointSpec("radgraph.tar.gz", str(tmp_path / "radgraph")),
    )
    monkeypatch.setattr(
        compat, "CHEXBERT_CHECKPOINT",
        compat.CheckpointSpec("chexbert.pth", str(tmp_path / "chexbert")),
    )

    def fake_download(*, repo_id, filename, cache_dir):
        return str(Path(cache_dir) / "snapshots" / "rev1" / filename)

    def fake_copy(src, dst):
        Path(dst).write_bytes(b"fake-bytes")

    results = compat.preplace_all_checkpoints(hf_hub_download_fn=fake_download, copy_fn=fake_copy)
    assert set(results.keys()) == {"radgraph.tar.gz", "chexbert.pth"}
    for result in results.values():
        assert result.newly_placed is True
        assert result.resolved_revision == "rev1"
