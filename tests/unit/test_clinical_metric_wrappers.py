"""Unit tests for the clinical metric wrappers in
src/evaluation/generation_metrics.py (Milestone 2.6, Cell 32).

Scope: F1RadGraphMetric, DatasetF1ChexbertMetric, InstanceF1ChexbertMetric,
the CHEXBERT_5_INDICES reduction helper, the instance-agreement formula
(cross-checked against src.baseline.pair_mining.similarity.chexbert_similarity
for equivalence), and the expanded metric registry. Every real wrapper is
exercised only via injected fakes -- radgraph/f1chexbert are not
installed in this sandbox (same guarantee already established in Cell
31's own test suite), so a test that forgets to inject a fake fails
loudly with a real ImportError classified as "missing_dependency"."""

import ast
import dataclasses
import inspect

import pytest

import src.evaluation.generation_metrics as gm
from src.evaluation.generation_metrics import (
    DatasetF1ChexbertMetric,
    F1RadGraphMetric,
    GenerationMetricsConfig,
    GenerationMetricWrapper,
    InstanceF1ChexbertMetric,
    MockGenerationMetricWrapper,
    build_metric_registry,
    build_mock_metric_registry,
)
from src.common.exceptions import EvaluationError
from src.baseline.pair_mining.similarity import chexbert_similarity


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _CountingFactory:
    def __init__(self, build):
        self._build = build
        self.call_count = 0

    def __call__(self):
        self.call_count += 1
        return self._build()


class _FakeF1RadGraph:
    def __init__(self, mean_reward, reward_list, hyp_ann=None, ref_ann=None):
        self._mean_reward = mean_reward
        self._reward_list = reward_list
        self._hyp_ann = hyp_ann if hyp_ann is not None else {}
        self._ref_ann = ref_ann if ref_ann is not None else {}
        self.calls = []

    def __call__(self, hyps, refs):
        self.calls.append((list(hyps), list(refs)))
        return self._mean_reward, self._reward_list, self._hyp_ann, self._ref_ann


class _RaisingF1RadGraph:
    def __init__(self, exc):
        self._exc = exc

    def __call__(self, hyps, refs):
        raise self._exc


class _FakeF1Chexbert:
    """Fake for both dataset-level (__call__) and instance-level
    (get_label) usage -- exactly matching the real f1chexbert.F1CheXbert
    class's dual role that both wrapper classes exploit."""

    def __init__(self, *, micro_f1=None, label_map=None):
        self._micro_f1 = micro_f1
        self._label_map = label_map or {}
        self.call_args = []
        self.get_label_calls = []

    def __call__(self, hyps, refs):
        self.call_args.append((list(hyps), list(refs)))
        class_report_5 = {"micro avg": {"f1-score": self._micro_f1}}
        return None, None, None, class_report_5

    def get_label(self, text):
        self.get_label_calls.append(text)
        return self._label_map[text]


class _RaisingF1Chexbert:
    def __init__(self, exc, *, raise_on="call"):
        self._exc = exc
        self._raise_on = raise_on

    def __call__(self, hyps, refs):
        if self._raise_on == "call":
            raise self._exc
        return None, None, None, {"micro avg": {"f1-score": 0.0}}

    def get_label(self, text):
        if self._raise_on == "get_label":
            raise self._exc
        return [0] * 14


def _label14(*ones_at):
    vec = [0] * 14
    for i in ones_at:
        vec[i] = 1
    return vec


# ---------------------------------------------------------------------------
# Module-scope import discipline (lazy dependency loading)
# ---------------------------------------------------------------------------


def test_module_never_imports_radgraph_or_f1chexbert_at_module_scope():
    source = inspect.getsource(gm)
    tree = ast.parse(source)
    top_level_imports = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert "radgraph" not in top_level_imports
    assert "f1chexbert" not in top_level_imports


# A prior version of this file asserted radgraph/f1chexbert are not
# installed, as a belt-and-suspenders sandbox guarantee. Removed: that
# is an environment property, not a property of this module's own
# code, and it is FALSE (correctly, expectedly) in a real Colab
# session where Cell 14 (Milestone 2.2) already installed both --
# confirmed live when this exact test failed there. The real guarantee
# that no test here accidentally depends on a real library is already
# structural: every test injects its own fake factory, and
# test_module_never_imports_radgraph_or_f1chexbert_at_module_scope
# (above) statically verifies the module itself never imports either
# at module scope, regardless of what happens to be installed.


# ---------------------------------------------------------------------------
# Uniform interface conformance
# ---------------------------------------------------------------------------


def test_all_three_clinical_wrappers_share_identical_compute_signature():
    for cls in (F1RadGraphMetric, DatasetF1ChexbertMetric, InstanceF1ChexbertMetric):
        sig = inspect.signature(cls.compute)
        assert list(sig.parameters) == ["self", "hyps", "refs"]
        assert issubclass(cls, GenerationMetricWrapper)


# ---------------------------------------------------------------------------
# The two F1CheXbert formulas must never be conflated
# ---------------------------------------------------------------------------


def test_dataset_and_instance_chexbert_are_genuinely_different_classes():
    assert DatasetF1ChexbertMetric is not InstanceF1ChexbertMetric
    assert not issubclass(DatasetF1ChexbertMetric, InstanceF1ChexbertMetric)
    assert not issubclass(InstanceF1ChexbertMetric, DatasetF1ChexbertMetric)


def test_dataset_and_instance_chexbert_have_distinct_names_matching_cell_30_schema():
    dataset_metric = DatasetF1ChexbertMetric(
        GenerationMetricsConfig(), f1chexbert_factory=lambda: _FakeF1Chexbert(micro_f1=0.7)
    )
    instance_metric = InstanceF1ChexbertMetric(
        GenerationMetricsConfig(),
        f1chexbert_factory=lambda: _FakeF1Chexbert(label_map={"h": _label14(1), "r": _label14(1)}),
    )
    assert dataset_metric.name == "f1chexbert"
    assert instance_metric.name == "f1chexbert_instance"
    assert dataset_metric.name != instance_metric.name
    # Cell 30's own output dataclasses use these exact field names --
    # this is not a coincidence, it's the disambiguation mechanism.
    result_fields = {f.name for f in dataclasses.fields(gm.GenerationMetricsResult)}
    example_fields = {f.name for f in dataclasses.fields(gm.GenerationExampleScore)}
    assert "f1_chexbert" in result_fields
    assert "f1_chexbert_instance" in example_fields


def test_dataset_and_instance_chexbert_produce_different_scores_on_the_same_fake_backend():
    # Same underlying fake object, both wrappers -- proves the two
    # computations are wired to genuinely different methods
    # (__call__ vs. get_label), not accidentally aliased.
    fake = _FakeF1Chexbert(
        micro_f1=0.7,
        label_map={
            "hyp one": _label14(1, 4),
            "ref one": _label14(1),
        },
    )
    dataset_metric = DatasetF1ChexbertMetric(GenerationMetricsConfig(), f1chexbert_factory=lambda: fake)
    instance_metric = InstanceF1ChexbertMetric(GenerationMetricsConfig(), f1chexbert_factory=lambda: fake)

    dataset_result = dataset_metric.compute(["hyp one"], ["ref one"])
    instance_result = instance_metric.compute(["hyp one"], ["ref one"])

    assert dataset_result.corpus_score == pytest.approx(0.7)
    assert dataset_result.per_example_scores is None
    # 5-class reduction of indices (1,4,5,7,9): hyp has 1s at (1,4) -> [1,1,0,0,0];
    # ref has 1 at (1) -> [1,0,0,0,0]; agreement = 4/5 = 0.8
    assert instance_result.per_example_scores == (pytest.approx(0.8),)
    assert dataset_result.corpus_score != instance_result.corpus_score
    assert fake.call_args == [(["hyp one"], ["ref one"])]
    assert fake.get_label_calls == ["hyp one", "ref one"]


# ---------------------------------------------------------------------------
# _reduce_chexbert_14_to_5 / _chexbert_instance_agreement
# ---------------------------------------------------------------------------


def test_reduce_chexbert_14_to_5_extracts_correct_indices():
    vec = _label14(1, 4, 5, 7, 9)  # all five subset positions set
    assert gm._reduce_chexbert_14_to_5(vec, metric_name="x") == (1, 1, 1, 1, 1)


def test_reduce_chexbert_14_to_5_ignores_non_subset_positions():
    vec = _label14(0, 2, 3, 6, 8, 10, 11, 12, 13)  # every non-subset index set
    assert gm._reduce_chexbert_14_to_5(vec, metric_name="x") == (0, 0, 0, 0, 0)


@pytest.mark.parametrize("bad_length", [0, 5, 13, 15])
def test_reduce_chexbert_14_to_5_rejects_wrong_length(bad_length):
    with pytest.raises(EvaluationError, match="14-class"):
        gm._reduce_chexbert_14_to_5([0] * bad_length, metric_name="x")


def test_chexbert_instance_agreement_matches_baseline_similarity_module_exactly():
    # Explicit cross-check that the two independent implementations
    # (this module's own, and src.baseline.pair_mining.similarity's,
    # kept separate to avoid a layering inversion -- see module
    # docstring) agree on every input, not just by construction.
    cases = [
        ((1, 0, 0, 0, 0), (1, 0, 0, 0, 0)),
        ((1, 1, 1, 1, 1), (0, 0, 0, 0, 0)),
        ((1, 0, 1, 0, 1), (1, 1, 0, 0, 1)),
        ((0, 0, 0, 0, 0), (0, 0, 0, 0, 0)),
    ]
    for label_a, label_b in cases:
        assert gm._chexbert_instance_agreement(label_a, label_b) == chexbert_similarity(
            label_a, label_b
        )


def test_chexbert_instance_agreement_identical_labels_is_one():
    assert gm._chexbert_instance_agreement((1, 0, 1, 0, 1), (1, 0, 1, 0, 1)) == pytest.approx(1.0)


def test_chexbert_instance_agreement_rejects_mismatched_length():
    with pytest.raises(EvaluationError, match="equal-length"):
        gm._chexbert_instance_agreement((1, 0), (1, 0, 0))


def test_chexbert_instance_agreement_rejects_empty():
    with pytest.raises(EvaluationError, match="non-empty"):
        gm._chexbert_instance_agreement((), ())


# ---------------------------------------------------------------------------
# F1RadGraphMetric
# ---------------------------------------------------------------------------


def test_f1radgraph_metric_computes_corpus_and_per_example_scores():
    fake = _FakeF1RadGraph(mean_reward=0.35, reward_list=[0.2, 0.5])
    metric = F1RadGraphMetric(GenerationMetricsConfig(), f1radgraph_factory=lambda: fake)
    result = metric.compute(["h1", "h2"], ["r1", "r2"])
    assert result.metric_name == "f1radgraph"
    assert result.corpus_score == pytest.approx(0.35)
    assert result.per_example_scores == (pytest.approx(0.2), pytest.approx(0.5))
    assert fake.calls == [(["h1", "h2"], ["r1", "r2"])]


def test_f1radgraph_metric_is_lazy_and_caches_the_constructed_instance():
    counting = _CountingFactory(lambda: _FakeF1RadGraph(mean_reward=0.1, reward_list=[0.1]))
    metric = F1RadGraphMetric(GenerationMetricsConfig(), f1radgraph_factory=counting)
    assert counting.call_count == 0
    metric.compute(["h"], ["r"])
    metric.compute(["h"], ["r"])
    assert counting.call_count == 1


def test_f1radgraph_metric_construction_failure_is_classified_and_wrapped():
    def raising_factory():
        raise ImportError("no module named radgraph")

    metric = F1RadGraphMetric(GenerationMetricsConfig(), f1radgraph_factory=raising_factory)
    with pytest.raises(EvaluationError, match="missing_dependency"):
        metric.compute(["h"], ["r"])


def test_f1radgraph_metric_computation_failure_is_classified_and_wrapped():
    metric = F1RadGraphMetric(
        GenerationMetricsConfig(),
        f1radgraph_factory=lambda: _RaisingF1RadGraph(ValueError("bad input")),
    )
    with pytest.raises(EvaluationError, match="invalid_input"):
        metric.compute(["h"], ["r"])


def test_f1radgraph_metric_rejects_reward_list_length_mismatch():
    fake = _FakeF1RadGraph(mean_reward=0.5, reward_list=[0.1])  # only 1, but 2 examples given
    metric = F1RadGraphMetric(GenerationMetricsConfig(), f1radgraph_factory=lambda: fake)
    with pytest.raises(EvaluationError):
        metric.compute(["h1", "h2"], ["r1", "r2"])


def test_default_f1radgraph_factory_lazily_imports_compat_and_radgraph(monkeypatch):
    import sys

    import src.baseline.radgraph as radgraph_package

    patched = []
    preplaced = []
    constructed_kwargs = {}

    class _FakeCompatModule:
        @staticmethod
        def patch_all():
            patched.append(True)

        @staticmethod
        def preplace_all_checkpoints():
            preplaced.append(True)

    class _FakeF1RadGraphClass:
        def __init__(self, **kwargs):
            constructed_kwargs.update(kwargs)

    fake_radgraph_module = type("fake_radgraph_module", (), {"F1RadGraph": _FakeF1RadGraphClass})()
    # `from src.baseline.radgraph import compat` resolves via the already-
    # imported package object's own `compat` attribute whenever some
    # other test in this session has already triggered a real import of
    # it (very likely -- this project's own compat tests do exactly
    # that) -- patching sys.modules alone is not reliable once that has
    # happened, so the package attribute itself is patched directly.
    monkeypatch.setattr(radgraph_package, "compat", _FakeCompatModule(), raising=False)
    monkeypatch.setitem(sys.modules, "radgraph", fake_radgraph_module)

    config = GenerationMetricsConfig(radgraph_reward_level="exact", radgraph_model_type="radgraph")
    gm._default_f1radgraph_factory(config)

    assert patched == [True]
    assert preplaced == [True]
    assert constructed_kwargs == {"reward_level": "exact", "model_type": "radgraph"}


# ---------------------------------------------------------------------------
# DatasetF1ChexbertMetric
# ---------------------------------------------------------------------------


def test_dataset_f1chexbert_metric_computes_corpus_score_only():
    fake = _FakeF1Chexbert(micro_f1=0.62)
    metric = DatasetF1ChexbertMetric(GenerationMetricsConfig(), f1chexbert_factory=lambda: fake)
    result = metric.compute(["h1", "h2"], ["r1", "r2"])
    assert result.metric_name == "f1chexbert"
    assert result.corpus_score == pytest.approx(0.62)
    assert result.per_example_scores is None
    assert fake.call_args == [(["h1", "h2"], ["r1", "r2"])]


def test_dataset_f1chexbert_metric_is_lazy_and_caches_the_constructed_instance():
    counting = _CountingFactory(lambda: _FakeF1Chexbert(micro_f1=0.5))
    metric = DatasetF1ChexbertMetric(GenerationMetricsConfig(), f1chexbert_factory=counting)
    assert counting.call_count == 0
    metric.compute(["h"], ["r"])
    metric.compute(["h"], ["r"])
    assert counting.call_count == 1


def test_dataset_f1chexbert_metric_construction_failure_is_classified_and_wrapped():
    def raising_factory():
        raise ModuleNotFoundError("no module named f1chexbert")

    metric = DatasetF1ChexbertMetric(GenerationMetricsConfig(), f1chexbert_factory=raising_factory)
    with pytest.raises(EvaluationError, match="missing_dependency"):
        metric.compute(["h"], ["r"])


def test_dataset_f1chexbert_metric_computation_failure_is_classified_and_wrapped():
    metric = DatasetF1ChexbertMetric(
        GenerationMetricsConfig(),
        f1chexbert_factory=lambda: _RaisingF1Chexbert(RuntimeError("device mismatch"), raise_on="call"),
    )
    with pytest.raises(EvaluationError, match="computation_failed"):
        metric.compute(["h"], ["r"])


# ---------------------------------------------------------------------------
# InstanceF1ChexbertMetric
# ---------------------------------------------------------------------------


def test_instance_f1chexbert_metric_computes_per_example_agreement():
    fake = _FakeF1Chexbert(
        label_map={
            "h1": _label14(1, 4, 5, 7, 9),  # all-5-subset positive
            "r1": _label14(1, 4, 5, 7, 9),  # identical -> agreement 1.0
            "h2": _label14(1),
            "r2": _label14(),  # disjoint on the one subset position that differs -> 4/5
        }
    )
    metric = InstanceF1ChexbertMetric(GenerationMetricsConfig(), f1chexbert_factory=lambda: fake)
    result = metric.compute(["h1", "h2"], ["r1", "r2"])
    assert result.metric_name == "f1chexbert_instance"
    assert result.per_example_scores[0] == pytest.approx(1.0)
    assert result.per_example_scores[1] == pytest.approx(4 / 5)
    assert result.corpus_score == pytest.approx((1.0 + 4 / 5) / 2)


def test_instance_f1chexbert_metric_is_lazy_and_caches_the_constructed_instance():
    counting = _CountingFactory(lambda: _FakeF1Chexbert(label_map={"h": _label14(), "r": _label14()}))
    metric = InstanceF1ChexbertMetric(GenerationMetricsConfig(), f1chexbert_factory=counting)
    assert counting.call_count == 0
    metric.compute(["h"], ["r"])
    metric.compute(["h"], ["r"])
    assert counting.call_count == 1


def test_instance_f1chexbert_metric_construction_failure_is_classified_and_wrapped():
    def raising_factory():
        raise ImportError("no module named f1chexbert")

    metric = InstanceF1ChexbertMetric(GenerationMetricsConfig(), f1chexbert_factory=raising_factory)
    with pytest.raises(EvaluationError, match="missing_dependency"):
        metric.compute(["h"], ["r"])


def test_instance_f1chexbert_metric_get_label_failure_is_classified_and_wrapped():
    metric = InstanceF1ChexbertMetric(
        GenerationMetricsConfig(),
        f1chexbert_factory=lambda: _RaisingF1Chexbert(ValueError("malformed report"), raise_on="get_label"),
    )
    with pytest.raises(EvaluationError, match="invalid_input"):
        metric.compute(["h"], ["r"])


def test_instance_f1chexbert_metric_rejects_malformed_label_vector_from_get_label():
    fake = _FakeF1Chexbert(label_map={"h": [0, 1, 0], "r": _label14()})  # h has only 3 elements
    metric = InstanceF1ChexbertMetric(GenerationMetricsConfig(), f1chexbert_factory=lambda: fake)
    with pytest.raises(EvaluationError, match="14-class"):
        metric.compute(["h"], ["r"])


def test_default_f1chexbert_factory_lazily_imports_and_passes_device(monkeypatch):
    import sys

    import src.baseline.radgraph as radgraph_package

    patched = []
    constructed_kwargs = {}

    class _FakeCompatModule:
        @staticmethod
        def patch_all():
            patched.append(True)

        @staticmethod
        def preplace_all_checkpoints():
            pass

    class _FakeF1ChexbertClass:
        def __init__(self, **kwargs):
            constructed_kwargs.update(kwargs)

    fake_f1chexbert_module = type(
        "fake_f1chexbert_module", (), {"F1CheXbert": _FakeF1ChexbertClass}
    )()
    # See test_default_f1radgraph_factory_lazily_imports_compat_and_radgraph's
    # own comment: the package attribute must be patched directly, not
    # just sys.modules, once compat has already been really imported
    # elsewhere in this test session.
    monkeypatch.setattr(radgraph_package, "compat", _FakeCompatModule(), raising=False)
    monkeypatch.setitem(sys.modules, "f1chexbert", fake_f1chexbert_module)

    config = GenerationMetricsConfig(chexbert_device="cuda")
    gm._default_f1chexbert_factory(config)

    assert patched == [True]
    assert constructed_kwargs == {"device": "cuda"}


# ---------------------------------------------------------------------------
# Shared validation (length mismatch / zero examples) for the clinical wrappers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper_factory",
    [
        lambda: F1RadGraphMetric(
            GenerationMetricsConfig(), f1radgraph_factory=lambda: _FakeF1RadGraph(0.1, [0.1])
        ),
        lambda: DatasetF1ChexbertMetric(
            GenerationMetricsConfig(), f1chexbert_factory=lambda: _FakeF1Chexbert(micro_f1=0.1)
        ),
        lambda: InstanceF1ChexbertMetric(
            GenerationMetricsConfig(),
            f1chexbert_factory=lambda: _FakeF1Chexbert(label_map={"a": _label14(), "b": _label14()}),
        ),
    ],
)
def test_clinical_wrappers_reject_length_mismatch(wrapper_factory):
    wrapper = wrapper_factory()
    with pytest.raises(EvaluationError, match="same length"):
        wrapper.compute(["a", "b"], ["only one"])


@pytest.mark.parametrize(
    "wrapper_factory",
    [
        lambda: F1RadGraphMetric(
            GenerationMetricsConfig(), f1radgraph_factory=lambda: _FakeF1RadGraph(0.0, [])
        ),
        lambda: DatasetF1ChexbertMetric(
            GenerationMetricsConfig(), f1chexbert_factory=lambda: _FakeF1Chexbert(micro_f1=0.0)
        ),
        lambda: InstanceF1ChexbertMetric(
            GenerationMetricsConfig(), f1chexbert_factory=lambda: _FakeF1Chexbert(label_map={})
        ),
    ],
)
def test_clinical_wrappers_reject_zero_examples(wrapper_factory):
    wrapper = wrapper_factory()
    with pytest.raises(EvaluationError, match="zero examples"):
        wrapper.compute([], [])


# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------


def test_build_metric_registry_now_includes_all_five_contract_metrics():
    registry = build_metric_registry(GenerationMetricsConfig())
    assert set(registry) == {"rouge_l", "bleu4", "bert_score", "f1radgraph", "f1chexbert", "f1chexbert_instance"}
    assert isinstance(registry["f1radgraph"], F1RadGraphMetric)
    assert isinstance(registry["f1chexbert"], DatasetF1ChexbertMetric)
    assert isinstance(registry["f1chexbert_instance"], InstanceF1ChexbertMetric)


def test_build_metric_registry_excludes_bleu4_when_disabled_still_includes_clinical_metrics():
    registry = build_metric_registry(GenerationMetricsConfig(bleu_enabled=False))
    assert "bleu4" not in registry
    assert {"f1radgraph", "f1chexbert", "f1chexbert_instance"}.issubset(registry)


def test_build_metric_registry_returns_independent_clinical_instances_across_calls():
    config = GenerationMetricsConfig()
    registry_one = build_metric_registry(config)
    registry_two = build_metric_registry(config)
    assert registry_one["f1radgraph"] is not registry_two["f1radgraph"]
    assert registry_one["f1chexbert"] is not registry_two["f1chexbert"]
    assert registry_one["f1chexbert_instance"] is not registry_two["f1chexbert_instance"]


def test_build_metric_registry_propagates_injected_clinical_factories():
    radgraph_fake = _FakeF1RadGraph(mean_reward=0.9, reward_list=[0.9])
    chexbert_fake = _FakeF1Chexbert(micro_f1=0.4, label_map={"h": _label14(1), "r": _label14(1)})
    registry = build_metric_registry(
        GenerationMetricsConfig(),
        f1radgraph_factory=lambda: radgraph_fake,
        f1chexbert_factory=lambda: chexbert_fake,
    )
    assert registry["f1radgraph"].compute(["h"], ["r"]).corpus_score == pytest.approx(0.9)
    assert registry["f1chexbert"].compute(["h"], ["r"]).corpus_score == pytest.approx(0.4)
    assert registry["f1chexbert_instance"].compute(["h"], ["r"]).corpus_score == pytest.approx(1.0)


def test_build_mock_metric_registry_now_covers_all_five_contract_metrics():
    mock_registry = build_mock_metric_registry()
    assert set(mock_registry) == {
        "rouge_l",
        "bleu4",
        "bert_score",
        "f1radgraph",
        "f1chexbert",
        "f1chexbert_instance",
    }
    for name, wrapper in mock_registry.items():
        assert isinstance(wrapper, MockGenerationMetricWrapper)
        assert wrapper.name == name


def test_build_mock_metric_registry_clinical_entries_are_deterministic_and_usable():
    registry = build_mock_metric_registry()
    for name in ("f1radgraph", "f1chexbert", "f1chexbert_instance"):
        first = registry[name].compute(["the lungs are clear"], ["the lungs are clear"])
        assert first.corpus_score == pytest.approx(1.0)
