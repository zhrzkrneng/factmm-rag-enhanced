"""Unit tests for the generation metric wrappers in
src/evaluation/generation_metrics.py (Milestone 2.6, Cell 31).

Scope: GenerationMetricWrapper interface, MetricComputation,
RougeLMetric, Bleu4Metric, BertScoreMetric, MockGenerationMetricWrapper,
build_metric_registry / build_mock_metric_registry, and the
dependency-error classifier. Every real wrapper is exercised only via
injected fakes -- rouge/evaluate/bert_score are not installed in this
sandbox (deliberately, to guarantee no test can accidentally depend on
them), so a test that forgets to inject a fake factory fails loudly
with a real ImportError classified as "missing_dependency"."""

import abc
import ast
import dataclasses
import inspect

import pytest

import src.evaluation.generation_metrics as gm
from src.evaluation.generation_metrics import (
    Bleu4Metric,
    BertScoreMetric,
    GenerationMetricsConfig,
    GenerationMetricWrapper,
    MetricComputation,
    MockGenerationMetricWrapper,
    RougeLMetric,
    build_metric_registry,
    build_mock_metric_registry,
)
from src.common.exceptions import EvaluationError


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _CountingFactory:
    """Wraps a zero-arg factory function, counting how many times it is
    actually invoked -- used to assert lazy construction happens at most
    once per wrapper instance."""

    def __init__(self, build):
        self._build = build
        self.call_count = 0

    def __call__(self):
        self.call_count += 1
        return self._build()


class _FakeRouge:
    def __init__(self, per_example_f_scores):
        self._scores = per_example_f_scores
        self.calls = []

    def get_scores(self, hyps, refs, avg=False):
        self.calls.append((list(hyps), list(refs), avg))
        return [{"rouge-l": {"f": score}} for score in self._scores]


class _RaisingRouge:
    def __init__(self, exc):
        self._exc = exc

    def get_scores(self, hyps, refs, avg=False):
        raise self._exc


class _FakeBleu:
    def __init__(self, precisions):
        self._precisions = precisions
        self.calls = []

    def compute(self, predictions, references):
        self.calls.append((list(predictions), list(references)))
        return {"precisions": self._precisions}


class _FakeTensor:
    def __init__(self, values):
        self._values = list(values)

    def tolist(self):
        return list(self._values)


class _FakeBertScorer:
    def __init__(self, per_example_f):
        self._f = per_example_f
        self.calls = []

    def score(self, cands, refs):
        self.calls.append((list(cands), list(refs)))
        return None, None, _FakeTensor(self._f)


# ---------------------------------------------------------------------------
# GenerationMetricWrapper / MetricComputation shape
# ---------------------------------------------------------------------------


def test_generation_metric_wrapper_is_abstract():
    with pytest.raises(TypeError):
        GenerationMetricWrapper()  # abstract -- cannot be instantiated directly


def test_metric_computation_is_immutable():
    computation = MetricComputation(metric_name="x", corpus_score=0.5, per_example_scores=(0.5,))
    with pytest.raises(dataclasses.FrozenInstanceError):
        computation.corpus_score = 1.0


def test_all_three_real_wrappers_share_identical_compute_signature():
    for cls in (RougeLMetric, Bleu4Metric, BertScoreMetric):
        sig = inspect.signature(cls.compute)
        assert list(sig.parameters) == ["self", "hyps", "refs"]
        assert issubclass(cls, GenerationMetricWrapper)


# ---------------------------------------------------------------------------
# Module-scope import discipline (lazy dependency loading)
# ---------------------------------------------------------------------------


def test_module_never_imports_metric_libraries_at_module_scope():
    source = inspect.getsource(gm)
    tree = ast.parse(source)
    top_level_imports = set()
    for node in tree.body:  # only direct module-level statements, not function bodies
        if isinstance(node, ast.Import):
            top_level_imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_level_imports.add(node.module.split(".")[0])
    assert "rouge" not in top_level_imports
    assert "evaluate" not in top_level_imports
    assert "bert_score" not in top_level_imports


# A prior version of this file asserted rouge/evaluate/bert_score are
# not installed, as a belt-and-suspenders sandbox guarantee. Removed:
# that is an environment property, not a property of this module's own
# code -- it happened to hold in every sandbox tried so far, but a
# Colab image that ships any of these preinstalled (plausible for a
# transformers-adjacent environment) would fail this test for reasons
# having nothing to do with this module's own correctness, exactly the
# failure mode caught live for the analogous
# test_radgraph_and_f1chexbert_are_not_installed_in_this_sandbox in
# Cell 32's own test file. The real guarantee -- no test here
# accidentally depends on a real library -- is already structural:
# every test injects its own fake factory, and
# test_module_never_imports_metric_libraries_at_module_scope (above)
# statically verifies the module itself never imports any of them at
# module scope, regardless of what happens to be installed.


# ---------------------------------------------------------------------------
# _validate_hyps_refs (shared across all wrappers, including the mock)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "wrapper_factory",
    [
        lambda: RougeLMetric(rouge_factory=lambda: _FakeRouge([1.0])),
        lambda: Bleu4Metric(bleu_factory=lambda: _FakeBleu([1, 1, 1, 1.0])),
        lambda: BertScoreMetric(
            GenerationMetricsConfig(), bert_scorer_factory=lambda: _FakeBertScorer([1.0])
        ),
        lambda: MockGenerationMetricWrapper(),
    ],
)
def test_compute_rejects_length_mismatch(wrapper_factory):
    wrapper = wrapper_factory()
    with pytest.raises(EvaluationError, match="same length"):
        wrapper.compute(["a", "b"], ["only one"])


@pytest.mark.parametrize(
    "wrapper_factory",
    [
        lambda: RougeLMetric(rouge_factory=lambda: _FakeRouge([])),
        lambda: Bleu4Metric(bleu_factory=lambda: _FakeBleu([0, 0, 0, 0])),
        lambda: BertScoreMetric(
            GenerationMetricsConfig(), bert_scorer_factory=lambda: _FakeBertScorer([])
        ),
        lambda: MockGenerationMetricWrapper(),
    ],
)
def test_compute_rejects_zero_examples(wrapper_factory):
    wrapper = wrapper_factory()
    with pytest.raises(EvaluationError, match="zero examples"):
        wrapper.compute([], [])


# ---------------------------------------------------------------------------
# _classify_metric_dependency_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc, expected_category",
    [
        (ImportError("no module named rouge"), "missing_dependency"),
        (ModuleNotFoundError("no module named bert_score"), "missing_dependency"),
        (ValueError("Hypothesis is empty"), "invalid_input"),
        (IndexError("list index out of range"), "invalid_input"),
        (KeyError("rouge-l"), "invalid_input"),
        (ConnectionError("could not connect to huggingface.co"), "network_error"),
        (TimeoutError("timed out"), "network_error"),  # "Timeout" substring matches the network category
        (RuntimeError("something else entirely"), "computation_failed"),
    ],
)
def test_classify_metric_dependency_error(exc, expected_category):
    category, detail = gm._classify_metric_dependency_error(exc)
    assert category == expected_category
    assert type(exc).__name__ in detail


class _FakeGatedRepoError(Exception):
    pass


class _FakeRepositoryNotFoundError(Exception):
    pass


def test_classify_metric_dependency_error_recognizes_hub_error_type_names():
    # These simulate real huggingface_hub exception *type names* (the
    # classifier matches on qualified type name substrings, not on
    # actually importing huggingface_hub -- see
    # _classify_generation_error's own established precedent).
    category, _ = gm._classify_metric_dependency_error(_FakeGatedRepoError("gated"))
    assert category == "authentication_required"
    category, _ = gm._classify_metric_dependency_error(_FakeRepositoryNotFoundError("missing"))
    assert category == "checkpoint_unavailable"


def test_classify_metric_dependency_error_walks_cause_chain():
    root = ImportError("no module named rouge")
    wrapper_exc = RuntimeError("construction failed")
    wrapper_exc.__cause__ = root
    category, _ = gm._classify_metric_dependency_error(wrapper_exc)
    assert category == "missing_dependency"


# ---------------------------------------------------------------------------
# RougeLMetric
# ---------------------------------------------------------------------------


def test_rouge_l_metric_computes_corpus_and_per_example_scores():
    fake = _FakeRouge([0.2, 0.8, 0.5])
    metric = RougeLMetric(rouge_factory=lambda: fake)
    result = metric.compute(["h1", "h2", "h3"], ["r1", "r2", "r3"])
    assert result.metric_name == "rouge_l"
    assert result.per_example_scores == (0.2, 0.8, 0.5)
    assert result.corpus_score == pytest.approx((0.2 + 0.8 + 0.5) / 3)
    assert fake.calls == [(["h1", "h2", "h3"], ["r1", "r2", "r3"], False)]


def test_rouge_l_metric_is_lazy_and_caches_the_constructed_instance():
    counting = _CountingFactory(lambda: _FakeRouge([1.0]))
    metric = RougeLMetric(rouge_factory=counting)
    assert counting.call_count == 0  # not constructed at __init__ time
    metric.compute(["h"], ["r"])
    assert counting.call_count == 1
    metric.compute(["h"], ["r"])
    assert counting.call_count == 1  # cached, not reconstructed


def test_rouge_l_metric_construction_failure_is_classified_and_wrapped():
    def raising_factory():
        raise ImportError("no module named rouge")

    metric = RougeLMetric(rouge_factory=raising_factory)
    with pytest.raises(EvaluationError, match="missing_dependency"):
        metric.compute(["h"], ["r"])


def test_rouge_l_metric_computation_failure_is_classified_and_wrapped():
    metric = RougeLMetric(rouge_factory=lambda: _RaisingRouge(ValueError("Hypothesis is empty.")))
    with pytest.raises(EvaluationError, match="invalid_input"):
        metric.compute([""], ["r"])


# ---------------------------------------------------------------------------
# Bleu4Metric
# ---------------------------------------------------------------------------


def test_bleu4_metric_computes_precisions_index_3_only():
    fake = _FakeBleu([0.9, 0.8, 0.7, 0.6])
    metric = Bleu4Metric(bleu_factory=lambda: fake)
    result = metric.compute(["h1", "h2"], ["r1", "r2"])
    assert result.metric_name == "bleu4"
    assert result.corpus_score == pytest.approx(0.6)
    assert result.per_example_scores is None  # no per-example decomposition, ever
    assert fake.calls == [(["h1", "h2"], [["r1"], ["r2"]])]


def test_bleu4_metric_is_lazy_and_caches_the_constructed_instance():
    counting = _CountingFactory(lambda: _FakeBleu([1, 1, 1, 1.0]))
    metric = Bleu4Metric(bleu_factory=counting)
    assert counting.call_count == 0
    metric.compute(["h"], ["r"])
    metric.compute(["h"], ["r"])
    assert counting.call_count == 1


def test_bleu4_metric_construction_failure_is_classified_and_wrapped():
    def raising_factory():
        raise ModuleNotFoundError("no module named evaluate")

    metric = Bleu4Metric(bleu_factory=raising_factory)
    with pytest.raises(EvaluationError, match="missing_dependency"):
        metric.compute(["h"], ["r"])


# ---------------------------------------------------------------------------
# BertScoreMetric
# ---------------------------------------------------------------------------


def test_bert_score_metric_computes_corpus_and_per_example_scores():
    fake = _FakeBertScorer([0.4, 0.6])
    metric = BertScoreMetric(GenerationMetricsConfig(), bert_scorer_factory=lambda: fake)
    result = metric.compute(["h1", "h2"], ["r1", "r2"])
    assert result.metric_name == "bert_score"
    assert result.per_example_scores == (0.4, 0.6)
    assert result.corpus_score == pytest.approx(0.5)
    assert fake.calls == [(["h1", "h2"], ["r1", "r2"])]


def test_bert_score_metric_is_lazy_and_caches_the_constructed_instance():
    counting = _CountingFactory(lambda: _FakeBertScorer([1.0]))
    metric = BertScoreMetric(GenerationMetricsConfig(), bert_scorer_factory=counting)
    assert counting.call_count == 0
    metric.compute(["h"], ["r"])
    metric.compute(["h"], ["r"])
    assert counting.call_count == 1


def test_bert_score_metric_construction_failure_is_classified_and_wrapped():
    def raising_factory():
        raise ImportError("no module named bert_score")

    metric = BertScoreMetric(GenerationMetricsConfig(), bert_scorer_factory=raising_factory)
    with pytest.raises(EvaluationError, match="missing_dependency"):
        metric.compute(["h"], ["r"])


def test_default_bert_scorer_factory_passes_config_fields_and_reuses_chexbert_device(monkeypatch):
    captured_kwargs = {}

    class _FakeBERTScorer:
        def __init__(self, **kwargs):
            captured_kwargs.update(kwargs)

    fake_module = type(
        "fake_bert_score_module", (), {"BERTScorer": _FakeBERTScorer}
    )()
    monkeypatch.setitem(__import__("sys").modules, "bert_score", fake_module)

    config = GenerationMetricsConfig(
        bert_score_model_type="some-other-model",
        bert_score_num_layers=7,
        bert_score_batch_size=16,
        bert_score_rescale_with_baseline=False,
        chexbert_device="cuda",
    )
    gm._default_bert_scorer_factory(config)

    assert captured_kwargs["model_type"] == "some-other-model"
    assert captured_kwargs["num_layers"] == 7
    assert captured_kwargs["batch_size"] == 16
    assert captured_kwargs["rescale_with_baseline"] is False
    assert captured_kwargs["device"] == "cuda"  # reuses chexbert_device, disclosed in the module docstring


def test_default_rouge_factory_lazily_imports_and_constructs(monkeypatch):
    constructed = []

    class _FakeRougeClass:
        def __init__(self):
            constructed.append(True)

    fake_module = type("fake_rouge_module", (), {"Rouge": _FakeRougeClass})()
    monkeypatch.setitem(__import__("sys").modules, "rouge", fake_module)

    instance = gm._default_rouge_factory()
    assert constructed == [True]
    assert isinstance(instance, _FakeRougeClass)


def test_default_bleu_factory_lazily_imports_and_loads(monkeypatch):
    loaded = []

    class _FakeEvaluateModule:
        @staticmethod
        def load(name):
            loaded.append(name)
            return "the-bleu-metric"

    monkeypatch.setitem(__import__("sys").modules, "evaluate", _FakeEvaluateModule())

    instance = gm._default_bleu_factory()
    assert loaded == ["bleu"]
    assert instance == "the-bleu-metric"


# ---------------------------------------------------------------------------
# MockGenerationMetricWrapper
# ---------------------------------------------------------------------------


def test_mock_metric_identical_text_scores_one():
    mock = MockGenerationMetricWrapper(name="mock")
    result = mock.compute(["same text here"], ["same text here"])
    assert result.per_example_scores == (1.0,)
    assert result.corpus_score == pytest.approx(1.0)


def test_mock_metric_disjoint_text_scores_zero():
    mock = MockGenerationMetricWrapper(name="mock")
    result = mock.compute(["alpha beta"], ["gamma delta"])
    assert result.per_example_scores == (0.0,)


def test_mock_metric_empty_string_scores_zero_without_raising():
    mock = MockGenerationMetricWrapper(name="mock")
    result = mock.compute([""], ["some ref text"])
    assert result.per_example_scores == (0.0,)


def test_mock_metric_is_deterministic_across_repeated_calls():
    mock = MockGenerationMetricWrapper(name="mock")
    first = mock.compute(["a b c", "d e"], ["a b x", "y z"])
    second = mock.compute(["a b c", "d e"], ["a b x", "y z"])
    assert first == second


def test_mock_metric_accepts_custom_score_fn():
    mock = MockGenerationMetricWrapper(name="custom", score_fn=lambda h, r: 0.42)
    result = mock.compute(["h1", "h2"], ["r1", "r2"])
    assert result.per_example_scores == (0.42, 0.42)
    assert result.corpus_score == pytest.approx(0.42)
    assert result.metric_name == "custom"


def test_mock_metric_name_defaults_and_is_settable():
    assert MockGenerationMetricWrapper().name == "mock_metric"
    assert MockGenerationMetricWrapper(name="rouge_l").name == "rouge_l"


# ---------------------------------------------------------------------------
# build_metric_registry / build_mock_metric_registry
# ---------------------------------------------------------------------------


def test_build_metric_registry_default_includes_bleu4():
    # Scoped to this cell's own 3 metrics: rouge_l/bleu4/bert_score must
    # be present, whatever else the registry may also contain -- Cell 32
    # (a later addition) legitimately expands this same registry with
    # f1radgraph/f1chexbert/f1chexbert_instance, and asserts the exact
    # full-registry shape itself, in its own test file.
    registry = build_metric_registry(GenerationMetricsConfig())
    assert {"rouge_l", "bleu4", "bert_score"}.issubset(registry)
    assert isinstance(registry["rouge_l"], RougeLMetric)
    assert isinstance(registry["bleu4"], Bleu4Metric)
    assert isinstance(registry["bert_score"], BertScoreMetric)


def test_build_metric_registry_excludes_bleu4_when_disabled():
    registry = build_metric_registry(GenerationMetricsConfig(bleu_enabled=False))
    assert "bleu4" not in registry
    assert {"rouge_l", "bert_score"}.issubset(registry)


def test_build_metric_registry_returns_independent_instances_across_calls():
    config = GenerationMetricsConfig()
    registry_one = build_metric_registry(config)
    registry_two = build_metric_registry(config)
    assert registry_one["rouge_l"] is not registry_two["rouge_l"]
    assert registry_one["bert_score"] is not registry_two["bert_score"]


def test_build_metric_registry_propagates_injected_factories_to_the_right_wrapper():
    rouge_fake = _FakeRouge([1.0])
    bleu_fake = _FakeBleu([1, 1, 1, 0.5])
    bert_fake = _FakeBertScorer([1.0])
    registry = build_metric_registry(
        GenerationMetricsConfig(),
        rouge_factory=lambda: rouge_fake,
        bleu_factory=lambda: bleu_fake,
        bert_scorer_factory=lambda: bert_fake,
    )
    assert registry["rouge_l"].compute(["h"], ["r"]).corpus_score == pytest.approx(1.0)
    assert registry["bleu4"].compute(["h"], ["r"]).corpus_score == pytest.approx(0.5)
    assert registry["bert_score"].compute(["h"], ["r"]).corpus_score == pytest.approx(1.0)


def test_build_mock_metric_registry_default_names_match_real_registry_shape():
    # See test_build_metric_registry_default_includes_bleu4's own note:
    # scoped to this cell's 3 metrics being present, not to the full
    # registry's exact shape (Cell 32 legitimately adds more entries).
    mock_registry = build_mock_metric_registry()
    assert {"rouge_l", "bleu4", "bert_score"}.issubset(mock_registry)
    for name, wrapper in mock_registry.items():
        assert isinstance(wrapper, MockGenerationMetricWrapper)
        assert wrapper.name == name


def test_build_mock_metric_registry_accepts_custom_names():
    mock_registry = build_mock_metric_registry(names=("only_this_one",))
    assert set(mock_registry) == {"only_this_one"}


def test_build_mock_metric_registry_is_usable_without_any_real_library():
    # The whole point: a registry-consuming caller (e.g. a future
    # evaluate_generation()) can be exercised end-to-end with zero real
    # metric libraries installed.
    registry = build_mock_metric_registry()
    for wrapper in registry.values():
        result = wrapper.compute(["the cat sat"], ["the cat sat"])
        assert result.corpus_score == pytest.approx(1.0)
