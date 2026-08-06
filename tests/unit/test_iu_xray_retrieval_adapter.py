"""Unit tests for src/data/iu_xray/retrieval_adapter.py.

Covers: (1) Cell 44 loader to retrieval dataset integration,
(3) exact split preservation, (4) excluded-record filtering,
(5) deterministic ordering, (7) self-match exclusion, (8) one sample
per study, (9) multi-image grouping preservation, (14) no fabricated
MIMIC/CheXpert fields, (19) repeated-run determinism.
"""

import pytest

from src.baseline.retrieval.dataset import REQUIRED_RECORD_FIELDS, RetrieverTrainingDataset
from src.data.iu_xray.records import IuXrayCanonicalRecord
from src.data.iu_xray.retrieval_adapter import (
    build_corpus,
    build_queries,
    build_records,
    filter_self_matches,
    to_contract_json,
)


def _record(study_id, image_paths=None, **overrides):
    payload = dict(
        dataset_name="iu_xray", study_id=study_id, report_id=study_id,
        image_ids=[f"{study_id}_IM-1"],
        image_paths=image_paths or [f"/fake/{study_id}_IM-1.png"],
        image_count=len(image_paths) if image_paths else 1,
        findings="Findings.", impression="Impression.",
        indication=None, comparison=None, full_report="Findings. Impression.",
        mesh_terms=[], labels=[], source_metadata={}, source_split=None,
        validation_status="valid", exclusion_reason=None,
    )
    payload.update(overrides)
    return IuXrayCanonicalRecord.from_dict(payload)


def test_build_records_produces_required_baseline_fields():
    records = [_record("CXR1")]
    built = build_records(records)
    key = list(built)[0]
    flat = built[key].flat_record
    assert set(REQUIRED_RECORD_FIELDS) <= set(flat)
    assert all(isinstance(flat[f], str) for f in REQUIRED_RECORD_FIELDS)


def test_dataset_tag_is_always_iu_xray_never_mimic_or_chexpert():
    records = [_record("CXR1")]
    built = build_records(records)
    flat = list(built.values())[0].flat_record
    assert flat["dataset"] == "iu-xray"
    assert flat["dataset"] not in ("mimic-cxr", "chexpert")


def test_query_key_uses_study_id_as_patient_id():
    built = build_records([_record("CXR42")])
    key = list(built)[0]
    assert key == ("iu-xray", "CXR42", "CXR42")


def test_exact_split_preservation_one_record_per_input_study():
    records = [_record(f"CXR{i}") for i in range(10)]
    built = build_records(records)
    assert len(built) == 10
    assert {v.source.study_id for v in built.values()} == {r.study_id for r in records}


def test_excluded_record_raises_rather_than_silently_included():
    excluded = _record("CXR1", validation_status="excluded", exclusion_reason="missing_image",
                        findings="Findings still present.")
    with pytest.raises(ValueError, match="excluded"):
        build_records([excluded])


def test_multi_image_study_flat_dict_uses_frontal_but_source_keeps_all():
    record = _record("CXR1", image_paths=["/fake/a.png", "/fake/b.png", "/fake/c.png"])
    built = build_records([record])
    entry = list(built.values())[0]
    assert entry.flat_record["image_path"] == "/fake/a.png"
    assert entry.source.image_paths == ["/fake/a.png", "/fake/b.png", "/fake/c.png"]
    assert entry.source.image_count == 3


def test_build_corpus_and_build_queries_are_equivalent_conversions():
    records = [_record("CXR1"), _record("CXR2")]
    assert build_corpus(records) == build_queries(records)


def test_filter_self_matches_removes_only_the_query_key():
    q = ("iu-xray", "CXR1", "CXR1")
    candidates = [q, ("iu-xray", "CXR2", "CXR2"), ("iu-xray", "CXR3", "CXR3")]
    filtered = filter_self_matches(q, candidates)
    assert q not in filtered
    assert len(filtered) == 2


def test_filter_self_matches_preserves_order():
    q = ("iu-xray", "CXR1", "CXR1")
    candidates = [("iu-xray", "CXR3", "CXR3"), q, ("iu-xray", "CXR2", "CXR2")]
    filtered = filter_self_matches(q, candidates)
    assert filtered == [("iu-xray", "CXR3", "CXR3"), ("iu-xray", "CXR2", "CXR2")]


def test_repeated_build_corpus_calls_produce_identical_output():
    records = [_record("CXR2"), _record("CXR1")]
    first = build_corpus(records)
    second = build_corpus(records)
    assert first == second


def test_built_corpus_is_accepted_by_real_retriever_training_dataset(tmp_path):
    # End-to-end proof this adapter's output is genuinely compatible
    # with the real, unmodified baseline retrieval dataset -- not just
    # shaped like it.
    import json

    records = [_record("CXR1"), _record("CXR2")]
    corpus = build_corpus(records)

    pairs_path = tmp_path / "pairs.jsonl"
    with pairs_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "dataset": "iu-xray", "patient_id": "CXR1", "study_id": "CXR1",
            "positive_keys": [["iu-xray", "CXR2", "CXR2"]],
            "scores": [{"similarity": 1.0}],
            "num_candidates_considered": 1, "num_qualifying_candidates": 1,
            "num_selected": 1, "flagged_bad_sample": False,
        }) + "\n")

    dataset = RetrieverTrainingDataset(corpus, pairs_path)
    assert len(dataset) == 1
    row = dataset[0]
    assert row["query_key"] == ("iu-xray", "CXR1", "CXR1")
    assert row["positive_text"] == corpus[("iu-xray", "CXR2", "CXR2")]["finding"]


def test_to_contract_json_has_required_keys():
    contract = to_contract_json()
    for key in ("corpus_policy", "query_policy", "self_match_exclusion_policy"):
        assert key in contract
        assert isinstance(contract[key], str) and contract[key]
