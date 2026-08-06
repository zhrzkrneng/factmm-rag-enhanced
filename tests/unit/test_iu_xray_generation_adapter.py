"""Unit tests for src/data/iu_xray/generation_adapter.py.

Covers: (2) Cell 44 loader to generation dataset integration,
(6) no split overlap, (8) one sample per study, (16) empty-evidence
behavior, (17) deterministic synthetic evidence, (18) Oracle-path
separation, (19) repeated-run determinism.
"""

from src.baseline.generation.dataset_builder import RAGDatasetBuilder, RAGDatasetBuilderConfig
from src.data.iu_xray.generation_adapter import (
    GENERATION_TEXT_FIELD,
    build_deterministic_synthetic_rankings,
    build_empty_rankings,
    build_generation_records,
    build_oracle_rankings,
    to_contract_json,
)
from src.data.iu_xray.records import IuXrayCanonicalRecord


def _record(study_id, **overrides):
    payload = dict(
        dataset_name="iu_xray", study_id=study_id, report_id=study_id,
        image_ids=[f"{study_id}_IM-1"], image_paths=[f"/fake/{study_id}_IM-1.png"],
        image_count=1, findings=f"Findings for {study_id}.", impression=f"Impression for {study_id}.",
        indication=None, comparison=None, full_report=f"Findings for {study_id}. Impression for {study_id}.",
        mesh_terms=[], labels=[], source_metadata={}, source_split=None,
        validation_status="valid", exclusion_reason=None,
    )
    payload.update(overrides)
    return IuXrayCanonicalRecord.from_dict(payload)


def _config():
    return RAGDatasetBuilderConfig(rag_data_mode=GENERATION_TEXT_FIELD, output_data_mode=GENERATION_TEXT_FIELD)


def test_build_generation_records_one_per_study():
    records = [_record("CXR1"), _record("CXR2"), _record("CXR3")]
    built = build_generation_records(records)
    assert len(built) == 3
    assert all(GENERATION_TEXT_FIELD in v and "image_path" in v for v in built.values())


def test_empty_rankings_result_in_all_rows_excluded():
    train = [_record("CXR1"), _record("CXR2")]
    corpus = build_generation_records(train)
    queries = corpus
    rankings = build_empty_rankings(list(queries))

    builder = RAGDatasetBuilder(queries, corpus, rankings, config=_config())
    rows = [builder.build_row(k) for k in queries]
    assert all(row.excluded for row in rows)
    assert all(row.retrieved_report_text is None for row in rows)


def test_deterministic_synthetic_rankings_produce_real_non_excluded_rows():
    train = [_record("CXR1"), _record("CXR2"), _record("CXR3")]
    corpus = build_generation_records(train)
    queries = corpus
    rankings = build_deterministic_synthetic_rankings(list(queries), list(corpus))

    builder = RAGDatasetBuilder(queries, corpus, rankings, config=_config())
    rows = {k: builder.build_row(k) for k in queries}

    for query_key, row in rows.items():
        assert row.excluded is False
        assert row.retrieved_key != query_key  # self-match rejected by RAGDatasetBuilder itself
        assert row.retrieved_report_text is not None


def test_deterministic_synthetic_rankings_are_reproducible():
    train = [_record("CXR2"), _record("CXR1")]
    corpus = build_generation_records(train)
    first = build_deterministic_synthetic_rankings(list(corpus), list(corpus))
    second = build_deterministic_synthetic_rankings(list(corpus), list(corpus))
    assert first == second


def test_oracle_rankings_isolated_and_result_in_exclusion_via_self_match_rejection():
    train = [_record("CXR1")]
    corpus = build_generation_records(train)
    queries = corpus
    oracle_rankings = build_oracle_rankings(list(queries))

    # Oracle ranking for a single-study corpus offers only the query's
    # own key -- RAGDatasetBuilder's real, unmodified self-study
    # rejection means this ends up excluded, proving the diagnostic
    # path never accidentally produces a usable "real" sample.
    builder = RAGDatasetBuilder(queries, corpus, oracle_rankings, config=_config())
    row = builder.build_row(list(queries)[0])
    assert row.excluded is True
    assert row.num_candidates_rejected_self_study == 1


def test_oracle_rankings_function_is_never_called_by_empty_or_synthetic_builders():
    # Direct structural check: the two "real" ranking builders don't
    # reference build_oracle_rankings at all.
    import inspect
    from src.data.iu_xray import generation_adapter

    empty_src = inspect.getsource(generation_adapter.build_empty_rankings)
    synthetic_src = inspect.getsource(generation_adapter.build_deterministic_synthetic_rankings)
    assert "build_oracle_rankings" not in empty_src
    assert "build_oracle_rankings" not in synthetic_src


def test_no_split_overlap_when_building_from_disjoint_study_id_sets():
    train = [_record(f"TRAIN{i}") for i in range(5)]
    validation = [_record(f"VAL{i}") for i in range(3)]
    train_records = build_generation_records(train)
    val_records = build_generation_records(validation)
    assert set(train_records).isdisjoint(set(val_records))


def test_to_contract_json_discloses_no_real_retrieval():
    contract = to_contract_json()
    assert contract["no_embeddings_computed"] is True
    assert contract["no_faiss_index_built"] is True
    assert contract["no_retriever_selected"] is True
