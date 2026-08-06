# Milestone 2.8 Cell 45 -- IU X-Ray Data-Pipeline Integration Report

## Created source files

- `src/data/iu_xray/target_report_policy.py` (new)
- `src/data/iu_xray/retrieval_adapter.py` (new)
- `src/data/iu_xray/generation_adapter.py` (new)
- `src/data/iu_xray/prompt_compatibility.py` (new)

## Created tests

- `tests/unit/test_iu_xray_target_report_policy.py` (new)
- `tests/unit/test_iu_xray_retrieval_adapter.py` (new)
- `tests/unit/test_iu_xray_generation_adapter.py` (new)
- `tests/unit/test_iu_xray_prompt_compatibility.py` (new)

## Integration targets (real, existing baseline interfaces)

- `src.baseline.retrieval.dataset.RetrieverTrainingDataset` (unmodified)
- `src.baseline.generation.dataset_builder.RAGDatasetBuilder` (unmodified)
- `src.baseline.generation.prompt_builder.PromptBuilder` (unmodified)

## NOT integrated -- confirmed unimplemented stubs in this repository

- `src/generation/rag_dataset.py`
- `src/generation/base.py`
- `src/generation/prompt_templates.py`
- `src/common/config.py`

## Policies

Corpus policy: Corpus = every usable ('valid') record in the split(s) passed to build_corpus(), converted 1:1 (never expanded per-image, never deduplicated beyond what Cell 43 already guarantees). Callers decide which split(s) form the corpus (this module does not hardcode 'train_only' -- see src.baseline.generation.dataset_builder.RAGDatasetBuilderConfig.corpus_scope for the convention this mirrors).
Query policy: Queries = every usable record in the split passed to build_queries(), one query per study/report (never per image). A query's own key is always excluded from its own candidate set by filter_self_matches() -- test/validation queries never retrieve themselves.
Self-match exclusion policy: By QueryKey equality (dataset, patient_id, study_id) -- since IU X-Ray's patient_id equals study_id (schema_mapping.PATIENT_ID_STRATEGY), self-study and self-patient exclusion collapse to the same check for this dataset. filter_self_matches() implements this directly; src.baseline.generation.dataset_builder.RAGDatasetBuilder additionally, independently enforces the same invariant when building generation rows (defense in depth, not redundant -- the two call sites are different).
Generation sample policy: One RAGDatasetRow per usable study (never per image), built by src.baseline.generation.dataset_builder.RAGDatasetBuilder itself (unmodified) from this module's query_records/corpus_records/knn_rankings. target_report_text is exactly the target-report-policy text (see src.data.iu_xray.target_report_policy) -- never altered by RAGDatasetBuilder, which only ever reads it verbatim from query_records[query_key][output_data_mode].

## Target report policy

- rank 1 (`findings_and_impression`): findings and impression concatenated with a single space (or just one copy if they are identical strings -- never duplicated)
- rank 2 (`findings_only`): findings text alone
- rank 3 (`impression_only`): impression text alone
- rank 4 (`full_report_only`): full_report text (comparison+indication+findings+impression, whichever were present) -- disclosed as practically unreachable for any Cell-43-valid record, since Cell 43's own validity rule already requires findings or impression to be non-None
- rank 5 (`unavailable`): no target report text could be derived; never fabricated -- callers must handle this explicitly (e.g. exclude the record), never substitute a placeholder string

## Prompt compatibility caveats

- PromptBuilder itself carries no dataset-specific identifiers or logic -- it only ever sees plain retrieved_report/target_report strings, so there is no MIMIC-CXR/CheXpert metadata for it to fabricate in the first place; this validation confirms that remains true for IU-X-Ray-derived strings too.
- Target report text is never included in retrieved evidence for non-Oracle rows -- checked directly (substring test), not just inferred from self-study exclusion, since two distinct studies could in principle share overlapping text.
- Empty evidence has explicit supported behavior via PromptMode.VQA_INFERENCE/VQA_TRAIN (no retrieved_report argument at all), not a special IU-X-Ray-specific empty-string path -- an excluded RAGDatasetRow (retrieved_report_text=None) is routed to VQA mode, never RAG mode with an empty string.
- Multi-image metadata is not visible to PromptBuilder at all (it only ever sees a single <image> token placeholder, never a real path) -- multi-image grouping is preserved one layer up, in IuXrayRetrievalRecord.source, not lost by this validation.
- Text truncation only occurs through PromptBuilderConfig.max_retrieved_report_length, explicitly configured -- never a hidden default truncation.

## Real-data validation

{
  "attempted": true,
  "external_manifest_status": {
    "available": true,
    "path": "/content/drive/MyDrive/FactMM-RAG-Enhanced/data/iu_xray/canonical/iu_xray_canonical_full.json",
    "error": null
  },
  "split_counts": {
    "train": 3060,
    "validation": 382,
    "test": 384
  },
  "usable_total": 3826,
  "usable_total_matches_3826": true,
  "linked_images_total": 7430,
  "linked_images_matches_7430": true,
  "no_split_overlap": true,
  "overlaps": {
    "train_validation": [],
    "train_test": [],
    "validation_test": []
  },
  "one_retrieval_query_per_usable_study": true,
  "one_generation_sample_per_usable_study": true,
  "no_self_match_in_non_oracle_fixtures": true,
  "multi_image_study_count": 3391,
  "single_image_study_count": 435,
  "multi_image_still_grouped_sample_ok": true,
  "findings_only_count": 6,
  "impression_only_count": 489,
  "missing_optional_metadata_count": 622,
  "repeated_run_determinism_ok": true,
  "prompt_compatibility_summary": {
    "row_count": 200,
    "any_target_leak": false,
    "all_deterministic": true,
    "excluded_count": 0,
    "rag_mode_count": 200,
    "caveats": [
      "PromptBuilder itself carries no dataset-specific identifiers or logic -- it only ever sees plain retrieved_report/target_report strings, so there is no MIMIC-CXR/CheXpert metadata for it to fabricate in the first place; this validation confirms that remains true for IU-X-Ray-derived strings too.",
      "Target report text is never included in retrieved evidence for non-Oracle rows -- checked directly (substring test), not just inferred from self-study exclusion, since two distinct studies could in principle share overlapping text.",
      "Empty evidence has explicit supported behavior via PromptMode.VQA_INFERENCE/VQA_TRAIN (no retrieved_report argument at all), not a special IU-X-Ray-specific empty-string path -- an excluded RAGDatasetRow (retrieved_report_text=None) is routed to VQA mode, never RAG mode with an empty string.",
      "Multi-image metadata is not visible to PromptBuilder at all (it only ever sees a single <image> token placeholder, never a real path) -- multi-image grouping is preserved one layer up, in IuXrayRetrievalRecord.source, not lost by this validation.",
      "Text truncation only occurs through PromptBuilderConfig.max_retrieved_report_length, explicitly configured -- never a hidden default truncation."
    ]
  }
}

## Test results

Cell 44 focused tests: 49 passed
Cell 45 focused tests: 38 passed
Full suite: 1008 passed
