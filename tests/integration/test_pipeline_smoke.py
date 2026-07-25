"""End-to-end smoke test for the Milestone 2.1 data pipeline.

Chains JsonReportParser -> IntegrityChecker -> PatientSplitValidator ->
ManifestBuilder over the synthetic fixtures in one test, exercising the
full parse -> validate -> manifest chain together rather than each
class in isolation (already covered by tests/unit/). Uses only
synthetic fixture data — no real patient data, per project rules.
"""

import os

from src.data.integrity import IntegrityChecker
from src.data.manifest import ManifestBuilder
from src.data.parsing import JsonReportParser
from src.data.splits import PatientSplitValidator

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def test_full_pipeline_end_to_end(tmp_path):
    # --- 1. Parse both datasets ---
    mimic_records = JsonReportParser(dataset="mimic-cxr").parse_file(
        os.path.join(FIXTURES_DIR, "sample_mimic.json")
    )
    chexpert_records = JsonReportParser(dataset="chexpert").parse_file(
        os.path.join(FIXTURES_DIR, "sample_chexpert.json")
    )
    assert len(mimic_records) == 2
    assert len(chexpert_records) == 1

    # Simulate a split: first mimic record -> train, second -> valid;
    # the chexpert record -> test, matching its real zero-shot-only role
    # (docs/paper_analysis.md Section 4).
    split_records = {
        "train": [mimic_records[0]],
        "valid": [mimic_records[1]],
        "test": chexpert_records,
    }

    # --- 2. Integrity check ---
    # Create real (empty) files at every referenced image path so the
    # missing_image_file check passes here; missing-file *detection*
    # itself is already covered by tests/unit/test_integrity.py, so
    # this smoke test isolates pipeline wiring instead of re-testing it.
    for records in split_records.values():
        for record in records:
            for image_path in record.image_paths:
                full_path = tmp_path / image_path
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text("")

    checker = IntegrityChecker(image_root=str(tmp_path))
    for records in split_records.values():
        checker.check_and_raise(records)  # must not raise

    # --- 3. Split validation ---
    # mimic's two patient_ids and chexpert's one patient_id are all
    # distinct, so no leakage is expected.
    PatientSplitValidator().check_and_raise(split_records)  # must not raise

    # --- 4. Manifest generation, one per split ---
    builder = ManifestBuilder()
    manifest_dir = tmp_path / "manifests"
    for split_name, records in split_records.items():
        manifest = builder.build(split_name, records)
        output_path = str(manifest_dir / f"{split_name}_manifest.json")
        builder.save(manifest, output_path)

        reloaded = builder.load(output_path)
        assert reloaded == manifest
        assert reloaded.record_count == len(records)

    produced_files = set(os.listdir(str(manifest_dir)))
    assert produced_files == {
        "train_manifest.json",
        "valid_manifest.json",
        "test_manifest.json",
    }
