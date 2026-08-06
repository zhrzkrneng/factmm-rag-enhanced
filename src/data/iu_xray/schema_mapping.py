"""IU X-Ray -> baseline ReportRecord schema mapping.

Responsibility: an explicit, inspectable mapping table from
IuXrayCanonicalRecord's 17 fields to src.data.schema.ReportRecord's 6
fields (the baseline sample contract every downstream stage --
retriever, generator, evaluation -- actually consumes), plus the mapper
function itself. Never modifies src.data.schema.ReportRecord or any
other baseline interface; this module only produces ReportRecord
instances the same way src.data.parsing.JsonReportParser already does
for MIMIC-CXR/CheXpert.

MAPPING_TABLE is written once here and exported verbatim (via
to_mapping_table_json()) into Cell 44's cell44_schema_mapping.json
artifact, so the two can never silently drift apart.
"""

from __future__ import annotations

from typing import List, Tuple

from src.data.iu_xray.exceptions import IuXraySchemaMappingError
from src.data.iu_xray.records import IuXrayCanonicalRecord
from src.data.schema import ReportRecord

# dataset tag used for every ReportRecord this mapper produces, parallel
# to the existing "mimic-cxr"/"chexpert" tags in src.data.parsing.
IU_XRAY_DATASET_TAG = "iu-xray"

# Identifier-mapping adapter rule (not a clinical-content fabrication):
# ReportRecord.patient_id is a required field, but IU X-Ray provides no
# patient-level identifier distinct from study_id (Cell 42's audit: no
# PhysioNet-style patient/study directory structure exists for this
# dataset). study_id is used as patient_id 1:1. Documented explicitly,
# not silently assumed -- see the "patient_id" row below and the
# caveat this produces for src.data.splits.PatientSplitValidator.
PATIENT_ID_STRATEGY = "study_id_as_patient_id"

MAPPING_TABLE: Tuple[dict, ...] = (
    {"iu_xray_field": "dataset_name", "baseline_field": "dataset",
     "transformation": f"NOT copied verbatim -- replaced with the "
                        f"constant {IU_XRAY_DATASET_TAG!r} (baseline's "
                        f"existing dataset-tag convention uses hyphens: "
                        f"'mimic-cxr', 'chexpert'; Cell 43's own "
                        f"canonical records use 'iu_xray' with an "
                        f"underscore internally)",
     "loss_or_caveat": "source value ('iu_xray') is discarded in favor "
                        "of the project-convention-matching tag; no "
                        "information loss (both name the same dataset)"},
    {"iu_xray_field": "study_id", "baseline_field": "study_id",
     "transformation": "direct copy",
     "loss_or_caveat": "none"},
    {"iu_xray_field": "(none)", "baseline_field": "patient_id",
     "transformation": f"adapter rule: {PATIENT_ID_STRATEGY}",
     "loss_or_caveat": (
         "IU X-Ray has no patient-level identifier distinct from "
         "study_id; using study_id as patient_id means "
         "PatientSplitValidator's cross-split leakage check becomes "
         "trivially satisfied for this dataset (each synthetic "
         "patient_id maps to exactly one study by construction) -- it "
         "provides no real patient-level leakage protection here."
     )},
    {"iu_xray_field": "report_id", "baseline_field": "(none)",
     "transformation": "dropped",
     "loss_or_caveat": "equals study_id for this dataset (Cell 43 XML "
                        "convention); not carried into ReportRecord"},
    {"iu_xray_field": "image_ids", "baseline_field": "(none)",
     "transformation": "dropped from ReportRecord",
     "loss_or_caveat": "baseline has no per-image-identifier concept, "
                        "only an ordered path list; preserved on "
                        "IuXrayCanonicalRecord itself"},
    {"iu_xray_field": "image_paths", "baseline_field": "image_paths",
     "transformation": "resolve ${IU_XRAY_STORAGE_ROOT} token per "
                        "entry, then copy in order",
     "loss_or_caveat": "baseline's frontal_image_path property assumes "
                        "image_paths[0] is the frontal view (MIMIC-CXR/"
                        "CheXpert convention); IU X-Ray provides no "
                        "confirmed view-position metadata to verify "
                        "this ordering (Cell 42 audit: UNKNOWN)"},
    {"iu_xray_field": "image_count", "baseline_field": "(none)",
     "transformation": "dropped", "loss_or_caveat": "redundant with "
                        "len(image_paths); not a ReportRecord field"},
    {"iu_xray_field": "findings", "baseline_field": "finding",
     "transformation": "direct copy",
     "loss_or_caveat": "ReportRecord.finding is a required non-null "
                        "str; a 'valid' IU X-Ray record with findings="
                        "None (e.g. impression-only) cannot be mapped "
                        "-- raises IuXraySchemaMappingError rather than "
                        "fabricating text or silently coercing to ''"},
    {"iu_xray_field": "impression", "baseline_field": "impression",
     "transformation": "direct copy",
     "loss_or_caveat": "same non-null requirement and failure mode as "
                        "findings above"},
    {"iu_xray_field": "indication", "baseline_field": "(none)",
     "transformation": "dropped",
     "loss_or_caveat": "no baseline field; MIMIC-CXR/CheXpert reports "
                        "have no Indication section in this project's "
                        "schema"},
    {"iu_xray_field": "comparison", "baseline_field": "(none)",
     "transformation": "dropped",
     "loss_or_caveat": "no baseline field; same reasoning as indication"},
    {"iu_xray_field": "full_report", "baseline_field": "(none)",
     "transformation": "dropped",
     "loss_or_caveat": "NOT the same concept as ReportRecord."
                        "combined_text (finding+' '+impression only, "
                        "per the official FactMM-RAG convention) -- "
                        "full_report additionally includes comparison/"
                        "indication when present; must not be conflated "
                        "with combined_text"},
    {"iu_xray_field": "mesh_terms", "baseline_field": "(none)",
     "transformation": "dropped",
     "loss_or_caveat": "no baseline field; not CheXpert-style structured "
                        "labels"},
    {"iu_xray_field": "labels", "baseline_field": "(none)",
     "transformation": "dropped", "loss_or_caveat": "no baseline field"},
    {"iu_xray_field": "source_metadata", "baseline_field": "(none)",
     "transformation": "dropped from ReportRecord",
     "loss_or_caveat": "preserved on IuXrayCanonicalRecord only"},
    {"iu_xray_field": "source_split", "baseline_field": "(none)",
     "transformation": "not embedded per-record",
     "loss_or_caveat": "split membership is a loader-level concern "
                        "(dataset.load_split()), not a ReportRecord "
                        "field"},
    {"iu_xray_field": "validation_status / exclusion_reason",
     "baseline_field": "(none)",
     "transformation": "drives inclusion filtering upstream of mapping",
     "loss_or_caveat": "excluded records are never passed to the "
                        "mapper by default (dataset.py's "
                        "include_excluded=False)"},
)


def to_mapping_table_json() -> List[dict]:
    """Returns MAPPING_TABLE as a plain list of dicts, JSON-ready."""
    return [dict(row) for row in MAPPING_TABLE]


def map_to_report_record(
    record: IuXrayCanonicalRecord, storage_root, resolve_path_fn
) -> ReportRecord:
    """Maps one IuXrayCanonicalRecord to a baseline ReportRecord.

    Args:
        record: the source IU X-Ray record.
        storage_root: resolved IU_XRAY_STORAGE_ROOT, passed through to
            `resolve_path_fn` for each image path.
        resolve_path_fn: `src.data.iu_xray.paths.resolve_path`, injected
            rather than imported directly so this function has no
            hidden global-state dependency.

    Raises:
        IuXraySchemaMappingError: if `record.findings` or
            `record.impression` is None -- the baseline contract
            requires both to be non-null strings, and this mapper never
            fabricates missing clinical text to satisfy that.
    """
    if record.findings is None or record.impression is None:
        missing = [
            name for name, value in
            (("findings", record.findings), ("impression", record.impression))
            if value is None
        ]
        raise IuXraySchemaMappingError(
            f"study_id={record.study_id!r} cannot be mapped to "
            f"ReportRecord: {', '.join(missing)} is None, but "
            f"ReportRecord.finding/impression are required non-null "
            f"fields. Not fabricated; see MAPPING_TABLE's 'findings'/"
            f"'impression' rows."
        )
    resolved_image_paths = [
        resolve_path_fn(path, storage_root) for path in record.image_paths
    ]
    return ReportRecord(
        image_paths=resolved_image_paths,
        finding=record.findings,
        impression=record.impression,
        patient_id=record.study_id,  # PATIENT_ID_STRATEGY
        study_id=record.study_id,
        dataset=IU_XRAY_DATASET_TAG,
    )


class MappingSkip:
    """One record that could not be mapped, and why."""

    __slots__ = ("study_id", "reason")

    def __init__(self, study_id: str, reason: str):
        self.study_id = study_id
        self.reason = reason

    def as_dict(self) -> dict:
        return {"study_id": self.study_id, "reason": self.reason}


def map_records(
    records: List[IuXrayCanonicalRecord], storage_root, resolve_path_fn
) -> Tuple[List[ReportRecord], List[MappingSkip]]:
    """Maps every record, collecting (never silently dropping) the ones
    that cannot be mapped, matching this project's established
    collect-then-report pattern (src.data.integrity.IntegrityChecker).

    Returns:
        (mapped_report_records, skipped) -- `skipped` is empty on a
        fully-compatible input set; non-empty entries must be surfaced
        in the caller's report (Cell 44's cell44_schema_mapping.json),
        never discarded.
    """
    mapped: List[ReportRecord] = []
    skipped: List[MappingSkip] = []
    for record in records:
        try:
            mapped.append(map_to_report_record(record, storage_root, resolve_path_fn))
        except IuXraySchemaMappingError as exc:
            skipped.append(MappingSkip(study_id=record.study_id, reason=str(exc)))
    return mapped, skipped
