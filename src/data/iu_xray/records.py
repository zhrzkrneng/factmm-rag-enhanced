"""IU X-Ray canonical record schema.

Responsibility: define the typed record structure matching Cell 43's
canonicalization output exactly (the `schema_fields` list recorded in
results/reproduction/milestone_2_8/iu_xray_canonical_manifest.json),
and provide (de)serialization to/from the plain-dict form used by the
external full canonical manifest's `records` array. Contains no
dataset-level validation, filtering, or splitting -- that belongs to
dataset.py, which operates over collections of these records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Must match iu_xray_canonical_manifest.json's "schema_fields" exactly
# (Cell 43's CANONICAL_SCHEMA_FIELDS) -- checked by manifest_reader.py
# before trusting any record built from the external manifest.
CANONICAL_SCHEMA_FIELDS = (
    "dataset_name",
    "study_id",
    "report_id",
    "image_ids",
    "image_paths",
    "image_count",
    "findings",
    "impression",
    "indication",
    "comparison",
    "full_report",
    "mesh_terms",
    "labels",
    "source_metadata",
    "source_split",
    "validation_status",
    "exclusion_reason",
)


@dataclass(frozen=True)
class IuXrayCanonicalRecord:
    """One IU X-Ray study/report, exactly as Cell 43 canonicalized it.

    One instance per study/report -- never one per image. All images
    belonging to one study are grouped in `image_ids`/`image_paths`,
    in the order Cell 43 discovered them.

    Attributes:
        dataset_name: source dataset tag, "iu_xray" for every record
            produced by Cell 43.
        study_id: de-identified study identifier (equals report_id for
            this dataset's XML layout -- IU X-Ray provides no separate
            patient-level identifier, see Cell 42's audit).
        report_id: de-identified report identifier.
        image_ids: every parentImage id referenced by this study's
            report XML, including ones that did not resolve to an
            actual file on disk (see source_metadata.unresolved_image_ids).
        image_paths: resolved, existing image file paths only -- may be
            shorter than image_ids if some referenced images were not
            found (never silently padded or reordered to match).
        image_count: len(image_paths) at canonicalization time.
        findings: "Findings" section text, or None if this report has
            no Findings section. Never inferred when missing.
        impression: "Impression" section text, or None. Never inferred.
        indication: "Indication" section text, or None. IU X-Ray-only
            section; MIMIC-CXR/CheXpert reports have no equivalent.
        comparison: "Comparison" section text, or None. IU X-Ray-only.
        full_report: comparison+indication+findings+impression
            concatenated in that order (only the present sections), or
            None if none were present. NOT the same concept as
            src.data.schema.ReportRecord.combined_text (finding+
            impression only) -- see schema_mapping.py.
        mesh_terms: MeSH terms found in this report's XML, deduplicated
            and sorted. Empty list if none found.
        labels: non-MeSH "problem"-style labels found in this report's
            XML, deduplicated and sorted. Empty list if none found.
        source_metadata: free-form provenance dict from Cell 43
            (source XML file path, unmapped AbstractText labels,
            unresolved image ids). Never re-derived here.
        source_split: split name if the source manifest already
            recorded one, else None (Cell 43 always recorded None here
            -- split assignment is a separate, later step; see
            dataset.py's load_split()).
        validation_status: "valid" or "excluded", exactly as Cell 43
            determined it. Never re-evaluated by this module.
        exclusion_reason: reason string when validation_status is
            "excluded" (e.g. "missing_image", "missing_report"), else
            None.
    """

    dataset_name: str
    study_id: str
    report_id: str
    image_ids: List[str]
    image_paths: List[str]
    image_count: int
    findings: Optional[str]
    impression: Optional[str]
    indication: Optional[str]
    comparison: Optional[str]
    full_report: Optional[str]
    mesh_terms: List[str]
    labels: List[str]
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    source_split: Optional[str] = None
    validation_status: str = "valid"
    exclusion_reason: Optional[str] = None

    @property
    def is_usable(self) -> bool:
        """True iff Cell 43 marked this record "valid"."""
        return self.validation_status == "valid"

    @property
    def is_multi_image(self) -> bool:
        """True iff this study has more than one resolved image."""
        return len(self.image_paths) > 1

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "IuXrayCanonicalRecord":
        """Builds one record from the external manifest's per-record dict.

        Raises:
            KeyError: if a required canonical field is missing from
                `payload` -- never defaulted or guessed.
        """
        return cls(
            dataset_name=payload["dataset_name"],
            study_id=payload["study_id"],
            report_id=payload["report_id"],
            image_ids=list(payload["image_ids"]),
            image_paths=list(payload["image_paths"]),
            image_count=payload["image_count"],
            findings=payload["findings"],
            impression=payload["impression"],
            indication=payload["indication"],
            comparison=payload["comparison"],
            full_report=payload["full_report"],
            mesh_terms=list(payload["mesh_terms"]),
            labels=list(payload["labels"]),
            source_metadata=dict(payload.get("source_metadata") or {}),
            source_split=payload.get("source_split"),
            validation_status=payload["validation_status"],
            exclusion_reason=payload.get("exclusion_reason"),
        )
