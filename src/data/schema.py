"""Report record schema.

Responsibility: define the typed record structure for a single
radiology study — image path(s), finding text, impression text, and
patient/study identifiers — matching the JSON schema used by the
official FactMM-RAG repository (`{"image": [...], "finding": ...,
"impression": ...}`), extended with explicit patient/study ID fields
for leakage checking.

This module intentionally contains no dataset-level validation (missing
files, duplicate studies, cross-split leakage): that belongs to
src/data/integrity.py and src/data/splits.py, which operate over
collections of records. ReportRecord itself only encodes structure and
the two derived text views the paper and official code actually use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class ReportRecord:
    """A single radiology study: image(s), report text, and identifiers.

    Attributes:
        image_paths: Relative image file paths for this study, in the
            order provided by the source dataset. By MIMIC-CXR/CheXpert
            convention (and the official FactMM-RAG code), the first
            entry is treated as the frontal view.
        finding: The report's "Findings" section text. This is the field
            used for RadGraph/CheXbert annotation and for all generation
            evaluation metrics (F1CheXbert, F1RadGraph, ROUGE-L,
            BERTScore) — never the concatenated form.
        impression: The report's "Impression" section text.
        patient_id: De-identified patient identifier, extracted from the
            source dataset's path structure.
        study_id: De-identified study identifier, extracted the same way.
        dataset: Short tag identifying the source dataset, e.g.
            "mimic-cxr" or "chexpert" — used to keep cross-dataset
            provenance visible (CheXpert is zero-shot-only; it must
            never be silently merged into a MIMIC-CXR training split).
    """

    image_paths: List[str]
    finding: str
    impression: str
    patient_id: str
    study_id: str
    dataset: str

    @property
    def combined_text(self) -> str:
        """Finding and impression concatenated with a single space.

        This is the text representation the official retriever code
        encodes as a candidate's report text (`finding + " " +
        impression`), and what the paper's implementation notes call
        "concatenate finding and impression to form the X-ray report."
        It is a *derived* view — `finding` and `impression` remain
        available separately because annotation and evaluation use
        `finding` alone (see class docstring).
        """
        return f"{self.finding} {self.impression}".strip()

    @property
    def frontal_image_path(self) -> str:
        """The frontal-view image path, by ordering convention.

        Matches the official FactMM-RAG code's convention of always
        using `image[0]` as the primary/frontal image. This is a
        convention-based fallback, not a verified view-position lookup —
        true metadata-based frontal-view selection (when available)
        happens upstream, during parsing (src/data/parsing.py), before a
        ReportRecord is constructed.

        Raises:
            ValueError: if this record has no image paths at all, which
                indicates a malformed upstream record rather than a
                normal empty-data case.
        """
        if not self.image_paths:
            raise ValueError(
                f"ReportRecord for study_id={self.study_id!r} has no "
                f"image_paths; cannot determine a frontal image."
            )
        return self.image_paths[0]
