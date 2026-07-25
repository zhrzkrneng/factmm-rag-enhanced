"""Report record schema.

Responsibility: define the typed record structure for a single
radiology study — image path(s), finding text, impression text, and
patient/study identifiers — matching the JSON schema used by the
official FactMM-RAG repository (`{"image": [...], "finding": ...,
"impression": ...}`), extended with explicit patient/study ID fields
for leakage checking. No implementation yet.
"""
