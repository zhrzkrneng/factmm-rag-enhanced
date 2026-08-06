"""IU X-Ray / Open-I dataset adaptation layer (Milestone 2.8, Cell 44).

Adapts Cell 43's canonicalized IU X-Ray records to this project's
existing baseline data contracts (src.data.schema.ReportRecord) without
modifying any baseline-v1.0 interface. CPU only; no models, no
embeddings, no retriever/generator code.
"""

from src.data.iu_xray.dataset import EXPECTED_SPLIT_COUNTS, SPLIT_NAMES, IuXrayDataset
from src.data.iu_xray.exceptions import (
    IuXrayManifestError,
    IuXrayPathResolutionError,
    IuXraySchemaMappingError,
    IuXraySplitError,
)
from src.data.iu_xray.records import CANONICAL_SCHEMA_FIELDS, IuXrayCanonicalRecord
from src.data.iu_xray.schema_mapping import (
    IU_XRAY_DATASET_TAG,
    MAPPING_TABLE,
    PATIENT_ID_STRATEGY,
    map_records,
    map_to_report_record,
)

__all__ = [
    "IuXrayDataset",
    "SPLIT_NAMES",
    "EXPECTED_SPLIT_COUNTS",
    "IuXrayCanonicalRecord",
    "CANONICAL_SCHEMA_FIELDS",
    "IuXrayManifestError",
    "IuXrayPathResolutionError",
    "IuXraySchemaMappingError",
    "IuXraySplitError",
    "map_to_report_record",
    "map_records",
    "MAPPING_TABLE",
    "IU_XRAY_DATASET_TAG",
    "PATIENT_ID_STRATEGY",
]
