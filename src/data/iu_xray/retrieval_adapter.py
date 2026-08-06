"""IU X-Ray -> baseline retrieval dataset adapter.

Responsibility: convert Cell 44's IuXrayCanonicalRecord splits into the
flat Dict[QueryKey, dict] shape src.baseline.retrieval.dataset already
consumes (REQUIRED_RECORD_FIELDS: dataset, patient_id, study_id,
finding, image_path) -- the exact same schema
src.baseline.retrieval.dataset.load_annotated_records() produces from a
JSONL file, produced here in-memory instead. RetrieverTrainingDataset
itself is never modified; this module only produces its expected input.

Real, documented limitation carried forward from schema_mapping.py:
REQUIRED_RECORD_FIELDS has room for exactly one image_path per query,
matching MIMIC-CXR/CheXpert's frontal-image convention -- so only
image_paths[0] is threaded into the flat dict. Full multi-image grouping
is NOT lost: build_records() also returns the untouched
IuXrayCanonicalRecord alongside each flat dict, so downstream code that
needs every image still has it (see IuXrayRetrievalRecord).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from src.baseline.pair_mining.mining import QueryKey
from src.data.iu_xray.records import IuXrayCanonicalRecord
from src.data.iu_xray.schema_mapping import IU_XRAY_DATASET_TAG
from src.data.iu_xray.target_report_policy import build_target_report

CORPUS_POLICY = (
    "Corpus = every usable ('valid') record in the split(s) passed to "
    "build_corpus(), converted 1:1 (never expanded per-image, never "
    "deduplicated beyond what Cell 43 already guarantees). Callers "
    "decide which split(s) form the corpus (this module does not "
    "hardcode 'train_only' -- see src.baseline.generation.dataset_"
    "builder.RAGDatasetBuilderConfig.corpus_scope for the convention "
    "this mirrors)."
)
QUERY_POLICY = (
    "Queries = every usable record in the split passed to "
    "build_queries(), one query per study/report (never per image). A "
    "query's own key is always excluded from its own candidate set by "
    "filter_self_matches() -- test/validation queries never retrieve "
    "themselves."
)
SELF_MATCH_EXCLUSION_POLICY = (
    "By QueryKey equality (dataset, patient_id, study_id) -- since IU "
    "X-Ray's patient_id equals study_id (schema_mapping."
    "PATIENT_ID_STRATEGY), self-study and self-patient exclusion "
    "collapse to the same check for this dataset. filter_self_matches() "
    "implements this directly; src.baseline.generation.dataset_builder."
    "RAGDatasetBuilder additionally, independently enforces the same "
    "invariant when building generation rows (defense in depth, not "
    "redundant -- the two call sites are different)."
)


@dataclass(frozen=True)
class IuXrayRetrievalRecord:
    """One study's flat baseline-retrieval dict, paired with its full
    IuXrayCanonicalRecord (all images, mesh_terms, labels, etc.) so
    downstream code needing more than the flat dict's single image_path
    still has it."""

    query_key: QueryKey
    flat_record: Dict[str, str]
    source: IuXrayCanonicalRecord
    target_report_policy_branch: str


def _study_id_key(study_id: str) -> QueryKey:
    # PATIENT_ID_STRATEGY: study_id doubles as patient_id.
    return (IU_XRAY_DATASET_TAG, study_id, study_id)


def build_records(
    records: List[IuXrayCanonicalRecord],
) -> Dict[QueryKey, IuXrayRetrievalRecord]:
    """Converts a list of usable IuXrayCanonicalRecords into
    QueryKey-addressed IuXrayRetrievalRecords, in input order (callers
    are expected to pass an already-deterministically-ordered list --
    see src.data.iu_xray.dataset.IuXrayDataset.load_split()).

    Raises:
        ValueError: if a record's target-report policy could not
            produce any text (branch "unavailable") -- never silently
            dropped or fabricated; the caller decides how to handle a
            genuinely un-mappable record, but this function refuses to
            produce a record with an empty/None "finding" field, since
            REQUIRED_RECORD_FIELDS requires a real string there. Also
            raised if any input record is not validation_status="valid"
            -- an excluded record (e.g. exclusion_reason="missing_image")
            can still have non-None findings/impression text, so this is
            checked explicitly rather than trusted to the caller (see
            src.data.iu_xray.dataset.IuXrayDataset.load_split's
            include_excluded=False default, which this defends in depth
            against a caller bypassing).
    """
    out: Dict[QueryKey, IuXrayRetrievalRecord] = {}
    for record in records:
        if not record.is_usable:
            raise ValueError(
                f"study_id={record.study_id!r}: validation_status="
                f"{record.validation_status!r} (exclusion_reason="
                f"{record.exclusion_reason!r}) -- excluded records must "
                f"never be converted into retrieval/generation records"
            )
        result = build_target_report(record)
        if result.text is None:
            raise ValueError(
                f"study_id={record.study_id!r}: target-report policy "
                f"produced no text (branch={result.policy_branch!r}); "
                f"cannot build a retrieval record without fabricating "
                f"the required 'finding' text"
            )
        key = _study_id_key(record.study_id)
        flat = {
            "dataset": IU_XRAY_DATASET_TAG,
            "patient_id": key[1],
            "study_id": key[2],
            "finding": result.text,
            "image_path": record.image_paths[0],
        }
        out[key] = IuXrayRetrievalRecord(
            query_key=key, flat_record=flat, source=record,
            target_report_policy_branch=result.policy_branch,
        )
    return out


def build_corpus(records: List[IuXrayCanonicalRecord]) -> Dict[QueryKey, dict]:
    """Builds the flat Dict[QueryKey, dict] corpus, directly consumable
    by src.baseline.retrieval.dataset.RetrieverTrainingDataset /
    src.baseline.generation.dataset_builder.RAGDatasetBuilder without
    modification."""
    return {k: v.flat_record for k, v in build_records(records).items()}


def build_queries(records: List[IuXrayCanonicalRecord]) -> Dict[QueryKey, dict]:
    """Same conversion as build_corpus(); a distinct name only to make
    caller intent explicit (see QUERY_POLICY/CORPUS_POLICY)."""
    return build_corpus(records)


def filter_self_matches(
    query_key: QueryKey, candidate_keys: List[QueryKey]
) -> List[QueryKey]:
    """Removes `query_key` from `candidate_keys`, preserving order.

    A query never retrieves itself -- enforced explicitly here for any
    code path that builds candidate lists directly (see
    SELF_MATCH_EXCLUSION_POLICY for how this relates to
    RAGDatasetBuilder's own independent enforcement)."""
    return [key for key in candidate_keys if key != query_key]


def to_contract_json() -> dict:
    return {
        "corpus_policy": CORPUS_POLICY,
        "query_policy": QUERY_POLICY,
        "self_match_exclusion_policy": SELF_MATCH_EXCLUSION_POLICY,
        "query_key_shape": "(dataset, patient_id, study_id) -- patient_id == study_id for iu-xray",
        "one_record_per_study": True,
        "image_paths_per_flat_record": 1,
        "multi_image_metadata_preserved_via": "IuXrayRetrievalRecord.source (full IuXrayCanonicalRecord)",
    }
