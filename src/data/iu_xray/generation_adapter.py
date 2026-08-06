"""IU X-Ray -> baseline generation/RAG dataset adapter.

Responsibility: convert Cell 44 splits into
src.baseline.generation.dataset_builder.RAGDatasetBuilder's expected
input shapes (query_records, corpus_records: Dict[QueryKey, dict];
knn_rankings: Dict[QueryKey, List[QueryKey]]) and provide three
explicitly-separated, clearly-labeled ranking sources for contract
validation -- empty, deterministic-synthetic, and Oracle -- none of
which is a real retriever result. RAGDatasetBuilder itself is never
modified; its own self-study/self-patient exclusion, patient-leakage
check, and target/retrieved-text handling all run unmodified against
this module's output.
"""

from __future__ import annotations

from typing import Dict, List

from src.baseline.pair_mining.mining import QueryKey
from src.data.iu_xray.records import IuXrayCanonicalRecord
from src.data.iu_xray.retrieval_adapter import build_records

# Reused as both RAGDatasetBuilderConfig.rag_data_mode and
# .output_data_mode -- IU X-Ray populates a single deterministic
# "finding" field (the target-report-policy text) for both roles,
# consistent with how the flat retrieval dict already works.
GENERATION_TEXT_FIELD = "finding"

GENERATION_SAMPLE_POLICY = (
    "One RAGDatasetRow per usable study (never per image), built by "
    "src.baseline.generation.dataset_builder.RAGDatasetBuilder itself "
    "(unmodified) from this module's query_records/corpus_records/"
    "knn_rankings. target_report_text is exactly the target-report-"
    "policy text (see src.data.iu_xray.target_report_policy) -- never "
    "altered by RAGDatasetBuilder, which only ever reads it verbatim "
    "from query_records[query_key][output_data_mode]."
)


def build_generation_records(
    records: List[IuXrayCanonicalRecord],
) -> Dict[QueryKey, dict]:
    """Flat Dict[QueryKey, dict] with 'image_path' and GENERATION_TEXT_FIELD
    populated -- directly usable as either query_records or
    corpus_records for RAGDatasetBuilder."""
    return {k: v.flat_record for k, v in build_records(records).items()}


def build_empty_rankings(query_keys: List[QueryKey]) -> Dict[QueryKey, List[QueryKey]]:
    """Every query maps to an empty candidate list.

    Used for contract validation only: RAGDatasetBuilder's own existing
    logic (unmodified) marks every resulting row excluded=True when
    reproduce_official_bug=False (the safe default) -- this module does
    not special-case that, it simply supplies no candidates and lets
    RAGDatasetBuilder's real behavior run."""
    return {key: [] for key in query_keys}


def build_deterministic_synthetic_rankings(
    query_keys: List[QueryKey], corpus_keys: List[QueryKey]
) -> Dict[QueryKey, List[QueryKey]]:
    """Deterministic, clearly-labeled SYNTHETIC rankings for contract
    validation -- NEVER a real retrieval result (no embeddings, no
    FAISS, no similarity scoring anywhere in this cell).

    Each query's ranking is simply every corpus key, sorted ascending,
    with the query's own key removed if present -- fully reproducible
    given the same inputs, and safe to feed straight into
    RAGDatasetBuilder to exercise its selection/rejection logic for
    real."""
    sorted_corpus = sorted(corpus_keys)
    return {
        query_key: [key for key in sorted_corpus if key != query_key]
        for query_key in query_keys
    }


def build_oracle_rankings(query_keys: List[QueryKey]) -> Dict[QueryKey, List[QueryKey]]:
    """DIAGNOSTIC-ONLY: each query's top (and only) candidate is itself.

    Isolated in its own function, never called from
    build_deterministic_synthetic_rankings/build_empty_rankings, and
    never used to build a "real" generation dataset -- its only purpose
    is to let a caller assert that RAGDatasetBuilder correctly rejects
    self-matches (num_candidates_rejected_self_study increments, the
    row ends up excluded=True since no other candidate exists), i.e. a
    negative-path diagnostic, not a data source."""
    return {query_key: [query_key] for query_key in query_keys}


def to_contract_json() -> dict:
    return {
        "generation_sample_policy": GENERATION_SAMPLE_POLICY,
        "text_field": GENERATION_TEXT_FIELD,
        "ranking_sources": {
            "empty": "every row excluded=True (contract validation only)",
            "deterministic_synthetic": "sorted corpus minus self -- NOT a real retriever result",
            "oracle": "diagnostic-only self-reference, isolated from the main flow",
        },
        "no_embeddings_computed": True,
        "no_faiss_index_built": True,
        "no_retriever_selected": True,
    }
