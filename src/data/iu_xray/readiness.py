"""Cell 46 -- IU X-Ray end-to-end CPU readiness dry-run arms + closure.

Responsibility: four clearly-separated diagnostic arms (DATA_ONLY,
MOCK_RETRIEVAL, ORACLE_PIPELINE_CHECK, MOCK_GENERATION_SMOKE), a
component readiness matrix, metric-readiness structural smoke checks,
and milestone-closure claim constants -- all built on top of Cell 44's
loader and Cell 45's retrieval/generation/prompt adapters, and on the
project's own real (never model-loading) mock interfaces:
src.baseline.generation.mock_adapter.MockGeneratorAdapter,
src.baseline.evaluation.oracle.OracleEvaluator (with injected fake
scorers, matching its own documented test discipline), and
src.evaluation.{retrieval_metrics,generation_metrics}.build_mock_*_
registry. No real retriever, no embeddings, no FAISS index, no Vicuna/
LLaVA, no real report generation, no scientific metric value is ever
produced or reported by this module -- every arm's output is explicitly
labeled diagnostic-only.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple

from src.baseline.evaluation.oracle import OracleConfig, OracleEvaluator
from src.baseline.generation.adapter import GeneratorResult
from src.baseline.generation.dataset_builder import RAGDatasetRow
from src.baseline.generation.generator import GenerationConfig
from src.baseline.generation.mock_adapter import MockGeneratorAdapter
from src.baseline.generation.prompt_builder import PromptBuilder, PromptBuilderConfig, PromptMode
from src.baseline.pair_mining.mining import QueryKey
from src.data.iu_xray import generation_adapter, retrieval_adapter
from src.data.iu_xray.records import IuXrayCanonicalRecord
from src.data.iu_xray.schema_mapping import IU_XRAY_DATASET_TAG, map_records
from src.data.iu_xray.target_report_policy import build_target_report
from src.data.schema import ReportRecord

# ---------------------------------------------------------------------------
# Readiness statuses
# ---------------------------------------------------------------------------

READY = "READY"
PARTIALLY_READY = "PARTIALLY_READY"
DEFERRED = "DEFERRED"
REMOVED = "REMOVED"
BLOCKED = "BLOCKED"

VALID_STATUSES = (READY, PARTIALLY_READY, DEFERRED, REMOVED, BLOCKED)


def validate_readiness_matrix(matrix: Dict[str, dict]) -> List[str]:
    """Returns a list of validation problems (empty if the matrix is
    well-formed): every entry must have a status in VALID_STATUSES and a
    non-empty rationale -- never an unexplained classification."""
    problems = []
    for component, entry in matrix.items():
        status = entry.get("status")
        rationale = entry.get("rationale")
        if status not in VALID_STATUSES:
            problems.append(f"{component}: invalid status {status!r}")
        if not rationale:
            problems.append(f"{component}: missing rationale")
    return problems


# ---------------------------------------------------------------------------
# ARM 1: DATA_ONLY
# ---------------------------------------------------------------------------


def run_data_only_arm(records: List[IuXrayCanonicalRecord]) -> dict:
    """Validates loader output + schema mapping + target-report policy
    over `records`, with NO retrieval and NO generation involved.

    Schema mapping is exercised via the established
    schema_mapping.map_records() collect-then-report pattern, not the
    strict map_to_report_record() directly -- a real, usable IU X-Ray
    record legitimately has findings=None (impression-only) or
    impression=None (findings-only) (this is exactly why the 4-branch
    target-report policy above exists), and map_to_report_record()
    correctly refuses to fabricate the missing half rather than being
    loosened here to accept it.

    Note on `deterministic_ordering`:
        This checks that reprocessing the exact same input `records`
        yields the exact same per-study output every time -- NOT that
        `records` itself must be sorted ascending by study_id.
        IuXrayDataset.load_split() only guarantees ascending-sorted
        order WITHIN its own single split (see that method's own
        docstring); a caller-built multi-split sample (e.g. train +
        validation + test concatenated for a readiness dry run) is
        legitimately NOT globally sorted, and this arm must not reject
        it as if it were non-deterministic.
    """
    if any(not r.is_usable for r in records):
        raise ValueError("run_data_only_arm: received an excluded record")

    study_ids = [r.study_id for r in records]

    policy_results = {r.study_id: build_target_report(r) for r in records}
    all_policy_texts_present = all(res.text is not None for res in policy_results.values())

    rerun_study_ids = [r.study_id for r in records]
    rerun_policy_branches = [build_target_report(r).policy_branch for r in records]
    original_policy_branches = [policy_results[sid].policy_branch for sid in study_ids]
    deterministic_order = (
        rerun_study_ids == study_ids and rerun_policy_branches == original_policy_branches
    )

    mapped, skipped = map_records(records, "not-used", lambda p, s: p)
    mapped_study_ids = {r.study_id for r in mapped}
    skipped_study_ids = {s.study_id for s in skipped}
    one_sample_per_study = (
        len(mapped) + len(skipped) == len(records)
        and mapped_study_ids.isdisjoint(skipped_study_ids)
        and (mapped_study_ids | skipped_study_ids) == set(study_ids)
        and len(records) == len(set(study_ids))
    )

    multi_image_grouping_preserved = all(
        len(r.image_paths) == r.image_count for r in records
    )

    return {
        "arm": "DATA_ONLY",
        "diagnostic_label": "DATA_ONLY -- loader/schema validation, no retrieval, no generation",
        "record_count": len(records),
        "deterministic_ordering": deterministic_order,
        "one_sample_per_study": one_sample_per_study,
        "multi_image_grouping_preserved": multi_image_grouping_preserved,
        "all_target_report_policy_texts_present": all_policy_texts_present,
        "policy_branch_counts": _count_branches(policy_results),
        "schema_mapping_full_record_count": len(mapped),
        "schema_mapping_skipped_count": len(skipped),
        "schema_mapping_skip_reasons": [s.reason for s in skipped],
        "no_excluded_records": True,
    }


def _count_branches(policy_results: Dict[str, object]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for result in policy_results.values():
        counts[result.policy_branch] = counts.get(result.policy_branch, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# ARM 2: MOCK_RETRIEVAL
# ---------------------------------------------------------------------------


def run_mock_retrieval_arm(records: List[IuXrayCanonicalRecord], top_k: int = 5) -> dict:
    """Deterministic MOCK_RETRIEVAL diagnostic -- NOT a real retriever.

    Reuses generation_adapter.build_deterministic_synthetic_rankings
    (the same "sorted corpus minus self" deterministic construction
    Cell 45 already validated), truncated to top_k, as the mock top-k
    evidence. No embedding model, no FAISS, no MARVEL/MedCPT/BGE-M3/
    CLIP/T5-ANCE anywhere in this function.
    """
    built = retrieval_adapter.build_records(records)
    query_keys = list(built)
    rankings = generation_adapter.build_deterministic_synthetic_rankings(query_keys, query_keys)
    top_k_rankings = {qk: candidates[:top_k] for qk, candidates in rankings.items()}

    no_self_match = all(qk not in cands for qk, cands in top_k_rankings.items())
    evidence_from_different_study = all(
        all(built[c].source.study_id != built[qk].source.study_id for c in cands)
        for qk, cands in top_k_rankings.items()
    )
    rankings_rerun = generation_adapter.build_deterministic_synthetic_rankings(query_keys, query_keys)
    deterministic = rankings == rankings_rerun

    return {
        "arm": "MOCK_RETRIEVAL",
        "diagnostic_label": "MOCK_RETRIEVAL -- deterministic synthetic evidence, "
                             "NOT baseline retrieval performance",
        "query_count": len(query_keys),
        "top_k": top_k,
        "no_self_match": no_self_match,
        "evidence_from_different_study": evidence_from_different_study,
        "deterministic_across_repeated_runs": deterministic,
        "no_real_embedding_model": True,
        "no_real_faiss_index": True,
        "forbidden_retrievers_used": [],
    }


# ---------------------------------------------------------------------------
# ARM 3: ORACLE_PIPELINE_CHECK
# ---------------------------------------------------------------------------

ORACLE_ARM_LABEL = "ORACLE_PIPELINE_CHECK -- NON-DEPLOYABLE DIAGNOSTIC upper-bound pipeline check, never a model-performance claim"


def _deterministic_fake_score(text_a: str, text_b: str, salt: str) -> float:
    """Small, deterministic, no-library fake scorer in [0, 1] -- exactly
    the discipline OracleEvaluator's own module docstring calls for
    ("small, deterministic FAKE scorer functions, never a real CheXbert/
    RadGraph instance-scoring call"). sha256-seeded, never Python's
    randomized hash()."""
    digest = hashlib.sha256(f"{salt}||{text_a}||{text_b}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def run_oracle_diagnostic_arm(
    records: List[IuXrayCanonicalRecord], query_study_ids: Sequence[str]
) -> dict:
    """Isolated Oracle diagnostic: real OracleEvaluator, fake injected
    scorers (never real CheXbert/RadGraph), corpus = all `records`,
    queries = the subset named by `query_study_ids`. Returns results
    clearly labeled NON-DEPLOYABLE DIAGNOSTIC -- never included in any
    baseline scientific result table by this function or its caller.

    Raises:
        ValueError: if any of `records` is not validation_status="valid"
            -- same defense-in-depth guard as
            retrieval_adapter.build_records, applied here directly since
            this function maps records itself rather than delegating to
            that one.

    Note:
        Builds each ReportRecord's `.finding` from the same
        target_report_policy.build_target_report() text
        retrieval_adapter.build_records() already uses -- NOT via
        map_to_report_record(), which requires findings and impression
        to both be non-None and would incorrectly reject a real, usable
        impression-only or findings-only study. OracleEvaluator.build_row
        only ever reads `.finding`; `.impression` is populated with the
        same derived text solely to satisfy ReportRecord's non-null
        contract and is never read by the Oracle scoring path.
    """
    if any(not r.is_usable for r in records):
        raise ValueError("run_oracle_diagnostic_arm: received an excluded record")

    def _to_report_record(r: IuXrayCanonicalRecord) -> ReportRecord:
        result = build_target_report(r)
        if result.text is None:
            raise ValueError(
                f"study_id={r.study_id!r}: target-report policy produced no "
                f"text (branch={result.policy_branch!r}); cannot build an "
                f"Oracle diagnostic record without fabricating text"
            )
        return ReportRecord(
            image_paths=list(r.image_paths),
            finding=result.text,
            impression=result.text,
            patient_id=r.study_id,
            study_id=r.study_id,
            dataset=IU_XRAY_DATASET_TAG,
        )

    mapped_by_study_id = {r.study_id: _to_report_record(r) for r in records}
    corpus_records: Dict[QueryKey, object] = {
        (rec.dataset, rec.patient_id, rec.study_id): rec for rec in mapped_by_study_id.values()
    }
    query_records = {
        key: rec for key, rec in corpus_records.items() if key[2] in set(query_study_ids)
    }

    evaluator = OracleEvaluator(
        query_records, corpus_records, config=OracleConfig(exclude_self=True),
        chexbert_instance_scorer=lambda a, b: _deterministic_fake_score(a, b, "chexbert"),
        radgraph_instance_scorer=lambda a, b: _deterministic_fake_score(a, b, "radgraph"),
    )
    rows = [evaluator.build_row(key) for key in query_records]

    no_self_reference = all(
        row.winning_key is None or row.winning_key != row.query_key for row in rows
    )
    target_reference_isolated = all(
        row.winning_key is None or row.winning_key[2] != row.query_key[2] for row in rows
    )

    return {
        "arm": "ORACLE_PIPELINE_CHECK",
        "diagnostic_label": ORACLE_ARM_LABEL,
        "query_count": len(rows),
        "no_self_reference_in_winning_key": no_self_reference,
        "target_reference_isolated_from_winning_key": target_reference_isolated,
        "uses_real_chexbert_or_radgraph": False,
        "excluded_from_baseline_result_tables": True,
    }


# ---------------------------------------------------------------------------
# ARM 4: MOCK_GENERATION_SMOKE
# ---------------------------------------------------------------------------

MOCK_GENERATION_LABEL = "MOCK_GENERATION_SMOKE -- diagnostic only, no real LLM loaded, no medical-performance claim"


def run_mock_generation_smoke_arm(rag_rows: List[RAGDatasetRow]) -> dict:
    """Builds a prompt + calls MockGeneratorAdapter.generate() for every
    non-excluded row -- proves prompt construction, adapter interface
    compatibility, output collection, and evaluation-input construction
    all work, without loading any real model."""
    prompt_config = PromptBuilderConfig()
    builder = PromptBuilder(prompt_config)
    adapter = MockGeneratorAdapter()
    gen_config = GenerationConfig(seed=42)

    results: List[GeneratorResult] = []
    reference_rows = []
    prediction_rows = []
    for row in rag_rows:
        if row.excluded:
            continue
        prompt = builder.build(PromptMode.RAG_INFERENCE, retrieved_report=row.retrieved_report_text)
        result = adapter.generate(row.query_key, row.image_path, prompt.text, gen_config)
        results.append(result)
        reference_rows.append({"query_key": row.query_key, "finding": row.target_report_text})
        prediction_rows.append({
            "query_key": result.query_key,
            "retrieved_finding": [result.generated_report_text] if result.error is None else None,
            "error": result.error,
        })

    stable_ids = [r.query_key for r in results]
    results_rerun = [
        adapter.generate(
            row.query_key, row.image_path,
            builder.build(PromptMode.RAG_INFERENCE, retrieved_report=row.retrieved_report_text).text,
            gen_config,
        )
        for row in rag_rows if not row.excluded
    ]
    deterministic = [r.generated_report_text for r in results] == [r.generated_report_text for r in results_rerun]

    return {
        "arm": "MOCK_GENERATION_SMOKE",
        "diagnostic_label": MOCK_GENERATION_LABEL,
        "input_row_count": len(rag_rows),
        "output_count": len(results),
        "one_output_per_non_excluded_input": len(results) == sum(1 for r in rag_rows if not r.excluded),
        "stable_output_identifiers": len(set(stable_ids)) == len(stable_ids),
        "deterministic_across_repeated_runs": deterministic,
        "no_real_model_loaded": True,
        "reference_row_count": len(reference_rows),
        "prediction_row_count": len(prediction_rows),
    }
