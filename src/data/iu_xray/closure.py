"""Cell 46 -- Milestone 2.8 component readiness matrix, metric
readiness, and closure claims.

Responsibility: the fixed, explicit classification data this project's
Cell 46 reports against (never inferred ad hoc), plus a metric-
readiness smoke-test function that exercises this project's own real
mock metric registries (src.evaluation.retrieval_metrics.
build_mock_retrieval_metric_registry, src.evaluation.generation_metrics.
build_mock_metric_registry) and the real, model-free
paired_bootstrap_significance function -- proving interface
compatibility only, never producing or reporting a scientific result.
"""

from __future__ import annotations

from typing import Dict

from src.evaluation.generation_metrics import GenerationMetricsConfig, build_mock_metric_registry
from src.evaluation.retrieval_metrics import RetrievalMetricsConfig, build_mock_retrieval_metric_registry
from src.evaluation.significance import SignificanceConfig, paired_bootstrap_significance

from src.data.iu_xray.readiness import BLOCKED, DEFERRED, PARTIALLY_READY, READY, REMOVED

# ---------------------------------------------------------------------------
# Component readiness matrix
# ---------------------------------------------------------------------------

COMPONENT_READINESS_MATRIX: Dict[str, dict] = {
    "official_iu_xray_acquisition": {
        "status": READY,
        "rationale": "Cell 43: real download from openi.nlm.nih.gov, checksum-verified, real-data-validated in Colab.",
    },
    "archive_checksums": {
        "status": READY,
        "rationale": "Cell 43: SHA-256 recorded and reused; Cell 44/45/46 all re-verify against it before any real-data step.",
    },
    "safe_extraction": {
        "status": READY,
        "rationale": "Cell 43: path-traversal-safe tarfile extraction, real-data-verified (11,425 files extracted).",
    },
    "canonical_manifest": {
        "status": READY,
        "rationale": "Cell 43: 3,955 records canonicalized, 3,826 usable, real-data-verified; lightweight index + external full manifest, hash-verified on every later cell.",
    },
    "deterministic_split": {
        "status": READY,
        "rationale": "Cell 43: seed=42 80/10/10 split, real-data-verified counts (3060/382/384), no overlap.",
    },
    "loader": {
        "status": READY,
        "rationale": "Cell 44: IuXrayDataset, real-data-validated against the Drive-backed manifest.",
    },
    "schema_mapping": {
        "status": READY,
        "rationale": "Cell 44: IU X-Ray -> ReportRecord mapping, real-data-validated, explicit documented gaps (17-row mapping table).",
    },
    "target_report_policy": {
        "status": READY,
        "rationale": "Cell 45: 4-branch deterministic policy, real-data-validated; branch 4 disclosed as practically unreachable given Cell 43's own validity rule.",
    },
    "retrieval_query_corpus_adapter": {
        "status": READY,
        "rationale": "Cell 45: build_records/build_corpus/build_queries, real-data-validated, accepted by the real RetrieverTrainingDataset.",
    },
    "self_match_exclusion": {
        "status": READY,
        "rationale": "Cell 45/46: filter_self_matches + RAGDatasetBuilder's own independent enforcement, real-data-verified across all three splits.",
    },
    "generation_dataset_adapter": {
        "status": READY,
        "rationale": "Cell 45: build_generation_records + ranking builders, real-data-validated against the real RAGDatasetBuilder.",
    },
    "prompt_builder_compatibility": {
        "status": READY,
        "rationale": "Cell 45: PromptBuilder validated unmodified against IU-X-Ray-derived text, real-data-verified, no target-report leakage.",
    },
    "mock_retrieval_diagnostics": {
        "status": READY,
        "rationale": "Cell 46 Arm 2: deterministic MOCK_RETRIEVAL diagnostic, explicitly labeled, never presented as retrieval performance.",
    },
    "oracle_diagnostic_arm": {
        "status": READY,
        "rationale": "Cell 46 Arm 3: real OracleEvaluator with injected fake scorers, isolated, labeled NON-DEPLOYABLE DIAGNOSTIC.",
    },
    "mock_generator_smoke_path": {
        "status": READY,
        "rationale": "Cell 46 Arm 4: real MockGeneratorAdapter, no ML dependency, labeled diagnostic-only.",
    },
    "retrieval_metric_interfaces": {
        "status": PARTIALLY_READY,
        "rationale": "src.evaluation.retrieval_metrics interfaces (MRR/Recall/NDCG) structurally smoke-tested via the real build_mock_retrieval_metric_registry -- interface proven compatible, but no real retrieval has run, so no scientific measurement exists yet.",
    },
    "generation_metric_interfaces": {
        "status": PARTIALLY_READY,
        "rationale": "src.evaluation.generation_metrics interfaces (ROUGE-L/BLEU-4/BERTScore) structurally smoke-tested via build_mock_metric_registry -- same caveat as retrieval metrics.",
    },
    "clinical_metric_interfaces": {
        "status": PARTIALLY_READY,
        "rationale": "F1RadGraph/F1CheXbert interfaces structurally smoke-tested via the mock registry, but both were trained/validated on MIMIC-CXR-style text; applicability to IU X-Ray text is UNKNOWN and not empirically tested by this project (see metric readiness caveats).",
    },
    "bootstrap_significance_framework": {
        "status": READY,
        "rationale": "src.evaluation.significance.paired_bootstrap_significance has no model/embedding dependency at all -- fully exercisable today on any valid paired score arrays, IU-X-Ray-derived or otherwise; smoke-tested for real with synthetic scores.",
    },
    "replacement_retriever_selection": {
        "status": DEFERRED,
        "rationale": "Explicitly deferred per the Cell 42 scope migration -- not selected or implemented in Cells 42-46.",
    },
    "embedding_generation": {
        "status": DEFERRED,
        "rationale": "No embeddings computed anywhere in Cells 42-46; requires a selected retriever first.",
    },
    "vector_index": {
        "status": DEFERRED,
        "rationale": "No FAISS or other vector index built anywhere in Cells 42-46; requires real embeddings first.",
    },
    "vicuna_llava_access": {
        "status": DEFERRED,
        "rationale": "Per Cell 42 scope migration: Vicuna-7B-v1.5 remains the intended generator backbone but is deferred to a later generator stage; no download or HF token request has occurred.",
    },
    "llava_vision_tower": {
        "status": DEFERRED,
        "rationale": "Same generator-stage deferral as Vicuna/LLaVA access.",
    },
    "mm_projector": {
        "status": DEFERRED,
        "rationale": "Cell 42 scope migration: reassessed under whatever adapted architecture is eventually chosen; not resolved by Cells 42-46.",
    },
    "marvel_warm_start_checkpoint": {
        "status": REMOVED,
        "rationale": "Explicitly removed from required resources per the Cell 42 scope migration.",
    },
    "mimic_cxr_dataset": {
        "status": DEFERRED,
        "rationale": "Cell 42 scope migration: DEFERRED/NOT REQUIRED under the new IU X-Ray primary-dataset scope.",
    },
    "chexpert_dataset": {
        "status": DEFERRED,
        "rationale": "Cell 42 scope migration: DEFERRED/NOT REQUIRED, same reasoning as MIMIC-CXR.",
    },
    "real_generator_inference": {
        "status": DEFERRED,
        "rationale": "No real generator has been loaded or run anywhere in Cells 42-46 -- explicitly out of this cell's scope.",
    },
    "real_baseline_result_table": {
        "status": DEFERRED,
        "rationale": "No scientific baseline result exists yet -- requires a selected retriever, real embeddings, and real generation, none of which have occurred.",
    },
    "innovation_phase_entry_readiness": {
        "status": PARTIALLY_READY,
        "rationale": "The data/schema/retrieval/generation/prompt pipeline is CPU-ready end-to-end (Cells 44-46), but Innovation-phase work also needs a selected retriever and generator stage, both still DEFERRED.",
    },
}


def readiness_totals(matrix: Dict[str, dict] = None) -> Dict[str, int]:
    matrix = matrix if matrix is not None else COMPONENT_READINESS_MATRIX
    totals = {"READY": 0, "PARTIALLY_READY": 0, "DEFERRED": 0, "REMOVED": 0, "BLOCKED": 0}
    for entry in matrix.values():
        totals[entry["status"]] = totals.get(entry["status"], 0) + 1
    return totals


# ---------------------------------------------------------------------------
# Metric readiness (real structural smoke test, mock inputs only)
# ---------------------------------------------------------------------------

CLINICAL_METRIC_CAVEATS = (
    "F1RadGraph and F1CheXbert were both trained/validated on MIMIC-CXR-"
    "style Findings/Impression text (Cell 42's audit); their behavior on "
    "IU X-Ray's four-section, generally shorter reports is UNKNOWN and "
    "not empirically tested by this project -- classified PARTIALLY_READY, "
    "never READY, until that is done with real generated text.",
    "IU X-Ray provides no confirmed official structured image labels "
    "(Cell 42 audit) -- F1CheXbert's dataset-level (not instance-level) "
    "variant, which compares against ground-truth structured labels, has "
    "no confirmed IU-X-Ray-native label source to compare against.",
)


def run_metric_readiness_smoke() -> dict:
    """Real, in-process smoke test of the metric machinery's own
    interfaces using each subsystem's own already-implemented mock
    registry / model-free function -- never a real metric library call,
    never a value presented as a baseline result."""
    retrieval_registry = build_mock_retrieval_metric_registry(RetrievalMetricsConfig(k_values=(1, 2)))
    positives = {("iu-xray", "CXR1", "CXR1"): [("iu-xray", "CXR2", "CXR2")]}
    ranked = {("iu-xray", "CXR1", "CXR1"): [("iu-xray", "CXR3", "CXR3"), ("iu-xray", "CXR2", "CXR2")]}
    retrieval_smoke = {
        name: metric.compute(positives, ranked).scores_at_k
        for name, metric in retrieval_registry.items()
    }

    generation_registry = build_mock_metric_registry()
    hyps = ["a mock generated report"]
    refs = ["a mock target report"]
    generation_smoke = {
        name: metric.compute(hyps, refs) for name, metric in generation_registry.items()
    }

    significance_smoke = paired_bootstrap_significance(
        [0.5, 0.6, 0.7], [0.4, 0.5, 0.6],
        metric_name="mock_metric", system_a_name="system_a", system_b_name="system_b",
        config=SignificanceConfig(num_bootstrap_samples=100),
    )

    return {
        "retrieval_metrics_smoke_ok": all(retrieval_smoke.values()),
        "retrieval_metrics_tested": sorted(retrieval_registry),
        "generation_metrics_smoke_ok": all(bool(v) for v in generation_smoke.values()),
        "generation_metrics_tested": sorted(generation_registry),
        "significance_smoke_ok": significance_smoke.metric_name == "mock_metric",
        "clinical_metric_caveats": list(CLINICAL_METRIC_CAVEATS),
        "metric_status": {
            "recall_at_k": PARTIALLY_READY, "ndcg_at_k": PARTIALLY_READY, "mrr": PARTIALLY_READY,
            "rouge_l": PARTIALLY_READY, "bleu4": PARTIALLY_READY, "bert_score": PARTIALLY_READY,
            "f1_radgraph": PARTIALLY_READY, "f1_chexbert": PARTIALLY_READY,
            "bootstrap_significance": READY,
        },
        "values_are_mock_never_scientific": True,
    }


# ---------------------------------------------------------------------------
# Milestone closure claims
# ---------------------------------------------------------------------------

MUST_NOT_CLAIM = (
    "exact FactMM-RAG reproduction",
    "direct comparability to MIMIC-CXR/CheXpert paper scores",
    "real retriever execution",
    "real Vicuna/LLaVA generation",
    "scientific performance from mock outputs",
    "completed baseline result table",
)

MAY_CLAIM = (
    "architecture-faithful adaptation inspired by FactMM-RAG",
    "official public IU X-Ray acquisition and deterministic preprocessing",
    "validated loader and schema adaptation",
    "validated retrieval/generation/prompt interfaces",
    "CPU-only end-to-end readiness",
    "readiness to implement the selected retriever and generator",
    "controlled future baseline-versus-innovation design",
)

CELL_AUDIT = {
    "cell_40": "Resource discovery -- 10-resource inventory (2 datasets, 7 checkpoints, 1 code dependency) under the (now-superseded) MIMIC-CXR/CheXpert/MARVEL scope.",
    "cell_41": "Acquisition planning -- Category A/B/C classification of Cell 40's inventory.",
    "cell_42": "IU X-Ray scope migration -- MIMIC-CXR/CheXpert DEFERRED, MARVEL REMOVED, Vicuna DEFERRED to generator stage; new scientific claim adopted; IU X-Ray audited (7,470 images / 3,955 reports).",
    "cell_43": "Official IU X-Ray acquisition (openi.nlm.nih.gov, checksum-verified), safe extraction, canonicalization (3,826 usable / 129 excluded), deterministic 80/10/10 split (3060/382/384).",
    "cell_44": "IU X-Ray loader (src/data/iu_xray/) + schema mapping to ReportRecord; real-data-validated against the Drive-backed manifest.",
    "cell_45": "Retrieval/generation/prompt pipeline integration against the real, unmodified src.baseline interfaces; target-report policy; self-match exclusion; real-data-validated.",
    "cell_46": "CPU-only end-to-end readiness dry run (4 diagnostic arms), component readiness matrix, metric readiness smoke test, Milestone 2.8 closure.",
}

SUPERSEDED_ASSUMPTIONS = (
    "MIMIC-CXR/CheXpert as the primary executable benchmark (Cell 42).",
    "MARVEL as the required retriever warm-start checkpoint (Cell 42).",
    "Exact numerical reproduction of the FactMM-RAG paper's MIMIC-CXR/CheXpert results (Cell 42).",
)

REMAINING_GPU_MODEL_NEEDS = (
    "Selecting and implementing a replacement retriever (embedding model + FAISS index).",
    "Acquiring and loading Vicuna-7B-v1.5 (or another selected generator backbone).",
    "Real generator inference (LLaVA-style multimodal generation).",
    "Empirical validation of F1RadGraph/F1CheXbert behavior on real IU X-Ray-derived generated text.",
    "Building the first real baseline-vs-innovation result table.",
)
