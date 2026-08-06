# Milestone 3.0 Cell 47 -- Innovation Architecture + Experimental Contract Report

## Detected REPO_ROOT
`/home/user/factmm-rag-enhanced`

## Branch + HEAD
branch=`reproduction/milestone-2.7`, HEAD=`92e63bfe299b1a381cf947153fe42979b87a99f2` ("Add Milestone 2.8 real Colab validation artifacts")

## Existing innovation scaffolding audit

{
  "src/innovation/adaptive_retrieval/adaptive_k.py": {
    "classification": "EXTEND",
    "rationale": "Pre-existing stub (docstring only, 'No implementation yet') at exactly I1's intended location. Extended with AdaptiveKConfig (validated) and AdaptiveKDecision -- the contract Section 3 asks for. The selection algorithm itself is deferred to Cell 48 per Section 10."
  },
  "src/innovation/factual_reranking/reranker.py": {
    "classification": "EXTEND",
    "rationale": "Pre-existing stub at exactly I2's intended location. Extended with RerankConfig (validated), RerankResult, a pluggable CompatibilityScorer interface, and a deterministic neutral fallback scorer -- no clinical model invented, per Section 4's explicit signal audit requirement."
  },
  "src/innovation/multi_report_fusion/fusion.py": {
    "classification": "EXTEND",
    "rationale": "Pre-existing stub at exactly I3's fusion half. Extended with FusionConfig (validated), FusionDecision, and the EvidenceItem/EvidenceContext structured-evidence contract (dedup + budget enforcement validated at construction time)."
  },
  "src/innovation/evaluation/innovation_metrics.py": {
    "classification": "EXTEND",
    "rationale": "Pre-existing stub. Extended with the Section 8 metrics contract, reusing (never reimplementing) the real metric names from src.evaluation.retrieval_metrics/generation_metrics's own registries, plus the innovation-specific diagnostic-name registry."
  },
  "src/innovation/confidence_gating/gate.py": {
    "classification": "UNUSED",
    "rationale": "Pre-existing stub for the ORIGINAL, broader Innovation D scope (docs/innovation_proposals.md: 'a gate deciding whether to retrieve at all'). Milestone 3.0's I3 ('Confidence-Gated Multi-Report Fusion') is a narrower, different decision (should_fuse over already-retrieved evidence, not retrieve-vs-not-retrieve) -- fully expressed instead as FusionDecision in multi_report_fusion/fusion.py, per that module's own docstring note. gate.py is left completely untouched (not one of the three required innovations for this cell) rather than rewritten or deleted, per Section 1's explicit instruction."
  },
  "src/innovation/uncertainty/detector.py": {
    "classification": "UNUSED",
    "rationale": "Pre-existing stub for the ORIGINAL Innovation E (uncertainty/ contradiction detection), which is not one of Milestone 3.0's exactly-three required innovations (I1/I2/I3). Left completely untouched, not deleted or rewritten."
  },
  "docs/innovation_proposals.md": {
    "classification": "REUSED_AS_HISTORICAL_CONTEXT",
    "rationale": "Original A-F innovation proposal document, read for context. Not modified (Cell 47's Document Update Policy: historical docs are not rewritten). I1/I2/I3 supersede/narrow A/B/(C+D) respectively for Milestone 3.0's actual scope; this supersession is recorded only in this cell's own new artifacts, never by editing the historical document."
  },
  "docs/phase_3_innovation_contract.md": {
    "classification": "REUSED_AS_HISTORICAL_CONTEXT",
    "rationale": "Earlier, more elaborate 6-innovation design contract (pre-dates the Milestone 2.8 IU X-Ray pivot). Its Section 13 baseline/innovation import rule (one-directional: innovation may import baseline, never the reverse) is the rule this cell actually follows -- see design caveats in cell47_report.md for why, over the narrower claim in src/innovation/__init__.py's own docstring."
  }
}

## Final architecture

{
  "pipeline_stages": [
    "query",
    "initial_retrieval",
    "adaptive_retrieval_dynamic_k",
    "factual_clinical_reranking",
    "confidence_gated_multi_report_fusion",
    "evidence_context",
    "prompt_builder",
    "generator",
    "generated_report"
  ],
  "innovation_stages": {
    "I1_adaptive_retrieval_dynamic_k": "adaptive_retrieval_dynamic_k",
    "I2_factual_clinical_reranking": "factual_clinical_reranking",
    "I3_confidence_gated_multi_report_fusion": "confidence_gated_multi_report_fusion"
  },
  "stage_gate_flags": {
    "adaptive_retrieval_dynamic_k": "adaptive_retrieval_enabled",
    "factual_clinical_reranking": "factual_reranking_enabled",
    "confidence_gated_multi_report_fusion": "confidence_gated_fusion_enabled"
  },
  "baseline_stages_when_all_disabled": [
    "query",
    "initial_retrieval",
    "evidence_context",
    "prompt_builder",
    "generator",
    "generated_report"
  ],
  "independently_switchable": true,
  "note": "Design/contract only -- no retrieval, reranking, or fusion algorithm is implemented by this module. See src/innovation/adaptive_retrieval/adaptive_k.py, src/innovation/factual_reranking/reranker.py, and src/innovation/multi_report_fusion/fusion.py for the per-innovation config/result contracts (Cells 48-50 implement the actual decision algorithms).",
  "inference_context_fields": [
    "query_key",
    "candidate_keys",
    "candidate_scores",
    "candidate_texts"
  ]
}

## E0-E7 experiment matrix

{
  "arms": [
    {
      "arm_id": "E0",
      "description": "baseline (no innovation enabled)",
      "config": {
        "adaptive_retrieval_enabled": false,
        "factual_reranking_enabled": false,
        "confidence_gated_fusion_enabled": false,
        "adaptive_k": {
          "min_k": 1,
          "max_k": 5,
          "default_k": 1,
          "confidence_threshold": 0.5,
          "margin_threshold": 0.0
        },
        "reranking": {
          "alpha": 1.0,
          "beta": 0.0,
          "gamma": 0.0
        },
        "fusion": {
          "max_evidence_count": 3,
          "context_budget_chars": 4000,
          "confidence_threshold": 0.5,
          "disagreement_threshold": 0.3
        },
        "random_seed": 42
      }
    },
    {
      "arm_id": "E1",
      "description": "Adaptive-K only",
      "config": {
        "adaptive_retrieval_enabled": true,
        "factual_reranking_enabled": false,
        "confidence_gated_fusion_enabled": false,
        "adaptive_k": {
          "min_k": 1,
          "max_k": 5,
          "default_k": 1,
          "confidence_threshold": 0.5,
          "margin_threshold": 0.0
        },
        "reranking": {
          "alpha": 1.0,
          "beta": 0.0,
          "gamma": 0.0
        },
        "fusion": {
          "max_evidence_count": 3,
          "context_budget_chars": 4000,
          "confidence_threshold": 0.5,
          "disagreement_threshold": 0.3
        },
        "random_seed": 42
      }
    },
    {
      "arm_id": "E2",
      "description": "Reranking only",
      "config": {
        "adaptive_retrieval_enabled": false,
        "factual_reranking_enabled": true,
        "confidence_gated_fusion_enabled": false,
        "adaptive_k": {
          "min_k": 1,
          "max_k": 5,
          "default_k": 1,
          "confidence_threshold": 0.5,
          "margin_threshold": 0.0
        },
        "reranking": {
          "alpha": 1.0,
          "beta": 0.0,
          "gamma": 0.0
        },
        "fusion": {
          "max_evidence_count": 3,
          "context_budget_chars": 4000,
          "confidence_threshold": 0.5,
          "disagreement_threshold": 0.3
        },
        "random_seed": 42
      }
    },
    {
      "arm_id": "E3",
      "description": "Confidence-Gated Fusion only",
      "config": {
        "adaptive_retrieval_enabled": false,
        "factual_reranking_enabled": false,
        "confidence_gated_fusion_enabled": true,
        "adaptive_k": {
          "min_k": 1,
          "max_k": 5,
          "default_k": 1,
          "confidence_threshold": 0.5,
          "margin_threshold": 0.0
        },
        "reranking": {
          "alpha": 1.0,
          "beta": 0.0,
          "gamma": 0.0
        },
        "fusion": {
          "max_evidence_count": 3,
          "context_budget_chars": 4000,
          "confidence_threshold": 0.5,
          "disagreement_threshold": 0.3
        },
        "random_seed": 42
      }
    },
    {
      "arm_id": "E4",
      "description": "Adaptive-K + Reranking",
      "config": {
        "adaptive_retrieval_enabled": true,
        "factual_reranking_enabled": true,
        "confidence_gated_fusion_enabled": false,
        "adaptive_k": {
          "min_k": 1,
          "max_k": 5,
          "default_k": 1,
          "confidence_threshold": 0.5,
          "margin_threshold": 0.0
        },
        "reranking": {
          "alpha": 1.0,
          "beta": 0.0,
          "gamma": 0.0
        },
        "fusion": {
          "max_evidence_count": 3,
          "context_budget_chars": 4000,
          "confidence_threshold": 0.5,
          "disagreement_threshold": 0.3
        },
        "random_seed": 42
      }
    },
    {
      "arm_id": "E5",
      "description": "Adaptive-K + Fusion",
      "config": {
        "adaptive_retrieval_enabled": true,
        "factual_reranking_enabled": false,
        "confidence_gated_fusion_enabled": true,
        "adaptive_k": {
          "min_k": 1,
          "max_k": 5,
          "default_k": 1,
          "confidence_threshold": 0.5,
          "margin_threshold": 0.0
        },
        "reranking": {
          "alpha": 1.0,
          "beta": 0.0,
          "gamma": 0.0
        },
        "fusion": {
          "max_evidence_count": 3,
          "context_budget_chars": 4000,
          "confidence_threshold": 0.5,
          "disagreement_threshold": 0.3
        },
        "random_seed": 42
      }
    },
    {
      "arm_id": "E6",
      "description": "Reranking + Fusion",
      "config": {
        "adaptive_retrieval_enabled": false,
        "factual_reranking_enabled": true,
        "confidence_gated_fusion_enabled": true,
        "adaptive_k": {
          "min_k": 1,
          "max_k": 5,
          "default_k": 1,
          "confidence_threshold": 0.5,
          "margin_threshold": 0.0
        },
        "reranking": {
          "alpha": 1.0,
          "beta": 0.0,
          "gamma": 0.0
        },
        "fusion": {
          "max_evidence_count": 3,
          "context_budget_chars": 4000,
          "confidence_threshold": 0.5,
          "disagreement_threshold": 0.3
        },
        "random_seed": 42
      }
    },
    {
      "arm_id": "E7",
      "description": "Adaptive-K + Reranking + Fusion",
      "config": {
        "adaptive_retrieval_enabled": true,
        "factual_reranking_enabled": true,
        "confidence_gated_fusion_enabled": true,
        "adaptive_k": {
          "min_k": 1,
          "max_k": 5,
          "default_k": 1,
          "confidence_threshold": 0.5,
          "margin_threshold": 0.0
        },
        "reranking": {
          "alpha": 1.0,
          "beta": 0.0,
          "gamma": 0.0
        },
        "fusion": {
          "max_evidence_count": 3,
          "context_budget_chars": 4000,
          "confidence_threshold": 0.5,
          "disagreement_threshold": 0.3
        },
        "random_seed": 42
      }
    }
  ],
  "required_statistical_comparisons": [
    [
      "E0",
      "E1"
    ],
    [
      "E0",
      "E2"
    ],
    [
      "E0",
      "E3"
    ],
    [
      "E0",
      "E7"
    ]
  ],
  "random_seed": 42,
  "identical_across_arms": [
    "dataset splits",
    "corpus",
    "generator (once real generation is introduced)",
    "prompt template (unless an innovation strictly requires evidence-formatting changes)",
    "random seed",
    "evaluation metrics"
  ]
}

## Metrics contract

{
  "metrics": {
    "retrieval": [
      "mrr",
      "recall",
      "ndcg"
    ],
    "generation": [
      "rouge_l",
      "bleu4",
      "bert_score"
    ],
    "clinical": [
      "f1radgraph",
      "f1chexbert"
    ]
  },
  "innovation_diagnostics": {
    "adaptive_k": [
      "mean_selected_k",
      "k_distribution",
      "evidence_reduction_vs_fixed_k"
    ],
    "reranking": [
      "rank_change_rate",
      "mean_rank_displacement",
      "top1_replacement_rate"
    ],
    "fusion": [
      "fusion_activation_rate",
      "mean_evidence_count",
      "evidence_context_reduction"
    ],
    "efficiency": [
      "latency_per_query",
      "evidence_count",
      "approximate_context_length"
    ]
  },
  "note": "Contract only -- no scientific performance is reported by Cell 47."
}

## Design caveats

- The three baseline/innovation-separation statements in this repo
  disagree in scope: `src/innovation/__init__.py`'s own docstring claims
  innovation code "never imports from src/baseline"; the more detailed
  `docs/phase_3_innovation_contract.md` Section 13 states a
  one-directional rule (innovation MAY import baseline; baseline must
  never import innovation). This cell follows the Section 13 version,
  because the real, load-bearing Milestone 2.8 interfaces (QueryKey,
  RAGDatasetRow, etc.) live under `src/baseline/**` -- `src/retrieval/`
  and `src/generation/` (the packages the narrower docstring named
  instead) are themselves unimplemented stubs. This is disclosed, not
  silently decided.
- `src/innovation/confidence_gating/gate.py` (original Innovation D:
  retrieve-or-not gating) and `src/innovation/uncertainty/detector.py`
  (original Innovation E) are classified UNUSED for Milestone 3.0's
  exactly-three-innovation scope and left completely untouched, per
  Section 1's explicit "do not rewrite or delete" instruction.
- I3's confidence gate (`FusionDecision.should_fuse`) intentionally
  narrows/differs from `confidence_gating/gate.py`'s original broader
  "retrieve at all" framing -- documented directly in
  `multi_report_fusion/fusion.py`'s module docstring.
- No Adaptive-K/reranking/fusion DECISION algorithm is implemented in
  this cell (Section 10's explicit prohibition) -- only the
  configuration/result contracts. The shared confidence formula IS
  implemented (Section 6 explicitly asks for a working, tested
  confidence representation), and is fully documented in
  `src/innovation/shared/confidence.py`.

## Test results

Cell 47 focused tests: 46 passed in 0.07s
All existing IU X-Ray tests: PASS
Full suite: 1090 passed in 4.78s
Baseline test count recorded before Cell 47 changes (this session): 1044

## Working tree

```
 M src/innovation/adaptive_retrieval/adaptive_k.py
 M src/innovation/evaluation/innovation_metrics.py
 M src/innovation/factual_reranking/reranker.py
 M src/innovation/multi_report_fusion/fusion.py
?? src/innovation/architecture.py
?? src/innovation/experiment_contract.py
?? src/innovation/shared/
?? tests/unit/test_innovation_architecture.py
?? tests/unit/test_innovation_confidence.py
?? tests/unit/test_innovation_config.py
?? tests/unit/test_innovation_experiment_matrix.py
?? tests/unit/test_innovation_metrics_contract.py
?? tests/unit/test_innovation_reranker_fusion_contracts.py

```

## Verdict

CELL 47: PASS
