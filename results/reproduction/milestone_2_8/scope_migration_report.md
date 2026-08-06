# Milestone 2.8 Cell 42 -- Dataset/Resource Scope Migration Report

Planning/compatibility-validation only. No dataset or checkpoint was downloaded. No existing contract or prior artifact was edited -- see the migration-impact list below for what will need editing later, under separate, explicit authorization.

## 1. IU X-Ray / Open-I dataset audit

See `iu_xray_dataset_audit.json` for the full field-by-field, source-labeled audit. Summary:

- Image count: 7470 (WEBFETCH_PAGE_READ)
- Report count: 3955 (WEBFETCH_PAGE_READ)
- Report sections: Comparison, Indication, Findings, Impression (WEBFETCH_PAGE_READ)
- Official NLM primary source pages (openi.nlm.nih.gov homepage and /faq) both returned HTTP 403 on every WebFetch attempt this session and could not be read directly. Community mirror pages (Kaggle, Academic Torrents, a Hugging Face mirror) also returned HTTP 403. One community page (GitHub, openmedlab/Awesome-Medical-Dataset) was successfully fetched and read this session and is this audit's strongest single source.
- Fields marked UNKNOWN: ['registration_requirement_official_nlm']
- Exact official license text, exact archive sizes for the official NLM tgz files, and the official-NLM registration requirement all remain UNKNOWN -- not confirmed by any source read this session, and not invented.

## 2. New scientific claim

**Removed:**

> This project no longer claims exact reproduction of the FactMM-RAG paper's reported results, and no longer claims direct numerical comparability with the paper's MIMIC-CXR/CheXpert results tables (the ones already extracted, PAPER_EXPLICIT-labeled, in docs/milestone_2_7_reproduction_validation_contract.md).

**Now claimed instead:**

- An architecture-faithful baseline adaptation inspired by FactMM-RAG (same retrieval-augmented generation architecture family; same evaluation-metric machinery already built in src/evaluation/), run on a different, lightweight public benchmark.
- Evaluation on IU X-Ray / Open-I, a lightweight public radiology report-generation benchmark, in place of MIMIC-CXR/CheXpert.
- A controlled baseline-vs-innovation comparison on the identical IU X-Ray split (adapted baseline vs. Innovation 1 vs. cumulative innovations), which remains scientifically meaningful even though it is no longer comparable to the paper's own MIMIC-CXR/CheXpert numbers.

## 3. Resource reclassification

| Resource | New status |
|---|---|
| chexpert | DEFERRED_NOT_REQUIRED |
| clip_vit_b32_retriever | REASSESS_UNDER_ADAPTED_ARCHITECTURE_NO_AUTO_DOWNLOAD |
| llava_base_vicuna | DEFERRED_TO_GENERATOR_STAGE |
| llava_fork_haotian_liu | RETAIN_ONLY_IF_REQUIRED_BY_GENERATOR_ARCHITECTURE |
| llava_generator_vision_tower_clip_l14 | RETAIN_ONLY_IF_REQUIRED_BY_GENERATOR_ARCHITECTURE |
| llava_mm_projector_checkpoint | REASSESS_UNDER_ADAPTED_ARCHITECTURE_NO_AUTO_DOWNLOAD |
| marvel_warm_start_checkpoint | REMOVED_FROM_REQUIRED_RESOURCES |
| mimic_cxr | DEFERRED_NOT_REQUIRED |
| t5_ance | REASSESS_UNDER_ADAPTED_ARCHITECTURE_NO_AUTO_DOWNLOAD |
| unlabeled_rag_section_checkpoint | REASSESS_UNDER_ADAPTED_ARCHITECTURE_NO_AUTO_DOWNLOAD |

Full rationale per resource is in `scope_migration.json`.

## 4. Migration-impact list (documents/artifacts needing later reinterpretation)

Not edited in this cell -- listed only, per Task 4's explicit instruction.

1. **docs/milestone_2_7_reproduction_validation_contract.md** -- Its reproduction targets (paper Table 2 values, MIMIC-CXR/CheXpert retrieval-only + backbone-variation settings) are all PAPER_EXPLICIT figures for a dataset this project no longer executes against as primary. Will need a note that these targets are retained only as historical paper-reference figures, not reproduction goals under the new scope.
2. **docs/milestone_2_8_resource_acquisition_contract.md** -- Per-resource acquisition instructions for MIMIC-CXR, CheXpert, and the MARVEL warm-start checkpoint are superseded by this migration's reclassification (DEFERRED_NOT_REQUIRED / REMOVED_FROM_REQUIRED_RESOURCES).
3. **results/reproduction/milestone_2_7/ (Cell 37/38/39 outputs: paper_values.json, reproduction_config.json, dataset_requirements.json, checkpoint_requirements.json, audit_summary.md, runtime_validation.json, dataset_validation.json, checkpoint_validation.json, readiness_report.md, blocker_status.json, cell39_readiness.md)** -- All built around the now-superseded MIMIC-CXR/CheXpert/MARVEL primary-dataset assumption; remain accurate as historical records of the paper-reproduction investigation but no longer describe this project's current execution plan.
4. **results/reproduction/milestone_2_8/ (Cell 40/41 outputs: resource_inventory.json, resource_matrix.json, resource_discovery_report.md, download_plan.md, resource_categories.json, acquisition_plan.json, manual_steps.md, cell41_summary.md)** -- Cell 40's 10-resource inventory and Cell 41's Category A/B/C classification both need reconciling against this cell's new reclassification -- most notably marvel_warm_start_checkpoint, which Cell 41 placed in Category A (public/automatic) but this migration removes from required resources entirely.
5. **Phase 3 innovation contract (not yet located/read this session -- named per this cell's Task 4 requirement; its exact path was not confirmed this session)** -- Any innovation description that assumes MIMIC-CXR/CheXpert scale or the MARVEL warm-start checkpoint as a starting point needs updating for the smaller IU X-Ray scale and the not-yet-chosen replacement retriever.
6. **Innovation roadmap (not yet located/read this session -- named per this cell's Task 4 requirement)** -- Same reasoning as the Phase 3 innovation contract -- any dataset- or retriever-specific milestones need reinterpretation under the new scope.
7. **Comparison-table wording (wherever paper-vs-reproduction results tables are presented, e.g. in future milestone reports)** -- Must be reworded to never present IU X-Ray scores next to the paper's MIMIC-CXR/CheXpert scores as if directly comparable (Task 5's explicit prohibition).
8. **Acceptance criteria (wherever this project's own 'reproduction succeeded' criteria are defined)** -- Criteria phrased in terms of matching paper MIMIC-CXR/CheXpert numbers no longer apply; need reframing around the new claim (architecture-faithful adaptation + controlled baseline-vs-innovation comparison on IU X-Ray).
9. **Branch/tag strategy** -- The branch name reproduction/milestone-2.7 and any future tags describing this work as a 'reproduction' may need reconsideration given the new claim explicitly disclaims exact reproduction; a naming decision is deferred, not made here.

## 5. Fair experiment design

- Adapted baseline: the architecture-faithful FactMM-RAG-inspired baseline, run on IU X-Ray with a not-yet-chosen replacement retriever and the deferred Vicuna-7B-v1.5 generator.
- Innovation 1: run on the identical IU X-Ray split as the adapted baseline.
- Cumulative innovations: run on the identical IU X-Ray split as the adapted baseline and Innovation 1.
- Held identical across baseline and every innovation run, except the explicitly ablated component: generator, retriever, preprocessing, prompts, decoding parameters, random seeds, and evaluation metrics.
- Explicitly forbidden: presenting IU X-Ray scores as if numerically comparable to the FactMM-RAG paper's MIMIC-CXR/CheXpert scores. Any comparison table must visually and textually separate 'paper reference values (MIMIC-CXR/CheXpert, not reproduced)' from 'this project's own IU X-Ray baseline-vs-innovation values'.

## 6. Metric validity on IU X-Ray

| Metric | Valid on IU X-Ray | Caveat |
|---|---|---|
| ROUGE-L | True | Reference-free of clinical-schema assumptions; directly applicable. |
| BLEU-4 | True | Same as ROUGE-L; short IU X-Ray reports may yield noisier n-gram statistics than MIMIC-CXR's larger corpus, given the ~32x smaller report count (3,955 vs. this project's recorded 125,417+991+1,624 MIMIC-CXR pairs). |
| BERTScore | True | Directly applicable; model choice (e.g. a clinical BERT variant) should stay identical between baseline and innovation runs, per the fair-comparison requirement above. |
| F1RadGraph | REASSESS | RadGraph was trained/validated on MIMIC-CXR-style Findings/Impression text; its behavior on IU X-Ray's differently-structured four-section reports (and its much shorter report text on average) is UNKNOWN and not tested this session -- flagged as needing empirical validation before being treated as a load-bearing metric. |
| F1CheXbert | REASSESS | CheXbert was trained on MIMIC-CXR/CheXpert-style reports and a fixed 14-observation label schema; its applicability to IU X-Ray text (different institution, different report-writing conventions, no confirmed official structured labels per this cell's audit) is UNKNOWN and not tested this session. |
| Recall@K | True | Retrieval-quality metric, dataset-structure-agnostic; directly applicable once a replacement retriever and its candidate corpus are defined (not done in this cell). |
| NDCG@K | True | Same as Recall@K. |
| MRR | True | Same as Recall@K. |
| bootstrap_significance | True | Directly applicable, though IU X-Ray's much smaller test-split size (test: 1,496 images / 790 reports per the community split found this session, vs. MIMIC-CXR's 1,624-pair test set recorded in Cell 37) means wider confidence intervals and lower statistical power should be expected and reported, not treated as equivalent to the paper's own MIMIC-CXR-scale significance results. |

## Explicit non-actions this cell

- IU X-Ray was not downloaded.
- Vicuna-7B-v1.5 was not downloaded or loaded; no HF token was requested or stored.
- MARVEL was not downloaded or used.
- No replacement retriever was chosen or implemented.
- No file under src/ or tests/ was modified.
- No existing contract or prior Milestone 2.7/2.8 artifact was edited -- only listed above as needing later reinterpretation.
- No inference or training was started. Cell 43 was not started.
