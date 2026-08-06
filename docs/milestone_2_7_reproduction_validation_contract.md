# Milestone 2.7 — Baseline Reproduction Validation: Design Contract

> **DESIGN CONTRACT — NOT EXECUTABLE CODE.** Every schema, table, and
> plan below is a specification, not implemented source or an executed
> experiment. No file under `src/baseline/`, `src/evaluation/`,
> `src/common/`, `src/data/`, `src/retrieval/`, `src/generation/`,
> `src/innovation/**`, or `tests/` is created or modified by this
> document. No Cell 37 is added. No experiment has run. **No claim of
> successful reproduction is made anywhere in this document** — every
> comparison table below is reserved and empty pending real benchmark
> results on real, credentialed data. This contract is produced through
> a fresh audit — paper PDF re-extracted via `pdftotext -layout` this
> session, official repository re-checked, `baseline-v1.0` inspected —
> before any reproduction tooling is written.

## Table of Contents

1. [Status and Scope](#1-status-and-scope)
2. [Reproduction Objective](#2-reproduction-objective)
3. [Source Hierarchy](#3-source-hierarchy)
4. [Exact Reproduction Matrix](#4-exact-reproduction-matrix)
5. [Required Datasets and Credential Constraints](#5-required-datasets-and-credential-constraints)
6. [Dataset Placement and Expected File Manifests](#6-dataset-placement-and-expected-file-manifests)
7. [Required Checkpoints and Revisions](#7-required-checkpoints-and-revisions)
8. [Environment and Dependency Pins](#8-environment-and-dependency-pins)
9. [Baseline Run Configuration](#9-baseline-run-configuration)
10. [Retrieval Evaluation Plan](#10-retrieval-evaluation-plan)
11. [Generation Evaluation Plan](#11-generation-evaluation-plan)
12. [Paper-Value Extraction Table](#12-paper-value-extraction-table)
13. [Our-Result Table Template](#13-our-result-table-template)
14. [Absolute and Relative Delta Formulas](#14-absolute-and-relative-delta-formulas)
15. [Reproduction Acceptance Criteria](#15-reproduction-acceptance-criteria)
16. [Statistical Comparison Plan](#16-statistical-comparison-plan)
17. [Seed/Repeat Policy](#17-seedrepeat-policy)
18. [Determinism Policy](#18-determinism-policy)
19. [Resource Estimation](#19-resource-estimation)
20. [Small-Scale Dry-Run Plan](#20-small-scale-dry-run-plan)
21. [Full-Scale Benchmark Plan](#21-full-scale-benchmark-plan)
22. [Failure Taxonomy](#22-failure-taxonomy)
23. [Discrepancy-Analysis Protocol](#23-discrepancy-analysis-protocol)
24. [Artifact/Output Schemas](#24-artifactoutput-schemas)
25. [Colab Execution Plan](#25-colab-execution-plan)
26. [Commit Plan](#26-commit-plan)
27. [Known Risks](#27-known-risks)
28. [Open Questions](#28-open-questions)
29. [Required Comparison Tables (Reserved)](#29-required-comparison-tables-reserved)
30. [Branch Strategy Recommendation](#30-branch-strategy-recommendation)

---

## 1. Status and Scope

Milestone 2.7 begins with Phase 3 Innovation implementation explicitly
**paused** and validates `baseline-v1.0` — the exact tagged, merged
commit (`main` merge of PR #1, Milestones 2.1–2.6, `CELL 36: PASS`,
921/921 tests) — against the original FactMM-RAG paper and official
repository. This is the audit-first design contract; no reproduction
tooling exists yet. **No baseline architecture, evaluation code, or
test is modified anywhere in this milestone.** Any bug this audit
surfaces is documented and fixed only via a separate, dedicated patch
(§30), never folded silently into reproduction scripts.

## 2. Reproduction Objective

Answer, with evidence rather than assertion:

1. **Does our baseline reproduce the paper's reported results?**
2. **If not, how large is the gap?**
3. **Which differences come from environment/version/data/checkpoint
   discrepancies** (vs. a genuine implementation defect)?
4. **Is the baseline sufficiently faithful to serve as the control for
   Phase 3 Innovations?**

Explicit non-goal, per instruction: passing this project's own unit and
integration test suite (921/921, Milestone 2.6) is **not** reproduction
evidence — those tests prove software correctness on synthetic data,
never agreement with the paper's real, credentialed-data benchmark
numbers. Reproduction is only established once real benchmark results
exist and are compared (§15).

## 3. Source Hierarchy

Every value in this contract is labeled with exactly one of:

| Label | Meaning | Precedence |
|---|---|---|
| `PAPER` | Directly stated in `paper/factmm_rag.pdf` (re-extracted this session via `pdftotext -layout`) | Primary reproduction target — this is what Milestone 2.7 exists to reproduce |
| `OFFICIAL_REPOSITORY` | Verified by reading `reference/original_repository/FactMM-RAG/` directly | Used to resolve implementation detail the paper omits; when it **disagrees** with `PAPER` (top_k, temperature — see §4), both are run and reported, never silently merged |
| `VERIFIED_BASELINE` | Our own `baseline-v1.0` output, from an actually-executed run — never a code-reading claim | The subject of comparison against `PAPER`/`OFFICIAL_REPOSITORY`; empty (§13) until real experiments run |
| `REASONABLE_INFERENCE` | Inferred with explicit reasoning, not directly stated by any source | Used only where no direct source exists, always flagged as inference, never presented as fact |
| `UNKNOWN` | Genuinely unknown even with paper + official repository both available | Never silently filled with a guess (per explicit instruction) — listed in §28 |

No value in §12 (paper extraction) or §4 (reproduction matrix) is
recorded without one of these five labels attached.

## 4. Exact Reproduction Matrix

Extends `docs/reproduction_matrix.md` with Milestone 2.7's specific
lens: what must be **run for real** to validate each component, and
what the paper/official-repository sources say about how to run it.

| Component | Paper spec | Official-code spec | `baseline-v1.0` module | Reproduction status | Confidence |
|---|---|---|---|---|---|
| Data split | 125,417 train / 991 valid / 1,624 test (MIMIC-CXR); 1,000 CheXpert zero-shot (`PAPER`) | Split inherited externally (Delbrouck et al. 2023 / vilmedic) | `src/data/{schema,parsing,integrity,splits,manifest}.py` | **Untested on real data** — only synthetic fixtures exercised to date | High (spec), Low (real-data execution) |
| RadGraph/CheXbert annotation | No version stated (`PAPER`) | `radgraph==0.0.9`, `f1chexbert==0.0.2` (`OFFICIAL_REPOSITORY`) | `src/baseline/radgraph/{annotator,compat}.py` | Real-checkpoint-verified on synthetic input (Cells 14/16); never run at real corpus scale | Medium |
| Factual pair mining | top_k=**2** (`PAPER`) vs. top_k=**3** (`OFFICIAL_REPOSITORY`) — unresolved discrepancy | `chex_thresh=1.0`, `radg_thresh=0.4`, `n=125417` (`OFFICIAL_REPOSITORY`, `gen_topk_pos.sh`) | `src/baseline/pair_mining/mining.py` | Both `top_k` values implemented, neither run at real scale | High (spec), Low (real-data execution) |
| Retriever architecture | MARVEL (T5-ANCE + ViT) (`PAPER`) | `CLIPVisionModel` (`openai/clip-vit-base-patch32`) + `OpenMatch/t5-ance` (`OFFICIAL_REPOSITORY`) | `src/baseline/retrieval/model.py` | Real-weight forward pass verified (Milestone 2.4G); **never trained** | Medium-High (spec), None (trained-weight execution) |
| Retriever training | τ=**0.01 fixed** (`PAPER`) vs. **learned** `logit_scale` (`OFFICIAL_REPOSITORY`) — unresolved discrepancy | AdamW, epochs=15, early_stop=5, batch=32, lr=5e-6 (`OFFICIAL_REPOSITORY`, matches `PAPER` exactly except temperature) | `src/baseline/retrieval/trainer.py` | Synthetic-data gradient flow verified only; **no real-data training run has ever occurred** (`docs/risk_register.md` #17) | High (spec), None (real training) |
| Retrieval-only evaluation | F1CheXbert/F1RadGraph/ROUGE-L/BERTScore reported (Table 2 top, `PAPER`); Recall/NDCG/MRR **not tabulated anywhere in the paper** | `evaluate_retriever.py`: MRR/Recall/NDCG @ {100,200,500,1000} via `pytrec_eval` (`OFFICIAL_REPOSITORY`) | `src/evaluation/retrieval_metrics.py` | Metrics real and `CELL 36: PASS`-verified on synthetic data; **never run against a real trained retriever's real corpus output** | High (metric implementation), None (real-value comparison — see §10) |
| RAG generation | Single top-1 retrieved report, fixed prompt template (`PAPER`, matches `build_rag_dataset.py` exactly) | Same-study/same-patient exclusion, fallback-to-raw-top-1-with-logged-anomaly (`OFFICIAL_REPOSITORY`) | `src/baseline/generation/{prompt_builder,adapter}.py` | Real HF adapter compatibility verified (Milestone 2.5G); **generator has never been fine-tuned or run end-to-end on real data** | Medium (spec), None (real generation) |
| Generation evaluation | ROUGE-L, BERTScore, F1RadGraph, F1CheXbert reported (Table 1); BLEU-4 computed in code, **never reported in paper** | `src/evaluation.py` (`OFFICIAL_REPOSITORY`) | `src/evaluation/generation_metrics.py` | Real and `CELL 36: PASS`-verified on synthetic data; **never run against real generated reports** | High (metric implementation), None (real-value comparison) |
| Oracle | `argmax` over corpus of `F1RadGraph + F1CheXbert` instance score, self-excluded for train queries (`PAPER`, Appendix A.3, verbatim confirmed this session) | Not implemented as a standalone script in the official repository | `src/baseline/evaluation/oracle.py` | Real and verified at tiny synthetic scale (Milestone 2.6, Cell 35); **never run at real corpus scale** | High (spec + implementation), None (real-scale execution) |
| Significance testing | "p-value < 0.05" (Table 1 caption, `PAPER`); exact test method **not stated** | Not in official code | `src/evaluation/significance.py` | Real, deterministic, verified (Milestone 2.6); cannot be compared against the paper's own statistic since the paper's exact method and per-example data are both unpublished (§16) | N/A — our own methodological choice, not a reproduction target |

## 5. Required Datasets and Credential Constraints

- **MIMIC-CXR**: requires PhysioNet credentialing (CITI training +
  signed Data Use Agreement). `PAPER` confirms the authors themselves
  completed this and that sharing access with third parties is
  prohibited (Section 8, Ethics). This project has never held, cached,
  or attempted automated access to MIMIC-CXR.
- **CheXpert**: requires Stanford AIMI Shared Datasets registration
  (`stanfordaimi.azurewebsites.net`), used **only** for the 1,000-pair
  zero-shot test set (`PAPER`, confirmed — never a training source).
- **The repository must not download either dataset automatically.**
  No script under `src/` or any reproduction tooling built for this
  milestone may fetch credentialed data — every download step is
  user-initiated, outside automation, per `docs/data_requirements.md`
  (unchanged, not modified by this contract).
- **Until real files are supplied by the user at the exact paths in §6,
  only synthetic/small-fixture dry runs (§20) are permitted.** Synthetic
  runs prove tooling correctness only — they **cannot** validate
  paper-level reproduction (§2's objective requires real data).

## 6. Dataset Placement and Expected File Manifests

Unchanged from `docs/data_requirements.md`, restated here for this
contract's self-containment:

```
data/                      # NEVER committed
├── mimic/
│   ├── train.json         # expected 125,417 records
│   ├── valid.json         # expected 991 records
│   └── test.json          # expected 1,624 records
├── chexpert/
│   └── test.json          # expected 1,000 records (zero-shot only)
└── manifests/
    ├── train_manifest.json
    ├── valid_manifest.json
    └── test_manifest.json
```

Record schema (`OFFICIAL_REPOSITORY`, unchanged):

```json
{"image": ["path/to/frontal.jpg", "path/to/lateral.jpg"], "finding": "...", "impression": "..."}
```

**Manifest check before any reproduction run proceeds** (§24/§25):
record count per split matches the expected counts above exactly (or
the run is halted and the discrepancy reported, never silently
proceeding on a partial/mismatched dataset); no `patient_id` appears in
more than one split (`IntegrityChecker`/`PatientSplitValidator`,
Milestone 2.1, reused unmodified); every referenced image path exists
on disk.

## 7. Required Checkpoints and Revisions

| Checkpoint | Source | Status |
|---|---|---|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32`, resolved revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | Real forward pass verified (Milestone 2.4G) |
| T5-ANCE | `OpenMatch/t5-ance`, resolved revision `bf70ee32b49c3e8c1d40982feebbc3b9930eeab4` | Real forward pass verified (Milestone 2.4G) |
| MARVEL warm-start | `OpenMatch/marvel-ance-clueweb`, resolved revision `19bd4191e36a285ffa13cad901c670cd785a4aec`, candidate file `model.best.pt` | Repository reachability confirmed; **full-weight load never attempted** (`docs/risk_register.md` #4) — parameter-name mapping to `MultiModalRetriever` (a from-scratch reimplementation, not a direct port) is likely required and untested |
| RadGraph | PyPI `radgraph==0.0.9` (`OFFICIAL_REPOSITORY` pin; paper states no version) | Real, verified (Milestone 2.2/2.6) |
| CheXbert | PyPI `f1chexbert==0.0.2`, checkpoint `chexbert.pth` (~1.25GB, publicly downloadable, no auth) | Real, verified (Milestone 2.2/2.6) |
| LLaVA base | `vicuna-7b-v1.5` (`PAPER`, Appendix A.2); fork `haotian-liu/LLaVA` at an official-repo-pinned commit | **`UNKNOWN`** — the exact pinned commit hash has not been recorded anywhere in this project's docs to date (§28); never downloaded, never run |
| LLaVA-1.6 | Named in `PAPER` Table 2 backbone-variation ablation only | Lower priority — not required for the primary Table 1 reproduction target |

## 8. Environment and Dependency Pins

Two mutually incompatible environments, unchanged from
`docs/compute_requirements.md`/`docs/paper_analysis.md` §18, restated:

- **Retriever-stage** (`OFFICIAL_REPOSITORY`): `torch==1.13.1`,
  `transformers==4.23.1`. **Not installable on Python 3.12**
  (`docs/risk_register.md` #8c) — this project uses
  `transformers==4.57.6` as a forced, disclosed deviation, `torch`
  left at whatever is already present (`2.11.0+cpu` in this sandbox).
- **Generator-stage (LLaVA)**: `transformers==4.36.2`, `peft==0.10.0`
  — never installed or exercised in this project to date.
- **Evaluation-stage** (Milestone 2.6, real-verified): `rouge==1.0.1`,
  `evaluate==0.4.6`, `bert-score==0.3.13`, `pytrec_eval==0.5`,
  `radgraph==0.0.9`, `f1chexbert==0.0.2`.

Every reproduction run's output (§24) records the exact installed
version of every package above — never assumed, always captured at run
time, matching Milestone 2.6's own `_detect_package_versions()`
discipline (reused unmodified).

## 9. Baseline Run Configuration

Because §4 documents two genuine, unresolved paper-vs-code
discrepancies (`top_k` ∈ {2, 3}; `temperature_mode` ∈ {fixed, learned}),
every reproduction run must declare which variant it used — **never a
single silently-chosen default**. A `ReproductionRunConfig` (spec only,
no code) is expected to carry: `top_k`, `temperature_mode`, dataset
split identifiers, checkpoint revisions (§7), dependency versions (§8),
and `seed` (§17) — every field explicit, none inferred from
environment state at run time.

## 10. Retrieval Evaluation Plan

Two genuinely different retrieval-evaluation targets exist, and this
milestone must not conflate them:

1. **Retrieval-only setting, reproducing `PAPER` Table 2 top half**:
   encode test-set images, retrieve the nearest training-corpus report
   by cosine similarity, score that *retrieved report's own text*
   directly against the ground-truth reference using the generation
   metrics (F1CheXbert, F1RadGraph, ROUGE-L, BERTScore) — this has a
   real, tabulated `PAPER` comparison point (§12).
2. **Recall@K/NDCG@K/MRR (via `pytrec_eval`, `OFFICIAL_REPOSITORY`'s
   `evaluate_retriever.py`)**: computable with `baseline-v1.0`'s real,
   verified `RecallAtKMetric`/`NdcgAtKMetric`/`MRRMetric`
   (`k_values=(100,200,500,1000)`, matching the official script's own
   default cutoffs exactly), but **the paper never tabulates a Recall
   or NDCG value anywhere**, and MRR appears only as an axis in
   Figures 3–4 (a qualitative threshold-sensitivity plot, not a
   reported point value for any specific configuration). This means
   Recall@K/NDCG@K/MRR are `NOT_COMPARABLE`-by-design against `PAPER`
   (§29) — they remain useful as an internal self-consistency
   diagnostic, or as a genuine comparison point only if this project
   separately runs the official `evaluate_retriever.py` script itself
   and treats that as an `OFFICIAL_REPOSITORY`-sourced number instead.

## 11. Generation Evaluation Plan

Reproduces `PAPER` Table 1 exactly: for each of MIMIC-CXR (in-domain)
and CheXpert (zero-shot), run the frozen `baseline-v1.0` RAG pipeline
(retrieval → prompt construction → generation) and score the output
via `EvaluationRunner` (Milestone 2.6, unmodified) on ROUGE-L, BLEU-4
(repo-only, no paper comparison target), BERTScore, F1RadGraph,
F1CheXbert. Requires a real-data-trained retriever **and** a real
fine-tuned generator (§7/§19) — this is the single largest blocker in
this milestone (§21).

## 12. Paper-Value Extraction Table

Every value below is `PAPER`, freshly re-extracted this session via
`pdftotext -layout paper/factmm_rag.pdf`, cross-checked against the
already-committed `docs/paper_analysis.md` §17 (Table 1, unchanged) and
newly extracting Table 2 in full (previously only summarized as "see
the PDF directly" — now captured verbatim below, closing that gap).

**Table 1 — RAG setting** (already in `docs/paper_analysis.md` §17,
reproduced here for self-containment): see §29's reserved comparison
table for the exact FactMM-RAG/Med-MARVEL/Oracle rows this milestone
targets.

**Table 2 — Retrieval-only setting** (`PAPER`, newly extracted this
session):

| Model | MIMIC-CXR F1CheXbert | MIMIC-CXR F1RadGraph | MIMIC-CXR ROUGE-L | MIMIC-CXR BERTScore | CheXpert F1CheXbert | CheXpert F1RadGraph | CheXpert ROUGE-L | CheXpert BERTScore |
|---|---|---|---|---|---|---|---|---|
| Med-MARVEL | 0.550 | 0.212 | 0.279 | 0.525 | 0.479 | 0.160 | 0.222 | 0.454 |
| **FactMM-RAG** | **0.605** | **0.249** | **0.297** | **0.547** | **0.491** | **0.174** | **0.237** | **0.467** |
| Oracle | 0.992 | 0.429 | 0.399 | 0.612 | 0.999 | 0.438 | 0.362 | 0.554 |

**Table 2 — Backbone variation, RAG setting** (`PAPER`, newly
extracted; confirms Table 1's headline FactMM-RAG row = `ClueWeb-
LLaVA1.5` exactly):

| Backbone | MIMIC-CXR F1CheXbert | MIMIC-CXR F1RadGraph | MIMIC-CXR ROUGE-L | MIMIC-CXR BERTScore | CheXpert F1CheXbert | CheXpert F1RadGraph | CheXpert ROUGE-L | CheXpert BERTScore |
|---|---|---|---|---|---|---|---|---|
| ClueWeb-LLaVA1.5 (= Table 1 FactMM-RAG) | 0.602 | 0.257 | 0.307 | 0.561 | 0.495 | 0.180 | 0.239 | 0.473 |
| WebQA-LLaVA1.5 | 0.572 | 0.262 | 0.304 | 0.562 | 0.456 | 0.184 | 0.237 | 0.474 |
| Med-MARVEL-LLaVA1.5 | 0.581 | 0.260 | 0.311 | 0.563 | 0.475 | 0.185 | 0.236 | 0.474 |
| ClueWeb-LLaVA1.6 | 0.601 | 0.252 | 0.303 | 0.558 | 0.492 | 0.178 | 0.237 | 0.471 |

**Recall@K / NDCG@K / MRR**: `UNKNOWN` — confirmed via full-text search
of the re-extracted PDF: no table anywhere in the paper reports a
Recall, NDCG, or point-value MRR number. MRR appears only as a
continuous axis (~0–20, later ~4.2–7.4) in Figures 3–4, a
threshold-sensitivity visualization, not a tabulated result for any
specific model configuration comparable to `baseline-v1.0`'s own
`k_values=(100,200,500,1000)` output.

**Oracle definition** (`PAPER`, Appendix A.3, verbatim confirmed this
session): `Oracle(q_i) = argmax_{j∈corpus, j≠i} s(q_i, d_j)`, where
`s(q,d)` is the **sum** of F1-RadGraph and F1-CheXbert instance-wise
scores — matches `docs/paper_analysis.md`'s existing description and
`OracleEvaluator`'s implementation (Milestone 2.6) exactly; no
discrepancy found on this re-check.

**BLEU-4**: confirmed `OFFICIAL_REPOSITORY`-only — computed by
`src/evaluation.py` but never reported in any paper table (re-confirmed
by full-text search this session, not just the prior audit's claim).

## 13. Our-Result Table Template

Empty placeholder, identical shape to §12/§29, to be filled only after
a real reproduction run executes:

```json
{
  "run_id": null,
  "dataset": null,
  "model": "FactMM-RAG (baseline-v1.0)",
  "top_k": null,
  "temperature_mode": null,
  "metrics": {
    "rouge_l": null, "bleu4": null, "bert_score": null,
    "f1radgraph": null, "f1chexbert": null,
    "recall": {"100": null, "200": null, "500": null, "1000": null},
    "ndcg": {"100": null, "200": null, "500": null, "1000": null},
    "mrr": null
  },
  "status": "NOT_YET_RUN"
}
```

## 14. Absolute and Relative Delta Formulas

For every metric with both a `PAPER` value and a `VERIFIED_BASELINE`
value:

```
absolute_delta = our_value - paper_value
relative_delta = absolute_delta / paper_value   (undefined / NOT_COMPARABLE if paper_value == 0)
```

All metrics in scope (§29) are higher-is-better, so a positive
`absolute_delta` means our baseline exceeds the paper's reported value
and a negative one means it falls short — both are reported as-is,
never truncated or reframed to look more favorable.

## 15. Reproduction Acceptance Criteria

**No single global threshold.** Per-metric, per-cause bands, distinct
from each other:

- **Exact implementation equivalence**: applies only to
  deterministic, formula-level agreement (e.g., our `F1RadGraph`
  wrapper's output on the *exact same* generated text the paper's own
  evaluation script would score) — `absolute_delta` should be at or
  near `0.0` (floating-point tolerance only, `<1e-6`). This band is
  **not achievable without the paper's own generated outputs**, which
  are unpublished — so this band is expected to be unreachable in
  practice and is recorded as such, not silently waived.
- **Expected stochastic tolerance**: applies to any metric downstream
  of a real, independently-trained retriever/generator (i.e., nearly
  everything in §12). Training is inherently stochastic (data
  ordering, hardware nondeterminism, exact seed) even when hyperparameters
  match exactly. Proposed band: `CLOSE` if `|relative_delta| <= 5%`,
  `DIVERGENT` if `|relative_delta| > 15%`, an intermediate zone left for
  case-by-case judgment via §23. These are proposed starting bands, not
  paper- or code-derived constants — recorded as our own methodological
  choice, exactly as `docs/risk_register.md` #13 already requires for
  significance testing.
- **Checkpoint/version tolerance**: applies specifically where §7/§8
  document a forced deviation (e.g. `transformers==4.57.6` instead of
  the paper's `4.23.1`, or an untested MARVEL warm-start). Any
  divergence must first be checked against §23's protocol before being
  attributed to a real implementation gap.
- **Non-comparable cases**: Recall@K/NDCG@K/MRR (§10, §12) — `UNKNOWN`
  paper target, so `status = NOT_COMPARABLE`, never forced into a
  MATCH/CLOSE/DIVERGENT judgment. BLEU-4 similarly — no paper value to
  compare against, `status = NOT_COMPARABLE`.

**Passing 921/921 unit/integration tests proves software correctness
only.** Reproduction requires real, executed benchmark agreement per
the bands above — this is stated explicitly to prevent ever declaring
"successfully reproduced" on the strength of the test suite alone.

## 16. Statistical Comparison Plan

`paired_bootstrap_significance` (Milestone 2.6, unmodified) is reused
for comparisons **within our own reproduction** — e.g., our
FactMM-RAG-configuration run vs. our own Med-MARVEL-configuration
control run, both executed by us on the same real query set. It
**cannot** be used to test our result against the paper's own
published Table 1/2 numbers directly, because the paper publishes only
aggregate values, never per-example scores — there is no paired data to
bootstrap against. This is stated explicitly to prevent a false claim
of "statistically compared against the paper." The paper's own "p-value
< 0.05" claim (Table 1 caption) used an unspecified, unpublished
per-example dataset and test method (`docs/paper_analysis.md` §15,
`docs/risk_register.md` #13) — not something this project can
re-execute or verify, only note as the paper's own unverifiable claim.

## 17. Seed/Repeat Policy

- **Evaluation-only runs** (scoring already-generated/already-retrieved
  output): a single deterministic pass is sufficient — metric
  computation itself is deterministic given fixed input (verified,
  Milestone 2.6).
- **Any real training run** (retriever stage, generator fine-tune, once
  attempted): recommend **≥3 seeds** per configuration before reporting
  a mean/CI, given training's inherent stochasticity (§15) — a single
  seed's result must never be presented as *the* reproduction number
  without disclosing it is a single run.
- Default seed: `42`, matching every other established convention in
  this project (`EvaluationRunConfig`, `SignificanceConfig`,
  `IterativeRefinementConfig`).

## 18. Determinism Policy

Reuses `EvaluationRunner`'s already-verified rerun-×2 byte-identical
determinism discipline (Milestone 2.6) for the evaluation stage of any
reproduction run, unmodified. Extended here: the full reproduction
pipeline (retrieval inference → prompt construction → generation →
evaluation) must be verified deterministic, given fixed checkpoints and
a fixed seed, via a rerun-×2 comparison at small/dry-run scale (§20) —
this determinism check is **not** performed at full paper scale (§21),
where compute cost makes a full rerun prohibitive; full-scale runs
instead record every input (checkpoint revisions, seed, dependency
versions) so a rerun *could* be performed later if ever needed.

## 19. Resource Estimation

Unchanged from `docs/compute_requirements.md`, restated for this
contract:

| Stage | Full-scale (paper-level) | This project's available environment |
|---|---|---|
| Retriever training | 1x A6000-class GPU, ~10h (`PAPER`, measured) | Not yet provisioned — no GPU training has occurred at any milestone |
| Factual similarity scoring | O(n²) over 125,417 reports, chunked SLURM (`OFFICIAL_REPOSITORY`) | Requires heavy sub-sampling or a comparable cluster |
| Generator (LLaVA) fine-tune | **8x A6000, ~4h** (`PAPER`, measured) | Well beyond a single-GPU Colab session; not provisioned |
| Evaluation | CPU-feasible at small scale; GPU helpful at full scale | Already real-verified at small scale (Milestone 2.6) |

**Full paper-scale reproduction of the generator stage is likely
infeasible in this project's currently available environment** without
a comparable multi-GPU cluster — flagged explicitly, not glossed over.
The retrieval-only setting (§10, §21) is comparatively far more
tractable and is recommended as the near-term, achievable reproduction
target.

## 20. Small-Scale Dry-Run Plan

A synthetic-fixture dry run, matching Cell 36's own established
discipline: tiny synthetic MIMIC-CXR-shaped fixture data (already
present under `tests/fixtures/`, never real patient data), run through
the **full** reproduction pipeline — retrieval-only evaluation,
RAG-setting generation evaluation, comparison-table generation (§29),
and the discrepancy-analysis protocol (§23) — end to end. Its only
purpose is proving the **tooling** (once built, in a future milestone)
works correctly; per explicit instruction, **synthetic runs cannot
validate paper-level reproduction** and must never be reported or
mistaken as such. Every dry-run output is labeled `SYNTHETIC_DRY_RUN`,
never presented alongside real comparison numbers without that label.

## 21. Full-Scale Benchmark Plan

Gated entirely on real, user-supplied, credentialed data (§5/§6) and
real checkpoints (§7). Two achievable tiers, recommended in this order:

1. **Retrieval-only reproduction (§10, Table 2 top half)** — requires
   only a real-data-trained retriever, not the generator. Comparatively
   tractable (§19: 1x A6000-class GPU, ~10h) and has a real `PAPER`
   comparison point. Recommended as the first full-scale target.
2. **Full RAG generation reproduction (§11, Table 1)** — additionally
   requires a real fine-tuned LLaVA generator (§19: 8x A6000, ~4h),
   which this project's environment does not currently provide. This
   tier is recommended as a **later**, resource-gated milestone, not
   attempted until GPU capacity comparable to the paper's own setup is
   available and confirmed by the user.

Neither tier begins until this contract is approved and a follow-up
implementation milestone is separately authorized (§1: no code yet).

## 22. Failure Taxonomy

Extends Milestone 2.6's six-category taxonomy
(`_classify_metric_dependency_error`: `missing_dependency`,
`authentication_required`, `checkpoint_unavailable`, `network_error`,
`invalid_input`, `computation_failed`) with reproduction-specific
categories:

- `DATA_UNAVAILABLE` — credentialed dataset not yet supplied by the
  user at the expected path (§6).
- `CHECKPOINT_UNAVAILABLE` — a required checkpoint (§7) cannot be
  reached or its full weights cannot be loaded (e.g. MARVEL's untested
  warm-start).
- `ENVIRONMENT_MISMATCH` — a dependency pin (§8) cannot be installed as
  specified (matching the already-documented `transformers`/`torch`
  Python-3.12 incompatibility, `docs/risk_register.md` #8c).
- `CONFIG_AMBIGUITY` — a paper-vs-code discrepancy (§4: `top_k`,
  `temperature_mode`) has not yet been resolved by an actual comparison
  run; the run is recorded as blocked on this ambiguity, not silently
  defaulted.
- `NUMERIC_DIVERGENCE` — the run completed successfully but a metric
  falls outside its acceptance band (§15); routed to §23.

## 23. Discrepancy-Analysis Protocol

For every metric classified `DIVERGENT` (§15) or `NUMERIC_DIVERGENCE`
(§22), check, **in this order**, before concluding anything:

1. **Environment/version pins** (§8) — does the installed dependency
   set differ from the paper's/official repository's pins in a way
   plausibly affecting this metric?
2. **Config variant used** (§9) — was this run's `top_k`/
   `temperature_mode` the same as whichever variant (if either) the
   paper's headline numbers actually used? (Genuinely unknown for
   certain per §4 — both must be tried.)
3. **Checkpoint revision** (§7) — does the exact checkpoint/revision
   hash match what was used to produce the reference numbers, or is it
   a later/different snapshot?
4. **Data/preprocessing/split** (§5, §6) — does the manifest match the
   expected counts and construction exactly?

The discrepancy is documented in `discrepancy_report.md` (§24) citing
which of the above (if any) explains the gap, with supporting evidence
— or marked **`UNEXPLAINED`** if none of the four account for it. A
divergence is never silently attributed to "expected noise" without
this checklist being run first.

## 24. Artifact/Output Schemas

Every output file below carries the same traceability fields Milestone
2.6/Phase 3 already established (`package_versions`, `hardware`,
`git_commit`, `generated_timestamp_utc`, `seed`) plus reproduction-
specific fields (`baseline_tag`, `checkpoint_revisions`,
`dataset_manifest_hashes`):

- **`paper_values.json`** — §12's extraction table, machine-readable,
  every value tagged `PAPER` with a citation (table/page reference).
- **`reproduction_config.json`** — the `ReproductionRunConfig` (§9)
  actually used for a given run.
- **`dataset_manifest.json`** — per-split record counts, sha256 hashes,
  patient/study leakage check result (§6).
- **`checkpoint_manifest.json`** — every checkpoint's resolved revision
  hash actually loaded (§7).
- **`retrieval_metrics.json`** — real `VERIFIED_BASELINE` Recall@K/
  NDCG@K/MRR output (§10), tagged `NOT_COMPARABLE` against `PAPER`.
- **`generation_metrics.json`** — real `VERIFIED_BASELINE` ROUGE-L/
  BLEU-4/BERTScore/F1RadGraph/F1CheXbert output (§11).
- **`reproduction_comparison.csv`** / **`reproduction_comparison.md`**
  — §29's table, populated with real deltas and statuses.
- **`discrepancy_report.md`** — §23's protocol output, per divergent
  metric.
- **`run_metadata.json`** — the full traceability bundle above, once
  per run.
- **`logs/`** — raw stdout/stderr of every stage, for post hoc audit.

All outputs are deterministic (§18), versioned (never overwritten,
append-only per run_id — matching Cells 30–36's own atomic-write
discipline), and fully traceable per the fields above.

## 25. Colab Execution Plan

**No Cell 37 is created by this contract.** The intended future
structure (prose only, for a later, separately-authorized
implementation milestone): a first cell running the small-scale dry run
(§20) against synthetic fixtures only, followed — only once real data
is confirmed present at the exact paths in §6 and the manifest check
passes — by a second, explicitly gated cell that runs the real
reproduction pipeline. The gate itself (manifest presence + count +
leakage check) must fail loudly and halt, never silently falling back
to synthetic data if real data is expected but missing.

## 26. Commit Plan

All additive, under a `reproduction/` tooling namespace (final module
path TBD at implementation time), never touching `src/baseline/`,
`src/evaluation/`, or any other frozen baseline path:

1. `Implement paper-value extraction and reproduction config schemas`
2. `Implement dataset/checkpoint manifest validators`
3. `Implement retrieval-only and generation evaluation reproduction runners`
4. `Implement comparison table and discrepancy report generators`
5. `Run small-scale synthetic dry run and document tooling verification`
6. `(gated on real data) Run retrieval-only full-scale reproduction`
7. `(gated on GPU capacity) Run full RAG generation reproduction`

Each commit stays scoped to exactly what it states, matching the
one-cohesive-unit-per-commit discipline used throughout this project.

## 27. Known Risks

Directly relevant `docs/risk_register.md` entries, unmodified,
inherited into this milestone's scope: #1b (`top_k` discrepancy), #1c
(temperature discrepancy), #2 (credentialed data), #3 (generator stage
unverified), #4 (MARVEL checkpoint untested), #6 (full-scale compute
infeasibility), #8/#8c (RadGraph/environment version pins), #15
(F1CheXbert run-to-run label instability — directly threatens any
reported F1CheXbert reproduction number's trustworthiness), #16
(CPU-only verification to date, GPU behavior unverified), #17 (no
real-data retriever training ever performed), #18 (train-time vs.
inference-time candidate representation mismatch, reproduced as-is),
#19 (Stage 2/ANCE loss divergence from official flat-pool scheme, not
yet empirically compared).

**New, milestone-2.7-specific risks**: no per-example paper data exists
for a true paired statistical comparison against the paper itself
(§16); Recall@K/NDCG@K/MRR have zero paper-published reference point
(§10/§12) and can only ever be `NOT_COMPARABLE` against `PAPER`; the
generator stage is entirely unverified end-to-end (never trained, never
run on real data), making full Table 1 reproduction the single largest
open risk in this milestone.

## 28. Open Questions

- **`UNKNOWN`**: the exact pinned commit hash of the `haotian-liu/
  LLaVA` fork the official repository's `install_llava.sh` clones —
  not recorded anywhere in this project's docs to date; must be
  extracted from the official repository's script directly before any
  generator-stage work begins.
- **`UNKNOWN`**: the paper's exact significance-test method (only
  "p-value < 0.05" is stated, Table 1 caption) — cannot be reproduced,
  only disclosed as unknown (§16).
- Should `top_k=2` (`PAPER`) and `top_k=3` (`OFFICIAL_REPOSITORY`) both
  be run as separate, fully reported reproduction arms, or is there a
  principled way to determine which one actually produced the paper's
  Table 1 numbers before running both? Current recommendation: run
  both, report both, never silently pick (per §9's stated discipline).
- Same question for `temperature_mode` (`fixed` τ=0.01 vs. `learned`
  `logit_scale`).
- Is full LLaVA fine-tuning (§19, §21) feasible at all within this
  project's available compute, or should the reproduction objective be
  explicitly narrowed to the retrieval-only setting (§10) as the
  practical ceiling?
- Was `evaluate_retriever.py`'s Recall@K/NDCG@K/MRR output ever
  published anywhere outside the paper itself (supplementary material,
  project page, author blog) that this project has not yet checked? Not
  found in this session's audit — currently treated as `UNKNOWN`, not
  assumed absent forever.

## 29. Required Comparison Tables (Reserved)

**Nothing below is populated.** Every `VERIFIED_BASELINE` cell is
empty pending a real reproduction run; `status` is `UNKNOWN` (never
run) until then, not a placeholder MATCH/CLOSE/DIVERGENT guess.

### Generation (RAG setting, per dataset)

| Metric | Dataset | Paper value | Official-repo value | Our baseline value | Absolute delta | Relative delta | Status |
|---|---|---|---|---|---|---|---|
| ROUGE-L | MIMIC-CXR | 0.307 | N/A (not separately published) | — | — | — | `UNKNOWN` |
| BLEU-4 | MIMIC-CXR | N/A (not paper-reported) | computed by `src/evaluation.py`, value not published | — | — | — | `NOT_COMPARABLE` |
| BERTScore | MIMIC-CXR | 0.561 | N/A | — | — | — | `UNKNOWN` |
| F1RadGraph | MIMIC-CXR | 0.257 | N/A | — | — | — | `UNKNOWN` |
| F1CheXbert | MIMIC-CXR | 0.602 | N/A | — | — | — | `UNKNOWN` |
| ROUGE-L | CheXpert (zero-shot) | 0.236 | N/A | — | — | — | `UNKNOWN` |
| BLEU-4 | CheXpert (zero-shot) | N/A | N/A | — | — | — | `NOT_COMPARABLE` |
| BERTScore | CheXpert (zero-shot) | 0.475 | N/A | — | — | — | `UNKNOWN` |
| F1RadGraph | CheXpert (zero-shot) | 0.185 | N/A | — | — | — | `UNKNOWN` |
| F1CheXbert | CheXpert (zero-shot) | 0.475 | N/A | — | — | — | `UNKNOWN` |

### Retrieval

| Metric | Paper value | Official-repo value | Our baseline value | Absolute delta | Relative delta | Status |
|---|---|---|---|---|---|---|
| Recall@100 | not tabulated (`UNKNOWN`) | computable via `evaluate_retriever.py`, not published | — | — | — | `NOT_COMPARABLE` |
| Recall@200 | not tabulated (`UNKNOWN`) | computable, not published | — | — | — | `NOT_COMPARABLE` |
| Recall@500 | not tabulated (`UNKNOWN`) | computable, not published | — | — | — | `NOT_COMPARABLE` |
| Recall@1000 | not tabulated (`UNKNOWN`) | computable, not published | — | — | — | `NOT_COMPARABLE` |
| NDCG@100 | not tabulated (`UNKNOWN`) | computable, not published | — | — | — | `NOT_COMPARABLE` |
| NDCG@200 | not tabulated (`UNKNOWN`) | computable, not published | — | — | — | `NOT_COMPARABLE` |
| NDCG@500 | not tabulated (`UNKNOWN`) | computable, not published | — | — | — | `NOT_COMPARABLE` |
| NDCG@1000 | not tabulated (`UNKNOWN`) | computable, not published | — | — | — | `NOT_COMPARABLE` |
| MRR | shown only as a figure axis, no point value (`UNKNOWN`) | computable, not published | — | — | — | `NOT_COMPARABLE` |

### Retrieval-only setting (Table 2 top half — has a real paper target)

| Metric | Dataset | Paper value (FactMM-RAG) | Our baseline value | Absolute delta | Relative delta | Status |
|---|---|---|---|---|---|---|
| F1CheXbert | MIMIC-CXR | 0.605 | — | — | — | `UNKNOWN` |
| F1RadGraph | MIMIC-CXR | 0.249 | — | — | — | `UNKNOWN` |
| ROUGE-L | MIMIC-CXR | 0.297 | — | — | — | `UNKNOWN` |
| BERTScore | MIMIC-CXR | 0.547 | — | — | — | `UNKNOWN` |
| F1CheXbert | CheXpert | 0.491 | — | — | — | `UNKNOWN` |
| F1RadGraph | CheXpert | 0.174 | — | — | — | `UNKNOWN` |
| ROUGE-L | CheXpert | 0.237 | — | — | — | `UNKNOWN` |
| BERTScore | CheXpert | 0.467 | — | — | — | `UNKNOWN` |

## 30. Branch Strategy Recommendation

**Recommended: a dedicated branch, `reproduction/milestone-2.7`,
branched from `main`/`baseline-v1.0` directly — not
`innovation/iterative-rag`.**

Rationale:

- **Strict separation from Innovation work.** `innovation/
  iterative-rag`'s own contract (`docs/phase_3_innovation_
  contract.md` §13) already establishes a one-directional dependency
  and anti-contamination discipline for Innovation code specifically.
  Reproduction tooling is a *different* kind of non-baseline work — it
  validates the baseline, it does not build on top of it the way
  Innovation modules do — and conflating the two under one branch would
  blur that distinction and complicate later review/history.
- **Independent lifecycles.** Innovation implementation is currently
  paused; reproduction validation may finish, stall on data access
  (§5), or need revisiting independently of whatever state Innovation
  work is in. A dedicated branch lets each proceed (or pause) without
  entangling the other's commit history.
- **Mirrors the already-established pattern.** `innovation/
  iterative-rag` itself was branched directly from `main`/
  `baseline-v1.0`, not from another feature branch (`docs/phase_3_
  innovation_contract.md` §14) — `reproduction/milestone-2.7` follows
  the identical pattern for the identical reason: every non-baseline
  work stream branches from the frozen baseline tag directly.
- **Baseline protection.** Neither branch, by construction, can modify
  `main`/`baseline-v1.0` — reproduction tooling reads the baseline's
  public API (matching the same read-only dependency direction as
  Innovation code) and never edits it.

**If a baseline bug is discovered during reproduction work**: it is
(1) documented (in `discrepancy_report.md` and/or `docs/risk_
register.md`), (2) reproduced with a new failing test demonstrating the
bug, (3) fixed in a **separate, dedicated patch branch/commit** off
`main` (never on `reproduction/milestone-2.7` and never silently folded
into a reproduction script as a workaround), and (4) only after that
patch is merged does `baseline-v1.0` get superseded by a new tag
(e.g. `baseline-v1.1`) for future work to build on — `baseline-v1.0`
itself is never retroactively modified.
