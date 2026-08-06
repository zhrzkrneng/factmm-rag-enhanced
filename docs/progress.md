# Project Progress

High-level status tracker. For per-cell execution detail, see
`docs/colab_execution_log.md`. For the reproduction target/status per
component, see `docs/reproduction_matrix.md`.

## Phase Status

| Phase | Status |
|---|---|
| Phase 0 — Repository and Source Audit | COMPLETE |
| Phase 1 — Project Architecture | COMPLETE |
| Phase 2 — Baseline Reproduction | IN PROGRESS |
| Phase 3 — Small-Scale Reproduction | NOT STARTED |
| Phase 4 — Proposed Innovations | NOT STARTED |
| Phase 5 — Experiment Design | NOT STARTED |
| Phase 6 — Research Quality and Documentation | NOT STARTED |

## Phase 2 Milestone Status

| Milestone | Status |
|---|---|
| 2.1 — Data pipeline | COMPLETE (unit + integration tested in Colab) |
| 2.2 — RadGraph processing | COMPLETE (compatibility layer + `annotator.py` implementation, merged into `main`, verified end-to-end via real Cell 16 smoke test) |
| 2.3 — Fact-aware pair mining | COMPLETE (`src/baseline/pair_mining/`, merged into `main` via PR #3, merge commit `d54df1e`) |
| 2.4 — Retriever | **Implementation complete, synthetic training validated, and real Hugging Face adapter forward-compatibility verified.** Four commits (`1fa883f`, `5d4d6b1`, `4e7759d`, `813be31`) pushed to `claude/factmm-rag-repo-setup-o04jx3`; open draft PR #4 targeting `main`, not yet merged |
| 2.5 — Retrieval-augmented generator | NOT STARTED |
| 2.6 — Evaluation | NOT STARTED |

## Milestone 2.1 — Colab Cell Checklist

Fixed cell order per the project's Colab workflow (setup cells 01–07,
then one cell at a time through the data pipeline classes described in
the Milestone 2.1 architecture explanation).

- [x] Cell 01 — Environment Inspection — **SUCCESS**
- [x] Cell 02 — Mount Google Drive — **SUCCESS**
- [x] Cell 03 — Repository Cloning — **SUCCESS** (public repo, no PAT needed)
- [x] Cell 04 — Dependency Installation — **SUCCESS** (faiss-cpu newly installed; rest already satisfied)
- [x] Cell 05 — Import and Version Verification — **SUCCESS** (transformers 5.13.1 noted as a future-milestone risk, not a current blocker)
- [x] Cell 06 — Configuration and Reproducibility Setup — **SUCCESS** (repo importable, SEED=42, configs readable)
- [x] Cell 07 — Data Availability and Directory Validation — **SUCCESS** (no real data present yet, as expected; skeleton created)
- [x] Cell 08 — `ReportRecord` implementation + unit tests — **SUCCESS** (7/7 tests passed)
- [x] Cell 09 — `JsonReportParser` implementation + unit tests — **SUCCESS** (12/12 tests passed)
- [x] Cell 10 — `IntegrityChecker` + exception types implementation + unit tests — **SUCCESS** (21/21 tests passed)
- [x] Cell 11 — `PatientSplitValidator` implementation + unit tests — **SUCCESS** (28/28 tests passed)
- [x] Cell 12 — `ManifestBuilder` + schema implementation + unit tests — **SUCCESS** (33/33 tests passed; demo manifest verified leak-free on Drive)
- [x] Cell 13 — end-to-end pipeline smoke test — **SUCCESS** (34/34 tests passed)

**Milestone 2.1 (data pipeline): COMPLETE.** All 5 planned classes
(`ReportRecord`, `JsonReportParser`, `IntegrityChecker`,
`PatientSplitValidator`, `ManifestBuilder`) implemented, unit-tested,
and verified end-to-end in the user's Colab environment. Awaiting
approval to begin Milestone 2.2 (RadGraph processing).

## Implementation Deviations From the Original Phase 1 Sketch

Documented here per CLAUDE.md's "document assumptions and deviations"
rule — both are simplifications discovered once the classes were
actually implemented, not silent changes:

1. **One parser class, not two.** Phase 1 planned separate
   `MimicCxrParser`/`CheXpertParser` classes. Both datasets turned out
   to share the identical JSON schema (`{"image", "finding",
   "impression"}`), differing only in their patient/study directory
   naming convention embedded in image paths. `JsonReportParser`,
   parameterized by `dataset`, replaces both without duplicating logic.
2. **One manifest class, not two.** Phase 1 planned
   `DatasetManifest`+`SplitManifest`. Nothing needs a combined
   multi-split wrapper — a plain `Dict[str, List[ReportRecord]]` (what
   `PatientSplitValidator` already accepts) covers that need — so only
   `SplitManifest` (matching the actual per-split output file) exists.

## Milestone 2.2 — Environment Audit (Cell 14), Compatibility Layer (Cell 15), Annotation Pipeline (Cell 16)

**Status: Milestone 2.2 is approved as complete.** The compatibility
layer (`src/baseline/radgraph/compat.py`, 29/29 unit tests) and the
RadGraph/CheXbert annotation pipeline (`src/baseline/radgraph/
annotator.py`, part of a 90/90-passing suite) are both implemented,
merged into `main` (PR #1), and independently verified end-to-end
against the real Colab environment: the compatibility layer via the
Cell 14 rerun, and the annotation pipeline itself via the real Cell 16
smoke test (see `docs/colab_execution_log.md`). This is an
execution-compatibility and pipeline-behavior verification — it is
**not** a clinical-correctness claim; see the Cell 16 summary below and
the scope reminders throughout this section.

- Full detail: `docs/colab_execution_log.md`'s Cell 14, "Cell 14 —
  Rerun After Runtime Restart," and Cell 15 entries.
- **Key outcome (original audit)**: `radgraph==0.0.9` and
  `f1chexbert==0.0.2` (matching the official repo's exact pins) both
  construct and run correctly on this Colab's Python 3.12 runtime,
  after 6 compatibility shims and 2 cache-path fixes for
  incompatibilities between `radgraph`'s ~2022-era vendored AllenNLP
  code and the modern package ecosystem. `transformers` had to be
  pinned to `4.57.6` instead of the official `4.23.1` — the exact pin
  is not installable on Python 3.12 (no compatible `tokenizers` wheel
  exists), so `4.57.6` (last pre-"V5" 4.x release) is used instead, a
  documented, forced deviation.
- **Confidence levels**: 5 of the 6 shims are provably
  behavior-preserving (restoring unchanged, well-documented legacy
  method semantics). One shim (`preprocess_reports`'s missing
  `"dataset"` field) supplies a required data value rather than
  restoring documented behavior, and is flagged for re-examination
  during Milestone 2.3 validation.
- **Compatibility layer moved into version control**: all 6 shims (5
  permanent, 1 experimental-gated) now live in
  `src/baseline/radgraph/compat.py`, version-gated via
  `check_environment()`/`CompatibilityError` so they refuse to silently
  apply to an unvalidated future package version. 29 unit tests
  (`tests/unit/test_radgraph_compat.py`) exercise the shim logic
  against injected fakes — no heavy ML dependencies required to run
  them. Full suite: 63/63 passing (no Milestone 2.1 regressions).
- **Independent re-verification (Cell 14 rerun)**: after this Colab
  session's kernel accumulated stale in-memory `transformers` state (a
  kernel-local bug, root-caused via a fresh-subprocess-vs-in-kernel
  comparison — not a 7th package incompatibility) and was fixed by a
  full runtime restart, a self-contained restoration of Cell 14 (same 6
  shims, same order, independent of `compat.py`) was re-run from
  scratch: `RadGraph()` and `F1CheXbert()` both constructed
  successfully and produced real annotation/inference output against
  the actual downloaded checkpoints.
- **Observed discrepancy, recorded not silently resolved**: this Cell
  14 rerun's F1CheXbert output (`[0,1,0,...]`, index 1 = "Cardiomegaly")
  differs from the original v15 run's output (index 13, "No Finding")
  for the identical input sentence and identical pinned versions. See
  `docs/risk_register.md` row 15 — root cause unconfirmed, tracked as
  hypotheses only. **Does not block the compatibility layer from being
  considered complete** (this does not mean Milestone 2.2 itself is
  complete — see below) — clinical/metric correctness and run-to-run
  determinism are explicitly out of this milestone's scope, deferred to
  Milestone 2.3/2.6.
- **Scope reminder**: Cell 14 verifies execution compatibility only —
  it does not verify clinical or metric correctness. Neither smoke-test
  output (original or rerun) should be read as a correctness claim
  about either model.
- **Cell 16 — real annotation-pipeline smoke test (`PASS`)**: run
  against the real `RadGraphAnnotator` merged into `main` (PR #1, merge
  commit `2910a979...`). `RadGraphAnnotator()` constructed real
  `RadGraph()` and `F1CheXbert()` models via the compatibility layer.
  Three synthetic (non-PHI) `ReportRecord`s were successfully annotated
  by `annotate_records()`. Output JSONL schema, stable
  `(dataset, patient_id, study_id)` identifiers, the 14-class
  `chexbert_labels_14` vector, the `chexbert_labels_5` subset, the
  sidecar run metadata (`run_status`, `summary`), and resume behavior
  (re-running the same records skipped all 3 with 0 reprocessed, 0
  duplicate output lines) were all validated. Both checkpoints were
  reused from the runtime's cache — the metadata recorded
  `revision_source: "preexisting_file"` for each, confirming no
  re-download occurred. Full detail: `docs/colab_execution_log.md`,
  "Cell 16 — Real Annotation Pipeline Smoke Test."
- **What this does and does not establish**: Cell 16 verifies execution
  compatibility and pipeline behavior only, not clinical correctness —
  the same scope boundary as Cell 14. The F1CheXbert determinism
  question (`docs/risk_register.md` row 15, an open, unconfirmed-root-
  cause risk from the Cell 14 rerun) remains open; Cell 16 did not
  investigate it and its resolution is not required for Milestone 2.2.

## Milestone 2.3 — Fact-Aware Pair Mining

**Status: COMPLETE.** `src/baseline/pair_mining/similarity.py` and
`src/baseline/pair_mining/mining.py` implement the paper's Eq. 1/2
factual-similarity mining procedure (`chexbert_similarity`,
`radgraph_similarity`, `combined_score`, `PairMiningConfig`,
`PairMiner`), with both the paper-explicit (`top_k=2`) and
official-repository (`top_k=3`, thresholds `chex=1.0`/`radg=0.4`)
defaults recorded and never silently conflated (see
`docs/risk_register.md` #1b). Implemented, unit-tested, and verified
against a real end-to-end synthetic smoke test in the user's Colab
environment (15-record dataset, 6 hand-verified scenarios, resume and
shuffle-determinism checks). Merged into `main` via a new draft PR #3
("Milestone 2.3: implement baseline Fact-Aware Pair Mining"), merge
commit `d54df1e`.

## Milestone 2.4 — Retriever

**Status: implementation complete, synthetic training validated, and
real Hugging Face adapter forward-compatibility verified.**

### Implementation (four commits, pushed, open draft PR #4)

- `1fa883f` — **Retriever core**: `RetrieverConfig`, `MultiModalRetriever`
  (`src/baseline/retrieval/model.py`) — CLIP-ViT vision tower + T5
  patch-splicing fusion, three audited encode paths
  (`encode_images_only`/`encode_text_only`/`encode_images_with_text`),
  both temperature modes (`fixed` vs. `learned_logit_scale`) kept
  explicitly configurable (resolves `docs/risk_register.md` #1c as "both
  implemented, never silently chosen"), warm-start loading with
  always-reported missing/unexpected keys — plus `loss.py`'s
  single-direction `contrastive_loss`, matching the official code exactly.
- `5d4d6b1` — **Dataset and collator**: `src/baseline/retrieval/dataset.py`
  — `RetrievalTrainingRow`, `RetrieverTrainingDataset`,
  `RetrievalCollator` (Stage 1 rectangular in-batch-negative batching;
  Stage 2 padded/masked per-query candidate batching — a disclosed,
  deliberate divergence from the official code's flat/whole-batch-
  negative-pool scheme, see `docs/risk_register.md`).
- `4e7759d` — **Hard-negative mining**: `src/baseline/retrieval/
  hard_negatives.py` — `HardNegativeMiningConfig`, `HardNegativeMiner`,
  reproducing the official `gen_hard_negatives.py`'s embedding-top-N +
  factual-dissimilarity selection (without FAISS, by design for this
  milestone) plus deliberate patient-leakage/cross-dataset safety checks
  beyond what the official script itself does.
- `813be31` — **Embedding export, FAISS indexing, and a lightweight
  trainer**: `src/retrieval/embeddings.py` (`export_embeddings`),
  `src/retrieval/index.py` (`FaissFlatIPIndex`, exact FlatIP, matching
  the official code's own index type), and
  `src/baseline/retrieval/trainer.py` (`RetrieverTrainer`,
  `RetrieverTrainerConfig`) — a smoke-test-scale training loop (not the
  paper-scale reproduction) with checkpoint save/load, sha256 integrity
  verification, and genuine train-resume.

**321/321 full project test suite passing** across all four commits.
Synthetic end-to-end verification (fake encoders, tiny data only)
confirmed real gradient flow: forward pass → contrastive loss →
backward → optimizer step → checkpoint save → reload → embedding
export → FAISS search, all in one smoke test
(`test_synthetic_end_to_end_smoke_lifecycle`).

### Milestone 2.4G — Real Hugging Face Adapter + Checkpoint Dry Run

A follow-up verification step, explicitly **not** Generator, Evaluation,
or Innovation work — its only purpose was confirming
`MultiModalRetriever` (unmodified) can actually connect to and run
through the real Hugging Face ecosystem, with no training, no
fine-tuning, no dataset download, and no paper-scale experiment. Full
detail: `docs/colab_execution_log.md`, "Milestone 2.4G" entry.

**Runtime**: Python 3.12.13, `torch` 2.11.0+cpu, `transformers` 4.57.6,
`tokenizers` 0.22.2, `accelerate` 1.14.0, `faiss` 1.14.3, CPU-only (no
CUDA device).

**Real image side**: `openai/clip-vit-base-patch32` (resolved revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`) — real `CLIPVisionModel` +
`CLIPImageProcessor` constructed (hidden size 768, image size 224), real
forward pass succeeded.

**Real text side**: `OpenMatch/t5-ance` (resolved revision
`bf70ee32b49c3e8c1d40982feebbc3b9930eeab4`) — real `T5Tokenizer` +
`T5EncoderModel` constructed (hidden size 768, vocab size 32100), real
forward pass succeeded. (`T5EncoderModel` here is a standalone
connectivity probe, distinct from `MultiModalRetriever`'s own
`_default_t5_factory`, which constructs the full encoder-decoder
`T5Model` its `_pool()` actually needs — see the model docstring and
`docs/colab_execution_log.md` for why these are deliberately two
different checks, not a contradiction.)

**Real `MultiModalRetriever` integration** (project's own real default
factories, architecture unmodified): image-only embedding shape
`(1, 768)`, text-only embedding shape `(1, 768)`, joint image+text
embedding shape `(1, 768)`, all outputs finite (no NaN/Inf), L2
normalization verified (query embedding norm `0.9999999403953552` ≈
1.0), scaled similarity verified, both `learned_logit_scale` and
`fixed` temperature modes verified against the real weights.

**Warm-start checkpoint**: `OpenMatch/marvel-ance-clueweb` confirmed
reachable (resolved revision `19bd4191e36a285ffa13cad901c670cd785a4aec`),
candidate checkpoint file `model.best.pt` identified. Full checkpoint
download/load was **intentionally not attempted** (kept as a disclosed
`WARNING`, not a `FAIL` — this is a scope decision, not a failure; see
`docs/risk_register.md`).

**Scope**: this verifies real-model construction and forward
compatibility only. No training, fine-tuning, dataset download, or
paper-scale reproduction occurred, and no claim about retrieval quality
or scientific performance is made.

**No compatibility issue was discovered** — `MultiModalRetriever` was
not modified as a result of this dry run.

## Next Step

Milestone 2.4 (Retriever) is implementation-complete, synthetically
validated, and real-Hugging-Face-adapter-verified. Draft PR #4
("Milestone 2.4: complete baseline Retriever stack") is open against
`main`, not yet merged — awaiting approval. Remaining open risks (full
MARVEL warm-start weight loading untested, CPU-only runtime, no
real-data training performed, the train/inference representation
mismatch, and the Stage 2 batch/loss divergence from official code) are
tracked in `docs/risk_register.md` and do not block moving to Milestone
2.5 once approved. Milestone 2.5 (retrieval-augmented generator) has
not been started and awaits explicit approval, per the project's
per-milestone-approval workflow.


## Milestone 2.8 -- IU X-Ray Scope Migration, Acquisition, and Pipeline Integration (Cells 40-46)

**Status: Milestone 2.8 is ready to close.** IU X-Ray/Open-I replaces MIMIC-CXR/CheXpert as the primary executable benchmark (Cell 42 scope migration); MARVEL removed from required resources; Vicuna-7B-v1.5 deferred to a later generator stage. Official acquisition, deterministic canonicalization/split, loader, schema mapping, and retrieval/generation/prompt pipeline integration are all real-data-validated (Cells 43-45). This Cell 46 CPU-only end-to-end dry run exercises four labeled diagnostic arms (DATA_ONLY, MOCK_RETRIEVAL, ORACLE_PIPELINE_CHECK, MOCK_GENERATION_SMOKE) -- none of which is real retriever inference, real generation, or a scientific result.

Readiness totals: {'READY': 16, 'PARTIALLY_READY': 4, 'DEFERRED': 10, 'REMOVED': 1, 'BLOCKED': 0}.

This project does not claim exact FactMM-RAG reproduction, direct comparability to the paper's MIMIC-CXR/CheXpert scores, real retriever execution, real Vicuna/LLaVA generation, scientific performance from mock outputs, or a completed baseline result table. It does claim an architecture-faithful adaptation inspired by FactMM-RAG, official public IU X-Ray acquisition and deterministic preprocessing, a validated loader/schema/retrieval/generation/prompt pipeline, CPU-only end-to-end readiness, and readiness to implement the selected retriever and generator. Full detail: `results/reproduction/milestone_2_8/cell46_milestone_closure.json`.

Remaining GPU/model work before a real baseline result exists: Selecting and implementing a replacement retriever (embedding model + FAISS index).; Acquiring and loading Vicuna-7B-v1.5 (or another selected generator backbone).; Real generator inference (LLaVA-style multimodal generation).; Empirical validation of F1RadGraph/F1CheXbert behavior on real IU X-Ray-derived generated text.; Building the first real baseline-vs-innovation result table..