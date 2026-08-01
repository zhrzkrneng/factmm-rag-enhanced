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
| 2.2 — RadGraph processing | COMPATIBILITY LAYER COMPLETE AND TESTED (Cell 14 rerun + Cell 15); `annotator.py` implementation not started |
| 2.3 — Fact-aware pair mining | NOT STARTED |
| 2.4 — Retriever | NOT STARTED |
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

## Milestone 2.2 — Environment Audit (Cell 14) and Compatibility Layer (Cell 15)

**Status: the compatibility layer is complete and validated —
`src/baseline/radgraph/compat.py` exists, is unit-tested (29/29), and
has been verified against the real Colab environment via an
independent Cell 14 rerun. This refers only to the compatibility
layer, not Milestone 2.2 as a whole: the RadGraph annotation pipeline
itself (`src/baseline/radgraph/annotator.py` and downstream annotation
logic) remains unimplemented, so Milestone 2.2 is NOT complete.**

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

## Next Step

The compatibility layer (Cells 14 + 15) is complete and end-to-end
tested. **This is not the same as Milestone 2.2 being complete** — the
annotation pipeline itself has not been started. Awaiting explicit
approval before any commit/push, and before starting
`src/baseline/radgraph/annotator.py` (the actual annotation-pipeline
implementation built on top of `compat.py`), which is what would
actually complete Milestone 2.2.
