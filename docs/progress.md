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
| 2.2 — RadGraph processing | NOT STARTED |
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

## Next Step

Milestone 2.1 complete. Awaiting approval to begin Milestone 2.2
(RadGraph processing).
