# Milestone 2.8 — Resource Acquisition & Environment Completion: Execution Contract

> **EXECUTION CONTRACT — NOT EXECUTABLE CODE.** This document defines
> how the resources blocking Milestone 2.7 (real MIMIC-CXR, real
> CheXpert, the MARVEL warm-start checkpoint, and the LLaVA base
> checkpoint — `results/reproduction/milestone_2_7/blocker_status.json`,
> `all_blockers_resolved: false`) will eventually be acquired, verified,
> and stored, once the user acts on it. **No script, notebook, or config
> file is created by this document — contract only.** No dataset or
> checkpoint is downloaded here. No baseline source file or test is
> modified. Cell 40 does not begin here. Every value below is labeled
> with its source; every unresolved item is marked `UNKNOWN` explicitly,
> never guessed.

## Table of Contents

1. [Purpose and Scope](#1-purpose-and-scope)
2. [Current Project Status](#2-current-project-status)
3. [Resource Inventory](#3-resource-inventory)
4. [Per-Resource Specification](#4-per-resource-specification)
5. [Dataset Acquisition Workflow](#5-dataset-acquisition-workflow)
6. [Checkpoint Acquisition Workflow](#6-checkpoint-acquisition-workflow)
7. [Environment Validation Workflow](#7-environment-validation-workflow)
8. [Storage Layout](#8-storage-layout)
9. [Download Integrity Verification](#9-download-integrity-verification)
10. [Risk Analysis](#10-risk-analysis)
11. [Failure Recovery](#11-failure-recovery)
12. [Reproducibility Guarantees](#12-reproducibility-guarantees)
13. [Security / Licensing Considerations](#13-security--licensing-considerations)
14. [Expected Outputs](#14-expected-outputs)
15. [Success Criteria](#15-success-criteria)
16. [Stopping Criteria](#16-stopping-criteria)
17. [Open Questions](#17-open-questions)
18. [Branch Strategy](#18-branch-strategy)

---

## 1. Purpose and Scope

Milestone 2.7 (Cells 37–39) established, through real, executed,
never-fabricated checks, that four prerequisites block any real
baseline reproduction: the MIMIC-CXR dataset, the CheXpert dataset, the
MARVEL warm-start checkpoint, and the LLaVA base checkpoint (fork
commit unresolved). Milestone 2.8's sole purpose is to define **how**
those four resources — plus the two already-available real checkpoints
(CLIP, T5-ANCE) — are acquired, where they are stored, and how their
integrity is verified, **before** any acquisition is attempted. This is
a contract, not an acquisition: nothing is downloaded, no automation is
built, and no runtime code exists as a result of this document. A
future, separately-authorized milestone implements the actual
acquisition tooling this contract specifies.

**In scope**: resource identification, licensing/access documentation,
storage layout, integrity-verification method, and the acquisition
workflow's shape.

**Out of scope**: any download, any checkpoint fetch, any training, any
inference, any change to `src/baseline/`, `src/evaluation/`,
`src/common/`, `src/data/`, `src/retrieval/`, `src/generation/`,
`src/innovation/`, or `tests/`, and any script/notebook/config file.

## 2. Current Project Status

- **`baseline-v1.0`**: the tagged, merged, real-verified baseline
  (Milestones 2.1–2.6, `CELL 36: PASS`, 921/921 tests) — unchanged,
  frozen, the fixed reference point for all reproduction work.
- **Milestone 2.7, Cells 37–39: complete.** Cell 37 extracted every
  paper/repo value into deterministic manifests. Cell 38 performed real,
  offline, no-download environment/dataset/checkpoint validation. Cell
  39 freshly re-checked all four blockers and confirmed, via a hard
  gate, that no reproduction inference has been or could be started.
  Both this project's sandbox and the user's Colab environment converged
  on the same set of unresolved blockers.
- **Current blockers** (from `results/reproduction/milestone_2_7/
  blocker_status.json`, `all_blockers_resolved: false`):
  - `mimic_cxr`: `MISSING` — no file present at any expected path.
  - `chexpert`: `MISSING` — no file present at any expected path.
  - `marvel_warm_start_checkpoint`: `MISSING` — not cached locally in
    either environment checked to date.
  - `llava_base_checkpoint`: `BLOCKED` — the fork's pinned commit hash
    is itself `UNKNOWN`, so there is nothing yet to check availability
    of.
  - For reference (not a blocker): the CLIP and T5-ANCE checkpoints were
    found `AVAILABLE` (locally cached) in the user's Colab environment
    during Cell 38, though not in this project's own sandbox — both
    real weights, already forward-pass-verified in Milestone 2.4G.

## 3. Resource Inventory

| # | Resource | Currently blocking? |
|---|---|---|
| 1 | MIMIC-CXR | Yes |
| 2 | CheXpert | Yes |
| 3 | MARVEL warm-start checkpoint | Yes |
| 4 | LLaVA base (`vicuna-7b-v1.5` + `haotian-liu/LLaVA` fork) | Yes |
| 5 | CLIP (`openai/clip-vit-base-patch32`) | No — already real, verified, and (in at least one environment) locally cached |
| 6 | T5-ANCE (`OpenMatch/t5-ance`) | No — already real, verified, and (in at least one environment) locally cached |

## 4. Per-Resource Specification

### 4.1 MIMIC-CXR

- **Source**: PhysioNet (`physionet.org`), processed per Delbrouck et
  al. 2023 (vilmedic ACL-2023 RadSum23 split), linked via
  `vilmedic.app/papers/acl2023/` in the official repository's README.
- **License**: PhysioNet Credentialed Health Data License.
- **Access method**: manual, credentialed download by the user via
  PhysioNet's own web interface — never automated by this project.
- **Authentication requirements**: PhysioNet account + completed CITI
  "Data or Specimens Only Research" training + signed Data Use
  Agreement (paper, Section 8, `PAPER_EXPLICIT`).
- **Expected size**: `UNKNOWN` — neither the paper nor the official
  repository states a download size in GB; not independently measured
  by this project (no access held).
- **Checksum/version**: `UNKNOWN` — PhysioNet does not publish a
  project-independent checksum for this specific processed split in any
  source this project has read; the split is defined by record counts
  only (125,417 / 991 / 1,624), not a file hash.
- **Destination path**: `data/mimic/{train,valid,test}.json` (never
  committed — see §8).
- **Validation method**: record-count match against the paper-stated
  split sizes, plus `IntegrityChecker`/`PatientSplitValidator`
  (Milestone 2.1, reused unmodified) for patient-leakage-free splits —
  see §9.

### 4.2 CheXpert

- **Source**: Stanford AIMI Shared Datasets
  (`stanfordaimi.azurewebsites.net`), the specific dataset referenced in
  the paper's Appendix A.3 footnote (the "hidden test set" images for
  the 1,000 MIMIC-CXR-RRS zero-shot test reports).
- **License**: Stanford AIMI dataset license (exact terms `UNKNOWN` —
  not independently reviewed by this project; access has never been
  held).
- **Access method**: manual, credentialed registration by the user via
  Stanford AIMI's own portal — never automated by this project.
- **Authentication requirements**: Stanford AIMI account registration
  (exact requirements, e.g. institutional affiliation, `UNKNOWN` —
  not verified).
- **Expected size**: `UNKNOWN`.
- **Checksum/version**: `UNKNOWN`.
- **Destination path**: `data/chexpert/test.json` (never committed).
- **Validation method**: record-count match against 1,000 (paper-stated,
  zero-shot only), same integrity-checker reuse as MIMIC-CXR.

### 4.3 MARVEL warm-start checkpoint

- **Source**: Hugging Face Hub, `OpenMatch/marvel-ance-clueweb`.
- **License**: `UNKNOWN` — not independently reviewed by this project;
  must be checked against the actual HF repo card before any use, per
  `docs/data_requirements.md`'s existing note ("Check HF repo's own
  license before use").
- **Access method**: Hugging Face Hub download (`huggingface_hub`/
  `transformers` standard resolution), publicly reachable per Milestone
  2.4G's confirmed repository reachability — no special credential
  needed beyond normal HF Hub access.
- **Authentication requirements**: none known/found to date (public
  repo); reconfirm at acquisition time, since this could change.
- **Expected size**: `UNKNOWN` — the candidate file `model.best.pt` was
  identified (Milestone 2.4G) but its exact byte size was not recorded.
- **Checksum/version**: resolved revision
  `19bd4191e36a285ffa13cad901c670cd785a4aec` (confirmed real, Milestone
  2.4G / Cell 37's `checkpoint_requirements.json`); no separate
  file-level checksum (e.g. sha256) has been recorded for
  `model.best.pt` itself — `UNKNOWN`.
- **Destination path**: standard Hugging Face cache
  (`~/.cache/huggingface/hub/models--OpenMatch--marvel-ance-clueweb/`)
  — not a project-tracked path, consistent with how CLIP/T5-ANCE are
  already handled.
- **Validation method**: `transformers.AutoConfig.from_pretrained(...,
  local_files_only=True)` (proves local availability, zero network,
  reused from Cells 38/39 unmodified) as a pre-check; full weight load
  + `load_warm_start_checkpoint(strict=False)` (Milestone 2.4,
  `src/baseline/retrieval/model.py`) as the real validation once
  downloaded, with the resolved revision hash always recorded — a
  parameter-name mapping step is likely required first (`docs/
  risk_register.md` #4, since `MultiModalRetriever` is a from-scratch
  reimplementation, not a direct port).

### 4.4 LLaVA base (`vicuna-7b-v1.5` + `haotian-liu/LLaVA` fork)

- **Source**: `vicuna-7b-v1.5` (base weights, exact hosting location as
  used by the official repo not yet re-confirmed by this project —
  `UNKNOWN` pending re-check of `install_llava.sh`); fork
  `haotian-liu/LLaVA` on GitHub, cloned by the official repo's
  `install_llava.sh` at an unspecified-in-this-project's-docs pinned
  commit.
- **License**: LLaVA itself Apache-2.0 per `docs/data_requirements.md`'s
  existing note, **but not yet verified against the actual pinned
  commit** (`docs/risk_register.md` #14, still `Open`); `vicuna-7b-v1.5`
  carries its own separate license terms (LLaMA-derived — exact terms
  `UNKNOWN`, not yet reviewed by this project).
- **Access method**: `UNKNOWN` in full detail — the official
  `install_llava.sh` script performs the fork clone and weight
  acquisition, but this project has not yet re-read that script in this
  milestone to extract its exact steps.
- **Authentication requirements**: `UNKNOWN` — likely none for the
  Apache-2.0-licensed fork code itself, but `vicuna-7b-v1.5`'s own
  weight distribution terms are unconfirmed.
- **Expected size**: `UNKNOWN` (a 7B-parameter model; order-of-magnitude
  estimate would be `REASONABLE_INFERENCE` only, not stated by any
  source read to date — not recorded as a number here per the
  instruction against inventing missing information).
- **Checksum/version**: `UNKNOWN` — **the single largest open item in
  this entire contract**: the exact `haotian-liu/LLaVA` fork commit
  pinned by the official repository has never been recorded anywhere in
  this project's documentation (first surfaced as an open question in
  `docs/milestone_2_7_reproduction_validation_contract.md` §28, reused
  here unresolved).
- **Destination path**: `UNKNOWN` — cannot be specified until the
  acquisition method itself is confirmed (§17).
- **Validation method**: `UNKNOWN` pending the above; at minimum, once
  a commit hash is identified, the same `local_files_only=True`-style
  pre-check pattern used for the other three Hub checkpoints should
  apply for any Hub-hosted artifact, plus a fork-commit-hash match
  against whatever is ultimately pinned in this project's own
  reproduction config.

### 4.5 CLIP (`openai/clip-vit-base-patch32`)

- **Source**: Hugging Face Hub.
- **License**: `UNKNOWN` in full detail — not independently reviewed by
  this project beyond noting it is a public, unrestricted HF repo;
  MIT-style OpenAI CLIP licensing is `REASONABLE_INFERENCE` from general
  knowledge of the OpenAI CLIP release, not confirmed against this
  specific HF repo's card by this project.
- **Access method**: standard Hugging Face Hub download — already
  exercised for real (Milestone 2.4G).
- **Authentication requirements**: none (public repo, already confirmed
  reachable and loadable for real).
- **Expected size**: `UNKNOWN` (not recorded during Milestone 2.4G).
- **Checksum/version**: resolved revision
  `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` (real, confirmed, Milestone
  2.4G).
- **Destination path**: standard Hugging Face cache — already present
  in at least the user's Colab environment (Cell 38, `AVAILABLE`).
- **Validation method**: already validated — real forward pass, correct
  embedding dimensions, finite outputs (Milestone 2.4G); re-validated as
  `AVAILABLE` via `local_files_only=True` in Cell 38.

### 4.6 T5-ANCE (`OpenMatch/t5-ance`)

- **Source**: Hugging Face Hub.
- **License**: `UNKNOWN` in full detail — not independently reviewed by
  this project.
- **Access method**: standard Hugging Face Hub download — already
  exercised for real (Milestone 2.4G).
- **Authentication requirements**: none (public repo, already confirmed
  reachable and loadable for real).
- **Expected size**: `UNKNOWN` (not recorded during Milestone 2.4G).
- **Checksum/version**: resolved revision
  `bf70ee32b49c3e8c1d40982feebbc3b9930eeab4` (real, confirmed, Milestone
  2.4G).
- **Destination path**: standard Hugging Face cache — already present
  in at least the user's Colab environment (Cell 38, `AVAILABLE`).
- **Validation method**: already validated — real forward pass, correct
  embedding dimensions, finite outputs (Milestone 2.4G); re-validated as
  `AVAILABLE` via `local_files_only=True` in Cell 38.

## 5. Dataset Acquisition Workflow

For both MIMIC-CXR and CheXpert, in the same shape (specifics per §4):

1. **User-side credentialing** (outside this project entirely): the
   user completes the dataset's own credentialing process (PhysioNet
   CITI + DUA for MIMIC-CXR; Stanford AIMI registration for CheXpert).
   This project neither performs nor automates any part of this step.
2. **User-side download**: the user downloads the dataset through the
   provider's own official interface, to a location of their choosing
   outside this repository's tracked tree.
3. **User-side placement**: the user places (or a future,
   separately-authorized helper script — not built by this contract —
   assists in placing) the processed files at the exact paths in §8.
4. **Manifest generation**: a future milestone's tooling (not built
   here) runs `ManifestBuilder` (Milestone 2.1, reused unmodified) to
   produce `data/manifests/{train,valid,test}_manifest.json`.
5. **Integrity verification**: §9.

**This project's own tooling never initiates step 1 or 2.** No script
produced by any future milestone may assume credentials exist or
attempt an automated fetch.

## 6. Checkpoint Acquisition Workflow

For CLIP, T5-ANCE, and the MARVEL warm-start checkpoint (all Hugging
Face Hub-hosted):

1. **Reachability pre-check**: `local_files_only=True` probe (Cells
   38/39's method, reused) to confirm current local-cache state before
   attempting anything.
2. **Download** (future milestone, not this contract): standard `transformers`/
   `huggingface_hub` resolution, pinned to the exact resolved revision
   recorded in §4 — never an unpinned "latest" fetch, to keep the
   checkpoint identity fixed and reproducible.
3. **Revision confirmation**: after download, the locally resolved
   revision hash is compared against the pinned value in §4; any
   mismatch is treated as a blocker requiring investigation (§10), never
   silently accepted.
4. **Validation**: §9.

For the LLaVA base checkpoint: **blocked entirely on §17's open
question** (the fork's pinned commit). No acquisition workflow can be
specified for it until that is resolved.

## 7. Environment Validation Workflow

Reuses Cells 38/39's exact discipline, unmodified, as the standing
validation method for this milestone and beyond:

1. Runtime probe (Python version, CUDA visibility, every required
   package's installed version) — Cell 38 §1, reused.
2. Dataset/manifest presence + record-count + patient-leakage checks —
   Cell 38 §2 / this contract §9, reused.
3. Checkpoint local-cache presence via `local_files_only=True` (zero
   network calls, proven via direct socket instrumentation, Cell 39's
   `no_network_calls()`, reused) — never a live fetch during validation.
4. Repository validation (required baseline modules import, config
   dataclasses construct with defaults) — Cell 38 §4, reused.

Any future acquisition milestone re-runs this exact validation workflow
after each acquisition step, never assuming success without a fresh,
real check.

## 8. Storage Layout

```
data/                          # NEVER committed (.gitignore)
├── mimic/
│   ├── train.json             # 125,417 records expected
│   ├── valid.json             # 991 records expected
│   └── test.json              # 1,624 records expected
├── chexpert/
│   └── test.json              # 1,000 records expected (zero-shot only)
└── manifests/
    ├── train_manifest.json
    ├── valid_manifest.json
    └── test_manifest.json

~/.cache/huggingface/hub/      # standard HF cache, NOT project-tracked
├── models--openai--clip-vit-base-patch32/
├── models--OpenMatch--t5-ance/
└── models--OpenMatch--marvel-ance-clueweb/

<LLaVA destination>            # UNKNOWN -- see SS4.4, SS17
```

`results/reproduction/milestone_2_7/` (and any future
`milestone_2_8/` outputs, if this contract's future implementation
phase produces any) remain under the existing `results/*` `.gitignore`
rule, force-added individually per this project's established practice
— this contract does not propose changing that rule.

## 9. Download Integrity Verification

- **Datasets**: record count per split matches the expected count
  exactly (§4); `IntegrityChecker`/`PatientSplitValidator` (Milestone
  2.1, reused unmodified) confirm no `patient_id` appears in more than
  one split and no referenced image path is missing. No dataset-wide
  cryptographic checksum is available from either provider in any
  source read to date (`UNKNOWN`) — record-count and leakage checks are
  therefore the primary integrity signal, not a substitute for one that
  doesn't exist.
- **Checkpoints (CLIP, T5-ANCE, MARVEL)**: resolved revision hash
  (Hugging Face's own commit-hash mechanism) compared against the
  pinned value in §4 — this is the checksum-equivalent mechanism HF Hub
  provides; no separate file-level sha256 has been independently
  computed by this project for any of the three.
- **LLaVA**: `UNKNOWN` pending §17.
- **Failure handling**: any integrity check failure halts acquisition
  and is logged (§11) — never silently retried with a relaxed check,
  never silently proceeded past.

## 10. Risk Analysis

Inherited directly from `docs/risk_register.md` (unmodified) and
`docs/milestone_2_7_reproduction_validation_contract.md` §27, applied
specifically to acquisition:

- **#2** (credentialed data access): the dominant risk for this entire
  milestone — acquisition cannot proceed at all without the user
  completing external credentialing processes this project has no
  control over.
- **#4** (MARVEL checkpoint): full-weight load has never been attempted;
  parameter-name mapping to `MultiModalRetriever` is likely required and
  unverified — acquisition alone does not guarantee successful loading.
- **#14** (LLaVA licensing): not yet reviewed against the actual pinned
  commit — must be resolved before any LLaVA acquisition, not after.
- **New, milestone-2.8-specific risk**: the LLaVA fork's pinned commit
  being `UNKNOWN` is a hard blocker on specifying (not just executing) an
  acquisition workflow for that one resource — every other resource's
  workflow is at least fully specified even though not yet executed;
  LLaVA's is not.
- **New risk**: dataset/checkpoint sizes are unknown, so storage
  capacity planning for wherever acquisition eventually happens
  (local disk, Colab session storage, or an external volume) cannot be
  done precisely yet — only qualitatively (large enough to need
  planning, per `docs/compute_requirements.md`'s existing guidance).

## 11. Failure Recovery

- **Dataset acquisition failure** (credentialing rejected, download
  interrupted, record count mismatch): halt, log the exact failure mode
  and step, do not retry automatically, surface to the user for a
  manual decision — matching this project's established
  never-silently-retry discipline (Cells 30–39).
- **Checkpoint acquisition failure** (network error, revision mismatch,
  license concern discovered late): halt, log, do not substitute a
  different checkpoint or revision without explicit approval — matching
  the same "never silently substitute" discipline established for
  `pytrec_eval` (Milestone 2.6) and the paper/code discrepancies
  (`top_k`, temperature).
- **Partial acquisition state** (e.g. `train.json` present but
  `valid.json`/`test.json` missing): treated as `MISSING` overall for
  that dataset (matching Cell 38/39's `overall_status` logic — a
  dataset is not `AVAILABLE` until every expected file is present), not
  as a partial-credit state.
- **Recovery is always resumable, never destructive**: a partially
  acquired resource is left in place, never deleted or overwritten by
  a retry, until integrity verification (§9) explicitly passes.

## 12. Reproducibility Guarantees

- Every checkpoint acquisition pins an exact resolved revision hash
  (§4/§6) — never an unpinned "latest," so the exact weights used are
  always identifiable after the fact.
- Every dataset acquisition records its manifest (record counts, patient
  IDs, image paths — never raw PHI text/images, per `docs/data_
  requirements.md`'s existing constraint) as a traceable artifact.
- Environment validation (§7) is re-run and its real output recorded
  after every acquisition step — no acquisition is considered complete
  without a fresh, real, post-acquisition validation pass, not merely a
  successful download.
- All of the above feeds directly into `docs/milestone_2_7_
  reproduction_validation_contract.md` §16's reproducibility-metadata
  scheme (`checkpoint_manifest.json`, `dataset_manifest.json`) —
  Milestone 2.8's outputs are designed to populate those exact fields,
  not a parallel, inconsistent scheme.

## 13. Security / Licensing Considerations

- **No credentialed data or credential material (API keys, DUA
  documents, account tokens) may ever be committed to this repository**
  — consistent with `docs/risk_register.md` #10 and this project's
  existing `.gitignore` discipline.
- **PHI handling**: MIMIC-CXR/CheXpert are de-identified per their own
  provider processes, but this project additionally commits to never
  storing raw report text or images in version control or logs (`docs/
  data_requirements.md`, unchanged) — manifests record only IDs, paths,
  and counts.
- **License review gap**: CLIP, T5-ANCE, and MARVEL's exact HF repo
  license terms have not been independently reviewed by this project
  (only inferred/assumed at a general level) — flagged as an open item
  (§17), to be closed before any of these checkpoints' weights are used
  in a reported result, not just before download.
- **LLaVA/vicuna licensing**: explicitly unresolved (`docs/risk_
  register.md` #14, still `Open`) — must be resolved before any
  acquisition attempt, not discovered after weights are already in
  hand.
- **Third-party redistribution prohibition**: MIMIC-CXR's own usage
  policy prohibits sharing access with third parties (paper, Section 8)
  — this project's storage layout (§8, never committed) is designed
  specifically to never violate this.

## 14. Expected Outputs

This contract itself is the only output of Milestone 2.8's
*documentation* phase. Once a future, separately-authorized
*implementation* phase acts on this contract, expected future outputs
(not produced now) include: populated `data/` and HF cache directories
(never committed), `data/manifests/*.json`, and updated
`results/reproduction/milestone_2_7/{dataset_validation,
checkpoint_validation,blocker_status}.json` reflecting newly resolved
blockers (via a future Cell 39 re-run, not a new file scheme).

## 15. Success Criteria

Milestone 2.8's **documentation** phase (this contract) succeeds when:
this document exists, covers all required sections, marks every
unverified value `UNKNOWN` rather than inventing it, and is approved.

A future **implementation** phase (out of scope here) would succeed
when a subsequent Cell 39 re-run reports `all_blockers_resolved: true`
via real, fresh, no-fabrication checks — not when a human asserts the
resources are "probably fine."

## 16. Stopping Criteria

- This document stops at the contract. No script, notebook, or config
  file is created.
- Any future implementation phase stops immediately, without
  proceeding, if: a credentialing step is rejected or ambiguous, an
  integrity check fails, a license review surfaces a genuine
  restriction, or the LLaVA fork commit cannot be identified with
  confidence.
- No acquisition step, present or future, proceeds past an unresolved
  `UNKNOWN` in this contract without that `UNKNOWN` first being
  explicitly resolved and this contract updated to reflect it.

## 17. Open Questions

- **The exact `haotian-liu/LLaVA` fork commit** the official repository
  pins — the single largest blocker in this contract (§4.4). Must be
  extracted directly from `install_llava.sh` in the official
  repository, not guessed.
- **`vicuna-7b-v1.5`'s exact hosting location and access terms** as
  used by the official install path — not yet re-confirmed in this
  milestone.
- **Exact download sizes** for MIMIC-CXR, CheXpert, MARVEL's
  `model.best.pt`, and any LLaVA weights — unknown, relevant for storage
  planning.
- **Exact license terms** for the CLIP, T5-ANCE, and MARVEL Hugging
  Face repos — only generally assumed, not independently reviewed
  against each repo's actual license file/card.
- **`vicuna-7b-v1.5`'s license terms** specifically (distinct from the
  LLaVA fork's own Apache-2.0 code license) — unresolved.
- **Where acquired resources would physically live** for actual
  reproduction work — local disk, a persistent Colab-mounted volume, or
  an external storage service — not decided; depends on the user's
  available infrastructure, not something this project can presume.
- **Whether any semi-automated acquisition helper (e.g. a script that
  validates a user-provided local path against the expected manifest,
  never one that fetches credentialed data itself) would be built in a
  future milestone, or whether acquisition remains fully manual** —
  undecided, deferred to whenever an implementation phase is
  authorized.

## 18. Branch Strategy

**Recommended: continue this contract on `reproduction/milestone-2.7`
for now, matching the branch this document was authored on** (per this
turn's own stated "Current branch: reproduction/milestone-2.7"), since
Milestone 2.8 is a direct continuation of Milestone 2.7's own blocker
list and reuses its artifacts/validation methods throughout (§7, §12).

**When a future implementation phase begins** (actual acquisition
tooling, if ever authorized): recommend branching to a new, dedicated
branch — e.g. `reproduction/milestone-2.8` — from `main`/`baseline-v1.0`
directly, following the exact same pattern already established for
`innovation/iterative-rag` and `reproduction/milestone-2.7` (branch
from the frozen baseline tag, never from another in-progress branch),
preserving strict separation between baseline, Innovation, and
reproduction-tooling work streams. This is a recommendation for that
future phase, not an action taken by this contract.
