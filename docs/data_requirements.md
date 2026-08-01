# Data Requirements

## Datasets Referenced by the Official Repository

| Dataset | Access | License/DUA | Source (OFFICIAL_REPOSITORY) |
|---|---|---|---|
| MIMIC-CXR | Credentialed (PhysioNet), requires completing CITI training and signing a data use agreement | PhysioNet Credentialed Health Data License | Linked via `vilmedic.app/papers/acl2023/` preprocessed split in README; paper Section 8 confirms authors completed training + signed DUA, and that sharing access with third parties is prohibited |
| CheXpert | Credentialed registration | Stanford AIMI dataset license | Linked via `stanfordaimi.azurewebsites.net` in README |

**Confirmed by the paper** (`PAPER_EXPLICIT`, Section 4 "Dataset"):
MIMIC-CXR provides **125,417 training / 991 validation / 1,624 test**
image-report pairs, sourced from Beth Israel Deaconess Medical Center.
CheXpert is used **only** as a **1,000-pair zero-shot test set** (the
"hidden test set" from MIMIC-CXR-RRS, with images separately downloaded
from Stanford AIMI) — never for training. This confirms and replaces the
prior `REASONABLE_INFERENCE` that CheXpert was evaluation-only.
| MARVEL retriever checkpoint | Public HuggingFace repo (`OpenMatch/marvel-ance-clueweb`) | Check HF repo's own license before use | README |
| LLaVA base model / fork | Public GitHub (`haotian-liu/LLaVA`, pinned commit) | Apache-2.0 (LLaVA) — verify against pinned commit | `install_llava.sh` |

**This project will never automatically download any of the above.** All
downloads are user-initiated, outside of automated scripts that assume
credentials exist.

## Expected Local Data Layout (once user supplies data)

```
data/                      # NEVER committed — see .gitignore
├── mimic/
│   ├── train.json
│   ├── valid.json
│   └── test.json
├── chexpert/
│   └── test.json
└── manifests/
    ├── train_manifest.json
    ├── valid_manifest.json
    └── test_manifest.json
```

Record schema (`OFFICIAL_REPOSITORY`, `data/parse.py` / `data/label.py`):

```json
{
  "image": ["path/to/frontal.jpg", "path/to/lateral.jpg"],
  "finding": "...",
  "impression": "..."
}
```

## Patient-Level Split Requirements

- MIMIC-CXR and CheXpert both encode `patient_id` and `study_id` in their
  directory structure (confirmed by `build_rag_dataset.py`'s
  `extract_paths` helper, which parses `patient, study, file` from the last
  three path components — `OFFICIAL_REPOSITORY`).
- This project's data pipeline (Milestone 2.1) must **independently
  verify** — not just assume — that no `patient_id` appears in more than
  one of {train, valid, test}. The official repo does not include this
  check explicitly as a standalone validator; it only filters
  same-study/same-patient **retrieval candidates at RAG-construction time**
  (`build_rag_dataset.py`), which is a different, narrower guarantee than a
  clean split.
- A dataset manifest (JSON/CSV) will record, per split: patient IDs, study
  IDs, image paths (relative, not absolute/PHI-adjacent), and record
  counts — never raw report text or images themselves, to avoid
  accidentally leaking PHI into version control or logs.

## Integrity Checks Required (Milestone 2.1)

- Missing image files referenced by a manifest entry.
- Missing or empty `finding`/`impression` text.
- Duplicate `(patient_id, study_id)` pairs within a split.
- Any `patient_id` present in more than one split.
- Non-frontal-only studies where a frontal view cannot be identified (if
  view-position metadata is available upstream).

## Synthetic / Small-Scale Test Data

For Phase 3 (smallest viable reproduction) and for unit tests, this project
will generate **synthetic fixtures** (placeholder images, templated
finding/impression text, fabricated but internally-consistent patient/study
IDs) — never real patient data, and never data requiring the above
credentials. Fixtures live under `tests/fixtures/`.

## Open Questions (remaining after reading the paper)

- Any additional inclusion/exclusion criteria beyond the vilmedic/RadSum23
  processed MIMIC-CXR split's stated counts (125,417/991/1,624) — the paper
  cites the split as-is from Delbrouck et al. 2023 rather than re-deriving it.
- Whether the paper's authors independently re-verified patient/study
  leakage in that inherited split, or trusted it as-given — not stated, so
  this project verifies leakage itself regardless (see below).
