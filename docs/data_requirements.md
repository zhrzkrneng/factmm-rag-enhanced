# Data Requirements

## Datasets Referenced by the Official Repository

| Dataset | Access | License/DUA | Source (OFFICIAL_REPOSITORY) |
|---|---|---|---|
| MIMIC-CXR | Credentialed (PhysioNet), requires completing CITI training and signing a data use agreement | PhysioNet Credentialed Health Data License | Linked via `vilmedic.app/papers/acl2023/` preprocessed split in README |
| CheXpert | Credentialed registration | Stanford AIMI dataset license | Linked via `stanfordaimi.azurewebsites.net` in README |
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

## Open Questions (pending paper PDF)

- Exact split sizes and any additional inclusion/exclusion criteria the
  paper applies beyond MIMIC-CXR's official split.
- Whether CheXpert is used for training, evaluation only, or cross-dataset
  generalization testing (the shipped repo only ships a CheXpert `test.json`
  placeholder, suggesting evaluation-only, but this is `REASONABLE_INFERENCE`,
  not confirmed).
