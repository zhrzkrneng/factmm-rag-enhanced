# Milestone 2.8 Cell 43 -- IU X-Ray Acquisition + Canonicalization Report

## Source discipline

- **official_nlm_openi** (OFFICIAL, U.S. National Library of Medicine (NLM), Open-i service): SKIPPED -- not checked -- already satisfied by reused local files
- **kaggle_mirror_raddar** (MIRROR_OF_PUBLIC_DATASET, Kaggle user 'raddar' (community mirror)): BLOCKED -- HTTP 404
- **academic_torrents_mirror** (MIRROR_OF_PUBLIC_DATASET, Academic Torrents (community mirror)): REACHABLE -- HTTP 200

Source used: **official_nlm_openi**

## Acquisition status: SUCCEEDED (reused-existing and/or downloaded)

2 file(s) present and checksum-verified, 1361926760 bytes total.

## Canonicalization

Layout detected: official NLM XML
Canonicalization error/status: (none -- canonicalization ran)
- Total records: 3955
- Usable records: 3826
- Excluded records: 129
- Malformed XML files: 0
- Records with a missing-tag caveat: 129
- Duplicate study_ids: 0
- Duplicate image paths: 0
- Missing-image records: 104
- Missing-report records: 25
- Corrupt images: 0

## Split

Authoritative split available: False -- Cell 42's audit found only a de facto community-standard 70/10/20 split (Li et al. 2018 lineage), not an official NLM-published split, and did not independently verify Li et al. 2018 itself. Per instruction, only an AUTHORITATIVE and VERIFIED public split may override the required deterministic default -- this does not qualify, so the fixed 80/10/10, seed=42 default is used instead.
- train: 3060
- validation: 382
- test: 384

## Parser caveats

- The official-NLM XML parser discovers tag names at runtime (matches on lowercased substrings 'abstracttext', 'mesh', 'problem', 'parentimage', 'uid' rather than a rigid fixed XPath) because the exact schema was not independently verified from a primary NLM documentation source this session. Any AbstractText Label not in {COMPARISON, INDICATION, FINDINGS, IMPRESSION} is preserved under each record's source_metadata.unmapped_abstract_labels, never discarded.
- image_ids always lists every parentImage id referenced by a report's XML, even if no matching PNG file was found; unresolved ids are listed per-record under source_metadata.unresolved_image_ids.
