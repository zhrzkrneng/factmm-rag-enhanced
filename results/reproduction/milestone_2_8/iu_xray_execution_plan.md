# Milestone 2.8 Cell 42 -- IU X-Ray Execution Plan (forward-looking, planning only)

This is a plan for future, separately-authorized cells. Nothing in this document is executed by Cell 42 itself.

## Preconditions before any execution cell may begin

1. This scope migration (Cell 42) is reviewed and approved by the user.
2. A replacement retriever is explicitly chosen, under a separate decision (not this cell).
3. The generator stage (Vicuna-7B-v1.5 acquisition, any required HF token) is explicitly authorized, under a separate decision (not this cell).
4. The migration-impact list's named documents are actually updated (not done in this cell).

## Planned steps, in order (none executed yet)

1. Acquire IU X-Ray from the official NLM source (NLMCXR_png.tgz + NLMCXR_reports.tgz) if confirmed accessible without credentials at execution time; otherwise document the actual blocker encountered (mirroring this project's established 'never assume, always verify against the real environment' discipline from Cell 38/39).
2. Build an IU-X-Ray-specific parser analogous to src/data/parsing.py's MimicCxrParser/CheXpertParser, extending src/data/schema.py's ReportRecord (or introducing a compatible variant) to carry the Comparison and Indication sections that the current two-section (finding/impression) schema does not model.
3. Adopt the community-standard 70/10/20 patient-level split (Li et al. 2018 lineage) as the default, pending independent verification of that paper (not read this session) -- and record it explicitly as a community convention, not an official NLM split, per this cell's audit.
4. Wire the adapted baseline, Innovation 1, and cumulative-innovations runs to the identical IU X-Ray split, per the fair experiment design in `scope_migration_report.md` section 5.
5. Apply the metric-validity caveats from `scope_migration_report.md` section 6 -- treat F1RadGraph and F1CheXbert results on IU X-Ray as provisional pending empirical validation, not as drop-in equivalents of their MIMIC-CXR/CheXpert behavior.
6. Update every document named in the migration-impact list before or alongside presenting any results, so no reader can mistake IU X-Ray numbers for paper-reproduction numbers.

## Explicitly out of scope for this plan document

- Choosing the replacement retriever.
- Downloading or loading Vicuna-7B-v1.5, or requesting any HF token.
- Downloading IU X-Ray.
- Any code change under src/ or tests/.
