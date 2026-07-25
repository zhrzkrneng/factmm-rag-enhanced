# CLAUDE.md

Project rules for AI-assisted development on FactMM-RAG Enhanced.

## Development Rules

- Reproduce the original FactMM-RAG baseline before working on extensions.
- Keep baseline and innovation code separated (e.g., separate modules/directories).
- Do not claim a component works unless it has actually been executed and tested.
- Use configuration files instead of hard-coded values.
- Add type hints, docstrings, logging, and error handling to code.
- Write unit tests for major modules.
- Use patient-level splits for train/validation/test data.
- Prevent patient and study leakage across data splits.
- Never commit medical data, credentials, secrets, checkpoints, or environment files.
- Record random seeds, package versions, configs, and hardware used for each experiment.
- Inspect relevant files and propose a plan before making major edits.
- Run tests after changes and report exact results.
- Document assumptions and deviations from the paper.
