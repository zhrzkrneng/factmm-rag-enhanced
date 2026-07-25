# FactMM-RAG Enhanced

## Description

This repository contains a reproduction and extension of **FactMM-RAG**, a
retrieval-augmented generation approach for improving factual consistency in
radiology report generation. The project first reproduces the original
paper's baseline results, then explores extensions and enhancements on top
of that baseline.

## Reproduction Objective

Faithfully reproduce the original FactMM-RAG baseline, including its
retrieval mechanism, generation pipeline, and reported evaluation metrics,
before any modifications are attempted. Reproduction results (including
deviations from the paper) will be documented in `docs/`.

## Extension Objective

Once the baseline is reproduced and validated, extend the approach with
additional innovations. Extension work is kept clearly separated from
baseline code so that the two can be evaluated and compared independently.

## Current Project Status

**Initial setup.** The repository structure has been created. No model code,
data pipelines, or experiments have been implemented yet.

## Repository Structure

```
.
├── CLAUDE.md                       # Project rules and conventions for AI-assisted development
├── README.md                       # This file
├── .gitignore
├── requirements.txt
├── configs/                        # Configuration files (YAML/JSON) for experiments
├── docs/                           # Documentation, notes, and write-ups
├── notebooks/                      # Exploratory analysis notebooks
├── paper/                          # Reference paper materials
├── reference/
│   └── original_repository/        # Vendored/reference copy of the original FactMM-RAG code
├── results/                        # Experiment outputs, metrics, logs (not committed)
├── scripts/                        # CLI entry points / run scripts
├── src/                            # Source code (baseline and extension modules)
└── tests/                          # Unit tests
```

## Installation

> Placeholder — installation instructions will be added once the codebase
> is implemented.

```bash
# python -m venv .venv
# source .venv/bin/activate
# pip install -r requirements.txt
```

## Dataset Access Warning

This project involves medical imaging and clinical text datasets (e.g.,
chest X-ray reports) that may require credentialed access (such as
PhysioNet) and are subject to data use agreements. **Do not commit any
dataset files, patient data, or protected health information (PHI) to this
repository.** Ensure you have the appropriate permissions and have signed
any required data use agreements before downloading or using such datasets.

## Citation

> Placeholder — citation information for the original FactMM-RAG paper will
> be added here.

```bibtex
@article{factmm_rag_placeholder,
  title   = {FactMM-RAG},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD}
}
```
