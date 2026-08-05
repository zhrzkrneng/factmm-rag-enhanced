"""Shared exception types.

Responsibility: project-wide exception classes raised loudly rather
than swallowed, per CLAUDE.md's rule against hiding errors with broad
exception handling. Each is deliberately a plain, undecorated Exception
subclass — the calling code is responsible for constructing an
informative message, not this module.
"""


class DataIntegrityError(Exception):
    """Raised when dataset integrity checks find one or more violations.

    Covers missing image files, empty/missing report text, duplicate
    (patient_id, study_id) pairs, and records with no identifiable
    image. Raised only after all issues have been collected (see
    src.data.integrity.IntegrityChecker), so the message can summarize
    every problem found rather than just the first one.
    """


class PatientLeakageError(Exception):
    """Raised when a patient_id appears in more than one dataset split.

    Indicates a train/valid/test leak that must be resolved before any
    training or evaluation on the affected splits can be trusted.
    """


class ConfigValidationError(Exception):
    """Raised when a loaded configuration file is invalid.

    Covers missing required fields and values outside an expected
    range/type.
    """


class CompatibilityError(Exception):
    """Raised when a third-party dependency's installed version doesn't
    match what a compatibility shim layer has been validated against.

    Used by src.baseline.radgraph.compat to fail loudly rather than
    silently apply patches to an unvalidated package version, which
    could reintroduce a bug that version already fixed, or mask a
    different real problem.
    """


class AnnotationError(Exception):
    """Raised when a single record fails RadGraph/CheXbert annotation for
    a reason internal to this project's own schema checks (empty input
    text, a malformed CheXbert output vector, corrupt or duplicate
    existing output) -- as opposed to whatever exception type
    radgraph/f1chexbert themselves happen to raise on a real inference
    failure, which propagates as-is rather than being wrapped here.
    """


class PairMiningError(Exception):
    """Raised when factual pair mining's own schema/config checks fail:
    a malformed or duplicate-keyed existing output line, a referenced
    positive key missing from the current candidate corpus, a stale
    existing output/metadata produced under an incompatible
    PairMiningConfig or config_version, or a positive_keys/scores
    structural mismatch -- never silently ignored or treated as an
    empty resume.
    """

class RetrievalDatasetError(Exception):
    """Raised when the retriever training dataset's own schema checks
    fail: a malformed annotated-record or pair-mining-output JSONL
    line, a duplicate composite (dataset, patient_id, study_id) key, a
    positive or hard-negative key that violates the cross-dataset
    invariant, or a referenced positive/negative key missing from the
    loaded records -- never silently dropped or treated as an empty
    dataset.
    """


class RetrievalIndexError(Exception):
    """Raised when embedding export or FAISS index construction/search
    checks fail: an empty corpus, a duplicate QueryKey, an inconsistent
    or wrong embedding dimension, a non-finite (NaN/Inf) embedding
    value, a claimed-normalized vector that is not actually unit-norm,
    or an invalid top_k -- never silently coerced or ignored.
    """


class RetrieverTrainingError(Exception):
    """Raised when the retriever training loop's own runtime checks
    fail: a non-finite loss or gradient, a batch missing the keys its
    configured training stage requires, a checkpoint that fails
    integrity verification (sha256 mismatch) or is missing a required
    field, or an embedding-dimension mismatch between a query and
    candidate encoding -- never silently skipped, retried, or trusted.
    """


class RAGDatasetError(Exception):
    """Raised when the RAG dataset builder's own schema/config checks
    fail: a query/corpus record missing a configured rag_data_mode/
    output_data_mode/image_path field, a KNN ranking referencing a
    candidate key missing from the corpus or violating the
    cross-dataset invariant, a malformed or duplicate-keyed existing
    output line, or an existing output/metadata produced under an
    incompatible RAGDatasetBuilderConfig -- never silently dropped or
    treated as an empty resume.
    """


class EvaluationError(Exception):
    """Raised when the evaluation subsystem's own schema/config checks
    fail: a malformed or duplicate-keyed reference/prediction JSONL
    line, a prediction whose query_key has no matching reference (or a
    reference with no matching prediction row), a prediction with
    neither a generated result nor an error (or both), a blank/malformed
    hypothesis text under the strict default pairing mode, a
    ref/prediction row count mismatch under positional-pairing
    reproduction mode, or a config/library-provenance mismatch in
    resume-safe metadata -- never silently dropped, coerced, or treated
    as a zero score. Real exceptions raised by an external metric
    library (radgraph/f1chexbert/rouge/evaluate/bert_score/pytrec_eval)
    are wrapped and chained here rather than swallowed.
    """
