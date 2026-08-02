"""FAISS index construction and search.

Responsibility: a flat, exact inner-product FAISS index
(faiss.IndexFlatIP) over corpus embeddings, matching the official
FactMM-RAG code's own indexing approach (gen_hard_negatives.py builds
an IndexFlatIP over the corpus's text embeddings; the official
gen_embeddings.py scripts always L2-normalize before persisting, so in
practice IP over those vectors already computes cosine similarity).

Only FlatIP is implemented -- no IVF/approximate variant -- matching
what the official code actually uses; nothing beyond exact search is
paper-faithful here.

Cosine-vs-plain-inner-product is an explicit, non-silent
FaissFlatIPIndex(normalized=...) choice recorded in metadata's
similarity_type, never inferred: when normalized=True, add()/search()
additionally verify every vector is actually (near-)unit-norm, so a
caller cannot claim "cosine" while feeding un-normalized vectors.

Key ordering is preserved explicitly by this module (FAISS itself only
returns integer row indices) -- self._keys[i] is always the QueryKey
added at row i, in insertion order, across possibly multiple add()
calls.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from src.baseline.pair_mining.mining import QueryKey
from src.common.exceptions import RetrievalIndexError

_UNIT_NORM_TOLERANCE = 1e-3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _resolve_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).resolve().parent,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _resolve_faiss_version() -> Optional[str]:
    import faiss

    return getattr(faiss, "__version__", None)


@dataclass(frozen=True)
class FaissIndexMetadata:
    config_version: str
    embedding_dim: int
    normalized: bool
    similarity_type: str
    num_vectors: int
    build_timestamp: str
    git_commit: Optional[str]
    index_type: str
    faiss_version: Optional[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _validate_embeddings_2d(embeddings: np.ndarray, expected_dim: int) -> None:
    if embeddings.ndim != 2:
        raise RetrievalIndexError(
            f"embeddings must be a 2-D array [N, embedding_dim], got shape {embeddings.shape}"
        )
    if embeddings.shape[1] != expected_dim:
        raise RetrievalIndexError(
            f"embeddings dimension mismatch: index expects {expected_dim}, got {embeddings.shape[1]}"
        )
    if not np.isfinite(embeddings).all():
        raise RetrievalIndexError("embeddings contains non-finite (NaN/Inf) values")


class FaissFlatIPIndex:
    """Exact flat inner-product FAISS index with a QueryKey row mapping.

    normalized=True is a caller assertion that every vector added/
    searched is unit-norm (so inner product == cosine similarity) --
    add()/search() verify this within a small numerical tolerance and
    raise RetrievalIndexError otherwise, rather than silently returning
    a plain-dot-product score mislabeled as cosine.
    """

    def __init__(self, embedding_dim: int, *, normalized: bool, config_version: str = "1.0") -> None:
        if not isinstance(embedding_dim, int) or isinstance(embedding_dim, bool) or embedding_dim < 1:
            raise ValueError(f"embedding_dim must be a positive integer, got {embedding_dim!r}")
        if not isinstance(normalized, bool):
            raise ValueError("normalized must be a bool")
        if not config_version:
            raise ValueError("config_version must be a non-empty string")

        import faiss

        self._faiss = faiss
        self._index = faiss.IndexFlatIP(embedding_dim)
        self._embedding_dim = embedding_dim
        self._normalized = normalized
        self._config_version = config_version
        self._keys: List[QueryKey] = []

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def normalized(self) -> bool:
        return self._normalized

    @property
    def similarity_type(self) -> str:
        return "cosine" if self._normalized else "inner_product"

    @property
    def num_vectors(self) -> int:
        return len(self._keys)

    def _validate_norms(self, embeddings: np.ndarray) -> None:
        if not self._normalized:
            return
        norms = np.linalg.norm(embeddings, axis=1)
        if not np.allclose(norms, 1.0, atol=_UNIT_NORM_TOLERANCE):
            bad = np.argmax(np.abs(norms - 1.0))
            raise RetrievalIndexError(
                f"normalized=True but row {int(bad)} has norm {norms[bad]:.6f} "
                f"(expected ~1.0 within {_UNIT_NORM_TOLERANCE}) -- refusing to treat "
                f"inner product as cosine similarity for un-normalized vectors"
            )

    def add(self, embeddings: np.ndarray, keys: Sequence[QueryKey]) -> None:
        """Adds embeddings (and their QueryKeys) to the index, in order.

        Raises:
            RetrievalIndexError: embeddings is not 2-D, has the wrong
                dimension, contains a non-finite value, disagrees in
                length with keys, contains a key already present in
                this index or duplicated within this call, or (when
                normalized=True) contains a non-unit-norm row.
        """
        _validate_embeddings_2d(embeddings, self._embedding_dim)
        if embeddings.shape[0] != len(keys):
            raise RetrievalIndexError(
                f"embeddings has {embeddings.shape[0]} rows but {len(keys)} keys were given"
            )

        existing = set(self._keys)
        seen_in_call = set()
        for key in keys:
            if key in existing or key in seen_in_call:
                raise RetrievalIndexError(f"Duplicate key {key} rejected in FaissFlatIPIndex.add()")
            seen_in_call.add(key)

        self._validate_norms(embeddings)

        self._index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
        self._keys.extend(keys)

    def search(
        self, query_embeddings: np.ndarray, top_k: int
    ) -> Tuple[np.ndarray, List[List[QueryKey]]]:
        """Searches for the top_k nearest (highest inner-product) neighbors.

        top_k is clipped to self.num_vectors when it exceeds the
        index's current size (rather than raising or letting FAISS's
        own -1-padded-index behavior leak through). An empty index
        always returns empty results, regardless of top_k.

        Raises:
            RetrievalIndexError: query_embeddings is not 2-D, has the
                wrong dimension, contains a non-finite value, or (when
                normalized=True) contains a non-unit-norm row.
            ValueError: top_k is not a positive integer.
        """
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer, got {top_k!r}")

        _validate_embeddings_2d(query_embeddings, self._embedding_dim)
        self._validate_norms(query_embeddings)

        num_queries = query_embeddings.shape[0]
        if self.num_vectors == 0:
            return np.empty((num_queries, 0), dtype=np.float32), [[] for _ in range(num_queries)]

        effective_k = min(top_k, self.num_vectors)
        scores, indices = self._index.search(
            np.ascontiguousarray(query_embeddings, dtype=np.float32), effective_k
        )
        result_keys = [[self._keys[idx] for idx in row] for row in indices]
        return scores, result_keys

    def _build_metadata(self) -> FaissIndexMetadata:
        return FaissIndexMetadata(
            config_version=self._config_version,
            embedding_dim=self._embedding_dim,
            normalized=self._normalized,
            similarity_type=self.similarity_type,
            num_vectors=self.num_vectors,
            build_timestamp=_utc_now_iso(),
            git_commit=_resolve_git_commit(),
            index_type="FlatIP",
            faiss_version=_resolve_faiss_version(),
        )

    def save(self, directory: Path) -> None:
        """Writes directory/index.faiss and directory/metadata.json.

        metadata.json also carries the row -> QueryKey mapping (as
        "keys") beyond the documented metadata fields -- necessary for
        load() to reconstruct search() results as QueryKeys rather
        than bare row indices; not itself part of the requested
        metadata schema.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        self._faiss.write_index(self._index, str(directory / "index.faiss"))

        payload = self._build_metadata().to_dict()
        payload["keys"] = [list(key) for key in self._keys]
        (directory / "metadata.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> "FaissFlatIPIndex":
        """Reconstructs a FaissFlatIPIndex from directory/index.faiss +
        directory/metadata.json (see save()).

        Raises:
            RetrievalIndexError: metadata.json is missing, malformed,
                or its embedding_dim disagrees with the loaded FAISS
                index's own dimension.
        """
        directory = Path(directory)
        metadata_path = directory / "metadata.json"
        index_path = directory / "index.faiss"

        if not metadata_path.exists():
            raise RetrievalIndexError(f"Missing metadata file at {metadata_path}")
        if not index_path.exists():
            raise RetrievalIndexError(f"Missing FAISS index file at {index_path}")

        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise RetrievalIndexError(f"Malformed metadata at {metadata_path}: {exc}") from exc

        required_fields = (
            "config_version",
            "embedding_dim",
            "normalized",
            "keys",
        )
        missing = [field for field in required_fields if field not in payload]
        if missing:
            raise RetrievalIndexError(f"Metadata at {metadata_path} missing fields: {missing}")

        import faiss

        raw_index = faiss.read_index(str(index_path))
        if raw_index.d != payload["embedding_dim"]:
            raise RetrievalIndexError(
                f"Loaded FAISS index dimension ({raw_index.d}) does not match "
                f"metadata embedding_dim ({payload['embedding_dim']})"
            )

        instance = cls(
            embedding_dim=payload["embedding_dim"],
            normalized=payload["normalized"],
            config_version=payload["config_version"],
        )
        instance._index = raw_index
        instance._keys = [tuple(key) for key in payload["keys"]]
        if raw_index.ntotal != len(instance._keys):
            raise RetrievalIndexError(
                f"Loaded FAISS index has {raw_index.ntotal} vectors but metadata "
                f"lists {len(instance._keys)} keys"
            )
        return instance
