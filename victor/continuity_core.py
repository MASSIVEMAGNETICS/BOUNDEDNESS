# ======================================================================================================================
# FILE: victor/continuity_core.py
# UUID: d4e5f6a7-b8c9-0123-defa-234567890123
# VERSION: v1.0.0-CONTINUITY-CORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Continuity Core
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Manages episodic memory entries with embedding-based similarity retrieval and tag filtering.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# DATE: 04/13/2026
# PROJECT NAME: BORON
# ======================================================================================================================

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Optional numpy dependency for embedding similarity
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMPY_AVAILABLE = False

# ----------------------------------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------------------------------

class ContinuityError(Exception):
    """Base error for continuity core."""


# ----------------------------------------------------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class MemoryEntry:
    """A single episodic memory entry with an optional embedding vector."""

    content: str
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    # Use default_factory=list so each instance gets its own list (avoids shared-mutable default)
    tags: List[str] = field(default_factory=list)
    embedding: Optional[List[float]] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "content": self.content,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "embedding": self.embedding,
            "meta": self.meta,
        }


# ----------------------------------------------------------------------------------------------------------------------
# ContinuityCore
# ----------------------------------------------------------------------------------------------------------------------

class ContinuityCore:
    """
    Episodic memory store with optional embedding-based similarity search.

    Entries are kept in insertion order.  When a *query_embedding* is provided,
    ``retrieve_relevant`` ranks entries by cosine similarity.  If numpy is unavailable
    the comparison falls back to a simple tag/keyword scan.
    """

    def __init__(self, max_entries: int = 10_000) -> None:
        self._lock = threading.RLock()
        self._entries: List[MemoryEntry] = []
        self._max_entries = max_entries
        self._step: int = 0
        logger.debug("ContinuityCore initialised (max_entries=%d)", max_entries)

    # ------------------------------------------------------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------------------------------------------------------

    def add_entry(self, entry: MemoryEntry) -> None:
        """Append *entry* to the memory store, evicting the oldest if at capacity."""
        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self._max_entries:
                self._entries = self._entries[-self._max_entries:]
            self._step += 1
        logger.debug("add_entry: id=%s tags=%s", entry.entry_id, entry.tags)

    def record(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        embedding: Optional[List[float]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> MemoryEntry:
        """Convenience wrapper: create a :class:`MemoryEntry` and add it."""
        entry = MemoryEntry(
            content=content,
            tags=tags if tags is not None else [],
            embedding=embedding,
            meta=meta if meta is not None else {},
        )
        self.add_entry(entry)
        return entry

    # ------------------------------------------------------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------------------------------------------------------

    def retrieve_relevant(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        tag_filter: Optional[List[str]] = None,
    ) -> List[MemoryEntry]:
        """
        Return up to *top_k* entries most similar to *query_embedding*.

        Similarity is measured by cosine similarity using numpy.  Entries without an
        embedding are excluded from ranking.

        Parameters
        ----------
        query_embedding:
            The query vector to compare against stored embeddings.
        top_k:
            Maximum number of results to return.
        tag_filter:
            If provided, only entries that carry **all** of the listed tags are
            considered.
        """
        if not _NUMPY_AVAILABLE:
            raise ContinuityError(
                "numpy is required for embedding-based retrieval; "
                "install it with: pip install numpy"
            )

        with self._lock:
            candidates = list(self._entries)

        # Apply tag filter
        if tag_filter:
            tag_set = set(tag_filter)
            candidates = [e for e in candidates if tag_set.issubset(set(e.tags))]

        # Rank by cosine similarity; skip entries without an embedding
        q = np.array(query_embedding, dtype=float)
        q_norm = float(np.linalg.norm(q))
        if q_norm == 0.0:
            return candidates[:top_k]

        scored: List[tuple[float, MemoryEntry]] = []
        for entry in candidates:
            if not entry.embedding:
                continue
            c = np.array(entry.embedding, dtype=float)
            c_norm = float(np.linalg.norm(c))
            if c_norm == 0.0:
                similarity = 0.0
            else:
                similarity = float(np.dot(q, c)) / (q_norm * c_norm)
            scored.append((similarity, entry))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [e for _, e in scored[:top_k]]

    def search_by_tag(self, *tags: str) -> List[MemoryEntry]:
        """Return all entries that carry **all** of the given *tags*."""
        tag_set = set(tags)
        with self._lock:
            return [e for e in self._entries if tag_set.issubset(set(e.tags))]

    def recent(self, n: int = 10) -> List[MemoryEntry]:
        """Return the *n* most recent entries."""
        with self._lock:
            return list(self._entries[-n:])

    def all_entries(self) -> List[MemoryEntry]:
        """Return a snapshot of all entries."""
        with self._lock:
            return list(self._entries)

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()
            self._step = 0

    @property
    def step(self) -> int:
        """Total number of entries ever added (monotonically increasing)."""
        return self._step

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
