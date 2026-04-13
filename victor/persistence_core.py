# ======================================================================================================================
# FILE: victor/persistence_core.py
# UUID: c3d4e5f6-a7b8-9012-cdef-123456789012
# VERSION: v1.0.0-PERSISTENCE-CORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Persistence Core
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: SQLite-backed key/value state persistence with versioned entries and TTL support.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# DATE: 04/13/2026
# PROJECT NAME: BORON
# ======================================================================================================================

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------------------------------

class PersistenceError(Exception):
    """Base error for persistence core."""


# ----------------------------------------------------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class StateEntry:
    """A single versioned state entry stored in the persistence layer."""

    key: str
    value: Any
    version: int = 1
    # Use field(default_factory=time.time) so each instance gets its own timestamp
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttl: Optional[float] = None  # seconds; None means no expiry
    tags: List[str] = field(default_factory=list)

    def is_expired(self) -> bool:
        if self.ttl is None:
            return False
        return time.time() > self.created_at + self.ttl

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ttl": self.ttl,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StateEntry":
        return cls(
            key=data["key"],
            value=data["value"],
            version=data.get("version", 1),
            created_at=data.get("created_at", time.time()),
            updated_at=data.get("updated_at", time.time()),
            ttl=data.get("ttl"),
            tags=data.get("tags", []),
        )


# ----------------------------------------------------------------------------------------------------------------------
# PersistenceCore
# ----------------------------------------------------------------------------------------------------------------------

class PersistenceCore:
    """
    SQLite-backed key/value store for system state persistence.

    - Keys are arbitrary non-empty strings.
    - Values are JSON-serialisable Python objects.
    - Supports versioning, TTL-based expiry, and tag filtering.
    - Thread-safe via an internal :class:`threading.RLock`.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Use ``":memory:"`` for an ephemeral
        in-memory database (useful for testing).
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        # Initialise logger BEFORE _init_db() so it can be used there safely
        self._logger = logger
        self._lock = threading.RLock()
        self._db_path = db_path
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()
        self._logger.debug("PersistenceCore initialised (db_path=%s)", db_path)

    # ------------------------------------------------------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _init_db(self) -> None:
        """Open the SQLite connection and create the schema if needed."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS state_entries (
                    key         TEXT PRIMARY KEY,
                    value_json  TEXT NOT NULL,
                    version     INTEGER NOT NULL DEFAULT 1,
                    created_at  REAL NOT NULL,
                    updated_at  REAL NOT NULL,
                    ttl         REAL,
                    tags_json   TEXT NOT NULL DEFAULT '[]'
                )
                """
            )
        self._logger.debug("Database schema ready")

    def _row_to_entry(self, row: sqlite3.Row) -> StateEntry:
        return StateEntry(
            key=row["key"],
            value=json.loads(row["value_json"]),
            version=row["version"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            ttl=row["ttl"],
            tags=json.loads(row["tags_json"]),
        )

    # ------------------------------------------------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------------------------------------------------

    def put(self, entry: StateEntry) -> None:
        """Insert or replace a :class:`StateEntry`."""
        if not entry.key:
            raise PersistenceError("StateEntry key must not be empty")
        value_json = json.dumps(entry.value)
        tags_json = json.dumps(entry.tags)
        with self._lock:
            assert self._conn is not None
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO state_entries
                        (key, value_json, version, created_at, updated_at, ttl, tags_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value_json = excluded.value_json,
                        version    = excluded.version,
                        updated_at = excluded.updated_at,
                        ttl        = excluded.ttl,
                        tags_json  = excluded.tags_json
                    """,
                    (entry.key, value_json, entry.version,
                     entry.created_at, entry.updated_at, entry.ttl, tags_json),
                )
        self._logger.debug("put: key=%s version=%d", entry.key, entry.version)

    def get(self, key: str) -> Optional[StateEntry]:
        """
        Return the :class:`StateEntry` for *key*, or ``None`` if absent or expired.

        Expired entries are lazily deleted on access.
        """
        with self._lock:
            assert self._conn is not None
            row = self._conn.execute(
                "SELECT * FROM state_entries WHERE key = ?", (key,)
            ).fetchone()
            if row is None:
                return None
            entry = self._row_to_entry(row)
            if entry.is_expired():
                self._delete_key(key)
                return None
            return entry

    def delete(self, key: str) -> bool:
        """Delete *key*. Returns True if it existed."""
        with self._lock:
            return self._delete_key(key)

    def _delete_key(self, key: str) -> bool:
        assert self._conn is not None
        cursor = self._conn.execute("DELETE FROM state_entries WHERE key = ?", (key,))
        self._conn.commit()
        return cursor.rowcount > 0

    def exists(self, key: str) -> bool:
        """Return True if *key* is present (and not expired)."""
        return self.get(key) is not None

    def list_keys(self, tag: Optional[str] = None) -> List[str]:
        """
        Return all non-expired keys, optionally filtered by *tag*.
        """
        with self._lock:
            assert self._conn is not None
            rows = self._conn.execute("SELECT key, created_at, ttl, tags_json FROM state_entries").fetchall()
            now = time.time()
            keys: List[str] = []
            for row in rows:
                ttl = row["ttl"]
                if ttl is not None and now > row["created_at"] + ttl:
                    continue  # expired; skip (lazy cleanup happens on get())
                if tag is not None:
                    tags = json.loads(row["tags_json"])
                    if tag not in tags:
                        continue
                keys.append(row["key"])
            return keys

    def purge_expired(self) -> int:
        """Delete all expired entries. Returns the number of rows removed."""
        now = time.time()
        with self._lock:
            assert self._conn is not None
            cursor = self._conn.execute(
                "DELETE FROM state_entries WHERE ttl IS NOT NULL AND created_at + ttl < ?",
                (now,),
            )
            self._conn.commit()
            count = cursor.rowcount
        self._logger.debug("purge_expired: removed %d entries", count)
        return count

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
        self._logger.debug("PersistenceCore closed")
