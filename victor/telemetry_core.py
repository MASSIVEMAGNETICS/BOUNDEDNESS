# ======================================================================================================================
# FILE: victor/telemetry_core.py
# UUID: a7b8c9d0-e1f2-3456-abcd-567890123456
# VERSION: v1.0.0-TELEMETRY-CORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Telemetry Core
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Collects, aggregates, and reports system metrics and structured events.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# DATE: 04/13/2026
# PROJECT NAME: BORON
# ======================================================================================================================

from __future__ import annotations

import collections
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------------------------------

class TelemetryError(Exception):
    """Base error for telemetry core."""


# ----------------------------------------------------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class MetricEvent:
    """A single metric data point."""

    name: str
    value: float
    unit: str = ""
    timestamp: float = field(default_factory=time.time)
    tags: Dict[str, str] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }


@dataclass
class StructuredEvent:
    """A free-form structured event (log entry, lifecycle event, etc.)."""

    kind: str
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }


# ----------------------------------------------------------------------------------------------------------------------
# TelemetryCore
# ----------------------------------------------------------------------------------------------------------------------

class TelemetryCore:
    """
    Lightweight in-process telemetry collector.

    Features:
    - Record :class:`MetricEvent` data points (gauge, counter, timing).
    - Record :class:`StructuredEvent` lifecycle/diagnostic events.
    - Query recent metrics by name with optional aggregation (mean, min, max, sum).
    - Register handler callbacks that are invoked on every metric or event.
    - Thread-safe; all public methods can be called from any thread.
    """

    def __init__(self, buffer_size: int = 10_000) -> None:
        self._lock = threading.RLock()
        self._buffer_size = buffer_size
        self._metrics: Deque[MetricEvent] = collections.deque(maxlen=buffer_size)
        self._events: Deque[StructuredEvent] = collections.deque(maxlen=buffer_size)
        self._metric_handlers: List[Callable[[MetricEvent], None]] = []
        self._event_handlers: List[Callable[[StructuredEvent], None]] = []
        logger.debug("TelemetryCore initialised (buffer_size=%d)", buffer_size)

    # ------------------------------------------------------------------------------------------------------------------
    # Handler registration
    # ------------------------------------------------------------------------------------------------------------------

    def add_metric_handler(self, handler: Callable[[MetricEvent], None]) -> None:
        """Register a callback invoked synchronously on every recorded metric."""
        with self._lock:
            self._metric_handlers.append(handler)

    def add_event_handler(self, handler: Callable[[StructuredEvent], None]) -> None:
        """Register a callback invoked synchronously on every recorded structured event."""
        with self._lock:
            self._event_handlers.append(handler)

    # ------------------------------------------------------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------------------------------------------------------

    def record_metric(
        self,
        name: str,
        value: float,
        unit: str = "",
        tags: Optional[Dict[str, str]] = None,
    ) -> MetricEvent:
        """Record a metric data point and return the created :class:`MetricEvent`."""
        event = MetricEvent(name=name, value=value, unit=unit, tags=tags or {})
        with self._lock:
            self._metrics.append(event)
            handlers = list(self._metric_handlers)
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001
                logger.exception("Metric handler raised an exception")
        return event

    def record_event(self, kind: str, payload: Optional[Dict[str, Any]] = None) -> StructuredEvent:
        """Record a structured event and return the created :class:`StructuredEvent`."""
        event = StructuredEvent(kind=kind, payload=payload or {})
        with self._lock:
            self._events.append(event)
            handlers = list(self._event_handlers)
        for handler in handlers:
            try:
                handler(event)
            except Exception:  # noqa: BLE001
                logger.exception("Event handler raised an exception")
        return event

    def timing(self, name: str, seconds: float, tags: Optional[Dict[str, str]] = None) -> MetricEvent:
        """Record a timing metric in seconds."""
        return self.record_metric(name, seconds, unit="s", tags=tags)

    # ------------------------------------------------------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------------------------------------------------------

    def recent_metrics(self, name: Optional[str] = None, limit: int = 100) -> List[MetricEvent]:
        """
        Return the most recent metric events, optionally filtered to *name*.
        """
        with self._lock:
            all_m = list(self._metrics)
        if name is not None:
            all_m = [m for m in all_m if m.name == name]
        return all_m[-limit:]

    def recent_events(self, kind: Optional[str] = None, limit: int = 100) -> List[StructuredEvent]:
        """
        Return the most recent structured events, optionally filtered to *kind*.
        """
        with self._lock:
            all_e = list(self._events)
        if kind is not None:
            all_e = [e for e in all_e if e.kind == kind]
        return all_e[-limit:]

    def aggregate(
        self,
        name: str,
        method: str = "mean",
        limit: int = 100,
    ) -> Optional[float]:
        """
        Aggregate recent values for metric *name*.

        *method* is one of ``"mean"``, ``"min"``, ``"max"``, ``"sum"``, ``"count"``.
        Returns ``None`` if there are no matching metrics.
        """
        events = self.recent_metrics(name=name, limit=limit)
        if not events:
            return None
        values = [e.value for e in events]
        if method == "mean":
            return sum(values) / len(values)
        if method == "min":
            return min(values)
        if method == "max":
            return max(values)
        if method == "sum":
            return sum(values)
        if method == "count":
            return float(len(values))
        raise TelemetryError(f"Unknown aggregation method: {method!r}")

    def report(self) -> Dict[str, Any]:
        """Return a summary snapshot of current telemetry buffers."""
        with self._lock:
            metric_count = len(self._metrics)
            event_count = len(self._events)
            # Collect distinct metric names
            names: Dict[str, int] = collections.Counter(m.name for m in self._metrics)
        return {
            "metric_count": metric_count,
            "event_count": event_count,
            "metric_names": dict(names),
            "buffer_size": self._buffer_size,
        }
