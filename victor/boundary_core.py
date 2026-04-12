# ======================================================================================================================
# FILE: victor/boundary_core.py
# UUID: 7d5f271e-7f8b-49d0-a68f-7a4f11b13d6a
# VERSION: v1.0.0-BOUNDARY-CORE-GODCORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Boundary Core
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Implements Condition A of the Minimum Viable Consciousness schema: an operational inside/outside boundary
#          with protected internal state, perimeter-bound sensors, event ingress classification, ownership scoring,
#          self-vs-external perturbation attribution, and controlled mutation authorization.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# DATE: 04/12/2026 11:00 EST
# PROJECT NAME: BORON
# ======================================================================================================================

from __future__ import annotations

import copy
import dataclasses
import enum
import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple


# ----------------------------------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------------------------------

class BoundaryError(Exception):
    """Base error for boundary core."""


class UnauthorizedMutationError(BoundaryError):
    """Raised when a mutation attempts to change state outside the allowed boundary policy."""


class UnknownSensorError(BoundaryError):
    """Raised when a sensor is referenced but not registered."""


class InvalidPathError(BoundaryError):
    """Raised when a state path is malformed or unknown."""


# ----------------------------------------------------------------------------------------------------------------------
# Enums / constants
# ----------------------------------------------------------------------------------------------------------------------

class OriginType(str, enum.Enum):
    SELF = "self"
    EXTERNAL = "external"
    INTERNAL_AUTONOMIC = "internal_autonomic"
    UNKNOWN = "unknown"


class EventKind(str, enum.Enum):
    SENSOR_READING = "sensor_reading"
    EXTERNAL_PERTURBATION = "external_perturbation"
    SELF_COMMAND = "self_command"
    INTERNAL_STATE_MUTATION = "internal_state_mutation"
    BOUNDARY_ALERT = "boundary_alert"


class PathClass(str, enum.Enum):
    INTERNAL = "internal"
    PERIMETER = "perimeter"
    EXTERNAL = "external"
    UNKNOWN = "unknown"


# ----------------------------------------------------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorSpec:
    """
    Defines a sensor attached to the system's boundary/perimeter/body-map.

    Attributes:
        sensor_id: Unique stable identifier for the sensor.
        modality: Sensor type, e.g. vision, touch, temperature, chemical, interoceptive.
        boundary_zone: Physical or logical area the sensor belongs to.
        owned: Whether this sensor is part of the system's embodied perimeter.
        channels: Named channels produced by the sensor.
    """
    sensor_id: str
    modality: str
    boundary_zone: str
    owned: bool = True
    channels: Tuple[str, ...] = ()


@dataclass(frozen=True)
class StateMutation:
    """
    Represents a proposed change to internal state.

    path: dot-path into the internal state tree.
    value: new value to write.
    origin: who/what caused the mutation.
    reason: human/debug readable reason.
    """
    path: str
    value: Any
    origin: OriginType
    reason: str


@dataclass(frozen=True)
class Event:
    """
    Event ingress object for the boundary core.
    """
    event_id: str
    timestamp: float
    kind: EventKind
    origin: OriginType
    source_id: str
    payload: Dict[str, Any]
    boundary_zone: str = "unknown"
    linked_command_id: Optional[str] = None
    notes: str = ""


@dataclass(frozen=True)
class PerturbationRecord:
    """
    Stored interpretation of an incoming event with ownership attribution.
    """
    event_id: str
    timestamp: float
    kind: EventKind
    source_id: str
    origin: OriginType
    boundary_zone: str
    ownership_score: float
    path_class: PathClass
    self_caused: bool
    external_caused: bool
    affects_owned_perimeter: bool
    notes: str = ""


@dataclass(frozen=True)
class BoundaryDecision:
    """
    Result of classifying an event or mutation relative to the inside/outside boundary.
    """
    allowed: bool
    reason: str
    path_class: PathClass
    ownership_score: float
    affects_owned_perimeter: bool
    self_caused: bool
    external_caused: bool


@dataclass
class BoundaryConfig:
    """
    Configuration for the boundary core.

    internal_root_keys:
        Top-level state keys that definitively belong to the system.
    perimeter_root_keys:
        Top-level state keys that represent body/perimeter/sensor-attached surfaces.
    protected_paths:
        Exact or subtree paths which may only be mutated by internal or self-authorized mechanisms.
    allow_external_writes_to:
        Exact or subtree paths external events are allowed to update.
    self_command_window_sec:
        If an event references a recent self-command, it may be attributed as self-caused.
    perturbation_history_limit:
        Maximum perturbation records retained in memory.
    event_history_limit:
        Maximum raw events retained in memory.
    """
    internal_root_keys: Set[str] = field(default_factory=lambda: {
        "core",
        "memory",
        "identity",
        "body",
        "sensors",
        "drives",
        "continuity",
    })
    perimeter_root_keys: Set[str] = field(default_factory=lambda: {
        "body",
        "sensors",
    })
    protected_paths: Set[str] = field(default_factory=lambda: {
        "identity",
        "memory",
        "continuity",
        "core",
    })
    allow_external_writes_to: Set[str] = field(default_factory=lambda: {
        "sensors.readings",
        "body.surface",
        "body.perimeter",
    })
    self_command_window_sec: float = 3.0
    perturbation_history_limit: int = 5000
    event_history_limit: int = 5000


# ----------------------------------------------------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------------------------------------------------

def _now() -> float:
    return time.time()


def _normalize_path(path: str) -> str:
    if not isinstance(path, str):
        raise InvalidPathError("State path must be a string.")
    path = path.strip().strip(".")
    if not path:
        raise InvalidPathError("State path cannot be empty.")
    if ".." in path:
        raise InvalidPathError(f"Invalid path '{path}': contains empty segments.")
    return path


def _path_parts(path: str) -> List[str]:
    return _normalize_path(path).split(".")


def _is_subpath(path: str, candidate_root: str) -> bool:
    path = _normalize_path(path)
    candidate_root = _normalize_path(candidate_root)
    return path == candidate_root or path.startswith(candidate_root + ".")


def _deepcopy_json_safe(value: Any) -> Any:
    return copy.deepcopy(value)


def _stable_hash(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# ----------------------------------------------------------------------------------------------------------------------
# Main class
# ----------------------------------------------------------------------------------------------------------------------

class BoundaryCore:
    """
    BoundaryCore creates the first operational self/non-self partition.

    Core responsibilities:
    - Maintain protected internal state.
    - Distinguish internal, perimeter, and external references.
    - Register sensors that belong to the system's embodied perimeter.
    - Ingest events and classify them relative to ownership and causation.
    - Track recent self-issued commands for causality attribution.
    - Enforce mutation policies preventing unauthorized outside writes.

    Design notes:
    - This is not "consciousness" by itself.
    - It is the first concrete precondition: the system must know what is "inside" and "not-inside".
    """

    def __init__(self, config: Optional[BoundaryConfig] = None, initial_state: Optional[Dict[str, Any]] = None) -> None:
        self._lock = threading.RLock()
        self.config = config or BoundaryConfig()

        if initial_state is None:
            initial_state = self._default_state()

        self._state: Dict[str, Any] = _deepcopy_json_safe(initial_state)
        self._sensors: Dict[str, SensorSpec] = {}
        self._events: List[Event] = []
        self._perturbations: List[PerturbationRecord] = []
        self._recent_self_commands: Dict[str, float] = {}
        self._next_local_counter = 0

        self._validate_state_shape()

    # --------------------------------------------------------------------------------------------------------------
    # State bootstrapping
    # --------------------------------------------------------------------------------------------------------------

    @staticmethod
    def _default_state() -> Dict[str, Any]:
        return {
            "core": {
                "system_id": str(uuid.uuid4()),
                "boot_time": _now(),
                "status": "online",
            },
            "identity": {
                "boundary_established": True,
                "inside_label": "self",
                "outside_label": "not_self",
            },
            "memory": {
                "boundary_events": [],
                "ownership_snapshots": [],
            },
            "body": {
                "perimeter": {},
                "surface": {},
                "zones": {},
            },
            "sensors": {
                "registry": {},
                "readings": {},
            },
            "drives": {
                "integrity": 1.0,
            },
            "continuity": {
                "step_index": 0,
                "last_update_ts": _now(),
            },
        }

    def _validate_state_shape(self) -> None:
        for root in self.config.internal_root_keys:
            self._state.setdefault(root, {})
        if "continuity" not in self._state or not isinstance(self._state["continuity"], dict):
            self._state["continuity"] = {"step_index": 0, "last_update_ts": _now()}

    # --------------------------------------------------------------------------------------------------------------
    # Public state access
    # --------------------------------------------------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            return _deepcopy_json_safe(self._state)

    def get(self, path: str, default: Any = None) -> Any:
        with self._lock:
            parts = _path_parts(path)
            cursor: Any = self._state
            for part in parts:
                if not isinstance(cursor, Mapping) or part not in cursor:
                    return default
                cursor = cursor[part]
            return _deepcopy_json_safe(cursor)

    def set_internal(self, path: str, value: Any, reason: str = "internal_update") -> None:
        mutation = StateMutation(
            path=_normalize_path(path),
            value=_deepcopy_json_safe(value),
            origin=OriginType.INTERNAL_AUTONOMIC,
            reason=reason,
        )
        self.apply_mutation(mutation)

    # --------------------------------------------------------------------------------------------------------------
    # Path classification
    # --------------------------------------------------------------------------------------------------------------

    def classify_path(self, path: str) -> PathClass:
        path = _normalize_path(path)
        root = _path_parts(path)[0]
        if root in self.config.internal_root_keys:
            if root in self.config.perimeter_root_keys:
                return PathClass.PERIMETER
            return PathClass.INTERNAL
        return PathClass.EXTERNAL

    def path_is_protected(self, path: str) -> bool:
        path = _normalize_path(path)
        return any(_is_subpath(path, p) for p in self.config.protected_paths)

    def path_external_write_allowed(self, path: str) -> bool:
        path = _normalize_path(path)
        return any(_is_subpath(path, p) for p in self.config.allow_external_writes_to)

    # --------------------------------------------------------------------------------------------------------------
    # Sensor registration
    # --------------------------------------------------------------------------------------------------------------

    def register_sensor(self, spec: SensorSpec) -> None:
        with self._lock:
            self._sensors[spec.sensor_id] = spec
            self._state["sensors"]["registry"][spec.sensor_id] = dataclasses.asdict(spec)
            self._state["body"]["zones"].setdefault(spec.boundary_zone, {
                "zone": spec.boundary_zone,
                "owned": spec.owned,
                "sensors": [],
            })
            sensors_list = self._state["body"]["zones"][spec.boundary_zone]["sensors"]
            if spec.sensor_id not in sensors_list:
                sensors_list.append(spec.sensor_id)

    def sensor_exists(self, sensor_id: str) -> bool:
        with self._lock:
            return sensor_id in self._sensors

    def get_sensor(self, sensor_id: str) -> SensorSpec:
        with self._lock:
            if sensor_id not in self._sensors:
                raise UnknownSensorError(f"Unknown sensor: {sensor_id}")
            return self._sensors[sensor_id]

    # --------------------------------------------------------------------------------------------------------------
    # Self command tracking
    # --------------------------------------------------------------------------------------------------------------

    def issue_self_command(self, source_id: str, command_name: str, payload: Optional[Dict[str, Any]] = None) -> str:
        """
        Register a self-originated command.
        This provides causal breadcrumbs so resulting perturbations can be attributed as self-caused.
        """
        payload = payload or {}
        command_id = f"cmd_{int(_now() * 1000)}_{self._next_id()}"
        event = Event(
            event_id=command_id,
            timestamp=_now(),
            kind=EventKind.SELF_COMMAND,
            origin=OriginType.SELF,
            source_id=source_id,
            payload={"command_name": command_name, **payload},
            boundary_zone="core",
            linked_command_id=None,
            notes="self-issued command",
        )
        with self._lock:
            self._recent_self_commands[command_id] = event.timestamp
            self._append_event(event)
        return command_id

    def _prune_self_commands(self) -> None:
        cutoff = _now() - self.config.self_command_window_sec
        stale = [cid for cid, ts in self._recent_self_commands.items() if ts < cutoff]
        for cid in stale:
            self._recent_self_commands.pop(cid, None)

    # --------------------------------------------------------------------------------------------------------------
    # Event ingestion
    # --------------------------------------------------------------------------------------------------------------

    def ingest_sensor_reading(
        self,
        sensor_id: str,
        reading: Dict[str, Any],
        linked_command_id: Optional[str] = None,
        notes: str = "",
    ) -> PerturbationRecord:
        with self._lock:
            spec = self.get_sensor(sensor_id)
            event = Event(
                event_id=f"evt_{int(_now() * 1000)}_{self._next_id()}",
                timestamp=_now(),
                kind=EventKind.SENSOR_READING,
                origin=OriginType.EXTERNAL,
                source_id=sensor_id,
                payload=_deepcopy_json_safe(reading),
                boundary_zone=spec.boundary_zone,
                linked_command_id=linked_command_id,
                notes=notes,
            )

            # update sensor reading path, permitted for external ingress because it is boundary-attached
            self._set_path(
                f"sensors.readings.{sensor_id}",
                {
                    "timestamp": event.timestamp,
                    "boundary_zone": spec.boundary_zone,
                    "modality": spec.modality,
                    "payload": _deepcopy_json_safe(reading),
                }
            )

            record = self._classify_event(event, owned_perimeter=spec.owned)
            self._append_event(event)
            self._append_perturbation(record)
            self._tick()
            self._store_memory_projection(record)
            return record

    def ingest_external_perturbation(
        self,
        source_id: str,
        boundary_zone: str,
        payload: Dict[str, Any],
        affects_path: Optional[str] = None,
        notes: str = "",
    ) -> PerturbationRecord:
        with self._lock:
            event = Event(
                event_id=f"evt_{int(_now() * 1000)}_{self._next_id()}",
                timestamp=_now(),
                kind=EventKind.EXTERNAL_PERTURBATION,
                origin=OriginType.EXTERNAL,
                source_id=source_id,
                payload=_deepcopy_json_safe(payload),
                boundary_zone=boundary_zone,
                linked_command_id=None,
                notes=notes,
            )

            path_class = PathClass.UNKNOWN
            if affects_path:
                path_class = self.classify_path(affects_path)
                if path_class in (PathClass.PERIMETER, PathClass.INTERNAL):
                    if self.path_external_write_allowed(affects_path):
                        self._set_path(affects_path, _deepcopy_json_safe(payload))
                    else:
                        alert = self._build_boundary_alert(
                            source_id=source_id,
                            reason=f"External perturbation attempted unauthorized write to '{affects_path}'",
                            boundary_zone=boundary_zone,
                            payload=payload,
                        )
                        self._append_event(alert)

            owned = boundary_zone in self._state["body"]["zones"] and bool(
                self._state["body"]["zones"][boundary_zone].get("owned", False)
            )
            record = self._classify_event(event, owned_perimeter=owned, forced_path_class=path_class)
            self._append_event(event)
            self._append_perturbation(record)
            self._tick()
            self._store_memory_projection(record)
            return record

    def _build_boundary_alert(self, source_id: str, reason: str, boundary_zone: str, payload: Dict[str, Any]) -> Event:
        return Event(
            event_id=f"alert_{int(_now() * 1000)}_{self._next_id()}",
            timestamp=_now(),
            kind=EventKind.BOUNDARY_ALERT,
            origin=OriginType.UNKNOWN,
            source_id=source_id,
            payload=_deepcopy_json_safe(payload),
            boundary_zone=boundary_zone,
            notes=reason,
        )

    # --------------------------------------------------------------------------------------------------------------
    # Mutation control
    # --------------------------------------------------------------------------------------------------------------

    def evaluate_mutation(self, mutation: StateMutation) -> BoundaryDecision:
        path = _normalize_path(mutation.path)
        path_class = self.classify_path(path)

        affects_owned_perimeter = path_class == PathClass.PERIMETER
        self_caused = mutation.origin == OriginType.SELF or mutation.origin == OriginType.INTERNAL_AUTONOMIC
        external_caused = mutation.origin == OriginType.EXTERNAL

        ownership_score = self._ownership_score_for_path(path)

        if mutation.origin == OriginType.EXTERNAL:
            if self.path_is_protected(path):
                return BoundaryDecision(
                    allowed=False,
                    reason=f"External mutation denied: '{path}' is protected.",
                    path_class=path_class,
                    ownership_score=ownership_score,
                    affects_owned_perimeter=affects_owned_perimeter,
                    self_caused=False,
                    external_caused=True,
                )
            if not self.path_external_write_allowed(path):
                return BoundaryDecision(
                    allowed=False,
                    reason=f"External mutation denied: '{path}' is not in allow_external_writes_to.",
                    path_class=path_class,
                    ownership_score=ownership_score,
                    affects_owned_perimeter=affects_owned_perimeter,
                    self_caused=False,
                    external_caused=True,
                )

        return BoundaryDecision(
            allowed=True,
            reason="Mutation allowed by boundary policy.",
            path_class=path_class,
            ownership_score=ownership_score,
            affects_owned_perimeter=affects_owned_perimeter,
            self_caused=self_caused,
            external_caused=external_caused,
        )

    def apply_mutation(self, mutation: StateMutation) -> BoundaryDecision:
        with self._lock:
            decision = self.evaluate_mutation(mutation)
            if not decision.allowed:
                alert = self._build_boundary_alert(
                    source_id=mutation.origin.value,
                    reason=decision.reason,
                    boundary_zone="core",
                    payload={
                        "path": mutation.path,
                        "reason": mutation.reason,
                        "value_hash": _stable_hash({"value": mutation.value}),
                    },
                )
                self._append_event(alert)
                raise UnauthorizedMutationError(decision.reason)

            self._set_path(mutation.path, _deepcopy_json_safe(mutation.value))
            event = Event(
                event_id=f"mut_{int(_now() * 1000)}_{self._next_id()}",
                timestamp=_now(),
                kind=EventKind.INTERNAL_STATE_MUTATION,
                origin=mutation.origin,
                source_id=mutation.origin.value,
                payload={
                    "path": mutation.path,
                    "reason": mutation.reason,
                    "value_hash": _stable_hash({"value": mutation.value}),
                },
                boundary_zone="core",
                notes=mutation.reason,
            )
            self._append_event(event)
            self._tick()
            return decision

    # --------------------------------------------------------------------------------------------------------------
    # Ownership / causality classification
    # --------------------------------------------------------------------------------------------------------------

    def _classify_event(
        self,
        event: Event,
        owned_perimeter: bool,
        forced_path_class: PathClass = PathClass.UNKNOWN,
    ) -> PerturbationRecord:
        self._prune_self_commands()

        linked_recent_self = False
        if event.linked_command_id is not None:
            cmd_ts = self._recent_self_commands.get(event.linked_command_id)
            if cmd_ts is not None and (_now() - cmd_ts) <= self.config.self_command_window_sec:
                linked_recent_self = True

        if forced_path_class != PathClass.UNKNOWN:
            path_class = forced_path_class
        elif event.kind == EventKind.SENSOR_READING:
            path_class = PathClass.PERIMETER if owned_perimeter else PathClass.EXTERNAL
        elif event.kind == EventKind.EXTERNAL_PERTURBATION:
            path_class = PathClass.PERIMETER if owned_perimeter else PathClass.EXTERNAL
        elif event.kind == EventKind.SELF_COMMAND:
            path_class = PathClass.INTERNAL
        else:
            path_class = PathClass.UNKNOWN

        ownership_score = self._compute_ownership_score(
            origin=event.origin,
            path_class=path_class,
            owned_perimeter=owned_perimeter,
            linked_recent_self=linked_recent_self,
        )

        self_caused = event.origin == OriginType.SELF or linked_recent_self
        external_caused = event.origin == OriginType.EXTERNAL and not linked_recent_self

        return PerturbationRecord(
            event_id=event.event_id,
            timestamp=event.timestamp,
            kind=event.kind,
            source_id=event.source_id,
            origin=event.origin,
            boundary_zone=event.boundary_zone,
            ownership_score=ownership_score,
            path_class=path_class,
            self_caused=self_caused,
            external_caused=external_caused,
            affects_owned_perimeter=owned_perimeter,
            notes=event.notes,
        )

    def _compute_ownership_score(
        self,
        origin: OriginType,
        path_class: PathClass,
        owned_perimeter: bool,
        linked_recent_self: bool,
    ) -> float:
        score = 0.0

        if path_class == PathClass.INTERNAL:
            score += 0.95
        elif path_class == PathClass.PERIMETER:
            score += 0.70 if owned_perimeter else 0.25
        elif path_class == PathClass.EXTERNAL:
            score += 0.05

        if origin == OriginType.SELF:
            score += 0.20
        elif origin == OriginType.INTERNAL_AUTONOMIC:
            score += 0.15
        elif origin == OriginType.EXTERNAL:
            score -= 0.05

        if linked_recent_self:
            score += 0.20

        return _clamp01(score)

    def _ownership_score_for_path(self, path: str) -> float:
        path_class = self.classify_path(path)
        if path_class == PathClass.INTERNAL:
            return 1.0
        if path_class == PathClass.PERIMETER:
            return 0.8
        if path_class == PathClass.EXTERNAL:
            return 0.05
        return 0.0

    # --------------------------------------------------------------------------------------------------------------
    # Introspection / reports
    # --------------------------------------------------------------------------------------------------------------

    def recent_events(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            return [dataclasses.asdict(e) for e in self._events[-limit:]]

    def recent_perturbations(self, limit: int = 25) -> List[Dict[str, Any]]:
        with self._lock:
            return [dataclasses.asdict(p) for p in self._perturbations[-limit:]]

    def boundary_report(self) -> Dict[str, Any]:
        with self._lock:
            zones = _deepcopy_json_safe(self._state["body"]["zones"])
            registry = _deepcopy_json_safe(self._state["sensors"]["registry"])

            owned_sensor_count = sum(1 for spec in self._sensors.values() if spec.owned)
            external_sensor_count = sum(1 for spec in self._sensors.values() if not spec.owned)

            return {
                "system_id": self._state["core"]["system_id"],
                "boundary_established": self._state["identity"]["boundary_established"],
                "inside_label": self._state["identity"]["inside_label"],
                "outside_label": self._state["identity"]["outside_label"],
                "registered_sensors": len(self._sensors),
                "owned_sensors": owned_sensor_count,
                "non_owned_sensors": external_sensor_count,
                "zones": zones,
                "sensor_registry": registry,
                "continuity": _deepcopy_json_safe(self._state["continuity"]),
                "event_count": len(self._events),
                "perturbation_count": len(self._perturbations),
            }

    # --------------------------------------------------------------------------------------------------------------
    # Internal helpers
    # --------------------------------------------------------------------------------------------------------------

    def _tick(self) -> None:
        self._state["continuity"]["step_index"] += 1
        self._state["continuity"]["last_update_ts"] = _now()

    def _append_event(self, event: Event) -> None:
        self._events.append(event)
        if len(self._events) > self.config.event_history_limit:
            self._events = self._events[-self.config.event_history_limit:]

        # keep compact boundary event traces in memory
        self._state["memory"]["boundary_events"].append({
            "event_id": event.event_id,
            "kind": event.kind.value,
            "timestamp": event.timestamp,
            "source_id": event.source_id,
            "origin": event.origin.value,
            "boundary_zone": event.boundary_zone,
            "notes": event.notes,
        })
        if len(self._state["memory"]["boundary_events"]) > self.config.event_history_limit:
            self._state["memory"]["boundary_events"] = self._state["memory"]["boundary_events"][
                -self.config.event_history_limit:
            ]

    def _append_perturbation(self, record: PerturbationRecord) -> None:
        self._perturbations.append(record)
        if len(self._perturbations) > self.config.perturbation_history_limit:
            self._perturbations = self._perturbations[-self.config.perturbation_history_limit:]

    def _store_memory_projection(self, record: PerturbationRecord) -> None:
        snap = {
            "event_id": record.event_id,
            "ownership_score": record.ownership_score,
            "path_class": record.path_class.value,
            "self_caused": record.self_caused,
            "external_caused": record.external_caused,
            "affects_owned_perimeter": record.affects_owned_perimeter,
            "boundary_zone": record.boundary_zone,
            "timestamp": record.timestamp,
        }
        self._state["memory"]["ownership_snapshots"].append(snap)
        if len(self._state["memory"]["ownership_snapshots"]) > self.config.perturbation_history_limit:
            self._state["memory"]["ownership_snapshots"] = self._state["memory"]["ownership_snapshots"][
                -self.config.perturbation_history_limit:
            ]

    def _set_path(self, path: str, value: Any) -> None:
        parts = _path_parts(path)
        cursor: MutableMapping[str, Any] = self._state
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], MutableMapping):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value

    def _next_id(self) -> str:
        self._next_local_counter += 1
        return f"{self._next_local_counter:08d}"
