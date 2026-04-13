# ======================================================================================================================
# FILE: boundedness_core.py
# UUID: a3f92b1d-6e4c-4d0f-b8a2-9c1e5f347210
# VERSION: v1.0.0-BOUNDEDNESS-CORE-GODCORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Boundedness Core
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Modular Boundedness core implementing boundary, homeostasis, sensor event management, and sensorimotor
#          core logic. Provides a reusable foundation for system boundary management and physical/virtual
#          embodiment support by composing BoundaryCore with HomeostasisCore and SensorimotorCore.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# DATE: 04/13/2026
# PROJECT NAME: BORON
# ======================================================================================================================

from __future__ import annotations

import copy
import dataclasses
import enum
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from victor.boundary_core import (
    BoundaryConfig,
    BoundaryCore,
    BoundaryDecision,
    BoundaryError,
    EventKind,
    InvalidPathError,
    OriginType,
    PathClass,
    PerturbationRecord,
    SensorSpec,
    StateMutation,
    UnauthorizedMutationError,
    UnknownSensorError,
)

__all__ = [
    # Re-exports from boundary_core
    "BoundaryConfig",
    "BoundaryCore",
    "BoundaryDecision",
    "BoundaryError",
    "EventKind",
    "InvalidPathError",
    "OriginType",
    "PathClass",
    "PerturbationRecord",
    "SensorSpec",
    "StateMutation",
    "UnauthorizedMutationError",
    "UnknownSensorError",
    # Homeostasis
    "DriveSpec",
    "HomeostasisSignal",
    "HomeostasisCore",
    # Sensorimotor
    "MotorCommand",
    "SensorimotorLoop",
    "MotorEvent",
    "SensorimotorCore",
    # Unified compositor
    "BoundednessCore",
]


# ----------------------------------------------------------------------------------------------------------------------
# Homeostasis
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class DriveSpec:
    """
    Specification for a regulated homeostatic drive variable.

    Attributes:
        drive_id: Unique stable identifier for the drive.
        setpoint: Target equilibrium value in the range [min_val, max_val].
        tolerance: Acceptable deviation from setpoint before an urgency signal is raised.
        min_val: Lower bound of the drive's physical range.
        max_val: Upper bound of the drive's physical range.
        label: Human-readable description of what this drive represents.
    """
    drive_id: str
    setpoint: float
    tolerance: float = 0.05
    min_val: float = 0.0
    max_val: float = 1.0
    label: str = ""


@dataclass(frozen=True)
class HomeostasisSignal:
    """
    Signal generated each time a drive value is updated, capturing deviation from setpoint.

    Attributes:
        drive_id: Identifier of the regulated drive.
        timestamp: Wall-clock time of measurement.
        current_value: Observed (clamped) value.
        setpoint: Target equilibrium value.
        error: Signed deviation: current_value - setpoint.
        abs_error: Absolute deviation from setpoint.
        in_tolerance: True when abs_error <= tolerance.
        urgency: Normalized urgency in [0, 1] — 0 means at setpoint, 1 means at range boundary.
        label: Human-readable drive description.
    """
    drive_id: str
    timestamp: float
    current_value: float
    setpoint: float
    error: float
    abs_error: float
    in_tolerance: bool
    urgency: float
    label: str = ""


class HomeostasisCore:
    """
    Maintains regulated homeostatic drive variables.

    Responsibilities:
    - Register named drive variables with setpoints and physical ranges.
    - Accept updated measurements and compute signed/absolute error signals.
    - Emit HomeostasisSignal records with urgency scores.
    - Report overall homeostatic integrity (1.0 = all drives at setpoint).
    """

    def __init__(self, history_limit: int = 2000) -> None:
        self._lock = threading.RLock()
        self._drives: Dict[str, DriveSpec] = {}
        self._values: Dict[str, float] = {}
        self._history: List[HomeostasisSignal] = []
        self._history_limit = history_limit

    def register_drive(self, spec: DriveSpec) -> None:
        """Register a new drive, initialised at its setpoint."""
        with self._lock:
            self._drives[spec.drive_id] = spec
            self._values[spec.drive_id] = spec.setpoint

    def drive_exists(self, drive_id: str) -> bool:
        with self._lock:
            return drive_id in self._drives

    def update_drive(self, drive_id: str, value: float) -> HomeostasisSignal:
        """
        Update a drive with a new observed value and return the resulting signal.

        Raises:
            KeyError: If drive_id has not been registered.
        """
        with self._lock:
            if drive_id not in self._drives:
                raise KeyError(f"Unknown homeostatic drive: {drive_id!r}")
            spec = self._drives[drive_id]
            clamped = max(spec.min_val, min(spec.max_val, value))
            self._values[drive_id] = clamped
            error = clamped - spec.setpoint
            abs_error = abs(error)
            in_tolerance = abs_error <= spec.tolerance
            drive_range = max(spec.max_val - spec.min_val, 1e-9)
            # Urgency is how far from centre (setpoint) the value is relative to half the range.
            urgency = min(1.0, abs_error / (drive_range / 2.0))
            sig = HomeostasisSignal(
                drive_id=drive_id,
                timestamp=time.time(),
                current_value=clamped,
                setpoint=spec.setpoint,
                error=error,
                abs_error=abs_error,
                in_tolerance=in_tolerance,
                urgency=urgency,
                label=spec.label,
            )
            self._history.append(sig)
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit:]
            return sig

    def current_values(self) -> Dict[str, float]:
        """Return a copy of the current drive values."""
        with self._lock:
            return dict(self._values)

    def integrity(self) -> float:
        """
        Overall homeostatic integrity: mean of (1 - urgency) across all registered drives.
        Returns 1.0 when no drives are registered or all drives are at setpoint.
        """
        with self._lock:
            if not self._drives:
                return 1.0
            urgencies: List[float] = []
            for drive_id, spec in self._drives.items():
                val = self._values.get(drive_id, spec.setpoint)
                abs_error = abs(val - spec.setpoint)
                drive_range = max(spec.max_val - spec.min_val, 1e-9)
                urgency = min(1.0, abs_error / (drive_range / 2.0))
                urgencies.append(urgency)
            return max(0.0, 1.0 - sum(urgencies) / len(urgencies))

    def recent_signals(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Return recent HomeostasisSignal records as plain dicts."""
        with self._lock:
            return [dataclasses.asdict(s) for s in self._history[-limit:]]

    def report(self) -> Dict[str, Any]:
        """Return a structured summary of all drives and overall integrity."""
        with self._lock:
            return {
                "drives": {
                    did: {
                        "current_value": self._values.get(did, spec.setpoint),
                        "setpoint": spec.setpoint,
                        "tolerance": spec.tolerance,
                        "min_val": spec.min_val,
                        "max_val": spec.max_val,
                        "label": spec.label,
                    }
                    for did, spec in self._drives.items()
                },
                "integrity": self.integrity(),
                "signal_count": len(self._history),
            }


# ----------------------------------------------------------------------------------------------------------------------
# Sensorimotor
# ----------------------------------------------------------------------------------------------------------------------

class MotorCommand(str, enum.Enum):
    """Canonical motor command vocabulary."""
    INCREASE = "increase"
    DECREASE = "decrease"
    HOLD = "hold"
    ACTIVATE = "activate"
    DEACTIVATE = "deactivate"


@dataclass
class SensorimotorLoop:
    """
    Maps a sensor modality to a motor channel with an optional response policy function.

    Attributes:
        loop_id: Unique identifier for this loop.
        sensor_modality: The sensor modality (e.g. "temperature", "vision") this loop listens to.
        motor_channel: The actuator/motor channel this loop drives.
        response_fn: Optional callable (payload -> MotorCommand | None).  When None, HOLD is assumed.
    """
    loop_id: str
    sensor_modality: str
    motor_channel: str
    response_fn: Optional[Callable[[Dict[str, Any]], Optional[MotorCommand]]] = field(
        default=None, compare=False, repr=False
    )


@dataclass(frozen=True)
class MotorEvent:
    """
    A motor output event generated by the sensorimotor core in response to a sensor reading.

    Attributes:
        event_id: Unique event identifier.
        timestamp: Wall-clock time the motor command was generated.
        loop_id: The sensorimotor loop that produced this event.
        motor_channel: Target actuator channel.
        command: The motor command issued.
        sensor_modality: Modality of the triggering sensor reading.
        trigger_payload: Copy of the sensor payload that triggered this event.
        efferent_copy: Always True; marks this event as a tracked self-issued signal.
    """
    event_id: str
    timestamp: float
    loop_id: str
    motor_channel: str
    command: MotorCommand
    sensor_modality: str
    trigger_payload: Dict[str, Any]
    efferent_copy: bool = True


class SensorimotorCore:
    """
    Coordinates sensor readings with motor/actuator channels.

    Responsibilities:
    - Register sensorimotor loops mapping sensor modalities to motor channels.
    - Process incoming sensor readings and route them through matching loops.
    - Store efferent copies of all issued motor commands for reafference tracking.
    """

    def __init__(self, event_limit: int = 2000) -> None:
        self._lock = threading.RLock()
        self._loops: Dict[str, SensorimotorLoop] = {}
        self._motor_events: List[MotorEvent] = []
        self._event_limit = event_limit
        self._counter = 0

    def register_loop(self, loop: SensorimotorLoop) -> None:
        """Register a sensorimotor loop."""
        with self._lock:
            self._loops[loop.loop_id] = loop

    def loop_exists(self, loop_id: str) -> bool:
        with self._lock:
            return loop_id in self._loops

    def process_sensor_reading(
        self, modality: str, payload: Dict[str, Any]
    ) -> List[MotorEvent]:
        """
        Process a sensor reading and generate motor events for every matching loop.

        For each loop whose sensor_modality matches, the loop's response_fn is called
        (if set) to determine the MotorCommand.  If response_fn is None, MotorCommand.HOLD
        is used.  Loops whose response_fn returns None produce no event.

        Returns:
            List of MotorEvent records (may be empty).
        """
        events: List[MotorEvent] = []
        with self._lock:
            for loop in self._loops.values():
                if loop.sensor_modality != modality:
                    continue
                if loop.response_fn is not None:
                    command = loop.response_fn(payload)
                else:
                    command = MotorCommand.HOLD
                if command is None:
                    continue
                self._counter += 1
                ev = MotorEvent(
                    event_id=f"mev_{int(time.time() * 1000)}_{self._counter:08d}",
                    timestamp=time.time(),
                    loop_id=loop.loop_id,
                    motor_channel=loop.motor_channel,
                    command=command,
                    sensor_modality=modality,
                    trigger_payload=copy.deepcopy(payload),
                    efferent_copy=True,
                )
                self._motor_events.append(ev)
                if len(self._motor_events) > self._event_limit:
                    self._motor_events = self._motor_events[-self._event_limit:]
                events.append(ev)
        return events

    def recent_motor_events(self, limit: int = 25) -> List[Dict[str, Any]]:
        """Return recent MotorEvent records as plain dicts."""
        with self._lock:
            return [dataclasses.asdict(e) for e in self._motor_events[-limit:]]

    def report(self) -> Dict[str, Any]:
        """Return a structured summary of registered loops and motor event count."""
        with self._lock:
            return {
                "registered_loops": len(self._loops),
                "loops": {
                    lid: {
                        "sensor_modality": lp.sensor_modality,
                        "motor_channel": lp.motor_channel,
                        "has_response_fn": lp.response_fn is not None,
                    }
                    for lid, lp in self._loops.items()
                },
                "motor_event_count": len(self._motor_events),
            }


# ----------------------------------------------------------------------------------------------------------------------
# Unified compositor
# ----------------------------------------------------------------------------------------------------------------------

class BoundednessCore:
    """
    Modular Boundedness core: composes BoundaryCore, HomeostasisCore, and SensorimotorCore
    into a single unified API.

    Roles:
    - BoundaryCore  — inside/outside boundary, ownership scoring, mutation policy enforcement.
    - HomeostasisCore — regulated drive variables with setpoints and error/urgency signals.
    - SensorimotorCore — sensor-to-motor loop routing with efferent-copy tracking.

    This class is the primary entry point for applications that require full embodiment support:
    physical or virtual systems that must simultaneously maintain an operational self/non-self
    boundary, regulate internal drives, and coordinate sensorimotor reflexes.
    """

    def __init__(
        self,
        boundary_config: Optional[BoundaryConfig] = None,
        initial_state: Optional[Dict[str, Any]] = None,
        homeostasis_history_limit: int = 2000,
        sensorimotor_event_limit: int = 2000,
    ) -> None:
        self.boundary = BoundaryCore(config=boundary_config, initial_state=initial_state)
        self.homeostasis = HomeostasisCore(history_limit=homeostasis_history_limit)
        self.sensorimotor = SensorimotorCore(event_limit=sensorimotor_event_limit)

    # ------------------------------------------------------------------------------------------------------------------
    # Registration helpers
    # ------------------------------------------------------------------------------------------------------------------

    def register_sensor(self, spec: SensorSpec) -> None:
        """Register a sensor with the boundary layer."""
        self.boundary.register_sensor(spec)

    def register_drive(self, spec: DriveSpec) -> None:
        """Register a homeostatic drive variable."""
        self.homeostasis.register_drive(spec)

    def register_loop(self, loop: SensorimotorLoop) -> None:
        """Register a sensorimotor loop."""
        self.sensorimotor.register_loop(loop)

    # ------------------------------------------------------------------------------------------------------------------
    # Sensor ingestion
    # ------------------------------------------------------------------------------------------------------------------

    def ingest_sensor_reading(
        self,
        sensor_id: str,
        reading: Dict[str, Any],
        linked_command_id: Optional[str] = None,
        notes: str = "",
        drive_updates: Optional[Dict[str, float]] = None,
    ) -> Tuple[PerturbationRecord, List[HomeostasisSignal], List[MotorEvent]]:
        """
        Process a sensor reading through all three subsystems.

        Steps:
        1. Classify the reading via BoundaryCore and produce a PerturbationRecord.
        2. Optionally update homeostatic drives (drive_updates: {drive_id: new_value}).
        3. Route the reading through matching SensorimotorCore loops.

        Args:
            sensor_id: Registered sensor identifier.
            reading: Sensor payload dictionary.
            linked_command_id: Optional ID of a prior self-command for causality attribution.
            notes: Human-readable annotation.
            drive_updates: Optional mapping of drive_id -> new value to apply to HomeostasisCore.

        Returns:
            A 3-tuple of (PerturbationRecord, List[HomeostasisSignal], List[MotorEvent]).
        """
        perturbation = self.boundary.ingest_sensor_reading(
            sensor_id=sensor_id,
            reading=reading,
            linked_command_id=linked_command_id,
            notes=notes,
        )

        homeo_signals: List[HomeostasisSignal] = []
        if drive_updates:
            for drive_id, value in drive_updates.items():
                if self.homeostasis.drive_exists(drive_id):
                    sig = self.homeostasis.update_drive(drive_id, value)
                    homeo_signals.append(sig)

        spec = self.boundary.get_sensor(sensor_id)
        motor_events = self.sensorimotor.process_sensor_reading(
            modality=spec.modality,
            payload=reading,
        )

        return perturbation, homeo_signals, motor_events

    def ingest_external_perturbation(
        self,
        source_id: str,
        boundary_zone: str,
        payload: Dict[str, Any],
        affects_path: Optional[str] = None,
        notes: str = "",
        drive_updates: Optional[Dict[str, float]] = None,
    ) -> Tuple[PerturbationRecord, List[HomeostasisSignal]]:
        """
        Process an external perturbation through the boundary and homeostasis subsystems.

        Args:
            source_id: Identifier of the external source.
            boundary_zone: Zone where the perturbation acts.
            payload: Perturbation payload.
            affects_path: Optional dot-path of the internal state affected.
            notes: Human-readable annotation.
            drive_updates: Optional mapping of drive_id -> new value.

        Returns:
            A 2-tuple of (PerturbationRecord, List[HomeostasisSignal]).
        """
        perturbation = self.boundary.ingest_external_perturbation(
            source_id=source_id,
            boundary_zone=boundary_zone,
            payload=payload,
            affects_path=affects_path,
            notes=notes,
        )

        homeo_signals: List[HomeostasisSignal] = []
        if drive_updates:
            for drive_id, value in drive_updates.items():
                if self.homeostasis.drive_exists(drive_id):
                    sig = self.homeostasis.update_drive(drive_id, value)
                    homeo_signals.append(sig)

        return perturbation, homeo_signals

    # ------------------------------------------------------------------------------------------------------------------
    # Command / mutation passthrough
    # ------------------------------------------------------------------------------------------------------------------

    def issue_self_command(
        self,
        source_id: str,
        command_name: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Issue a self-originated command and return its command ID."""
        return self.boundary.issue_self_command(source_id, command_name, payload)

    def apply_mutation(self, mutation: StateMutation) -> BoundaryDecision:
        """Apply a state mutation through boundary policy enforcement."""
        return self.boundary.apply_mutation(mutation)

    # ------------------------------------------------------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------------------------------------------------------

    def system_report(self) -> Dict[str, Any]:
        """Return a unified report spanning all three subsystems."""
        return {
            "boundary": self.boundary.boundary_report(),
            "homeostasis": self.homeostasis.report(),
            "sensorimotor": self.sensorimotor.report(),
        }

    def snapshot(self) -> Dict[str, Any]:
        """Return a deep copy of the boundary internal state."""
        return self.boundary.snapshot()


# ----------------------------------------------------------------------------------------------------------------------
# Embedded demo
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    def pretty(title: str, obj: Any) -> None:
        print(f"\n=== {title} ===")
        print(json.dumps(obj, indent=2, sort_keys=True, default=str))

    # ------------------------------------------------------------------
    # Build the system
    # ------------------------------------------------------------------
    core = BoundednessCore()

    # Register sensors attached to the system's embodied perimeter
    core.register_sensor(SensorSpec(
        sensor_id="thermo_skin",
        modality="temperature",
        boundary_zone="skin_surface",
        owned=True,
        channels=("celsius",),
    ))
    core.register_sensor(SensorSpec(
        sensor_id="cam_front",
        modality="vision",
        boundary_zone="front_surface",
        owned=True,
        channels=("rgb",),
    ))
    core.register_sensor(SensorSpec(
        sensor_id="intero_heart",
        modality="interoceptive",
        boundary_zone="internal_viscera",
        owned=True,
        channels=("bpm",),
    ))
    core.register_sensor(SensorSpec(
        sensor_id="net_feed",
        modality="remote_feed",
        boundary_zone="remote_network",
        owned=False,
        channels=("text",),
    ))

    # Register homeostatic drives
    core.register_drive(DriveSpec(
        drive_id="body_temp",
        setpoint=0.5,
        tolerance=0.05,
        min_val=0.0,
        max_val=1.0,
        label="core temperature (normalized)",
    ))
    core.register_drive(DriveSpec(
        drive_id="energy",
        setpoint=0.7,
        tolerance=0.10,
        min_val=0.0,
        max_val=1.0,
        label="energy reserve",
    ))
    core.register_drive(DriveSpec(
        drive_id="heart_rate",
        setpoint=0.4,
        tolerance=0.08,
        min_val=0.0,
        max_val=1.0,
        label="heart rate (normalized, 30-150 bpm)",
    ))

    # Register sensorimotor loops
    def _temp_response(payload: Dict[str, Any]) -> Optional[MotorCommand]:
        """Simple bang-bang thermoregulation: activate cooling above 38 °C, warming below 36 °C."""
        celsius = payload.get("celsius")
        if celsius is None:
            return None
        if celsius > 38.0:
            return MotorCommand.ACTIVATE   # cooling system on
        if celsius < 36.0:
            return MotorCommand.DEACTIVATE  # cooling system off (warming by default)
        return MotorCommand.HOLD

    core.register_loop(SensorimotorLoop(
        loop_id="thermo_to_cooling",
        sensor_modality="temperature",
        motor_channel="cooling_system",
        response_fn=_temp_response,
    ))

    def _vision_response(payload: Dict[str, Any]) -> Optional[MotorCommand]:
        """Increase attentional gain when scene confidence is high."""
        if payload.get("confidence", 0.0) > 0.8:
            return MotorCommand.INCREASE
        return MotorCommand.HOLD

    core.register_loop(SensorimotorLoop(
        loop_id="vision_to_attention",
        sensor_modality="vision",
        motor_channel="attentional_gain",
        response_fn=_vision_response,
    ))

    pretty("INITIAL SYSTEM REPORT", core.system_report())

    # ------------------------------------------------------------------
    # Self-command causality
    # ------------------------------------------------------------------
    cmd_id = core.issue_self_command(
        source_id="motor_controller",
        command_name="scan_environment",
        payload={"mode": "wide"},
    )
    print(f"\nIssued self-command: {cmd_id}")

    # Vision reading linked to the prior self-command (self-caused)
    pert1, homeo1, motors1 = core.ingest_sensor_reading(
        sensor_id="cam_front",
        reading={"objects": ["doorway", "chair"], "confidence": 0.92},
        linked_command_id=cmd_id,
        notes="scene scan after self-command",
    )
    pretty("BOUNDARY PERTURBATION — vision (self-caused)", dataclasses.asdict(pert1))
    pretty("MOTOR EVENTS — vision -> attentional gain", [dataclasses.asdict(e) for e in motors1])

    # ------------------------------------------------------------------
    # Thermoregulation
    # ------------------------------------------------------------------
    pert2, homeo2, motors2 = core.ingest_sensor_reading(
        sensor_id="thermo_skin",
        reading={"celsius": 38.6},
        notes="elevated skin temperature",
        drive_updates={"body_temp": (38.6 - 34.0) / 6.0},  # normalize 34–40 °C range
    )
    pretty("BOUNDARY PERTURBATION — temperature", dataclasses.asdict(pert2))
    if homeo2:
        pretty("HOMEOSTASIS SIGNAL — body_temp", dataclasses.asdict(homeo2[0]))
    if motors2:
        pretty("MOTOR EVENT — thermo -> cooling", dataclasses.asdict(motors2[0]))

    # ------------------------------------------------------------------
    # Interoception
    # ------------------------------------------------------------------
    pert3, homeo3, motors3 = core.ingest_sensor_reading(
        sensor_id="intero_heart",
        reading={"bpm": 58},
        notes="interoceptive heart rate reading",
        drive_updates={"heart_rate": (58 - 30) / 120.0},  # normalize 30–150 bpm range
    )
    pretty("BOUNDARY PERTURBATION — interoception", dataclasses.asdict(pert3))
    if homeo3:
        pretty("HOMEOSTASIS SIGNAL — heart_rate", dataclasses.asdict(homeo3[0]))

    # ------------------------------------------------------------------
    # External perturbation (non-self origin) with homeostasis update
    # ------------------------------------------------------------------
    pert4, homeo4 = core.ingest_external_perturbation(
        source_id="thermal_environment",
        boundary_zone="skin_surface",
        payload={"heat_flux_w_m2": 45.0},
        affects_path="body.surface.skin_surface",
        notes="ambient heat load on skin",
        drive_updates={"energy": 0.55},  # energy expenditure response
    )
    pretty("BOUNDARY PERTURBATION — external heat", dataclasses.asdict(pert4))
    if homeo4:
        pretty("HOMEOSTASIS SIGNAL — energy", dataclasses.asdict(homeo4[0]))

    # ------------------------------------------------------------------
    # Non-owned remote feed (classified as external/not-self)
    # ------------------------------------------------------------------
    pert5, homeo5, motors5 = core.ingest_sensor_reading(
        sensor_id="net_feed",
        reading={"headline": "External world event stream"},
        notes="non-owned remote data source",
    )
    pretty("BOUNDARY PERTURBATION — remote feed (non-self)", dataclasses.asdict(pert5))

    # ------------------------------------------------------------------
    # Mutation control
    # ------------------------------------------------------------------
    decision_ok = core.apply_mutation(StateMutation(
        path="core.status",
        value="alert",
        origin=OriginType.INTERNAL_AUTONOMIC,
        reason="homeostatic integrity drop detected",
    ))
    pretty("AUTHORIZED INTERNAL MUTATION", dataclasses.asdict(decision_ok))

    try:
        core.apply_mutation(StateMutation(
            path="identity.inside_label",
            value="compromised",
            origin=OriginType.EXTERNAL,
            reason="external identity rewrite attempt",
        ))
    except UnauthorizedMutationError as exc:
        print(f"\nExternal mutation correctly blocked: {exc}")

    # ------------------------------------------------------------------
    # Final reports
    # ------------------------------------------------------------------
    pretty("HOMEOSTASIS REPORT", core.homeostasis.report())
    pretty("SENSORIMOTOR REPORT", core.sensorimotor.report())
    pretty("FINAL SYSTEM REPORT", core.system_report())
    pretty("FINAL STATE SNAPSHOT", core.snapshot())
