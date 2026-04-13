# ======================================================================================================================
# FILE: demo_boundary_core.py
# UUID: 58eb52f3-5284-4bbb-b3f1-cfa2b8d36b9b
# VERSION: v1.0.0-BOUNDARY-DEMO-GODCORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Boundary Core Demo
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Runnable demonstration of the BoundaryCore showing sensor registration, self-command causality,
#          external perturbation classification, mutation control, and ownership reporting.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# DATE: 04/12/2026 11:00 EST
# PROJECT NAME: BORON
# ======================================================================================================================

from __future__ import annotations

import json
import time

from victor.boundary_core import (
    BoundaryCore,
    SensorSpec,
    StateMutation,
    OriginType,
    UnauthorizedMutationError,
)


def pretty(title: str, obj) -> None:
    print(f"\n=== {title} ===")
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


def main() -> None:
    core = BoundaryCore()

    # Register sensors physically/logically attached to the system perimeter
    core.register_sensor(SensorSpec(
        sensor_id="cam_front",
        modality="vision",
        boundary_zone="front_surface",
        owned=True,
        channels=("rgb",)
    ))
    core.register_sensor(SensorSpec(
        sensor_id="touch_left",
        modality="touch",
        boundary_zone="left_arm",
        owned=True,
        channels=("pressure", "temperature")
    ))
    core.register_sensor(SensorSpec(
        sensor_id="net_feed",
        modality="remote_feed",
        boundary_zone="remote_network",
        owned=False,
        channels=("text",)
    ))

    pretty("INITIAL BOUNDARY REPORT", core.boundary_report())

    # Self issues a command that may causally explain a near-future sensor change
    cmd_id = core.issue_self_command(
        source_id="motor_controller",
        command_name="turn_head_left",
        payload={"degrees": 15}
    )
    print(f"\nIssued self command: {cmd_id}")

    time.sleep(0.1)

    # Owned sensor receives resulting data, linked to the self command
    rec1 = core.ingest_sensor_reading(
        sensor_id="cam_front",
        reading={"object": "doorway", "angle_offset_deg": -13},
        linked_command_id=cmd_id,
        notes="visual scene changed after self movement"
    )
    pretty("SENSOR READING AFTER SELF COMMAND", rec1.__dict__)

    # External touch perturbation against owned perimeter
    rec2 = core.ingest_external_perturbation(
        source_id="outside_object",
        boundary_zone="left_arm",
        payload={"pressure": 0.82, "temperature_c": 28.4},
        affects_path="body.surface.left_arm",
        notes="contact at left arm"
    )
    pretty("EXTERNAL PERTURBATION ON OWNED PERIMETER", rec2.__dict__)

    # Non-owned data feed: should classify as mostly outside/not-self
    rec3 = core.ingest_sensor_reading(
        sensor_id="net_feed",
        reading={"headline": "External world event stream"},
        notes="remote non-owned data source"
    )
    pretty("NON-OWNED REMOTE FEED", rec3.__dict__)

    # Authorized internal mutation
    decision = core.apply_mutation(StateMutation(
        path="core.status",
        value="alert",
        origin=OriginType.INTERNAL_AUTONOMIC,
        reason="Integrity warning propagated to core status"
    ))
    pretty("AUTHORIZED INTERNAL MUTATION DECISION", decision.__dict__)

    # Unauthorized external mutation attempt on protected identity
    try:
        core.apply_mutation(StateMutation(
            path="identity.inside_label",
            value="compromised_self",
            origin=OriginType.EXTERNAL,
            reason="malicious rewrite attempt"
        ))
    except UnauthorizedMutationError as exc:
        print(f"\nUnauthorized external mutation blocked correctly: {exc}")

    # External write allowed to sensor readings subtree
    allowed_external = core.apply_mutation(StateMutation(
        path="sensors.readings.manual_probe",
        value={"probe": True, "value": 123},
        origin=OriginType.EXTERNAL,
        reason="permitted boundary-adjacent reading injection"
    ))
    pretty("ALLOWED EXTERNAL MUTATION TO SENSOR READINGS", allowed_external.__dict__)

    pretty("RECENT PERTURBATIONS", core.recent_perturbations(limit=10))
    pretty("RECENT EVENTS", core.recent_events(limit=20))
    pretty("FINAL BOUNDARY REPORT", core.boundary_report())
    pretty("FINAL STATE SNAPSHOT", core.snapshot())


if __name__ == "__main__":
    main()
