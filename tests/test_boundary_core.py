"""Tests for victor/boundary_core.py"""
from __future__ import annotations

import pytest
import time

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
    _clamp01,
    _is_subpath,
    _normalize_path,
    _path_parts,
    _stable_hash,
)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

class TestNormalizePath:
    def test_strips_whitespace_and_dots(self):
        assert _normalize_path("  .core.identity. ") == "core.identity"

    def test_raises_on_non_string(self):
        with pytest.raises(InvalidPathError):
            _normalize_path(123)  # type: ignore[arg-type]

    def test_raises_on_empty(self):
        with pytest.raises(InvalidPathError):
            _normalize_path("")

    def test_raises_on_double_dot(self):
        with pytest.raises(InvalidPathError):
            _normalize_path("core..identity")


class TestIsSubpath:
    def test_exact_match(self):
        assert _is_subpath("core", "core")

    def test_child_path(self):
        assert _is_subpath("core.status", "core")

    def test_no_match(self):
        assert not _is_subpath("memory", "core")

    def test_no_prefix_collision(self):
        # "corex" should not match root "core"
        assert not _is_subpath("corex.foo", "core")


class TestClamp01:
    def test_below_zero(self):
        assert _clamp01(-1.0) == 0.0

    def test_above_one(self):
        assert _clamp01(2.0) == 1.0

    def test_midrange(self):
        assert _clamp01(0.5) == 0.5


class TestStableHash:
    def test_deterministic(self):
        payload = {"a": 1, "b": [1, 2]}
        assert _stable_hash(payload) == _stable_hash(payload)

    def test_different_payloads(self):
        assert _stable_hash({"a": 1}) != _stable_hash({"a": 2})


# ---------------------------------------------------------------------------
# BoundaryCore construction
# ---------------------------------------------------------------------------

class TestBoundaryCoreInit:
    def test_default_init(self):
        bc = BoundaryCore()
        snap = bc.snapshot()
        assert snap["identity"]["boundary_established"] is True
        assert snap["continuity"]["step_index"] == 0

    def test_custom_initial_state(self):
        state = BoundaryCore._default_state()
        state["core"]["status"] = "custom"
        bc = BoundaryCore(initial_state=state)
        assert bc.get("core.status") == "custom"

    def test_get_missing_returns_default(self):
        bc = BoundaryCore()
        assert bc.get("nonexistent.path", "fallback") == "fallback"


# ---------------------------------------------------------------------------
# Path classification
# ---------------------------------------------------------------------------

class TestPathClassification:
    def setup_method(self):
        self.bc = BoundaryCore()

    def test_internal_path(self):
        assert self.bc.classify_path("memory.boundary_events") == PathClass.INTERNAL

    def test_perimeter_path(self):
        assert self.bc.classify_path("sensors.readings") == PathClass.PERIMETER

    def test_external_path(self):
        assert self.bc.classify_path("outside.world") == PathClass.EXTERNAL

    def test_protected_path(self):
        assert self.bc.path_is_protected("identity.inside_label")
        assert self.bc.path_is_protected("memory")
        assert not self.bc.path_is_protected("drives.integrity")

    def test_external_write_allowed(self):
        assert self.bc.path_external_write_allowed("sensors.readings.s1")
        assert self.bc.path_external_write_allowed("body.surface.left")
        assert not self.bc.path_external_write_allowed("identity.inside_label")


# ---------------------------------------------------------------------------
# Sensor registration
# ---------------------------------------------------------------------------

class TestSensorRegistration:
    def setup_method(self):
        self.bc = BoundaryCore()
        self.spec = SensorSpec(
            sensor_id="touch_left",
            modality="touch",
            boundary_zone="left_hand",
            owned=True,
            channels=("pressure", "temperature"),
        )

    def test_register_and_exists(self):
        self.bc.register_sensor(self.spec)
        assert self.bc.sensor_exists("touch_left")

    def test_get_sensor(self):
        self.bc.register_sensor(self.spec)
        retrieved = self.bc.get_sensor("touch_left")
        assert retrieved == self.spec

    def test_get_unknown_sensor_raises(self):
        with pytest.raises(UnknownSensorError):
            self.bc.get_sensor("nonexistent")

    def test_register_updates_state(self):
        self.bc.register_sensor(self.spec)
        registry = self.bc.get("sensors.registry")
        assert "touch_left" in registry
        zones = self.bc.get("body.zones")
        assert "left_hand" in zones
        assert "touch_left" in zones["left_hand"]["sensors"]

    def test_duplicate_register_does_not_duplicate_sensor_list(self):
        self.bc.register_sensor(self.spec)
        self.bc.register_sensor(self.spec)
        zones = self.bc.get("body.zones")
        assert zones["left_hand"]["sensors"].count("touch_left") == 1


# ---------------------------------------------------------------------------
# Mutation control
# ---------------------------------------------------------------------------

class TestMutationControl:
    def setup_method(self):
        self.bc = BoundaryCore()

    def test_internal_mutation_allowed(self):
        decision = self.bc.evaluate_mutation(
            StateMutation(path="drives.integrity", value=0.9, origin=OriginType.INTERNAL_AUTONOMIC, reason="test")
        )
        assert decision.allowed

    def test_external_mutation_to_protected_path_denied(self):
        decision = self.bc.evaluate_mutation(
            StateMutation(path="identity.inside_label", value="hacked", origin=OriginType.EXTERNAL, reason="test")
        )
        assert not decision.allowed

    def test_external_mutation_to_non_allowed_path_denied(self):
        decision = self.bc.evaluate_mutation(
            StateMutation(path="drives.integrity", value=0.5, origin=OriginType.EXTERNAL, reason="test")
        )
        assert not decision.allowed

    def test_external_mutation_to_allowed_path_permitted(self):
        decision = self.bc.evaluate_mutation(
            StateMutation(path="sensors.readings.s1", value={"v": 1}, origin=OriginType.EXTERNAL, reason="test")
        )
        assert decision.allowed

    def test_apply_mutation_writes_state(self):
        self.bc.apply_mutation(
            StateMutation(path="drives.integrity", value=0.42, origin=OriginType.INTERNAL_AUTONOMIC, reason="test")
        )
        assert self.bc.get("drives.integrity") == pytest.approx(0.42)

    def test_apply_unauthorized_mutation_raises(self):
        with pytest.raises(UnauthorizedMutationError):
            self.bc.apply_mutation(
                StateMutation(path="identity.inside_label", value="x", origin=OriginType.EXTERNAL, reason="attack")
            )

    def test_set_internal_convenience(self):
        self.bc.set_internal("drives.integrity", 0.77)
        assert self.bc.get("drives.integrity") == pytest.approx(0.77)

    def test_apply_mutation_increments_step(self):
        before = self.bc.get("continuity.step_index")
        self.bc.set_internal("drives.integrity", 0.5)
        after = self.bc.get("continuity.step_index")
        assert after == before + 1


# ---------------------------------------------------------------------------
# Sensor reading ingestion
# ---------------------------------------------------------------------------

class TestSensorReadingIngestion:
    def setup_method(self):
        self.bc = BoundaryCore()
        self.spec = SensorSpec(
            sensor_id="cam_front",
            modality="vision",
            boundary_zone="front_face",
            owned=True,
        )
        self.bc.register_sensor(self.spec)

    def test_ingest_returns_perturbation_record(self):
        record = self.bc.ingest_sensor_reading("cam_front", {"lux": 300})
        assert isinstance(record, PerturbationRecord)
        assert record.source_id == "cam_front"
        assert record.kind == EventKind.SENSOR_READING

    def test_ingest_updates_sensor_readings_state(self):
        self.bc.ingest_sensor_reading("cam_front", {"lux": 500})
        reading = self.bc.get("sensors.readings.cam_front")
        assert reading is not None
        assert reading["payload"]["lux"] == 500

    def test_ingest_unknown_sensor_raises(self):
        with pytest.raises(UnknownSensorError):
            self.bc.ingest_sensor_reading("unknown_sensor", {})

    def test_owned_sensor_reading_is_perimeter(self):
        record = self.bc.ingest_sensor_reading("cam_front", {"lux": 100})
        assert record.path_class == PathClass.PERIMETER
        assert record.affects_owned_perimeter is True

    def test_ingest_increments_step(self):
        before = self.bc.get("continuity.step_index")
        self.bc.ingest_sensor_reading("cam_front", {"lux": 100})
        assert self.bc.get("continuity.step_index") == before + 1

    def test_ingest_appends_to_events_and_perturbations(self):
        self.bc.ingest_sensor_reading("cam_front", {"lux": 100})
        report = self.bc.boundary_report()
        assert report["event_count"] >= 1
        assert report["perturbation_count"] >= 1

    def test_ingest_stores_memory_projection(self):
        self.bc.ingest_sensor_reading("cam_front", {"lux": 100})
        snaps = self.bc.get("memory.ownership_snapshots")
        assert len(snaps) >= 1
        snap = snaps[-1]
        assert "ownership_score" in snap
        assert "path_class" in snap


# ---------------------------------------------------------------------------
# External perturbation ingestion
# ---------------------------------------------------------------------------

class TestExternalPerturbationIngestion:
    def setup_method(self):
        self.bc = BoundaryCore()

    def test_ingest_basic_perturbation(self):
        record = self.bc.ingest_external_perturbation(
            source_id="env_wind",
            boundary_zone="skin_back",
            payload={"force": 0.3},
        )
        assert isinstance(record, PerturbationRecord)
        assert record.external_caused is True

    def test_perturbation_to_allowed_path_updates_state(self):
        self.bc.ingest_external_perturbation(
            source_id="env",
            boundary_zone="skin_back",
            payload={"color": "red"},
            affects_path="body.surface",
        )
        val = self.bc.get("body.surface")
        assert val == {"color": "red"}

    def test_perturbation_to_protected_path_raises_alert_not_write(self):
        before = self.bc.get("identity.inside_label")
        self.bc.ingest_external_perturbation(
            source_id="attacker",
            boundary_zone="core",
            payload={"inside_label": "hacked"},
            affects_path="identity.inside_label",
        )
        # state must be unchanged
        assert self.bc.get("identity.inside_label") == before
        # a boundary alert event should have been emitted
        events = self.bc.recent_events(limit=10)
        kinds = [e["kind"] for e in events]
        assert EventKind.BOUNDARY_ALERT.value in kinds


# ---------------------------------------------------------------------------
# Self command & causality attribution
# ---------------------------------------------------------------------------

class TestSelfCommandCausality:
    def setup_method(self):
        self.bc = BoundaryCore()
        self.spec = SensorSpec(
            sensor_id="arm_touch",
            modality="touch",
            boundary_zone="right_arm",
            owned=True,
        )
        self.bc.register_sensor(self.spec)

    def test_linked_command_marks_self_caused(self):
        cmd_id = self.bc.issue_self_command("motor", "reach_out")
        record = self.bc.ingest_sensor_reading("arm_touch", {"pressure": 0.5}, linked_command_id=cmd_id)
        assert record.self_caused is True
        assert record.external_caused is False

    def test_unlinked_reading_is_external_caused(self):
        record = self.bc.ingest_sensor_reading("arm_touch", {"pressure": 0.5})
        assert record.external_caused is True
        assert record.self_caused is False


# ---------------------------------------------------------------------------
# Ownership scoring
# ---------------------------------------------------------------------------

class TestOwnershipScoring:
    def setup_method(self):
        self.bc = BoundaryCore()

    def test_internal_path_score_is_one(self):
        assert self.bc._ownership_score_for_path("memory.boundary_events") == 1.0

    def test_perimeter_path_score(self):
        assert self.bc._ownership_score_for_path("sensors.readings") == pytest.approx(0.8)

    def test_external_path_score(self):
        assert self.bc._ownership_score_for_path("outside.world") == pytest.approx(0.05)


# ---------------------------------------------------------------------------
# Boundary report
# ---------------------------------------------------------------------------

class TestBoundaryReport:
    def setup_method(self):
        self.bc = BoundaryCore()

    def test_report_structure(self):
        report = self.bc.boundary_report()
        for key in ("system_id", "boundary_established", "inside_label", "outside_label",
                    "registered_sensors", "owned_sensors", "non_owned_sensors",
                    "zones", "sensor_registry", "continuity", "event_count", "perturbation_count"):
            assert key in report

    def test_report_counts_sensors(self):
        self.bc.register_sensor(SensorSpec("s1", "touch", "zone_a", owned=True))
        self.bc.register_sensor(SensorSpec("s2", "vision", "zone_b", owned=False))
        report = self.bc.boundary_report()
        assert report["registered_sensors"] == 2
        assert report["owned_sensors"] == 1
        assert report["non_owned_sensors"] == 1


# ---------------------------------------------------------------------------
# History limits
# ---------------------------------------------------------------------------

class TestHistoryLimits:
    def test_event_history_trimmed(self):
        cfg = BoundaryConfig(event_history_limit=5, perturbation_history_limit=5)
        bc = BoundaryCore(config=cfg)
        spec = SensorSpec("s1", "touch", "zone_a", owned=True)
        bc.register_sensor(spec)
        for _ in range(20):
            bc.ingest_sensor_reading("s1", {"v": 1})
        assert len(bc._events) <= 5

    def test_perturbation_history_trimmed(self):
        cfg = BoundaryConfig(event_history_limit=5, perturbation_history_limit=5)
        bc = BoundaryCore(config=cfg)
        spec = SensorSpec("s1", "touch", "zone_a", owned=True)
        bc.register_sensor(spec)
        for _ in range(20):
            bc.ingest_sensor_reading("s1", {"v": 1})
        assert len(bc._perturbations) <= 5


# ---------------------------------------------------------------------------
# _next_id
# ---------------------------------------------------------------------------

class TestNextId:
    def test_increments(self):
        bc = BoundaryCore()
        id1 = bc._next_id()
        id2 = bc._next_id()
        assert int(id2) == int(id1) + 1

    def test_zero_padded(self):
        bc = BoundaryCore()
        nid = bc._next_id()
        assert len(nid) == 8


# ---------------------------------------------------------------------------
# _set_path
# ---------------------------------------------------------------------------

class TestSetPath:
    def test_set_nested_path(self):
        bc = BoundaryCore()
        bc._set_path("drives.integrity", 0.33)
        assert bc._state["drives"]["integrity"] == pytest.approx(0.33)

    def test_set_creates_intermediate_dicts(self):
        bc = BoundaryCore()
        bc._set_path("drives.new_key.deep", "value")
        assert bc._state["drives"]["new_key"]["deep"] == "value"
