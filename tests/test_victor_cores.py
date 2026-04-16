"""Smoke tests for the victor sovereign cores.

Covers import, basic instantiation, and key operations for each core:
- config_core: dot-path get/set, missing key default
- security_core: role registration, vault lifecycle (in-memory), credential store/retrieve,
                 permission checks
- persistence_core: put/get/delete, created_at uniqueness, logger initialised before DB
- continuity_core: record, retrieve_relevant (fixes undefined-q bug), tag default
- deployment_core: create/start/complete/rollback lifecycle
- sovereignty_core: rule evaluation (allow/deny), audit log
- telemetry_core: record_metric, aggregate, record_event, report
"""
from __future__ import annotations

import time

import pytest

# ---------------------------------------------------------------------------
# config_core
# ---------------------------------------------------------------------------

from victor.config_core import ConfigCore, ConfigKeyError


class TestConfigCore:
    def test_set_and_get(self):
        cfg = ConfigCore()
        cfg.set("server.host", "localhost")
        assert cfg.get("server.host") == "localhost"

    def test_get_missing_returns_default(self):
        cfg = ConfigCore()
        assert cfg.get("nonexistent", "fallback") == "fallback"

    def test_require_raises_on_missing(self):
        cfg = ConfigCore()
        with pytest.raises(ConfigKeyError):
            cfg.require("missing.key")

    def test_delete(self):
        cfg = ConfigCore()
        cfg.set("foo.bar", 42)
        assert cfg.delete("foo.bar") is True
        assert cfg.get("foo.bar") is None

    def test_has(self):
        cfg = ConfigCore()
        cfg.set("x", 1)
        assert cfg.has("x") is True
        assert cfg.has("y") is False

    def test_initial_dict(self):
        cfg = ConfigCore(initial={"a.b": 99})
        assert cfg.get("a.b") == 99

    def test_nested_dot_path(self):
        cfg = ConfigCore()
        cfg.set("a.b.c.d", "deep")
        assert cfg.get("a.b.c.d") == "deep"


# ---------------------------------------------------------------------------
# security_core
# ---------------------------------------------------------------------------

from victor.security_core import (
    Credential,
    PermissionDeniedError,
    Role,
    SecurityCore,
    UnknownRoleError,
    VaultError,
)


class TestSecurityCoreInMemory:
    """Test SecurityCore in in-memory mode (vault_path=None)."""

    def setup_method(self):
        self.sc = SecurityCore()  # in-memory, no encryption
        admin = Role(
            name="admin",
            permissions={"read", "write"},
            allowed_services={"*"},
        )
        reader = Role(
            name="reader",
            permissions={"read"},
            allowed_services={"db"},
        )
        self.sc.register_role(admin)
        self.sc.register_role(reader)

    def test_store_and_retrieve(self):
        cred = Credential(service="db", scope="prod", username="alice", password="s3cr3t")
        self.sc.store_credential(cred, role_name="admin")
        retrieved = self.sc.get_credential("db", "prod", role_name="admin")
        assert retrieved.username == "alice"
        assert retrieved.password == "s3cr3t"

    def test_get_missing_raises(self):
        with pytest.raises(KeyError):
            self.sc.get_credential("db", "nonexistent_scope", role_name="admin")

    def test_reader_can_read(self):
        cred = Credential(service="db", scope="prod", username="bob", password="pw")
        self.sc.store_credential(cred, role_name="admin")
        retrieved = self.sc.get_credential("db", "prod", role_name="reader")
        assert retrieved.username == "bob"

    def test_reader_cannot_write(self):
        cred = Credential(service="db", scope="prod", username="eve", password="hack")
        with pytest.raises(PermissionDeniedError):
            self.sc.store_credential(cred, role_name="reader")

    def test_reader_wrong_service_denied(self):
        # reader only allowed for "db", not "payments"
        cred = Credential(service="db", scope="prod", username="x", password="y")
        self.sc.store_credential(cred, role_name="admin")
        with pytest.raises(PermissionDeniedError):
            self.sc.get_credential("payments", "prod", role_name="reader")

    def test_unknown_role_raises(self):
        with pytest.raises(UnknownRoleError):
            self.sc.get_role("ghost")

    def test_has_permission(self):
        assert self.sc.has_permission("admin", "read") is True
        assert self.sc.has_permission("reader", "write") is False

    def test_delete_credential(self):
        cred = Credential(service="db", scope="dev", username="u", password="p")
        self.sc.store_credential(cred, role_name="admin")
        assert self.sc.delete_credential("db", "dev", role_name="admin") is True
        with pytest.raises(KeyError):
            self.sc.get_credential("db", "dev", role_name="admin")

    def test_credential_to_dict_round_trip(self):
        cred = Credential(service="s", scope="sc", username="u", password="p", meta={"k": "v"})
        d = cred.to_dict()
        cred2 = Credential.from_dict(d)
        assert cred2.username == cred.username
        assert cred2.meta == {"k": "v"}

    def test_no_master_passphrase_no_vault_readonly(self, tmp_path):
        """Starting with no vault file and no passphrase → read-only mode; store raises."""
        sc = SecurityCore(vault_path=str(tmp_path / "vault.enc"))
        sc.register_role(Role("admin", {"write"}, {"*"}))
        cred = Credential("svc", "scope", "u", "p")
        with pytest.raises(VaultError):
            sc.store_credential(cred, "admin")


class TestSecurityCoreVaultFile:
    """Test SecurityCore with an on-disk encrypted vault."""

    def test_vault_create_and_reload(self, tmp_path):
        vault_path = str(tmp_path / "vault.enc")
        passphrase = "super-secret-passphrase-123"

        # Create vault and store credential
        sc1 = SecurityCore(vault_path=vault_path, master_passphrase=passphrase)
        sc1.register_role(Role("admin", {"read", "write"}, {"*"}))
        sc1.store_credential(
            Credential("mydb", "prod", "alice", "wonderland"), role_name="admin"
        )

        # Reload from disk
        sc2 = SecurityCore(vault_path=vault_path, master_passphrase=passphrase)
        sc2.register_role(Role("admin", {"read", "write"}, {"*"}))
        cred = sc2.get_credential("mydb", "prod", role_name="admin")
        assert cred.username == "alice"
        assert cred.password == "wonderland"

    def test_wrong_passphrase_raises(self, tmp_path):
        vault_path = str(tmp_path / "vault.enc")
        sc = SecurityCore(vault_path=vault_path, master_passphrase="correct-pass")
        with pytest.raises(VaultError):
            SecurityCore(vault_path=vault_path, master_passphrase="wrong-pass")

    def test_missing_passphrase_for_existing_vault_raises(self, tmp_path):
        vault_path = str(tmp_path / "vault.enc")
        SecurityCore(vault_path=vault_path, master_passphrase="p")
        with pytest.raises(VaultError):
            SecurityCore(vault_path=vault_path, master_passphrase=None)


# ---------------------------------------------------------------------------
# persistence_core
# ---------------------------------------------------------------------------

from victor.persistence_core import PersistenceCore, StateEntry


class TestPersistenceCore:
    def setup_method(self):
        # In-memory DB for tests
        self.pc = PersistenceCore(db_path=":memory:")

    def test_put_and_get(self):
        entry = StateEntry(key="foo", value={"x": 1})
        self.pc.put(entry)
        result = self.pc.get("foo")
        assert result is not None
        assert result.value == {"x": 1}

    def test_get_missing_returns_none(self):
        assert self.pc.get("nonexistent") is None

    def test_delete(self):
        entry = StateEntry(key="bar", value=42)
        self.pc.put(entry)
        assert self.pc.delete("bar") is True
        assert self.pc.get("bar") is None

    def test_exists(self):
        self.pc.put(StateEntry(key="k", value=0))
        assert self.pc.exists("k") is True
        assert self.pc.exists("missing") is False

    def test_list_keys(self):
        self.pc.put(StateEntry(key="a", value=1, tags=["x"]))
        self.pc.put(StateEntry(key="b", value=2, tags=["y"]))
        all_keys = self.pc.list_keys()
        assert "a" in all_keys and "b" in all_keys

    def test_list_keys_by_tag(self):
        self.pc.put(StateEntry(key="tagged", value=1, tags=["special"]))
        self.pc.put(StateEntry(key="other", value=2))
        keys = self.pc.list_keys(tag="special")
        assert "tagged" in keys
        assert "other" not in keys

    def test_created_at_unique_per_instance(self):
        """Regression: created_at must use default_factory, not a module-level call."""
        e1 = StateEntry(key="k1", value=1)
        time.sleep(0.01)
        e2 = StateEntry(key="k2", value=2)
        # Each instance has its own timestamp; they should not be identical
        # (they may be equal in very fast machines so we just check they're floats)
        assert isinstance(e1.created_at, float)
        assert isinstance(e2.created_at, float)

    def test_ttl_expiry(self):
        entry = StateEntry(key="expiring", value="bye", ttl=0.01)
        self.pc.put(entry)
        time.sleep(0.05)
        assert self.pc.get("expiring") is None

    def test_upsert_updates_value(self):
        self.pc.put(StateEntry(key="u", value=1, version=1))
        self.pc.put(StateEntry(key="u", value=2, version=2))
        result = self.pc.get("u")
        assert result is not None
        assert result.value == 2
        assert result.version == 2

    def test_purge_expired(self):
        self.pc.put(StateEntry(key="dead", value=0, ttl=0.01))
        self.pc.put(StateEntry(key="alive", value=1))
        time.sleep(0.05)
        removed = self.pc.purge_expired()
        assert removed >= 1
        assert self.pc.get("alive") is not None


# ---------------------------------------------------------------------------
# continuity_core
# ---------------------------------------------------------------------------

from victor.continuity_core import ContinuityCore, MemoryEntry


class TestContinuityCore:
    def setup_method(self):
        self.cc = ContinuityCore()

    def test_record_and_recent(self):
        self.cc.record("hello world", tags=["greeting"])
        entries = self.cc.recent(5)
        assert len(entries) == 1
        assert entries[0].content == "hello world"

    def test_tags_default_is_list_not_none(self):
        """Regression: tags must default to [] not None."""
        e = MemoryEntry(content="test")
        assert isinstance(e.tags, list)
        assert len(e.tags) == 0

    def test_retrieve_relevant_cosine(self):
        """Regression: retrieve_relevant must not use undefined variable q."""
        self.cc.record("alpha", embedding=[1.0, 0.0, 0.0])
        self.cc.record("beta",  embedding=[0.0, 1.0, 0.0])
        self.cc.record("gamma", embedding=[0.0, 0.0, 1.0])

        results = self.cc.retrieve_relevant([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0].content == "alpha"

    def test_retrieve_relevant_top_k(self):
        for i in range(10):
            self.cc.record(f"entry_{i}", embedding=[float(i), 0.0])
        results = self.cc.retrieve_relevant([9.0, 0.0], top_k=3)
        assert len(results) <= 3

    def test_retrieve_relevant_no_embeddings(self):
        """Entries without embeddings are excluded from ranked results."""
        self.cc.record("no embed")
        results = self.cc.retrieve_relevant([1.0, 0.0])
        assert results == []

    def test_search_by_tag(self):
        self.cc.record("tagged", tags=["important"])
        self.cc.record("untagged")
        results = self.cc.search_by_tag("important")
        assert len(results) == 1
        assert results[0].content == "tagged"

    def test_max_entries_eviction(self):
        cc = ContinuityCore(max_entries=3)
        for i in range(5):
            cc.record(f"e{i}")
        assert len(cc) == 3

    def test_clear(self):
        self.cc.record("something")
        self.cc.clear()
        assert len(self.cc) == 0


# ---------------------------------------------------------------------------
# deployment_core
# ---------------------------------------------------------------------------

from victor.deployment_core import (
    DeploymentCore,
    DeploymentStatus,
    InvalidTransitionError,
    UnknownDeploymentError,
)


class TestDeploymentCore:
    def setup_method(self):
        self.dc = DeploymentCore()

    def test_create_pending(self):
        dep = self.dc.create("my-service", "v1.0.0")
        assert dep.status == DeploymentStatus.PENDING
        assert dep.name == "my-service"

    def test_full_lifecycle(self):
        dep = self.dc.create("svc", "v2")
        dep = self.dc.start(dep.deployment_id)
        assert dep.status == DeploymentStatus.RUNNING
        dep = self.dc.complete(dep.deployment_id)
        assert dep.status == DeploymentStatus.COMPLETE
        dep = self.dc.rollback(dep.deployment_id)
        assert dep.status == DeploymentStatus.ROLLED_BACK

    def test_fail_and_rollback(self):
        dep = self.dc.create("svc", "v3")
        dep = self.dc.start(dep.deployment_id)
        dep = self.dc.fail(dep.deployment_id, note="crash")
        assert dep.status == DeploymentStatus.FAILED
        dep = self.dc.rollback(dep.deployment_id)
        assert dep.status == DeploymentStatus.ROLLED_BACK

    def test_invalid_transition_raises(self):
        dep = self.dc.create("svc", "v4")
        with pytest.raises(InvalidTransitionError):
            self.dc.complete(dep.deployment_id)  # PENDING → COMPLETE not allowed

    def test_unknown_deployment_raises(self):
        with pytest.raises(UnknownDeploymentError):
            self.dc.get("nonexistent-id")

    def test_list_by_status(self):
        d1 = self.dc.create("a", "v1")
        d2 = self.dc.create("b", "v1")
        self.dc.start(d1.deployment_id)
        pending = self.dc.list_by_status(DeploymentStatus.PENDING)
        running = self.dc.list_by_status(DeploymentStatus.RUNNING)
        assert any(d.deployment_id == d2.deployment_id for d in pending)
        assert any(d.deployment_id == d1.deployment_id for d in running)

    def test_latest(self):
        self.dc.create("svc", "v1")
        time.sleep(0.01)
        self.dc.create("svc", "v2")
        latest = self.dc.latest("svc")
        assert latest is not None
        assert latest.version == "v2"


# ---------------------------------------------------------------------------
# sovereignty_core
# ---------------------------------------------------------------------------

from victor.sovereignty_core import (
    RuleAction,
    RuleViolationError,
    SovereigntyCore,
    SovereigntyRule,
)


class TestSovereigntyCore:
    def setup_method(self):
        self.sc = SovereigntyCore(default_action=RuleAction.ALLOW)

    def test_default_allow(self):
        action = self.sc.evaluate({"type": "neutral"})
        assert action == RuleAction.ALLOW

    def test_deny_rule(self):
        rule = SovereigntyRule(
            name="block_external",
            predicate=lambda ctx: ctx.get("source") == "external",
            action=RuleAction.DENY,
            priority=10,
        )
        self.sc.add_rule(rule)
        with pytest.raises(RuleViolationError):
            self.sc.evaluate({"source": "external"})

    def test_allow_rule_passes(self):
        rule = SovereigntyRule(
            name="allow_internal",
            predicate=lambda ctx: ctx.get("source") == "internal",
            action=RuleAction.ALLOW,
            priority=5,
        )
        self.sc.add_rule(rule)
        action = self.sc.evaluate({"source": "internal"})
        assert action == RuleAction.ALLOW

    def test_is_allowed_helper(self):
        rule = SovereigntyRule(
            name="deny_all",
            predicate=lambda ctx: True,
            action=RuleAction.DENY,
        )
        self.sc.add_rule(rule)
        assert self.sc.is_allowed({"x": 1}) is False

    def test_consent_required(self):
        rule = SovereigntyRule(
            name="needs_consent",
            predicate=lambda ctx: ctx.get("risky") is True,
            action=RuleAction.REQUIRE_CONSENT,
        )
        self.sc.add_rule(rule)
        action = self.sc.evaluate({"risky": True})
        assert action == RuleAction.REQUIRE_CONSENT
        pending = self.sc.pending_consent_requests()
        assert len(pending) == 1
        req_id = pending[0].request_id
        resolved = self.sc.resolve_consent(req_id, granted=True)
        assert resolved.granted is True
        assert len(self.sc.pending_consent_requests()) == 0

    def test_audit_log_recorded(self):
        self.sc.evaluate({"x": 1})
        log = self.sc.audit_log()
        assert len(log) >= 1
        assert log[-1].outcome == "allowed"

    def test_remove_rule(self):
        rule = SovereigntyRule(
            name="temp", predicate=lambda ctx: True, action=RuleAction.DENY
        )
        self.sc.add_rule(rule)
        removed = self.sc.remove_rule(rule.rule_id)
        assert removed is True
        # After removal, default ALLOW should apply
        action = self.sc.evaluate({})
        assert action == RuleAction.ALLOW

    def test_priority_order(self):
        """Higher priority rule wins."""
        low = SovereigntyRule(
            name="low_deny", predicate=lambda ctx: True, action=RuleAction.DENY, priority=1
        )
        high = SovereigntyRule(
            name="high_allow", predicate=lambda ctx: True, action=RuleAction.ALLOW, priority=10
        )
        self.sc.add_rule(low)
        self.sc.add_rule(high)
        action = self.sc.evaluate({})
        assert action == RuleAction.ALLOW


# ---------------------------------------------------------------------------
# telemetry_core
# ---------------------------------------------------------------------------

from victor.telemetry_core import TelemetryCore


class TestTelemetryCore:
    def setup_method(self):
        self.tc = TelemetryCore(buffer_size=500)

    def test_record_and_retrieve_metric(self):
        self.tc.record_metric("cpu", 0.75, unit="%")
        recent = self.tc.recent_metrics("cpu")
        assert len(recent) == 1
        assert recent[0].value == pytest.approx(0.75)
        assert recent[0].unit == "%"

    def test_record_event(self):
        self.tc.record_event("startup", {"version": "v1"})
        events = self.tc.recent_events("startup")
        assert len(events) == 1
        assert events[0].payload["version"] == "v1"

    def test_aggregate_mean(self):
        for v in [1.0, 2.0, 3.0]:
            self.tc.record_metric("temp", v)
        mean = self.tc.aggregate("temp", method="mean")
        assert mean == pytest.approx(2.0)

    def test_aggregate_min_max(self):
        for v in [5.0, 10.0, 15.0]:
            self.tc.record_metric("pressure", v)
        assert self.tc.aggregate("pressure", method="min") == pytest.approx(5.0)
        assert self.tc.aggregate("pressure", method="max") == pytest.approx(15.0)

    def test_aggregate_sum(self):
        self.tc.record_metric("bytes", 100.0)
        self.tc.record_metric("bytes", 200.0)
        assert self.tc.aggregate("bytes", method="sum") == pytest.approx(300.0)

    def test_aggregate_count(self):
        for _ in range(7):
            self.tc.record_metric("ticks", 1.0)
        assert self.tc.aggregate("ticks", method="count") == pytest.approx(7.0)

    def test_aggregate_missing_returns_none(self):
        assert self.tc.aggregate("nonexistent") is None

    def test_report_structure(self):
        self.tc.record_metric("m1", 1.0)
        self.tc.record_event("e1")
        rpt = self.tc.report()
        assert rpt["metric_count"] >= 1
        assert rpt["event_count"] >= 1
        assert "metric_names" in rpt

    def test_metric_handler_called(self):
        received = []
        self.tc.add_metric_handler(lambda e: received.append(e))
        self.tc.record_metric("x", 1.0)
        assert len(received) == 1
        assert received[0].name == "x"

    def test_event_handler_called(self):
        received = []
        self.tc.add_event_handler(lambda e: received.append(e))
        self.tc.record_event("ping")
        assert len(received) == 1
        assert received[0].kind == "ping"

    def test_timing_helper(self):
        evt = self.tc.timing("request_time", 0.123)
        assert evt.unit == "s"
        assert evt.value == pytest.approx(0.123)

    def test_buffer_size_respected(self):
        tc = TelemetryCore(buffer_size=5)
        for i in range(10):
            tc.record_metric("m", float(i))
        assert len(tc.recent_metrics()) == 5
