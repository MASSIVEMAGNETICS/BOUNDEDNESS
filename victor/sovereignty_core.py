# ======================================================================================================================
# FILE: victor/sovereignty_core.py
# UUID: f6a7b8c9-d0e1-2345-fabc-456789012345
# VERSION: v1.0.0-SOVEREIGNTY-CORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Sovereignty Core
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Manages autonomy rules, consent gates, and sovereignty constraints for the Victor system.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# DATE: 04/13/2026
# PROJECT NAME: BORON
# ======================================================================================================================

from __future__ import annotations

import enum
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------------------------------

class SovereigntyError(Exception):
    """Base error for sovereignty core."""


class RuleViolationError(SovereigntyError):
    """Raised when an action is blocked by a sovereignty rule."""


class UnknownRuleError(SovereigntyError):
    """Raised when a referenced rule ID does not exist."""


# ----------------------------------------------------------------------------------------------------------------------
# Enumerations
# ----------------------------------------------------------------------------------------------------------------------

class RuleAction(str, enum.Enum):
    ALLOW = "allow"
    DENY  = "deny"
    REQUIRE_CONSENT = "require_consent"


# ----------------------------------------------------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class SovereigntyRule:
    """
    A named rule that governs whether a given action/context combination is permitted.

    *predicate* is a callable that receives a context dict and returns True when the
    rule *matches* (i.e., when the rule applies to that context).

    If the rule matches and *action* is ``DENY``, the action is blocked.
    If the rule matches and *action* is ``REQUIRE_CONSENT``, a consent gate is raised.
    If the rule matches and *action* is ``ALLOW``, the action is explicitly permitted
    (useful for overriding lower-priority DENY rules when rules are evaluated in order).
    """

    name: str
    predicate: Callable[[Dict[str, Any]], bool]
    action: RuleAction = RuleAction.DENY
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = 0  # Higher value = evaluated first
    meta: Dict[str, Any] = field(default_factory=dict)

    def matches(self, context: Dict[str, Any]) -> bool:
        try:
            return bool(self.predicate(context))
        except Exception:  # noqa: BLE001
            return False


@dataclass
class ConsentRequest:
    """Issued when a REQUIRE_CONSENT rule matches; must be resolved before the action proceeds."""

    rule_id: str
    rule_name: str
    context: Dict[str, Any]
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    resolved: bool = False
    granted: bool = False

    def resolve(self, granted: bool) -> None:
        self.resolved = True
        self.granted = granted


@dataclass
class AuditEntry:
    """Immutable audit log entry recording an evaluation outcome."""

    context: Dict[str, Any]
    outcome: str  # "allowed", "denied", "consent_required"
    rule_name: Optional[str]
    timestamp: float = field(default_factory=time.time)
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ----------------------------------------------------------------------------------------------------------------------
# SovereigntyCore
# ----------------------------------------------------------------------------------------------------------------------

class SovereigntyCore:
    """
    Evaluates registered :class:`SovereigntyRule` objects against an action context,
    returning an allow/deny/consent decision.

    Rules are evaluated in descending priority order.  The first matching rule wins.
    If no rule matches, the default action is ``ALLOW``.
    """

    def __init__(self, default_action: RuleAction = RuleAction.ALLOW) -> None:
        self._lock = threading.RLock()
        self._rules: Dict[str, SovereigntyRule] = {}
        self._consent_requests: Dict[str, ConsentRequest] = {}
        self._audit_log: List[AuditEntry] = []
        self._default_action = default_action
        logger.debug("SovereigntyCore initialised (default_action=%s)", default_action.value)

    # ------------------------------------------------------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------------------------------------------------------

    def add_rule(self, rule: SovereigntyRule) -> None:
        """Register a rule.  Replaces any existing rule with the same *rule_id*."""
        with self._lock:
            self._rules[rule.rule_id] = rule
        logger.debug("Rule added: %s (priority=%d action=%s)", rule.name, rule.priority, rule.action.value)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID. Returns True if it existed."""
        with self._lock:
            if rule_id in self._rules:
                del self._rules[rule_id]
                return True
        return False

    def get_rule(self, rule_id: str) -> SovereigntyRule:
        with self._lock:
            if rule_id not in self._rules:
                raise UnknownRuleError(f"Unknown rule: {rule_id!r}")
            return self._rules[rule_id]

    def list_rules(self) -> List[SovereigntyRule]:
        """Return rules sorted by descending priority."""
        with self._lock:
            return sorted(self._rules.values(), key=lambda r: r.priority, reverse=True)

    # ------------------------------------------------------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------------------------------------------------------

    def evaluate(self, context: Dict[str, Any]) -> RuleAction:
        """
        Evaluate all rules against *context* and return the winning :class:`RuleAction`.

        Records the decision in the audit log.

        Raises :class:`RuleViolationError` if the outcome is ``DENY``.
        Returns :class:`RuleAction` for ``ALLOW`` or ``REQUIRE_CONSENT``.
        """
        rules = self.list_rules()  # already sorted by priority desc
        matching_rule: Optional[SovereigntyRule] = None
        outcome_action = self._default_action

        for rule in rules:
            if rule.matches(context):
                matching_rule = rule
                outcome_action = rule.action
                break

        rule_name = matching_rule.name if matching_rule else None

        if outcome_action == RuleAction.DENY:
            audit = AuditEntry(context=context, outcome="denied", rule_name=rule_name)
            with self._lock:
                self._audit_log.append(audit)
            logger.warning("Sovereignty DENY: rule=%s context=%s", rule_name, context)
            raise RuleViolationError(
                f"Action blocked by sovereignty rule: {rule_name!r}"
            )

        if outcome_action == RuleAction.REQUIRE_CONSENT:
            req = ConsentRequest(
                rule_id=matching_rule.rule_id,  # type: ignore[union-attr]
                rule_name=rule_name or "",
                context=context,
            )
            with self._lock:
                self._consent_requests[req.request_id] = req
                self._audit_log.append(
                    AuditEntry(context=context, outcome="consent_required", rule_name=rule_name)
                )
            logger.info("Sovereignty CONSENT required: rule=%s request_id=%s", rule_name, req.request_id)
            return outcome_action

        # ALLOW
        with self._lock:
            self._audit_log.append(
                AuditEntry(context=context, outcome="allowed", rule_name=rule_name)
            )
        return outcome_action

    def is_allowed(self, context: Dict[str, Any]) -> bool:
        """Return True if *context* is allowed, False if denied or consent-required."""
        try:
            action = self.evaluate(context)
            return action == RuleAction.ALLOW
        except RuleViolationError:
            return False

    # ------------------------------------------------------------------------------------------------------------------
    # Consent management
    # ------------------------------------------------------------------------------------------------------------------

    def resolve_consent(self, request_id: str, granted: bool) -> ConsentRequest:
        """Resolve a pending consent request."""
        with self._lock:
            req = self._consent_requests.get(request_id)
            if req is None:
                raise SovereigntyError(f"Unknown consent request: {request_id!r}")
            req.resolve(granted)
        logger.info("Consent %s for request %s", "granted" if granted else "denied", request_id)
        return req

    def pending_consent_requests(self) -> List[ConsentRequest]:
        """Return unresolved consent requests."""
        with self._lock:
            return [r for r in self._consent_requests.values() if not r.resolved]

    # ------------------------------------------------------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------------------------------------------------------

    def audit_log(self, limit: int = 100) -> List[AuditEntry]:
        """Return the most recent *limit* audit entries."""
        with self._lock:
            return list(self._audit_log[-limit:])
