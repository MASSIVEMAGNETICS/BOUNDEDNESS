# ======================================================================================================================
# FILE: victor/deployment_core.py
# UUID: e5f6a7b8-c9d0-1234-efab-345678901234
# VERSION: v1.0.0-DEPLOYMENT-CORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Deployment Core
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Manages versioned deployments, rollout state, and rollback capability.
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
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------------------------------

class DeploymentError(Exception):
    """Base error for deployment core."""


class UnknownDeploymentError(DeploymentError):
    """Raised when a deployment ID is not found."""


class InvalidTransitionError(DeploymentError):
    """Raised when a status transition is not allowed."""


# ----------------------------------------------------------------------------------------------------------------------
# Enumerations
# ----------------------------------------------------------------------------------------------------------------------

class DeploymentStatus(str, enum.Enum):
    PENDING  = "pending"
    RUNNING  = "running"
    COMPLETE = "complete"
    FAILED   = "failed"
    ROLLED_BACK = "rolled_back"


# Allowed status transitions
_TRANSITIONS: Dict[DeploymentStatus, List[DeploymentStatus]] = {
    DeploymentStatus.PENDING:      [DeploymentStatus.RUNNING, DeploymentStatus.FAILED],
    DeploymentStatus.RUNNING:      [DeploymentStatus.COMPLETE, DeploymentStatus.FAILED],
    DeploymentStatus.COMPLETE:     [DeploymentStatus.ROLLED_BACK],
    DeploymentStatus.FAILED:       [DeploymentStatus.ROLLED_BACK],
    DeploymentStatus.ROLLED_BACK:  [],
}


# ----------------------------------------------------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class Deployment:
    """Represents a single versioned deployment unit."""

    name: str
    version: str
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: DeploymentStatus = DeploymentStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)
    log: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "deployment_id": self.deployment_id,
            "name": self.name,
            "version": self.version,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "meta": self.meta,
            "log": self.log,
        }


# ----------------------------------------------------------------------------------------------------------------------
# DeploymentCore
# ----------------------------------------------------------------------------------------------------------------------

class DeploymentCore:
    """
    Manages the lifecycle of versioned deployments with status transitions and rollback.

    Deployments progress through the following states::

        PENDING → RUNNING → COMPLETE → ROLLED_BACK
                          ↘ FAILED  → ROLLED_BACK
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._deployments: Dict[str, Deployment] = {}
        logger.debug("DeploymentCore initialised")

    # ------------------------------------------------------------------------------------------------------------------
    # Deployment lifecycle
    # ------------------------------------------------------------------------------------------------------------------

    def create(self, name: str, version: str, meta: Optional[Dict[str, Any]] = None) -> Deployment:
        """Create and register a new :class:`Deployment` in ``PENDING`` state."""
        dep = Deployment(name=name, version=version, meta=meta or {})
        dep.log.append(f"Deployment created at {dep.created_at:.3f}")
        with self._lock:
            self._deployments[dep.deployment_id] = dep
        logger.info("Deployment created: id=%s name=%s version=%s", dep.deployment_id, name, version)
        return dep

    def transition(self, deployment_id: str, new_status: DeploymentStatus, note: str = "") -> Deployment:
        """
        Transition *deployment_id* to *new_status*.

        Raises :class:`InvalidTransitionError` if the transition is not allowed.
        """
        with self._lock:
            dep = self._get(deployment_id)
            allowed = _TRANSITIONS.get(dep.status, [])
            if new_status not in allowed:
                raise InvalidTransitionError(
                    f"Cannot transition {dep.status.value!r} → {new_status.value!r} "
                    f"for deployment {deployment_id!r}"
                )
            dep.status = new_status
            dep.updated_at = time.time()
            msg = f"[{dep.updated_at:.3f}] Status → {new_status.value}"
            if note:
                msg += f" ({note})"
            dep.log.append(msg)
        logger.info("Deployment %s transitioned to %s", deployment_id, new_status.value)
        return dep

    def start(self, deployment_id: str, note: str = "") -> Deployment:
        """Transition to ``RUNNING``."""
        return self.transition(deployment_id, DeploymentStatus.RUNNING, note)

    def complete(self, deployment_id: str, note: str = "") -> Deployment:
        """Transition to ``COMPLETE``."""
        return self.transition(deployment_id, DeploymentStatus.COMPLETE, note)

    def fail(self, deployment_id: str, note: str = "") -> Deployment:
        """Transition to ``FAILED``."""
        return self.transition(deployment_id, DeploymentStatus.FAILED, note)

    def rollback(self, deployment_id: str, note: str = "") -> Deployment:
        """Transition to ``ROLLED_BACK``."""
        return self.transition(deployment_id, DeploymentStatus.ROLLED_BACK, note)

    # ------------------------------------------------------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------------------------------------------------------

    def _get(self, deployment_id: str) -> Deployment:
        """Return the :class:`Deployment`, raising :class:`UnknownDeploymentError` if absent."""
        dep = self._deployments.get(deployment_id)
        if dep is None:
            raise UnknownDeploymentError(f"Unknown deployment: {deployment_id!r}")
        return dep

    def get(self, deployment_id: str) -> Deployment:
        with self._lock:
            return self._get(deployment_id)

    def list_all(self) -> List[Deployment]:
        """Return a snapshot of all deployments, newest first."""
        with self._lock:
            return sorted(self._deployments.values(), key=lambda d: d.created_at, reverse=True)

    def list_by_status(self, status: DeploymentStatus) -> List[Deployment]:
        """Return all deployments with the given *status*."""
        with self._lock:
            return [d for d in self._deployments.values() if d.status == status]

    def latest(self, name: str) -> Optional[Deployment]:
        """Return the most recently created deployment with *name*, or None."""
        with self._lock:
            candidates = [d for d in self._deployments.values() if d.name == name]
        if not candidates:
            return None
        return max(candidates, key=lambda d: d.created_at)
