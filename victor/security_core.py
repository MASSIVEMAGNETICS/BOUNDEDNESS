# ======================================================================================================================
# FILE: victor/security_core.py
# UUID: b2c3d4e5-f6a7-8901-bcde-f12345678901
# VERSION: v1.0.0-SECURITY-CORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Security Core
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Role-based access control and Fernet-encrypted credential vault.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# DATE: 04/13/2026
# PROJECT NAME: BORON
# ======================================================================================================================

from __future__ import annotations

import base64
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Optional dependency: cryptography (Fernet)
# ----------------------------------------------------------------------------------------------------------------------

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    _CRYPTO_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CRYPTO_AVAILABLE = False

# ----------------------------------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------------------------------

class SecurityError(Exception):
    """Base error for security core."""


class PermissionDeniedError(SecurityError):
    """Raised when a role lacks the required permission or service access."""


class VaultError(SecurityError):
    """Raised for vault initialisation or decryption failures."""


class UnknownRoleError(SecurityError):
    """Raised when a role is referenced that has not been registered."""


# ----------------------------------------------------------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------------------------------------------------------

@dataclass
class Role:
    """A named role with a set of permissions and allowed services."""

    name: str
    permissions: Set[str] = field(default_factory=set)
    allowed_services: Set[str] = field(default_factory=set)
    meta: Dict[str, Any] = field(default_factory=dict)

    def can(self, permission: str) -> bool:
        return permission in self.permissions

    def allows_service(self, service: str) -> bool:
        return "*" in self.allowed_services or service in self.allowed_services


@dataclass
class Credential:
    """Stores a username/password pair plus arbitrary metadata for a service."""

    service: str
    scope: str
    username: str
    password: str
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "service": self.service,
            "scope": self.scope,
            "username": self.username,
            "password": self.password,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Credential":
        return cls(
            service=data["service"],
            scope=data["scope"],
            username=data["username"],
            password=data["password"],
            meta=data.get("meta", {}),
        )


# ----------------------------------------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------------------------------------

_PBKDF2_SALT = b"victor-security-core-salt-v1"
_PBKDF2_ITERATIONS = 200_000


def _derive_fernet_key(master_passphrase: str) -> bytes:
    """Derive a 32-byte Fernet key from *master_passphrase* via PBKDF2-HMAC-SHA256."""
    if not _CRYPTO_AVAILABLE:
        raise VaultError("cryptography package is not installed; cannot use encrypted vault")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_PBKDF2_SALT,
        iterations=_PBKDF2_ITERATIONS,
    )
    raw_key = kdf.derive(master_passphrase.encode("utf-8"))
    return base64.urlsafe_b64encode(raw_key)


# ----------------------------------------------------------------------------------------------------------------------
# SecurityCore
# ----------------------------------------------------------------------------------------------------------------------

class SecurityCore:
    """
    Role-based access control system with an optional Fernet-encrypted credential vault.

    Vault layout on disk (before encryption)::

        {
            "<service>": {
                "<scope>": { ...Credential.to_dict()... }
            }
        }

    Initialisation rules:
    - If *vault_path* is None, the vault operates in-memory only (no persistence).
    - If *vault_path* points to an existing file, *master_passphrase* is required to decrypt it.
    - If *vault_path* does not yet exist, *master_passphrase* is required to create it.
    - If *master_passphrase* is omitted and no vault file exists, the core starts in
      **read-only / no-vault** mode; store operations will raise :class:`VaultError`.
    """

    def __init__(
        self,
        vault_path: Optional[str] = None,
        master_passphrase: Optional[str] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._vault_path = vault_path
        self._fernet: Any = None  # Fernet instance or None
        self._vault: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._roles: Dict[str, Role] = {}
        self._readonly = False

        self._init_vault(master_passphrase)
        logger.debug("SecurityCore initialised (vault_path=%s, readonly=%s)", vault_path, self._readonly)

    # ------------------------------------------------------------------------------------------------------------------
    # Vault initialisation
    # ------------------------------------------------------------------------------------------------------------------

    def _init_vault(self, master_passphrase: Optional[str]) -> None:
        """Initialise the Fernet key and load (or create) the vault."""
        if not self._vault_path:
            # In-memory only — no encryption needed
            self._readonly = False
            return

        vault_exists = os.path.isfile(self._vault_path)

        if not master_passphrase:
            if vault_exists:
                raise VaultError(
                    f"Vault file {self._vault_path!r} exists but no master_passphrase was provided; "
                    "cannot decrypt."
                )
            # No vault file, no passphrase → read-only empty vault
            logger.warning(
                "No master_passphrase provided and no vault file found; starting in read-only/no-vault mode."
            )
            self._readonly = True
            return

        # We have a passphrase — derive the Fernet key
        fernet_key = _derive_fernet_key(master_passphrase)
        if _CRYPTO_AVAILABLE:
            self._fernet = Fernet(fernet_key)
        else:
            raise VaultError("cryptography package is not installed; cannot use encrypted vault")

        if vault_exists:
            self._vault = self._load_and_decrypt()
        else:
            # Create an empty vault and persist it immediately
            self._vault = {}
            self._save_vault()

    def _load_and_decrypt(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Load and decrypt the vault file. Raises VaultError on failure.

        Does NOT call _init_vault to avoid recursion.
        """
        if self._fernet is None:
            raise VaultError("Cannot decrypt vault: Fernet key not initialised.")
        try:
            with open(self._vault_path, "rb") as fh:  # type: ignore[arg-type]
                ciphertext = fh.read()
            plaintext = self._fernet.decrypt(ciphertext)
            data = json.loads(plaintext.decode("utf-8"))
            if not isinstance(data, dict):
                raise VaultError("Vault data is not a JSON object")
            return data
        except InvalidToken as exc:
            raise VaultError("Vault decryption failed — wrong master passphrase?") from exc
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise VaultError(f"Failed to load vault: {exc}") from exc

    def _save_vault(self) -> None:
        """Encrypt and persist the vault to disk."""
        if self._vault_path is None:
            return  # In-memory mode; nothing to save
        if self._readonly:
            raise VaultError("Vault is in read-only mode; cannot save.")
        if self._fernet is None:
            raise VaultError("Cannot save vault: Fernet key not initialised.")
        plaintext = json.dumps(self._vault, indent=2).encode("utf-8")
        ciphertext = self._fernet.encrypt(plaintext)
        with open(self._vault_path, "wb") as fh:
            fh.write(ciphertext)
        logger.debug("Vault saved to %s", self._vault_path)

    # ------------------------------------------------------------------------------------------------------------------
    # Role management
    # ------------------------------------------------------------------------------------------------------------------

    def register_role(self, role: Role) -> None:
        """Register a role."""
        with self._lock:
            self._roles[role.name] = role
        logger.debug("Role registered: %s", role.name)

    def get_role(self, role_name: str) -> Role:
        """Return the :class:`Role` with *role_name*, raising :class:`UnknownRoleError` if absent."""
        with self._lock:
            if role_name not in self._roles:
                raise UnknownRoleError(f"Role not registered: {role_name!r}")
            return self._roles[role_name]

    def has_permission(self, role_name: str, permission: str) -> bool:
        """Return True if the role has *permission*."""
        try:
            return self.get_role(role_name).can(permission)
        except UnknownRoleError:
            return False

    # ------------------------------------------------------------------------------------------------------------------
    # Credential vault operations
    # ------------------------------------------------------------------------------------------------------------------

    def _check_role_access(self, role_name: str, service: str, required_permission: str) -> None:
        """Raise :class:`PermissionDeniedError` unless *role_name* has *required_permission* on *service*."""
        role = self.get_role(role_name)
        if not role.can(required_permission):
            raise PermissionDeniedError(
                f"Role {role_name!r} lacks permission {required_permission!r}"
            )
        if not role.allows_service(service):
            raise PermissionDeniedError(
                f"Role {role_name!r} is not allowed to access service {service!r}"
            )

    def store_credential(self, credential: Credential, role_name: str) -> None:
        """
        Store *credential* in the vault using *role_name* for authorisation.

        The role must have ``"write"`` permission and service access.
        """
        if self._readonly:
            raise VaultError("Vault is in read-only mode; cannot store credentials.")
        with self._lock:
            self._check_role_access(role_name, credential.service, "write")
            self._vault.setdefault(credential.service, {})[credential.scope] = credential.to_dict()
            self._save_vault()
        logger.debug("Credential stored: service=%s scope=%s", credential.service, credential.scope)

    def get_credential(self, service: str, scope: str, role_name: str) -> Credential:
        """
        Retrieve a credential from the vault using *role_name* for authorisation.

        The role must have ``"read"`` permission and service access.
        """
        with self._lock:
            self._check_role_access(role_name, service, "read")
            service_vault = self._vault.get(service, {})
            cred_dict = service_vault.get(scope)
            if cred_dict is None:
                raise KeyError(f"No credential found for service={service!r} scope={scope!r}")
            return Credential.from_dict(cred_dict)

    def delete_credential(self, service: str, scope: str, role_name: str) -> bool:
        """Delete a credential. Returns True if it existed. Role requires ``"write"``."""
        if self._readonly:
            raise VaultError("Vault is in read-only mode; cannot delete credentials.")
        with self._lock:
            self._check_role_access(role_name, service, "write")
            service_vault = self._vault.get(service, {})
            if scope in service_vault:
                del service_vault[scope]
                if not service_vault:
                    del self._vault[service]
                self._save_vault()
                return True
            return False

    def list_services(self, role_name: str) -> List[str]:
        """List services visible to *role_name*."""
        role = self.get_role(role_name)
        with self._lock:
            if "*" in role.allowed_services:
                return list(self._vault.keys())
            return [s for s in self._vault if role.allows_service(s)]
