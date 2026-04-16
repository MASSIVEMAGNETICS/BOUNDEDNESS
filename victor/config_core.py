# ======================================================================================================================
# FILE: victor/config_core.py
# UUID: a1b2c3d4-e5f6-7890-abcd-ef1234567890
# VERSION: v1.0.0-CONFIG-CORE
# COMPAT: Victor>=v0.1.0 | SAVE3>=v0.1.0
# NAME: Config Core
# AUTHOR: Brandon "iambandobandz" Emery x Victor (Fractal Architect Mode)
# PURPOSE: Manages system configuration with dot-path access, JSON persistence, and thread safety.
# LICENSE: Proprietary - Massive Magnetics / Ethica AI / BHeard Network
# DATE: 04/13/2026
# PROJECT NAME: BORON
# ======================================================================================================================

from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------------------------------------------------
# Exceptions
# ----------------------------------------------------------------------------------------------------------------------

class ConfigError(Exception):
    """Base error for config core."""


class ConfigKeyError(ConfigError):
    """Raised when a config key is missing and no default is provided."""


# ----------------------------------------------------------------------------------------------------------------------
# ConfigCore
# ----------------------------------------------------------------------------------------------------------------------

class ConfigCore:
    """
    Thread-safe configuration manager with dot-path key access and optional JSON persistence.

    Keys are dot-separated strings like ``"server.host"`` or ``"logging.level"``.
    """

    def __init__(
        self,
        initial: Optional[Dict[str, Any]] = None,
        config_path: Optional[str] = None,
    ) -> None:
        self._lock = threading.RLock()
        self._data: Dict[str, Any] = {}
        self._config_path = config_path

        if config_path and os.path.isfile(config_path):
            self._load_from_file(config_path)

        if initial:
            for key, value in initial.items():
                self.set(key, value)

        logger.debug("ConfigCore initialised (path=%s)", config_path)

    # ------------------------------------------------------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------------------------------------------------------

    def _load_from_file(self, path: str) -> None:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ConfigError(f"Config file must contain a JSON object; got {type(data).__name__}")
            with self._lock:
                self._data.update(data)
            logger.info("Config loaded from %s", path)
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Failed to load config from {path}: {exc}") from exc

    def _save_to_file(self, path: str) -> None:
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(self._data, fh, indent=2)
            logger.debug("Config saved to %s", path)
        except OSError as exc:
            raise ConfigError(f"Failed to save config to {path}: {exc}") from exc

    @staticmethod
    def _split_key(key: str) -> List[str]:
        if not isinstance(key, str) or not key.strip():
            raise ConfigKeyError(f"Invalid config key: {key!r}")
        return key.strip().split(".")

    # ------------------------------------------------------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------------------------------------------------------

    def get(self, key: str, default: Any = None) -> Any:
        """Return the value at *key* (dot-path), or *default* if absent."""
        parts = self._split_key(key)
        with self._lock:
            node: Any = self._data
            for part in parts:
                if not isinstance(node, dict) or part not in node:
                    return default
                node = node[part]
            return node

    def require(self, key: str) -> Any:
        """Return the value at *key*, raising :class:`ConfigKeyError` if absent."""
        sentinel = object()
        value = self.get(key, sentinel)
        if value is sentinel:
            raise ConfigKeyError(f"Required config key missing: {key!r}")
        return value

    def set(self, key: str, value: Any) -> None:
        """Set the value at *key* (dot-path), creating intermediate dicts as needed."""
        parts = self._split_key(key)
        with self._lock:
            node: Dict[str, Any] = self._data
            for part in parts[:-1]:
                if part not in node or not isinstance(node[part], dict):
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
        logger.debug("Config set: %s = %r", key, value)

    def delete(self, key: str) -> bool:
        """Delete *key* from the config. Returns True if the key existed."""
        parts = self._split_key(key)
        with self._lock:
            node: Any = self._data
            for part in parts[:-1]:
                if not isinstance(node, dict) or part not in node:
                    return False
                node = node[part]
            if isinstance(node, dict) and parts[-1] in node:
                del node[parts[-1]]
                return True
            return False

    def has(self, key: str) -> bool:
        """Return True if *key* is present in the config."""
        sentinel = object()
        return self.get(key, sentinel) is not sentinel

    def all(self) -> Dict[str, Any]:
        """Return a shallow copy of the entire config dict."""
        with self._lock:
            return dict(self._data)

    def save(self, path: Optional[str] = None) -> None:
        """Persist the current config to *path* (or the path given at construction)."""
        target = path or self._config_path
        if not target:
            raise ConfigError("No config path specified for save")
        with self._lock:
            self._save_to_file(target)

    def reload(self) -> None:
        """Reload config from the file path given at construction."""
        if not self._config_path:
            raise ConfigError("No config path specified for reload")
        self._load_from_file(self._config_path)
