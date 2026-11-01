"""Configuration loading.

Loads ``config.yaml`` (non-secret, tunable parameters) and overlays secrets /
environment overrides from a ``.env`` file. Nothing here reaches out to the
network. Access the singleton via :func:`get_config`.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Project root = two levels up from this file (src/credit_risk/config.py).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class Config:
    """Thin wrapper around the parsed YAML dict with dotted-path access.

    Also carries resolved absolute paths and the (optional) LLM API key read
    from the environment, so the rest of the codebase never touches ``os.environ``
    directly.
    """

    def __init__(self, data: dict[str, Any], root: Path):
        self._data = data
        self.root = root

    # -- generic access -----------------------------------------------------
    def get(self, dotted_key: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted_key.split("."):
            if isinstance(node, dict) and part in node:
                node = node[part]
            else:
                return default
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    @property
    def raw(self) -> dict[str, Any]:
        return self._data

    # -- resolved paths (always absolute) -----------------------------------
    def path(self, dotted_key: str) -> Path:
        rel = self.get(dotted_key)
        if rel is None:
            raise KeyError(f"No path configured at '{dotted_key}'")
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    # -- LLM credentials / availability -------------------------------------
    @property
    def llm_provider(self) -> str:
        return str(self.get("llm.provider", "anthropic"))

    @property
    def llm_api_key(self) -> str | None:
        """Return the API key for the configured provider, if present."""
        env_var = {
            "anthropic": "ANTHROPIC_API_KEY",
            "openai": "OPENAI_API_KEY",
            "gemini": "GEMINI_API_KEY",
            "ollama": None,  # local, no key needed
        }.get(self.llm_provider)
        if env_var is None:
            return None
        return os.environ.get(env_var)

    @property
    def llm_available(self) -> bool:
        """True only if the LLM is enabled AND usable (key present or Ollama)."""
        if not bool(self.get("llm.enabled", False)):
            return False
        if self.llm_provider == "ollama":
            return True
        return self.llm_api_key is not None


@lru_cache(maxsize=1)
def get_config(config_path: str | None = None) -> Config:
    """Load and cache the configuration.

    Reads ``.env`` first (so env vars are available), then parses the YAML.
    """
    load_dotenv(PROJECT_ROOT / ".env")  # silently no-ops if the file is absent
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return Config(data=data, root=PROJECT_ROOT)
