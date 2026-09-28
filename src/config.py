"""Configuration loader and management module."""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, Any, Optional

DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "config.yaml"
)

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load and parse YAML configuration."""
    target_path = config_path or DEFAULT_CONFIG_PATH
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Configuration file not found: {target_path}")

    with open(target_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Base directory for resolving relative paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config["base_dir"] = base_dir

    return config

def resolve_path(base_dir: str, rel_path: str) -> str:
    """Resolve a relative path against project root if not already absolute."""
    if os.path.isabs(rel_path):
        return rel_path
    return os.path.normpath(os.path.join(base_dir, rel_path))
