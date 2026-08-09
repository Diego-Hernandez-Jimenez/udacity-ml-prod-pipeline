"""
Shared utilities for the ML pipeline.

Author: Diego Hernández Jiménez
"""

from pathlib import Path

import yaml

_CONFIG_PATH = Path(__file__).parent.parent / "ml_config.yaml"


def load_config() -> dict:
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f)
