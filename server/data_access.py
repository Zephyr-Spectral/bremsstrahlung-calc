"""Data access layer — loads and caches NASA TN D-4755 JSON data."""

from __future__ import annotations

import json
import logging
from typing import Any

import config

log = logging.getLogger(__name__)

# Module-level cache (loaded once on first access)
_nasa_data_cache: dict[str, Any] | None = None


def get_nasa_data() -> dict[str, Any]:
    """Load and return the NASA TN D-4755 data, caching after first read."""
    global _nasa_data_cache
    if _nasa_data_cache is not None:
        return _nasa_data_cache

    if not config.NASA_DATA_PATH.exists():
        log.warning("NASA data file not found at %s", config.NASA_DATA_PATH)
        return {}

    with config.NASA_DATA_PATH.open() as f:
        _nasa_data_cache = json.load(f)

    log.info("Loaded NASA data from %s", config.NASA_DATA_PATH)
    return _nasa_data_cache


def clear_cache() -> None:
    """Clear the data cache (useful for testing)."""
    global _nasa_data_cache
    _nasa_data_cache = None
