"""File-based cache for Monte Carlo simulation results.

Each cached result is a JSON file in data/mc_cache/ with the naming convention:
    {code}_{material}_{T0:.2f}_{angle}.json

The JSON contains the spectrum (photon energies + intensities) plus metadata
(n_events, runtime_seconds, timestamp, code version).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import config

log = logging.getLogger(__name__)

CACHE_DIR: Path = config.DATA_DIR / "mc_cache"


def _cache_path(code: str, material: str, t0_mev: float, angle_deg: float) -> Path:
    """Build the cache file path for a specific run configuration."""
    return CACHE_DIR / f"{code}_{material}_{t0_mev:.2f}_{angle_deg:.0f}.json"


def get_cached(code: str, material: str, t0_mev: float, angle_deg: float) -> dict[str, Any] | None:
    """Retrieve a cached Monte Carlo result, or None if not cached."""
    path = _cache_path(code, material, t0_mev, angle_deg)
    if not path.exists():
        return None
    with path.open() as f:
        data: dict[str, Any] = json.load(f)
    log.info("Cache hit: %s", path.name)
    return data


def save_cache(
    code: str,
    material: str,
    t0_mev: float,
    angle_deg: float,
    result: dict[str, Any],
) -> None:
    """Save a Monte Carlo result to the file cache."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(code, material, t0_mev, angle_deg)
    with path.open("w") as f:
        json.dump(result, f, separators=(",", ":"))
    log.info("Cached: %s", path.name)


def list_cached() -> list[dict[str, Any]]:
    """List all cached results with their metadata."""
    if not CACHE_DIR.exists():
        return []
    results: list[dict[str, Any]] = []
    for path in sorted(CACHE_DIR.glob("*.json")):
        parts = path.stem.split("_")
        if len(parts) >= 4:  # code_mat_t0_angle
            results.append(
                {
                    "file": path.name,
                    "code": parts[0],
                    "material": parts[1],
                    "t0_mev": parts[2],
                    "angle_deg": parts[3],
                }
            )
    return results
