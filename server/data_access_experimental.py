"""Data access layer for experimental validation datasets.

Currently supports:
  - Dance et al. (1968): Al and Fe thick-target bremsstrahlung, 0.5-2.8 MeV
"""

from __future__ import annotations

import json
import logging
from typing import Any

import config

log = logging.getLogger(__name__)

_dance_cache: dict[str, Any] | None = None


def _load_dance() -> dict[str, Any] | None:
    """Load and cache Dance et al. data."""
    global _dance_cache
    if _dance_cache is not None:
        return _dance_cache

    if not config.DANCE_1968_PATH.exists():
        log.debug("Dance 1968 data not found at %s", config.DANCE_1968_PATH)
        return None

    with config.DANCE_1968_PATH.open() as f:
        _dance_cache = json.load(f)

    return _dance_cache


def get_experimental_spectrum(
    material: str,
    electron_energy_mev: float,
    angle_deg: float,
) -> dict[str, Any] | None:
    """Get experimental data at (material, energy, angle) if available.

    Returns dict with photon_energy_mev, intensity, source, uncertainty_pct
    or None if no data at this point.
    """
    data = _load_dance()
    if data is None:
        return None

    spectra = data.get("spectra", {})
    if material not in spectra:
        return None

    energy_key = f"{electron_energy_mev:.2f}"
    if energy_key not in spectra[material]:
        return None

    angle_key = str(int(angle_deg))
    angles = spectra[material][energy_key].get("angles", {})
    if angle_key not in angles:
        return None

    entry = angles[angle_key]
    k_vals = entry.get("photon_energy_mev", [])
    i_vals = entry.get("intensity", [])

    if not k_vals:
        return None

    return {
        "photon_energy_mev": k_vals,
        "intensity": i_vals,
        "source": "Dance et al., J. Appl. Phys. 39, 2881 (1968)",
        "uncertainty_pct": data.get("metadata", {}).get("uncertainty_pct", 18),
    }


def list_experimental_data() -> list[dict[str, Any]]:
    """List all available experimental data points (non-empty entries)."""
    data = _load_dance()
    if data is None:
        return []

    available: list[dict[str, Any]] = []
    for material, energies in data.get("spectra", {}).items():
        for energy_key, energy_data in energies.items():
            for angle_key, angle_data in energy_data.get("angles", {}).items():
                if angle_data.get("photon_energy_mev"):
                    available.append(
                        {
                            "material": material,
                            "electron_energy_mev": float(energy_key),
                            "angle_deg": float(angle_key),
                            "n_points": len(angle_data["photon_energy_mev"]),
                            "source": "Dance et al. (1968)",
                        }
                    )
    return available


def clear_cache() -> None:
    """Clear the experimental data cache."""
    global _dance_cache
    _dance_cache = None
