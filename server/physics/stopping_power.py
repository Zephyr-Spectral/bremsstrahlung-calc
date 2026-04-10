"""Electron stopping power calculations.

Primary source: NIST ESTAR tabulated data (Berger & Seltzer, ICRU-37)
for all elements Z=1-92 at 81 energies from 10 keV to 1000 MeV.

Fallback: Bethe formula for electrons (Rohrlich & Carlson form) used
when ESTAR data is not available (e.g., effective-Z composites like SS304).

The ESTAR data includes shell corrections, density effect, and all
standard corrections that the Bethe formula approximates.  Using ESTAR
directly gives stopping powers within 1% of NIST reference values and
fixes the CSDA range errors that were 1.76x for W and 2.0x for Pb
with the Bethe formula alone.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import-untyped]

import config
from server.physics._validation import require_positive_energy, require_positive_z

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NIST ESTAR data cache (loaded once at first use)
# ---------------------------------------------------------------------------
_estar_cache: dict[str, Any] | None = None


def _get_estar_data() -> dict[str, Any]:
    """Load and cache NIST ESTAR element data."""
    global _estar_cache
    if _estar_cache is None:
        with config.ESTAR_DATA_PATH.open() as f:
            _estar_cache = json.load(f)
    return _estar_cache


def _estar_lookup(
    z: int,
    kinetic_energy_mev: float,
    field: str,
) -> float | None:
    """Look up a single ESTAR value by element Z and energy.

    Uses log-log interpolation on the 81-point NIST grid.

    Args:
        z: Atomic number (1-92).
        kinetic_energy_mev: Electron kinetic energy in MeV.
        field: One of 'S_collision', 'S_radiative', 'S_total', 'CSDA_range_g_cm2'.

    Returns:
        Interpolated value, or None if element not in ESTAR data.
    """
    estar = _get_estar_data()

    # Find element by Z
    element_data = None
    for edata in estar.values():
        if edata.get("Z") == z:
            element_data = edata
            break

    if element_data is None or not element_data.get("data"):
        return None

    rows = element_data["data"]
    energies = [r["energy_MeV"] for r in rows]
    values = [r[field] for r in rows]

    # Filter out zeros/negatives for log interpolation
    valid_e = []
    valid_v = []
    for e_val, v_val in zip(energies, values, strict=False):
        if e_val > 0 and v_val > 0:
            valid_e.append(e_val)
            valid_v.append(v_val)

    if len(valid_e) < 2:
        return None

    # Clamp to ESTAR energy range
    e_clamped = max(valid_e[0], min(valid_e[-1], kinetic_energy_mev))

    # Log-log interpolation (physical: power-law behavior)
    log_e = np.log(valid_e)
    log_v = np.log(valid_v)
    interp_fn = interp1d(log_e, log_v, kind="linear", fill_value="extrapolate")
    return float(np.exp(interp_fn(np.log(e_clamped))))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def collision_stopping_power(
    kinetic_energy_mev: float,
    z: int | float,
    a: float,
) -> float:
    """Collision stopping power for electrons in MeV cm^2/g.

    Uses NIST ESTAR tabulated values when available (Z=1-92 elements).
    Falls back to Bethe formula for non-integer Z (effective-Z composites).

    Args:
        kinetic_energy_mev: Electron kinetic energy in MeV.
        z: Atomic number of target material.
        a: Atomic weight in g/mol.

    Returns:
        Collision stopping power -dE/d(rho*x) in MeV cm^2/g.
    """
    require_positive_energy(kinetic_energy_mev, "Electron energy")
    require_positive_z(z)

    # Try NIST ESTAR first (integer Z only)
    z_int = round(z)
    if abs(z - z_int) < 0.01:
        result = _estar_lookup(z_int, kinetic_energy_mev, "S_collision")
        if result is not None:
            return result

    # Fallback: Bethe formula
    return _bethe_collision_stopping_power(kinetic_energy_mev, z, a)


def radiative_stopping_power(
    kinetic_energy_mev: float,
    z: int | float,
    a: float,
) -> float:
    """Radiative (bremsstrahlung) stopping power in MeV cm^2/g.

    Uses NIST ESTAR tabulated values when available.
    Falls back to Born approximation formula.

    Args:
        kinetic_energy_mev: Electron kinetic energy in MeV.
        z: Atomic number of target material.
        a: Atomic weight in g/mol.

    Returns:
        Radiative stopping power in MeV cm^2/g.
    """
    require_positive_energy(kinetic_energy_mev, "Electron energy")
    require_positive_z(z)

    z_int = round(z)
    if abs(z - z_int) < 0.01:
        result = _estar_lookup(z_int, kinetic_energy_mev, "S_radiative")
        if result is not None:
            return result

    return _bethe_radiative_stopping_power(kinetic_energy_mev, z, a)


def total_stopping_power(
    kinetic_energy_mev: float,
    z: int | float,
    a: float,
) -> float:
    """Total stopping power (collision + radiative) in MeV cm^2/g.

    Uses NIST ESTAR tabulated values when available.

    Args:
        kinetic_energy_mev: Electron kinetic energy in MeV.
        z: Atomic number of target material.
        a: Atomic weight in g/mol.

    Returns:
        Total stopping power in MeV cm^2/g.
    """
    require_positive_energy(kinetic_energy_mev, "Electron energy")
    require_positive_z(z)

    z_int = round(z)
    if abs(z - z_int) < 0.01:
        result = _estar_lookup(z_int, kinetic_energy_mev, "S_total")
        if result is not None:
            return result

    s_col = _bethe_collision_stopping_power(kinetic_energy_mev, z, a)
    s_rad = _bethe_radiative_stopping_power(kinetic_energy_mev, z, a)
    return s_col + s_rad


def estar_csda_range(
    kinetic_energy_mev: float,
    z: int | float,
) -> float | None:
    """Look up NIST ESTAR CSDA range directly (no numerical integration).

    Returns None if element not in ESTAR data (use csda_range() as fallback).

    Args:
        kinetic_energy_mev: Electron kinetic energy in MeV.
        z: Atomic number.

    Returns:
        CSDA range in g/cm^2, or None if not available.
    """
    z_int = round(z)
    if abs(z - z_int) < 0.01:
        return _estar_lookup(z_int, kinetic_energy_mev, "CSDA_range_g_cm2")
    return None


# ---------------------------------------------------------------------------
# Bethe formula fallback (for effective-Z composites)
# ---------------------------------------------------------------------------


def _bethe_collision_stopping_power(
    kinetic_energy_mev: float,
    z: int | float,
    a: float,
) -> float:
    """Bethe collision stopping power (Rohrlich & Carlson form).

    Used as fallback when NIST ESTAR is not available.
    Does NOT include density effect correction (matches Powell eq. 8).
    """
    z_f = float(z)
    beta = config.electron_beta(kinetic_energy_mev)
    beta2 = beta * beta
    tau = kinetic_energy_mev / config.ELECTRON_MASS_MEV

    i_mev = config.mean_ionization_potential_ev(z) * 1.0e-6

    prefactor = (
        2.0
        * math.pi
        * config.CLASSICAL_ELECTRON_RADIUS_CM**2
        * config.ELECTRON_MASS_MEV
        * config.AVOGADRO
        * z_f
        / (a * beta2)
    )

    log_arg = tau**2 * (tau + 2.0) / (2.0 * (i_mev / config.ELECTRON_MASS_MEV) ** 2)
    if log_arg <= 0:
        return 0.0

    f_tau = (
        1.0
        - beta2
        + (tau**2 / 8.0 - (2.0 * tau + 1.0) * math.log(2.0)) / (tau + 1.0) ** 2
    )

    return prefactor * (math.log(log_arg) + f_tau)


def _bethe_radiative_stopping_power(
    kinetic_energy_mev: float,
    z: int | float,
    a: float,
) -> float:
    """Approximate radiative stopping power (Born approximation).

    Used as fallback when NIST ESTAR is not available.
    """
    z_f = float(z)
    total_energy = config.electron_total_energy_mev(kinetic_energy_mev)
    sigma_0 = config.ALPHA_FINE * config.RE_SQUARED_CM2

    b_rad = 4.0 * (
        math.log(2.0 * total_energy / config.ELECTRON_MASS_MEV) - 1.0 / 3.0
    )
    b_rad = max(b_rad, 5.33)

    return config.AVOGADRO * z_f * (z_f + 1.0) * sigma_0 * total_energy * b_rad / a
