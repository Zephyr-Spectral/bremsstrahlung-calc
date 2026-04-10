"""Electron range calculations.

Primary source: NIST ESTAR tabulated CSDA ranges (direct lookup, no
numerical integration needed for Z=1-92 elements).

Fallback: Numerical integration of 1/S_total for effective-Z composites
where ESTAR data is not available.
"""

from __future__ import annotations

import logging

import numpy as np

import config
from server.physics._validation import require_positive_energy, require_positive_z
from server.physics.stopping_power import estar_csda_range, total_stopping_power

log = logging.getLogger(__name__)


def csda_range(
    kinetic_energy_mev: float,
    z: int | float,
    a: float,
    n_steps: int = 500,
    e_min_mev: float = config.CSDA_RANGE_EMIN_MEV,
) -> float:
    """Compute CSDA electron range.

    Uses NIST ESTAR tabulated range when available (exact, no integration
    error).  Falls back to numerical integration of 1/S_total for
    effective-Z composites (SS304, SS316).

    Args:
        kinetic_energy_mev: Incident electron kinetic energy in MeV.
        z: Atomic number of target material.
        a: Atomic weight in g/mol.
        n_steps: Number of integration steps (fallback only).
        e_min_mev: Lower energy cutoff in MeV (fallback only).

    Returns:
        Range in g/cm^2.

    Raises:
        ValueError: If energy or Z is non-positive.
    """
    require_positive_energy(kinetic_energy_mev, "Electron energy")
    require_positive_z(z)

    # Try NIST ESTAR direct lookup first
    estar_range = estar_csda_range(kinetic_energy_mev, z)
    if estar_range is not None:
        return estar_range

    # Fallback: numerical integration for non-ESTAR materials
    energies = np.linspace(e_min_mev, kinetic_energy_mev, n_steps)
    de = energies[1] - energies[0]

    inv_s = np.array([1.0 / total_stopping_power(e, z, a) for e in energies])
    return float(np.trapezoid(inv_s, dx=de))
