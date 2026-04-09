"""Electron range calculations.

CSDA (Continuous Slowing Down Approximation) range computed by
numerical integration of 1/S_total, following NASA TN D-4755 eq. 15.
"""

from __future__ import annotations

import logging

import numpy as np

import config
from server.physics._validation import require_positive_energy, require_positive_z
from server.physics.stopping_power import total_stopping_power

log = logging.getLogger(__name__)


def csda_range(
    kinetic_energy_mev: float,
    z: int | float,
    a: float,
    n_steps: int = 500,
    e_min_mev: float = config.CSDA_RANGE_EMIN_MEV,
) -> float:
    """Compute CSDA electron range by numerical integration.

    R = integral from e_min to T0 of dT / S_total(T)

    Args:
        kinetic_energy_mev: Incident electron kinetic energy in MeV.
        z: Atomic number of target material.
        a: Atomic weight in g/mol.
        n_steps: Number of integration steps (trapezoidal rule).
        e_min_mev: Lower energy cutoff in MeV to avoid divergence.

    Returns:
        Range in g/cm^2.

    Raises:
        ValueError: If energy or Z is non-positive.
    """
    require_positive_energy(kinetic_energy_mev, "Electron energy")
    require_positive_z(z)

    energies = np.linspace(e_min_mev, kinetic_energy_mev, n_steps)
    de = energies[1] - energies[0]

    # Trapezoidal integration of 1/S(T)
    inv_s = np.array([1.0 / total_stopping_power(e, z, a) for e in energies])
    return float(np.trapezoid(inv_s, dx=de))
