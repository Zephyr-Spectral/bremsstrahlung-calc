"""Electron range calculations.

CSDA (Continuous Slowing Down Approximation) range computed by
numerical integration of 1/S_total, following NASA TN D-4755 eq. 15.
"""

from __future__ import annotations

import logging

import numpy as np

from server.physics.stopping_power import total_stopping_power

log = logging.getLogger(__name__)


def csda_range(
    kinetic_energy_mev: float,
    z: int | float,
    a: float,
    n_steps: int = 500,
) -> float:
    """Compute CSDA electron range by numerical integration.

    R = integral from 0 to T0 of dT / S_total(T)

    Args:
        kinetic_energy_mev: Incident electron kinetic energy in MeV.
        z: Atomic number of target material.
        a: Atomic weight in g/mol.
        n_steps: Number of integration steps (trapezoidal rule).

    Returns:
        Range in g/cm².

    Raises:
        ValueError: If energy or Z is non-positive.
    """
    if kinetic_energy_mev <= 0:
        msg = f"Electron energy must be positive, got {kinetic_energy_mev} MeV"
        raise ValueError(msg)
    if z <= 0:
        msg = f"Atomic number must be positive, got Z={z}"
        raise ValueError(msg)

    # Integration from small energy (avoid zero) to T0
    e_min = 0.01  # MeV, lower cutoff to avoid divergence
    energies = np.linspace(e_min, kinetic_energy_mev, n_steps)
    de = energies[1] - energies[0]

    # Trapezoidal integration of 1/S(T)
    inv_s = np.array([1.0 / total_stopping_power(e, z, a) for e in energies])
    return float(np.trapezoid(inv_s, dx=de))
