"""Electron stopping power calculations.

Implements the Bethe formula for collision stopping power and radiative
stopping power for electrons in materials, following the formulation
in Berger & Seltzer (NASA SP-3012, 1964) as cited in NASA TN D-4755.
"""

from __future__ import annotations

import logging
import math

import config
from server.physics._validation import require_positive_energy, require_positive_z

log = logging.getLogger(__name__)


def collision_stopping_power(
    kinetic_energy_mev: float,
    z: int | float,
    a: float,
) -> float:
    """Bethe collision stopping power for electrons in MeV cm^2/g.

    Uses the Bethe-Ashkin formula (eq. 8 in NASA TN D-4755, ref. 5)
    for energy loss due to ionizing collisions.

    Args:
        kinetic_energy_mev: Electron kinetic energy in MeV.
        z: Atomic number of target material.
        a: Atomic weight in g/mol.

    Returns:
        Collision stopping power -dE/d(rho*x) in MeV cm^2/g.

    Raises:
        ValueError: If energy or Z is non-positive.
    """
    require_positive_energy(kinetic_energy_mev, "Electron energy")
    require_positive_z(z)

    z_f = float(z)

    # Relativistic quantities
    gamma = config.electron_gamma(kinetic_energy_mev)
    beta = config.electron_beta(kinetic_energy_mev)
    beta2 = beta * beta
    tau = kinetic_energy_mev / config.ELECTRON_MASS_MEV  # T / m0c2

    # Mean ionization potential in MeV
    i_mev = config.mean_ionization_potential_ev(z) * 1.0e-6

    # Bethe formula for electrons (Rohrlich & Carlson form)
    prefactor = (
        2.0
        * math.pi
        * config.CLASSICAL_ELECTRON_RADIUS_CM**2
        * config.ELECTRON_MASS_MEV
        * config.AVOGADRO
        * z_f
        / (a * beta2)
    )

    # Electron-specific logarithmic term
    log_arg = tau**2 * (tau + 2.0) / (2.0 * (i_mev / config.ELECTRON_MASS_MEV) ** 2)
    if log_arg <= 0:
        return 0.0

    # F(tau) correction for electrons (Rohrlich & Carlson)
    f_tau = 1.0 - beta2 + (tau**2 / 8.0 - (2.0 * tau + 1.0) * math.log(2.0)) / (tau + 1.0) ** 2

    # Density effect correction (simplified Sternheimer parameterization)
    delta = _density_effect_correction(beta * gamma, z)

    return prefactor * (math.log(log_arg) + f_tau - delta)


def radiative_stopping_power(
    kinetic_energy_mev: float,
    z: int | float,
    a: float,
) -> float:
    """Radiative (bremsstrahlung) stopping power in MeV cm^2/g.

    Approximate formula: S_rad ~ (alpha * Z * (T + m0c2) * sigma_0 * N_A) / A
    where sigma_0 = (alpha * r_e)^2 and alpha is fine structure constant.

    Args:
        kinetic_energy_mev: Electron kinetic energy in MeV.
        z: Atomic number of target material.
        a: Atomic weight in g/mol.

    Returns:
        Radiative stopping power in MeV cm^2/g.
    """
    require_positive_energy(kinetic_energy_mev, "Electron energy")
    require_positive_z(z)

    z_f = float(z)
    total_energy = config.electron_total_energy_mev(kinetic_energy_mev)

    # Born approximation radiation cross section
    sigma_0 = config.ALPHA_FINE * config.RE_SQUARED_CM2

    # Approximate radiation length formula
    b_rad = 4.0 * (math.log(2.0 * total_energy / config.ELECTRON_MASS_MEV) - 1.0 / 3.0)
    b_rad = max(b_rad, 5.33)  # floor for low energies

    return config.AVOGADRO * z_f * (z_f + 1.0) * sigma_0 * total_energy * b_rad / a


def total_stopping_power(
    kinetic_energy_mev: float,
    z: int | float,
    a: float,
) -> float:
    """Total stopping power (collision + radiative) in MeV cm^2/g.

    Args:
        kinetic_energy_mev: Electron kinetic energy in MeV.
        z: Atomic number of target material.
        a: Atomic weight in g/mol.

    Returns:
        Total stopping power in MeV cm^2/g.
    """
    s_col = collision_stopping_power(kinetic_energy_mev, z, a)
    s_rad = radiative_stopping_power(kinetic_energy_mev, z, a)
    return s_col + s_rad


def _density_effect_correction(beta_gamma: float, z: int | float) -> float:
    """Sternheimer density effect correction delta/2.

    Simplified parameterization for solids.

    Args:
        beta_gamma: Relativistic momentum beta * gamma.
        z: Atomic number (used to estimate plasma frequency).

    Returns:
        delta/2 correction term.
    """
    x = math.log10(beta_gamma)
    z_f = float(z)

    # Approximate parameters for condensed media
    x0 = 0.2 if z_f < 13 else 0.0
    x1 = 3.0
    c_bar = 2.0 * math.log(config.mean_ionization_potential_ev(z) / 28.816) + 1.0
    a_param = (c_bar - 4.606 * x0) / (x1 - x0) ** 3

    if x >= x1:
        return c_bar + 4.606 * x
    if x >= x0:
        return c_bar + 4.606 * x + a_param * (x1 - x) ** 3
    return 0.0
