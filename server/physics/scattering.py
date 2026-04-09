"""Multiple electron scattering in thick targets.

Implements Berger's Legendre polynomial expansion (ref. 3 in NASA TN D-4755)
for the angular distribution of electrons after multiple Coulomb scattering,
and the backscatter correction from Wright & Trump (ref. 6).
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.special import legendre

import config

log = logging.getLogger(__name__)


def scattering_probability(
    angle_rad: float,
    depth_fraction: float,
    z: int | float,
    electron_energy_mev: float,
    n_legendre: int = config.DEFAULT_N_LEGENDRE,
) -> float:
    """Angular probability density for electron at given depth.

    P_s(xi, t) from Berger's method (NASA TN D-4755 eq. 4).
    Uses Legendre polynomial expansion with scattering coefficients
    that depend on depth and material.

    Args:
        angle_rad: Electron angle from normal (radians).
        depth_fraction: Fractional depth in target (0 to 1).
        z: Atomic number of target.
        electron_energy_mev: Current electron energy at this depth.
        n_legendre: Number of Legendre terms.

    Returns:
        Probability density P_s (per steradian).
    """
    if depth_fraction <= 0:
        # At surface, electrons travel in forward direction
        return 1.0 / (2.0 * math.pi) if abs(angle_rad) < 0.01 else 0.0

    cos_xi = math.cos(angle_rad)

    # Scattering parameter: increases with depth and Z
    z_f = float(z)
    p_beta = config.electron_momentum_moc(electron_energy_mev) * config.electron_beta(
        electron_energy_mev
    )
    if p_beta <= 0:
        return 0.0

    # Characteristic scattering angle squared per unit depth
    chi_sq = config.MOLIERE_PREFACTOR * z_f * (z_f + 1.0) / (p_beta**2)

    # Transport mean free path parameter
    g_l_t = chi_sq * depth_fraction

    # Legendre series: P_s = sum (2l+1)/(4*pi) * exp(-l(l+1)*g) * P_l(cos xi)
    result = 0.0
    for l_idx in range(n_legendre):
        coeff = (2 * l_idx + 1) / (4.0 * math.pi)
        exp_factor = math.exp(-l_idx * (l_idx + 1) * g_l_t)
        p_l = float(legendre(l_idx)(cos_xi))
        result += coeff * exp_factor * p_l

    return max(result, 0.0)


def backscatter_fraction(z: int | float, electron_energy_mev: float) -> float:
    """Fraction of electrons backscattered from target surface.

    Delegates to config.backscatter_fraction.

    Args:
        z: Atomic number.
        electron_energy_mev: Incident electron kinetic energy in MeV.

    Returns:
        Backscatter fraction W (0 to 0.5).
    """
    return config.backscatter_fraction(z, electron_energy_mev)


def scattering_broadened_angle(
    detection_angle_deg: float,
    electron_angle_rad: float,
    azimuthal_angle_rad: float,
) -> float:
    """Compute photon emission angle theta_0 from spherical triangle.

    NASA TN D-4755 eq. 3:
    cos(theta_0) = cos(xi) * cos(phi_d) + sin(xi) * sin(phi_d) * cos(psi)

    where xi is the electron angle, phi_d is the detection angle,
    and psi is the azimuthal angle.

    Args:
        detection_angle_deg: Detection angle from target normal (degrees).
        electron_angle_rad: Electron direction angle from normal (radians).
        azimuthal_angle_rad: Azimuthal angle (radians).

    Returns:
        Emission angle theta_0 in radians.
    """
    phi_d = math.radians(detection_angle_deg)
    cos_theta0 = math.cos(electron_angle_rad) * math.cos(phi_d) + math.sin(
        electron_angle_rad
    ) * math.sin(phi_d) * math.cos(azimuthal_angle_rad)
    # Clamp to valid range for acos
    cos_theta0 = max(-1.0, min(1.0, cos_theta0))
    return math.acos(cos_theta0)


def average_scattering_probability(
    depth_fraction: float,
    z: int | float,
    electron_energy_mev: float,
    n_angles: int = 16,
    n_legendre: int = config.DEFAULT_N_LEGENDRE,
) -> np.ndarray:
    """Compute scattering probability distribution over angles.

    Returns array of (angle_rad, P_s) pairs at equally-spaced angles from 0 to pi.

    Args:
        depth_fraction: Fractional depth in target (0 to 1).
        z: Atomic number.
        electron_energy_mev: Electron energy at this depth.
        n_angles: Number of angle points.
        n_legendre: Number of Legendre terms.

    Returns:
        Array of shape (n_angles, 2) with columns [angle_rad, probability].
    """
    angles = np.linspace(0.0, math.pi, n_angles)
    probs = np.array(
        [
            scattering_probability(a, depth_fraction, z, electron_energy_mev, n_legendre)
            for a in angles
        ]
    )
    return np.column_stack([angles, probs])
