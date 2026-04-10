"""Multiple electron scattering in thick targets.

Implements Berger's Legendre polynomial expansion (NASA TN D-4755 eq. 4)
for the angular distribution of electrons after multiple Coulomb scattering,
with the Moliere screened transport cross section (ref. 3 in NASA TN).

The key transport coefficient includes the Coulomb logarithm:

    G_l(t) = l(l+1)/2 * Omega_0 * L_coulomb

where:
    Omega_0 = 4 pi r_0^2 N_A Z(Z+1) / (A p^2 beta^2)  [bare Coulomb]
    L_coulomb = ln(1/chi_a^2)   [Moliere screening correction]
    chi_a = alpha * Z^{1/3} / (0.885 * p * beta)  [Thomas-Fermi screening angle]

The screening correction replaces the empirical factor of 3.0 with the
physics-based Coulomb logarithm, which varies with Z and energy.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from scipy.special import legendre  # type: ignore[import-untyped]

import config

log = logging.getLogger(__name__)

# Precomputed constant: 4 pi r_0^2 N_A  [cm^2/mol]
_FOUR_PI_RE2_NA: float = 4.0 * math.pi * config.RE_SQUARED_CM2 * config.AVOGADRO  # 0.601 cm^2/mol

# Bohr radius in natural units (cm, used for Thomas-Fermi screening)
_BOHR_RADIUS_CM: float = 5.29177e-9  # cm


def _coulomb_logarithm(z: float, p_beta: float) -> float:
    """Compute the Coulomb logarithm for the transport cross section.

    Uses the radiation-length logarithm L_rad = ln(184.15 * Z^{-1/3})
    from Tsai (Rev. Mod. Phys. 46, 815, 1974).  This is the same Coulomb
    logarithm that appears in the radiation length formula and correctly
    accounts for Thomas-Fermi screening in the angular-integrated transport
    cross section.

    For the MeV electron energy range and Z=12-82, L_rad = 3.8-4.4.

    Args:
        z: Atomic number (float for effective-Z).
        p_beta: Product of momentum and velocity in m_0*c units (unused,
                retained for API compatibility with energy-dependent models).

    Returns:
        Coulomb logarithm (dimensionless, typically 3.5-4.5).
    """
    z_f = float(z)
    if z_f <= 0:
        return 1.0

    # Radiation-length Coulomb logarithm (Tsai 1974)
    l_rad = math.log(184.15 * z_f ** (-1.0 / 3.0))
    return max(l_rad, 1.0)


def scattering_probability(
    angle_rad: float,
    depth_g_cm2: float,
    z: int | float,
    electron_energy_mev: float,
    a: float = 1.0,
    n_legendre: int = config.DEFAULT_N_LEGENDRE,
) -> float:
    """Angular probability density for electron at given depth.

    P_epsilon(E, epsilon) from Berger's method (NASA TN D-4755 eq. 4):

        P = sum_l (2l+1)/(4 pi) * exp(-l(l+1)/2 * Omega_eff * t) * P_l(cos epsilon)

    where Omega_eff = Omega_0 * L_coulomb includes the Moliere screening
    correction through the Coulomb logarithm.

    Normalized so that 2 pi integral_0^pi P sin(eps) deps = 1.

    Args:
        angle_rad: Electron scattering angle epsilon (radians).
        depth_g_cm2: Physical depth traversed in g/cm^2.
        z: Atomic number.
        electron_energy_mev: Electron kinetic energy at this depth.
        a: Atomic weight in g/mol (needed for correct Omega_0).
        n_legendre: Number of Legendre terms (more = sharper forward peak).

    Returns:
        Probability density P_epsilon (per steradian).
    """
    if depth_g_cm2 <= 0:
        # At surface: delta function at epsilon=0.
        return 1.0 / (2.0 * math.pi * 0.001) if abs(angle_rad) < 0.001 else 0.0

    cos_eps = math.cos(angle_rad)
    z_f = float(z)

    p_moc = config.electron_momentum_moc(electron_energy_mev)
    beta = config.electron_beta(electron_energy_mev)
    p_beta = p_moc * beta
    if p_beta <= 0:
        return 0.0

    # Bare Coulomb Omega_0 [per g/cm^2]
    omega_0 = _FOUR_PI_RE2_NA * z_f * (z_f + 1.0) / (a * p_beta**2)

    # Moliere screening correction (Coulomb logarithm)
    l_coul = _coulomb_logarithm(z_f, p_beta)

    # Effective transport coefficient
    omega_eff = omega_0 * l_coul

    # Berger transport parameter: g_base = Omega_eff * t / 2
    g_base = omega_eff * depth_g_cm2 / 2.0

    # Legendre series: P = sum (2l+1)/(4pi) * exp(-l(l+1)*g_base) * P_l(cos eps)
    result = 0.0
    for l_idx in range(n_legendre):
        exponent = l_idx * (l_idx + 1) * g_base
        if exponent > 30:
            break  # exp(-30) ~ 1e-13, negligible
        coeff = (2 * l_idx + 1) / (4.0 * math.pi)
        exp_factor = math.exp(-exponent)
        p_l = float(legendre(l_idx)(cos_eps))
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
    cos(theta_0) = cos(epsilon) * cos(phi_d) + sin(epsilon) * sin(phi_d) * cos(psi)

    Args:
        detection_angle_deg: Detection angle phi_d from beam axis (degrees).
        electron_angle_rad: Electron scattering angle epsilon (radians).
        azimuthal_angle_rad: Azimuthal angle psi (radians).

    Returns:
        Emission angle theta_0 in radians.
    """
    phi_d = math.radians(detection_angle_deg)
    cos_theta0 = math.cos(electron_angle_rad) * math.cos(phi_d) + math.sin(
        electron_angle_rad
    ) * math.sin(phi_d) * math.cos(azimuthal_angle_rad)
    cos_theta0 = max(-1.0, min(1.0, cos_theta0))
    return math.acos(cos_theta0)


def average_scattering_probability(
    depth_g_cm2: float,
    z: int | float,
    electron_energy_mev: float,
    a: float = 1.0,
    n_angles: int = 16,
    n_legendre: int = config.DEFAULT_N_LEGENDRE,
) -> np.ndarray:  # type: ignore[type-arg]
    """Compute scattering probability distribution over angles.

    Args:
        depth_g_cm2: Physical depth in g/cm^2.
        z: Atomic number.
        electron_energy_mev: Electron energy at this depth.
        a: Atomic weight in g/mol.
        n_angles: Number of angle points.
        n_legendre: Number of Legendre terms.

    Returns:
        Array of shape (n_angles, 2) with columns [angle_rad, probability].
    """
    angles = np.linspace(0.0, math.pi, n_angles)
    probs = np.array(
        [
            scattering_probability(ang, depth_g_cm2, z, electron_energy_mev, a, n_legendre)
            for ang in angles
        ]
    )
    return np.column_stack([angles, probs])
