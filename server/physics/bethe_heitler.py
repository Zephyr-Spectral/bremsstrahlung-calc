"""Bethe-Heitler thin-target bremsstrahlung cross section.

Implements the Koch & Motz 2BS formula (Schiff approximation with Thomas-Fermi
screening) from Rev. Mod. Phys. 31, 920-955 (1959), as documented in the
Geant4 Physics Reference Manual (Livermore bremsstrahlung model).

The 2BS is preferred over the unscreened 2BN for numerical stability: the
2BN formula involves cancellation of large terms that produces negative
values at MeV energies, while the screening correction ln(M) in the 2BS
dominates and keeps the cross section positive.

References:
    Koch HW & Motz JW, Rev. Mod. Phys. 31, 920 (1959)
    Geant4 Physics Reference Manual, Ch. "Livermore Bremsstrahlung"
    NASA TN D-4755 (Powell, 1968) eq. 1
"""

from __future__ import annotations

import logging
import math

import numpy as np

import config
from server.physics._validation import require_positive_energy, require_positive_z

log = logging.getLogger(__name__)


def bethe_heitler_2bn(
    electron_energy_mev: float,
    photon_energy_mev: float,
    emission_angle_rad: float,
    z: int | float,
) -> float:
    """Koch & Motz 2BS: d^2 sigma / (dk dOmega) in cm^2/(MeV sr atom).

    Despite the function name (retained for API compatibility), this
    implements the 2BS formula (Born with Schiff screening) which is
    numerically stable across the full MeV energy range.

    The 2BS doubly-differential cross section is (Geant4 PRM notation):

        d^2 sigma      2 Z^2 r_0^2 alpha   E_0^2
        ----------- = ------------------- x ----- x [A + B + C ln(M)]
        dk_MeV dOmega        pi             k_MeV

    where A, B, C are angular/energy terms and M is the Thomas-Fermi
    screening function.

    Args:
        electron_energy_mev: Electron kinetic energy in MeV.
        photon_energy_mev: Photon energy k in MeV.
        emission_angle_rad: Angle between photon and electron direction (rad).
        z: Atomic number of target nucleus.

    Returns:
        Doubly-differential cross section in cm^2/(MeV sr atom).
        Returns 0.0 if photon energy >= electron kinetic energy.
    """
    require_positive_energy(electron_energy_mev, "Electron energy")
    require_positive_energy(photon_energy_mev, "Photon energy")
    require_positive_z(z)

    if photon_energy_mev >= electron_energy_mev:
        return 0.0

    m0c2 = config.ELECTRON_MASS_MEV
    z_f = float(z)

    # Natural units (energies in m0c2)
    e0 = config.electron_total_energy_mev(electron_energy_mev) / m0c2  # gamma
    k_nat = photon_energy_mev / m0c2
    e_final = e0 - k_nat  # final electron total energy / m0c2

    if e_final <= 1.0:
        return 0.0

    # Fractional photon energy and final electron fraction
    x = k_nat / e0  # k / E_0  (0 < x < 1)
    one_minus_x = 1.0 - x

    # Angular parameter
    theta = emission_angle_rad
    u = (e0 * theta) ** 2  # y^2 = (E_0 * theta)^2
    u_plus_1 = u + 1.0

    # ----- 2BS angular/energy terms (Geant4 PRM) -----
    #
    # A = 16 u (1-x) / (u+1)^4
    # B = -(2-x)^2 / (u+1)^2
    # C = [1 + (1-x)^2] / (u+1)^2  -  4 u (1-x) / (u+1)^4
    #
    term_a = 16.0 * u * one_minus_x / u_plus_1**4
    term_b = -((2.0 - x) ** 2) / u_plus_1**2
    term_c = (1.0 + one_minus_x**2) / u_plus_1**2 - 4.0 * u * one_minus_x / u_plus_1**4

    # ----- Screening function M(y) -----
    #
    # 1/M = (k / (2 E_0 E))^2 + (Z^{1/3} / (111 (u+1)))^2
    #
    # For x -> 0 or large Z: 1/M is small, M is large, ln M is large positive.
    # This is the dominant term that keeps the cross section positive.
    #
    inv_m_energy = x / (2.0 * one_minus_x)  # = k/(2 E_0 E) in natural units
    inv_m_screen = z_f ** (1.0 / 3.0) / (111.0 * u_plus_1)
    inv_m_sq = inv_m_energy**2 + inv_m_screen**2

    if inv_m_sq <= 0:
        return 0.0

    ln_m = -math.log(inv_m_sq)  # ln(M) = -ln(1/M^2)/2... wait
    # Actually: 1/M(y) is defined, so M = 1/sqrt(inv_m_sq) for 2BS
    # ln(M) = -0.5 * ln(inv_m_sq)
    # But the Geant4 formula has:
    #   1/M(y) = [k/(2E0*E)]^2 + [Z^{1/3}/(111(y^2+1))]^2
    # and then ln M(y) is used.  Since M = 1/[stuff], ln M = -ln(stuff).
    # Here "stuff" = inv_m_sq, so ln M = -ln(inv_m_sq).
    #
    # This is consistent with Koch & Motz: M is a large number for small
    # screening, and ln M is large and positive.
    ln_m = -math.log(inv_m_sq)

    bracket = term_a + term_b + term_c * ln_m
    if bracket <= 0:
        return 0.0

    # ----- Prefactor -----
    #
    # d^2 sigma / (dk_MeV dOmega) = (2 Z^2 r_0^2 alpha / pi) * (E_0^2 / k_MeV) * bracket
    #
    # Derivation: d sigma = (4 Z^2 r_0^2 / 137)(dk/k)(y dy){bracket}
    # where y dy corresponds to E_0^2 dOmega/(2 pi) for small angles.
    #
    prefactor = (
        2.0
        * z_f**2
        * config.RE_SQUARED_CM2
        * config.ALPHA_FINE
        / math.pi
        * e0**2
        / photon_energy_mev
    )

    return prefactor * bracket


def thin_target_spectrum(
    electron_energy_mev: float,
    z: int | float,
    a: float,
    photon_energies_mev: list[float] | None = None,
    n_points: int = 100,
    n_angles: int = config.DEFAULT_THIN_TARGET_N_ANGLES,
    include_electron_electron: bool = True,
) -> tuple[list[float], list[float]]:
    """Angle-integrated thin-target bremsstrahlung spectrum.

    Integrates the 2BS cross section over solid angle to get dsigma/dk.

    Args:
        electron_energy_mev: Electron kinetic energy in MeV.
        z: Atomic number.
        a: Atomic weight in g/mol.
        photon_energies_mev: Specific photon energies to evaluate at.
            If None, generates n_points energies from k_min to k_max.
        n_points: Number of photon energy points if photon_energies_mev is None.
        n_angles: Number of angle quadrature points for angular integration.
        include_electron_electron: If True, replace Z^2 with Z(Z+1).

    Returns:
        Tuple of (photon_energies, cross_sections) where cross sections are
        in cm^2/(MeV atom).
    """
    if photon_energies_mev is None:
        k_min = config.SPECTRUM_K_FRACTION_MIN * electron_energy_mev
        k_max = config.SPECTRUM_K_FRACTION_MAX * electron_energy_mev
        photon_energies_mev = list(np.linspace(k_min, k_max, n_points))

    z_f = float(z)
    z_correction = (z_f + 1.0) / z_f if include_electron_electron else 1.0

    cross_sections: list[float] = []

    # Integrate over angle using uniform quadrature
    angles = np.linspace(0.001, math.pi, n_angles)
    d_theta = angles[1] - angles[0]

    for k in photon_energies_mev:
        if k >= electron_energy_mev or k <= 0:
            cross_sections.append(0.0)
            continue

        # Integrate 2*pi * sin(theta) * d2sigma/(dk dOmega) * dtheta
        integral = 0.0
        for theta in angles:
            ds = bethe_heitler_2bn(electron_energy_mev, k, theta, z)
            integral += 2.0 * math.pi * math.sin(theta) * ds * d_theta

        cross_sections.append(integral * z_correction)

    return photon_energies_mev, cross_sections


def _safe_log_ratio(energy: float, momentum: float) -> float:
    """Compute ln((E+p)/(E-p)) safely, avoiding division by zero.

    This equals 2*atanh(p/E) but handled for edge cases.
    """
    if momentum <= 0 or energy <= momentum:
        return 0.0
    ratio = (energy + momentum) / (energy - momentum)
    if ratio <= 0:
        return 0.0
    return math.log(ratio)
