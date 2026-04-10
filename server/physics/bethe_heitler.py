"""Bethe-Heitler thin-target bremsstrahlung cross sections.

Implements both:
  - Koch & Motz 2BN (eq. 1 in NASA TN D-4755): full unscreened Born
    approximation, valid at ALL angles. This is what Powell used.
  - Koch & Motz 2BS (Schiff with screening): small-angle approximation,
    used as fallback when 2BN gives negative values due to numerical
    cancellation at extreme relativistic energies.

References:
    Koch HW & Motz JW, Rev. Mod. Phys. 31, 920 (1959)
    Powell CA, NASA TN D-4755 (1968), eq. 1
    Geant4 Physics Reference Manual, "Livermore Bremsstrahlung"
"""

from __future__ import annotations

import logging
import math

import numpy as np
import numpy.typing as npt

import config
from server.physics._validation import require_positive_energy, require_positive_z

log = logging.getLogger(__name__)


def bethe_heitler_2bn(
    electron_energy_mev: float,
    photon_energy_mev: float,
    emission_angle_rad: float,
    z: int | float,
) -> float:
    """Koch & Motz 2BN: d^2 sigma / (dk dOmega) in cm^2/(MeV sr atom).

    Full unscreened Born approximation (NASA TN D-4755 eq. 1).
    Valid at all angles, unlike the 2BS Schiff approximation.
    Falls back to 2BS if 2BN gives a negative result (rare numerical
    cancellation at extreme relativistic energies).

    All internal calculations use natural units (m0c^2 = 1).

    Args:
        electron_energy_mev: Electron kinetic energy in MeV.
        photon_energy_mev: Photon energy k in MeV.
        emission_angle_rad: Angle theta_0 between photon and electron (rad).
        z: Atomic number of target nucleus.

    Returns:
        Doubly-differential cross section in cm^2/(MeV sr atom).
    """
    require_positive_energy(electron_energy_mev, "Electron energy")
    require_positive_energy(photon_energy_mev, "Photon energy")
    require_positive_z(z)

    if photon_energy_mev >= electron_energy_mev:
        return 0.0

    m0c2 = config.ELECTRON_MASS_MEV
    z_f = float(z)

    # Natural units (energies in m0c2, momenta in m0c)
    e0 = config.electron_total_energy_mev(electron_energy_mev) / m0c2
    p0 = config.electron_momentum_moc(electron_energy_mev)
    k = photon_energy_mev / m0c2

    e_f = e0 - k  # final electron total energy
    if e_f <= 1.0:
        return 0.0
    p_f = math.sqrt(e_f**2 - 1.0)

    theta = emission_angle_rad
    cos_theta = math.cos(theta)
    sin2_theta = math.sin(theta) ** 2

    # Auxiliary variables (Koch & Motz / Powell notation)
    delta_0 = e0 - p0 * cos_theta
    if delta_0 <= 0:
        return 0.0

    q_sq = p0**2 + k**2 - 2.0 * p0 * k * cos_theta
    if q_sq <= 0:
        return 0.0
    q_val = math.sqrt(q_sq)

    # L = ln((E*E0 - 1 + p*p0) / (E*E0 - 1 - p*p0))
    ee0_m1 = e_f * e0 - 1.0
    pp0 = p_f * p0
    l_num = ee0_m1 + pp0
    l_den = ee0_m1 - pp0
    if l_den <= 0 or l_num <= 0:
        return _bethe_heitler_2bs(electron_energy_mev, photon_energy_mev, theta, z)
    big_l = math.log(l_num / l_den)

    # epsilon = ln((E + p) / (E - p))
    ep_num = e_f + p_f
    ep_den = e_f - p_f
    if ep_den <= 0:
        return _bethe_heitler_2bs(electron_energy_mev, photon_energy_mev, theta, z)
    eps = math.log(ep_num / ep_den)

    # epsilon^Q = ln((Q + p) / (Q - p))
    qp_num = q_val + p_f
    qp_den = q_val - p_f
    eps_q = 0.0 if qp_den <= 0 else math.log(qp_num / qp_den)

    p0_sq = p0**2
    delta_0_sq = delta_0**2

    # --- 2BN bracket (Powell eq. 1) ---
    # Term 1: 8 sin^2(theta) (2 E0^2 + 1) / (p0^2 Delta0^4)
    t1 = 8.0 * sin2_theta * (2.0 * e0**2 + 1.0) / (p0_sq * delta_0**4)

    # Term 2: -2(5 E0^2 + 2 E E0 + 3) / (p0^2 Delta0^2)
    t2 = -2.0 * (5.0 * e0**2 + 2.0 * e_f * e0 + 3.0) / (p0_sq * delta_0_sq)

    # Term 3: -2(p0^2 - k^2) / (Q^2 Delta0^2)
    t3 = -2.0 * (p0_sq - k**2) / (q_sq * delta_0_sq)

    # Term 4: +4 E / (p0^2 Delta0)
    t4 = 4.0 * e_f / (p0_sq * delta_0)

    # Term 5: L/(p p0) * [sub-bracket]
    l_over_pp0 = big_l / (p_f * p0)
    sub5_a = 4.0 * e0 * sin2_theta * (3.0 * k - p0_sq * e_f) / (p0_sq * delta_0**4)
    sub5_b = 4.0 * e0**2 * (e0**2 + e_f**2) / (p0_sq * delta_0_sq)
    sub5_c = (2.0 - 2.0 * (7.0 * e0**2 - 3.0 * e_f * e0 + e_f**2)) / (p0_sq * delta_0_sq)
    sub5_d = 2.0 * k * (e0**2 + e_f * e0 - 1.0) / (p0_sq * delta_0)
    t5 = l_over_pp0 * (sub5_a + sub5_b + sub5_c + sub5_d)

    # Term 6: -4 epsilon / (p Delta0)
    t6 = -4.0 * eps / (p_f * delta_0)

    # Term 7: epsilon^Q / (p Q) * [sub-bracket]
    if q_val > 0 and p_f > 0:
        eq_over_pq = eps_q / (p_f * q_val)
        sub7_a = 4.0 / delta_0_sq
        sub7_b = -6.0 * k / delta_0
        sub7_c = -2.0 * k * (p0_sq - k**2) / (q_sq * delta_0)
        t7 = eq_over_pq * (sub7_a + sub7_b + sub7_c)
    else:
        t7 = 0.0

    bracket = t1 + t2 + t3 + t4 + t5 + t6 + t7

    # If 2BN gives negative (rare numerical cancellation), fall back to 2BS
    if bracket <= 0:
        return _bethe_heitler_2bs(electron_energy_mev, photon_energy_mev, theta, z)

    # Prefactor: Z^2 r0^2 / (8 pi * 137) * (1/k) * (p/p0)
    # Result in cm^2/(m0c2 * sr * atom). Divide by m0c2 for per MeV.
    prefactor = z_f**2 * config.RE_SQUARED_CM2 / (8.0 * math.pi * 137.0) * (1.0 / k) * (p_f / p0)

    # Convert from per m0c2 to per MeV: divide by m0c2
    return prefactor * bracket / m0c2


def _bethe_heitler_2bs(
    electron_energy_mev: float,
    photon_energy_mev: float,
    emission_angle_rad: float,
    z: int | float,
) -> float:
    """Koch & Motz 2BS (Schiff with screening) fallback.

    Used when 2BN gives negative values due to numerical cancellation.
    Valid primarily at small angles.
    """
    m0c2 = config.ELECTRON_MASS_MEV
    z_f = float(z)

    e0 = config.electron_total_energy_mev(electron_energy_mev) / m0c2
    k_nat = photon_energy_mev / m0c2
    e_final = e0 - k_nat
    if e_final <= 1.0:
        return 0.0

    x = k_nat / e0
    one_minus_x = 1.0 - x
    u = (e0 * emission_angle_rad) ** 2
    u_plus_1 = u + 1.0

    # 2BS angular terms
    term_a = 16.0 * u * one_minus_x / u_plus_1**4
    term_b = -((2.0 - x) ** 2) / u_plus_1**2
    term_c = (1.0 + one_minus_x**2) / u_plus_1**2 - 4.0 * u * one_minus_x / u_plus_1**4

    # Screening function M(y)
    inv_m_energy = x / (2.0 * one_minus_x)
    inv_m_screen = z_f ** (1.0 / 3.0) / (111.0 * u_plus_1)
    inv_m_sq = inv_m_energy**2 + inv_m_screen**2
    if inv_m_sq <= 0:
        return 0.0
    ln_m = -math.log(inv_m_sq)

    bracket = term_a + term_b + term_c * ln_m
    if bracket <= 0:
        return 0.0

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

    Integrates the 2BN cross section over solid angle to get dsigma/dk.

    Args:
        electron_energy_mev: Electron kinetic energy in MeV.
        z: Atomic number.
        a: Atomic weight in g/mol.
        photon_energies_mev: Specific photon energies to evaluate at.
        n_points: Number of photon energy points if photon_energies_mev is None.
        n_angles: Number of angle quadrature points for angular integration.
        include_electron_electron: If True, replace Z^2 with Z(Z+1).

    Returns:
        Tuple of (photon_energies, cross_sections) in cm^2/(MeV atom).
    """
    if photon_energies_mev is None:
        k_min = config.SPECTRUM_K_FRACTION_MIN * electron_energy_mev
        k_max = config.SPECTRUM_K_FRACTION_MAX * electron_energy_mev
        photon_energies_mev = list(np.linspace(k_min, k_max, n_points))

    z_f = float(z)
    z_correction = (z_f + 1.0) / z_f if include_electron_electron else 1.0

    cross_sections: list[float] = []
    angles = np.linspace(0.001, math.pi, n_angles)
    d_theta = angles[1] - angles[0]

    for k in photon_energies_mev:
        if k >= electron_energy_mev or k <= 0:
            cross_sections.append(0.0)
            continue

        integral = 0.0
        for theta in angles:
            ds = bethe_heitler_2bn(electron_energy_mev, k, theta, z)
            integral += 2.0 * math.pi * math.sin(theta) * ds * d_theta

        cross_sections.append(integral * z_correction)

    return photon_energies_mev, cross_sections


# ---------------------------------------------------------------------------
# Vectorized variants (avoid Python-level loops for array of angles)
# ---------------------------------------------------------------------------


def _bethe_heitler_2bs_vec(
    electron_energy_mev: float,
    photon_energy_mev: float,
    emission_angles_rad: npt.NDArray[np.float64],
    z: int | float,
) -> npt.NDArray[np.float64]:
    """Vectorized 2BS fallback for an array of emission angles."""
    m0c2 = config.ELECTRON_MASS_MEV
    z_f = float(z)
    e0 = config.electron_total_energy_mev(electron_energy_mev) / m0c2
    k_nat = photon_energy_mev / m0c2
    e_final = e0 - k_nat
    if e_final <= 1.0:
        return np.zeros_like(emission_angles_rad)
    x = k_nat / e0
    one_minus_x = 1.0 - x
    u = (e0 * emission_angles_rad) ** 2
    u_plus_1 = u + 1.0
    term_a = 16.0 * u * one_minus_x / u_plus_1**4
    term_b = -((2.0 - x) ** 2) / u_plus_1**2
    term_c = (1.0 + one_minus_x**2) / u_plus_1**2 - 4.0 * u * one_minus_x / u_plus_1**4
    inv_m_energy = x / (2.0 * max(one_minus_x, 1e-30))
    inv_m_screen = z_f ** (1.0 / 3.0) / (111.0 * u_plus_1)
    inv_m_sq = inv_m_energy**2 + inv_m_screen**2
    ln_m = -np.log(np.maximum(inv_m_sq, 1e-30))
    bracket = np.maximum(term_a + term_b + term_c * ln_m, 0.0)
    prefactor = (
        2.0
        * z_f**2
        * config.RE_SQUARED_CM2
        * config.ALPHA_FINE
        / math.pi
        * e0**2
        / photon_energy_mev
    )
    return prefactor * bracket  # type: ignore[no-any-return]  # numpy product returns ndarray


def bethe_heitler_2bn_vec(
    electron_energy_mev: float,
    photon_energy_mev: float,
    emission_angles_rad: npt.NDArray[np.float64],
    z: int | float,
) -> npt.NDArray[np.float64]:
    """Vectorized Koch & Motz 2BN for an array of emission angles.

    Equivalent to calling bethe_heitler_2bn() for each angle but uses
    numpy broadcasting, eliminating the Python-level loop.  The 2BS
    fallback (for negative brackets) is also fully vectorized.

    Args:
        electron_energy_mev: Electron kinetic energy in MeV.
        photon_energy_mev: Photon energy k in MeV.
        emission_angles_rad: Array of emission angles in radians (any shape).
        z: Atomic number.

    Returns:
        d^2 sigma / (dk dOmega) in cm^2/(MeV sr atom), same shape as input.
    """
    require_positive_energy(electron_energy_mev, "Electron energy")
    require_positive_energy(photon_energy_mev, "Photon energy")
    require_positive_z(z)

    if photon_energy_mev >= electron_energy_mev:
        return np.zeros_like(emission_angles_rad)

    m0c2 = config.ELECTRON_MASS_MEV
    z_f = float(z)
    e0 = config.electron_total_energy_mev(electron_energy_mev) / m0c2
    p0 = config.electron_momentum_moc(electron_energy_mev)
    k = photon_energy_mev / m0c2
    e_f = e0 - k
    if e_f <= 1.0:
        return np.zeros_like(emission_angles_rad)
    p_f = math.sqrt(e_f**2 - 1.0)

    # Scalar invariants (angle-independent)
    ee0_m1 = e_f * e0 - 1.0
    pp0 = p_f * p0
    l_num, l_den = ee0_m1 + pp0, ee0_m1 - pp0
    ep_num, ep_den = e_f + p_f, e_f - p_f
    if l_den <= 0 or l_num <= 0 or ep_den <= 0:
        return _bethe_heitler_2bs_vec(
            electron_energy_mev, photon_energy_mev, emission_angles_rad, z
        )
    big_l = math.log(l_num / l_den)
    eps_val = math.log(ep_num / ep_den)

    # Array operations: all theta-dependent quantities
    cos_theta = np.cos(emission_angles_rad)
    sin2_theta = np.sin(emission_angles_rad) ** 2
    delta_0 = e0 - p0 * cos_theta
    q_sq = p0**2 + k**2 - 2.0 * p0 * k * cos_theta
    valid = (delta_0 > 0) & (q_sq > 0)

    # Safe denominators to avoid division by zero in invalid regions
    d0 = np.where(valid, delta_0, 1.0)
    qs = np.where(valid, q_sq, 1.0)
    qv = np.sqrt(qs)

    # eps_Q = log((Q + p_f) / (Q - p_f)) where it exists
    qp_den = qv - p_f
    valid_q = valid & (qp_den > 0)
    qp_ratio = np.where(valid_q, (qv + p_f) / np.where(qp_den > 0, qp_den, 1.0), 1.0)
    eps_q = np.where(valid_q, np.log(np.maximum(qp_ratio, 1e-30)), 0.0)

    p0_sq = p0**2

    # 2BN bracket terms (Powell eq. 1)
    t1 = 8.0 * sin2_theta * (2.0 * e0**2 + 1.0) / (p0_sq * d0**4)
    t2 = -2.0 * (5.0 * e0**2 + 2.0 * e_f * e0 + 3.0) / (p0_sq * d0**2)
    t3 = -2.0 * (p0_sq - k**2) / (qs * d0**2)
    t4 = 4.0 * e_f / (p0_sq * d0)
    l_pp0 = big_l / (p_f * p0)
    sub5 = (
        4.0 * e0 * sin2_theta * (3.0 * k - p0_sq * e_f) / (p0_sq * d0**4)
        + 4.0 * e0**2 * (e0**2 + e_f**2) / (p0_sq * d0**2)
        + (2.0 - 2.0 * (7.0 * e0**2 - 3.0 * e_f * e0 + e_f**2)) / (p0_sq * d0**2)
        + 2.0 * k * (e0**2 + e_f * e0 - 1.0) / (p0_sq * d0)
    )
    t5 = l_pp0 * sub5
    t6 = -4.0 * eps_val / (p_f * d0)
    eq_pq = np.where(valid_q, eps_q / (p_f * qv), 0.0)
    t7 = eq_pq * (4.0 / d0**2 - 6.0 * k / d0 - 2.0 * k * (p0_sq - k**2) / (qs * d0))
    bracket = t1 + t2 + t3 + t4 + t5 + t6 + t7

    prefactor = (
        z_f**2 * config.RE_SQUARED_CM2 / (8.0 * math.pi * 137.0) * (1.0 / k) * (p_f / p0) / m0c2
    )
    result = prefactor * bracket

    # Fallback to vectorized 2BS where 2BN gives negative or invalid
    neg_mask = ~valid | (bracket <= 0)
    if np.any(neg_mask):
        bh_2bs = _bethe_heitler_2bs_vec(
            electron_energy_mev, photon_energy_mev, emission_angles_rad, z
        )
        result = np.where(neg_mask, bh_2bs, result)

    return result  # type: ignore[no-any-return]  # numpy where/product returns ndarray
