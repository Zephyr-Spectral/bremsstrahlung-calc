"""Thick-target bremsstrahlung integration engine.

Depth-based slab integration over the electron CSDA track:

    I(k, phi_d) = k * (N_A/A) * (1-W) * (Z+1)/Z
                * integral_0^R_CSDA  DDCS(T(m), k, phi_d) * f(k, R-m) * dm

where m is the mass thickness (penetration depth in g/cm^2), T(m) is the
electron energy at depth m from the pre-computed ESTAR energy-depth profile,
R_CSDA is the total CSDA range, and:

  DDCS = BremsLib v2.0 partial-wave doubly-differential cross section
  f    = photon transmission/buildup through remaining target (R - m)
  W    = backscatter fraction (Wright & Trump fit)
  Z+1/Z = electron-electron bremsstrahlung correction

The scattering convolution broadens the angular distribution at each depth
using Berger's Legendre expansion with the true cumulative depth m.
"""

from __future__ import annotations

import logging
import math

import numpy as np

import config
from server.physics.attenuation import photon_transmission
from server.physics.bremslib import bremslib_ddcs_vec
from server.physics.electron_range import csda_range
from server.physics.scattering import (
    backscatter_fraction,
    scattering_probability_vec,
)
from server.physics.stopping_power import energy_at_depth

log = logging.getLogger(__name__)


def thick_target_intensity(
    electron_energy_mev: float,
    photon_energy_mev: float,
    detection_angle_deg: float,
    z: int | float,
    a: float,
    density_g_cm3: float,
    material_symbol: str | None = None,
    n_slabs: int = config.DEFAULT_N_SLABS,
    n_xi: int = config.DEFAULT_THICK_TARGET_N_XI,
    n_azimuth: int = config.DEFAULT_THICK_TARGET_N_AZIMUTH,
) -> float:
    """Compute thick-target bremsstrahlung intensity at a single point.

    Uses Powell's method: integrate thin-target BH contributions over
    the electron energy-loss path (eq. 1, NASA TN D-4755).

    Args:
        electron_energy_mev: Incident electron kinetic energy (MeV).
        photon_energy_mev: Photon energy (MeV).
        detection_angle_deg: Detection angle from beam axis (degrees).
        z: Atomic number.
        a: Atomic weight (g/mol).
        density_g_cm3: Material density (g/cm³).  Used only for photon attenuation.
        material_symbol: Material symbol for NASA attenuation data lookup.
        n_slabs: Number of electron-energy integration steps.
        n_xi: Number of electron scattering angle quadrature points.
        n_azimuth: Number of azimuthal quadrature points.

    Returns:
        Intensity I(k, phi_d) in MeV/(MeV sr electron).
    """
    if photon_energy_mev >= electron_energy_mev:
        return 0.0
    if photon_energy_mev <= 0 or electron_energy_mev <= 0:
        return 0.0

    z_f = float(z)

    # Z(Z+1) correction: replaces Z^2 in BH with Z(Z+1) for e-e brems
    ee_correction = (z_f + 1.0) / z_f

    # Backscatter correction (1 - W)
    bs_correction = 1.0 - backscatter_fraction(z, electron_energy_mev)

    # Total CSDA range [g/cm^2]
    total_range = csda_range(electron_energy_mev, z, a, n_steps=200)
    if total_range <= 0:
        return 0.0

    na_over_a = config.AVOGADRO / a  # N_A/A  [atoms/g]
    z_int = round(z_f)

    # --- Depth grid: uniform mass-thickness slabs ---
    # Pre-compute T(m) from ESTAR stopping power (cached per element/energy)
    delta_m = total_range / n_slabs  # uniform slab thickness [g/cm^2]
    m_mids = np.asarray((np.arange(n_slabs) + 0.5) * delta_m, dtype=np.float64)
    t_mids = energy_at_depth(m_mids, electron_energy_mev, z, a)  # vectorized

    # --- Scattering quadrature (log-spaced epsilon grid) ---
    eps_min = 0.001  # avoid eps=0 singularity
    eps_angles = np.logspace(np.log10(eps_min), np.log10(math.pi), n_xi)
    eps_widths = np.diff(eps_angles)
    eps_widths = np.append(eps_widths, eps_widths[-1])  # extend last bin

    psi_angles = np.linspace(0.0, 2.0 * math.pi, n_azimuth, endpoint=False)
    d_psi = 2.0 * math.pi / n_azimuth

    # theta_0 matrix: photon emission angle for each (eps, psi) pair
    # Independent of slab energy — compute once outside loop
    phi_d_rad = math.radians(detection_angle_deg)
    cos_phi_d = math.cos(phi_d_rad)
    sin_phi_d = math.sin(phi_d_rad)
    cos_theta0_mat = (
        np.cos(eps_angles)[:, np.newaxis] * cos_phi_d
        + np.sin(eps_angles)[:, np.newaxis] * sin_phi_d * np.cos(psi_angles)[np.newaxis, :]
    )
    theta_0_mat = np.arccos(np.clip(cos_theta0_mat, -1.0, 1.0))
    sin_eps = np.sin(eps_angles)

    # --- Slab loop: integrate over depth ---
    intensity_sum = 0.0

    for j in range(n_slabs):
        t_j = float(t_mids[j])
        if t_j <= photon_energy_mev:
            break

        m_j = float(m_mids[j])
        remaining_depth = max(total_range - m_j, 0.0)

        # Photon transmission through remaining target
        transmission = photon_transmission(
            photon_energy_mev,
            remaining_depth,
            detection_angle_deg,
            z,
            material_symbol,
        )

        # BremsLib partial-wave DDCS at electron energy T(m_j)
        ddcs_mat = bremslib_ddcs_vec(t_j, photon_energy_mev, theta_0_mat, z_int)

        # Scattering probability at true cumulative depth m_j
        p_eps_arr = scattering_probability_vec(eps_angles, m_j, z, t_j, a)

        # Weighted angular integration
        weights = p_eps_arr * sin_eps * eps_widths
        scatter_sum = float(np.dot(weights, ddcs_mat.sum(axis=1))) * d_psi

        # Accumulate: I += k * corrections * DDCS_convolved * dm * transmission
        intensity_sum += (
            photon_energy_mev
            * ee_correction
            * bs_correction
            * na_over_a
            * scatter_sum
            * delta_m
            * transmission
        )

    return intensity_sum


def thick_target_spectrum(
    electron_energy_mev: float,
    detection_angle_deg: float,
    z: int | float,
    a: float,
    density_g_cm3: float,
    material_symbol: str | None = None,
    n_points: int = config.DEFAULT_PHOTON_ENERGY_POINTS,
    n_slabs: int = config.DEFAULT_N_SLABS,
) -> tuple[list[float], list[float]]:
    """Compute full thick-target spectrum at a fixed detection angle.

    Args:
        electron_energy_mev: Incident electron kinetic energy (MeV).
        detection_angle_deg: Detection angle from beam axis (degrees).
        z: Atomic number.
        a: Atomic weight (g/mol).
        density_g_cm3: Material density (g/cm³).
        material_symbol: Material symbol for NASA data lookup.
        n_points: Number of photon energy points.
        n_slabs: Number of electron-energy integration steps.

    Returns:
        Tuple of (photon_energies_mev, intensities).
    """
    k_min = config.THICK_SPECTRUM_K_FRACTION_MIN * electron_energy_mev
    k_max = config.THICK_SPECTRUM_K_FRACTION_MAX * electron_energy_mev
    # Use log-spacing: bremsstrahlung spans decades and needs even sampling on log scale
    photon_energies = list(np.logspace(np.log10(k_min), np.log10(k_max), n_points))

    intensities = [
        thick_target_intensity(
            electron_energy_mev,
            k,
            detection_angle_deg,
            z,
            a,
            density_g_cm3,
            material_symbol,
            n_slabs,
        )
        for k in photon_energies
    ]

    return photon_energies, intensities


def angle_integrated_spectrum(
    electron_energy_mev: float,
    z: int | float,
    a: float,
    density_g_cm3: float,
    material_symbol: str | None = None,
    n_photon_points: int = config.DEFAULT_PHOTON_ENERGY_POINTS,
    n_angle_points: int = 19,
    n_slabs: int = config.DEFAULT_N_SLABS,
) -> tuple[list[float], list[float]]:
    """Compute angle-integrated thick-target spectrum.

    Integrates I(k, phi_d) * 2π sin(phi_d) over phi_d from 0 to π/2.

    Args:
        electron_energy_mev: Incident electron kinetic energy (MeV).
        z: Atomic number.
        a: Atomic weight (g/mol).
        density_g_cm3: Material density (g/cm³).
        material_symbol: Material symbol for NASA data lookup.
        n_photon_points: Number of photon energy points.
        n_angle_points: Number of angle integration points.
        n_slabs: Number of electron-energy integration steps.

    Returns:
        Tuple of (photon_energies_mev, integrated_intensities).
        Integrated intensities in MeV/(MeV electron).
    """
    angles_deg = list(np.linspace(0.0, 90.0, n_angle_points))
    d_angle_rad = math.radians(90.0 / (n_angle_points - 1))

    k_min = config.THICK_SPECTRUM_K_FRACTION_MIN * electron_energy_mev
    k_max = config.THICK_SPECTRUM_K_FRACTION_MAX * electron_energy_mev
    photon_energies = list(np.linspace(k_min, k_max, n_photon_points))

    integrated = []
    for k in photon_energies:
        total = 0.0
        for angle in angles_deg:
            intensity = thick_target_intensity(
                electron_energy_mev,
                k,
                angle,
                z,
                a,
                density_g_cm3,
                material_symbol,
                n_slabs,
            )
            total += intensity * 2.0 * math.pi * math.sin(math.radians(angle)) * d_angle_rad
        integrated.append(total)

    return photon_energies, integrated


def intensity_to_photon_rate(
    intensity: float,
    beam_current_ua: float,
) -> float:
    """Convert intensity to photon rate.

    photon_rate = intensity * (I_beam / e)

    where intensity is in MeV/(MeV sr electron) and the result is
    in photons/(MeV sr s).

    Args:
        intensity: Bremsstrahlung intensity in MeV/(MeV sr electron).
        beam_current_ua: Beam current in microamperes.

    Returns:
        Photon rate in photons/(MeV sr s). Returns 0.0 if beam_current is 0.
    """
    if beam_current_ua == 0:
        return 0.0
    electrons_per_second = (beam_current_ua * 1.0e-6) / config.ELEMENTARY_CHARGE_C
    return intensity * electrons_per_second
