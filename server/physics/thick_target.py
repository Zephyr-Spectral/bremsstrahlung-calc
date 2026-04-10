"""Thick-target bremsstrahlung integration engine.

Implements Powell's method (NASA TN D-4755, eqs. 1-14): integrate
thin-target contributions over the electron slowing-down path:

    I(k, phi_d) = k * (N_A/A) * (1-W) * (Z+1)/Z
                * integral_0^T0  BH(T, k, phi_d) * f(k, X) * dT/S_tot(T)

where:
  BH  = doubly-differential thin-target cross section [cm²/(MeV sr atom)]
  f   = photon transmission/buildup through remaining target
  X   = cumulative path length from surface [g/cm²]
  S   = total electron stopping power [MeV cm²/g]
  W   = backscatter fraction
  Z+1/Z = electron-electron bremsstrahlung correction (Z² → Z(Z+1))

The multiple-scattering angular redistribution is a correction to the
angular distribution shape; at the dominant forward angles it broadens
the peak but does not change the integral significantly.  Including it
properly requires Molière theory with physical path-length units which
is deferred to a later phase.
"""

from __future__ import annotations

import logging
import math

import numpy as np

import config
from server.physics.attenuation import photon_transmission
from server.physics.bethe_heitler import bethe_heitler_2bn
from server.physics.electron_range import csda_range
from server.physics.scattering import (
    backscatter_fraction,
    scattering_broadened_angle,
    scattering_probability,
)
from server.physics.stopping_power import total_stopping_power

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

    # Energy grid: n_slabs equal-energy steps from T0 down to threshold
    t_min = photon_energy_mev + config.ELECTRON_MASS_MEV * 0.01
    t_max = electron_energy_mev
    if t_min >= t_max:
        return 0.0

    delta_e = t_max / n_slabs  # ΔE = E0/n (eq. 10)
    na_over_a = config.AVOGADRO / a  # N_A/A  [atoms/g]

    # Scattering quadrature (eq. 14 notation: alpha=0..beta, gamma=0..delta)
    eps_angles = np.linspace(0.0, math.pi, n_xi + 1)[1:]  # skip eps=0 (delta fn)
    d_eps = math.pi / n_xi
    psi_angles = np.linspace(0.0, 2.0 * math.pi, n_azimuth, endpoint=False)
    d_psi = 2.0 * math.pi / n_azimuth

    intensity_sum = 0.0
    cumulative_depth = 0.0  # g/cm^2

    for i in range(n_slabs):
        # Electron kinetic energy at slab i (eq. 10: E_i = E0 - i*DE)
        t_i = t_max - i * delta_e
        if t_i <= photon_energy_mev:
            break

        s_tot = total_stopping_power(t_i, z, a)
        if s_tot <= 0:
            break

        # Path-length element for this energy step [g/cm^2]
        dt_i = delta_e / s_tot

        # Physical depth at midpoint of this slab [g/cm^2]
        depth_mid = cumulative_depth + dt_i * 0.5
        remaining_depth = max(total_range - depth_mid, 0.0)

        # Photon transmission through remaining target (eq. 12-13)
        transmission = photon_transmission(
            photon_energy_mev,
            remaining_depth,
            detection_angle_deg,
            z,
            material_symbol,
        )

        # --- Scattering convolution (eq. 14 inner double sum) ---
        # Sum over electron scattering angles (epsilon) and azimuthal (psi).
        # theta_0 = angle between scattered electron and photon direction
        # computed from spherical triangle (eq. 3).
        scatter_sum = 0.0

        for eps in eps_angles:
            p_eps = scattering_probability(eps, depth_mid, z, t_i, a)

            for psi in psi_angles:
                theta_0 = scattering_broadened_angle(detection_angle_deg, eps, psi)
                bh = bethe_heitler_2bn(t_i, photon_energy_mev, theta_0, z)
                scatter_sum += bh * p_eps * math.sin(eps) * d_eps * d_psi

        # Also add the epsilon=0 contribution (forward electrons)
        # At eps=0: theta_0 = phi_d, P_eps is the forward peak
        p_eps_0 = scattering_probability(0.001, depth_mid, z, t_i, a)
        theta_0_fwd = scattering_broadened_angle(detection_angle_deg, 0.001, 0.0)
        bh_fwd = bethe_heitler_2bn(t_i, photon_energy_mev, theta_0_fwd, z)
        scatter_sum += bh_fwd * p_eps_0 * math.sin(0.001) * d_eps * 2.0 * math.pi

        # Eq. 14: contribution from slab i
        intensity_sum += (
            photon_energy_mev
            * ee_correction
            * bs_correction
            * na_over_a
            * scatter_sum
            * dt_i
            * transmission
        )

        cumulative_depth += dt_i

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
