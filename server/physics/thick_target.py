"""Thick-target bremsstrahlung integration engine.

Implements Powell's method (NASA TN D-4755, eqs. 1-14): approximate a thick
target as a series of thin slabs, sum thin-target bremsstrahlung contributions
over the electron path through the target, including:
  1. Multiple electron scattering (Berger)
  2. Electron backscatter correction (Wright & Trump)
  3. Electron-electron bremsstrahlung: Z² → Z(Z+1)
  4. Photon attenuation and buildup in target
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
) -> float:
    """Compute thick-target bremsstrahlung intensity at a single point.

    Implements NASA TN D-4755 eq. 14: sums thin-target contributions
    over n_slabs, each treated as thin target for BH production.

    Args:
        electron_energy_mev: Incident electron kinetic energy (MeV).
        photon_energy_mev: Photon energy (MeV).
        detection_angle_deg: Detection angle from target normal (degrees).
        z: Atomic number.
        a: Atomic weight (g/mol).
        density_g_cm3: Material density (g/cm³).
        material_symbol: Material symbol for NASA data lookup.
        n_slabs: Number of thin slabs for integration.

    Returns:
        Intensity I(k, phi_d) in MeV/(MeV·sr·electron).
    """
    if photon_energy_mev >= electron_energy_mev:
        return 0.0
    if photon_energy_mev <= 0 or electron_energy_mev <= 0:
        return 0.0

    z_f = float(z)

    # Total target thickness = mean electron range
    total_range = csda_range(electron_energy_mev, z, a, n_steps=200)
    if total_range <= 0:
        return 0.0

    # Backscatter correction (1 - W)
    w = backscatter_fraction(z, electron_energy_mev)
    bs_correction = 1.0 - w

    # Electron-electron bremsstrahlung: Z(Z+1) instead of Z²
    ee_correction = (z_f + 1.0) / z_f

    # Energy decrement per slab
    de_slab = electron_energy_mev / n_slabs

    # Depth increment per slab (in g/cm²)
    dt_slab = total_range / n_slabs

    intensity_sum = 0.0

    # Number of azimuthal angles for scattering integration
    n_azimuth = 8
    azimuth_angles = np.linspace(0.0, 2.0 * math.pi, n_azimuth, endpoint=False)
    d_azimuth = 2.0 * math.pi / n_azimuth

    # Number of electron angles for scattering integration
    n_xi = 8
    xi_angles = np.linspace(0.01, math.pi / 2, n_xi)
    d_xi = xi_angles[1] - xi_angles[0]

    for i in range(n_slabs):
        # Electron energy at slab i (decreasing with depth)
        e_i = electron_energy_mev - i * de_slab
        if e_i <= photon_energy_mev + config.ELECTRON_MASS_MEV:
            break  # electron energy too low to produce this photon

        depth_fraction = (i + 0.5) / n_slabs
        remaining_depth = total_range * (1.0 - depth_fraction)

        # Photon transmission through remaining target
        transmission = photon_transmission(
            photon_energy_mev,
            remaining_depth,
            detection_angle_deg,
            z,
            material_symbol,
        )

        # Number density * slab thickness
        na_dt = config.AVOGADRO * dt_slab / a

        # Integrate over electron scattering angles
        slab_contribution = 0.0

        for xi in xi_angles:
            # Scattering probability at this depth and angle
            p_s = scattering_probability(xi, depth_fraction, z, e_i)

            for psi in azimuth_angles:
                # Photon emission angle relative to electron direction
                theta_0 = scattering_broadened_angle(detection_angle_deg, xi, psi)

                # Thin-target BH cross section at this angle
                dsigma = bethe_heitler_2bn(e_i, photon_energy_mev, theta_0, z)

                # Contribution: P_s * (d²sigma/dk dOmega) * sin(xi) * dxi * dpsi
                slab_contribution += p_s * dsigma * math.sin(xi) * d_xi * d_azimuth

        # Multiply by: photon_energy * na_dt * transmission * corrections
        intensity_sum += (
            photon_energy_mev
            * slab_contribution
            * na_dt
            * transmission
            * bs_correction
            * ee_correction
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
        detection_angle_deg: Detection angle from target normal (degrees).
        z: Atomic number.
        a: Atomic weight (g/mol).
        density_g_cm3: Material density (g/cm³).
        material_symbol: Material symbol for NASA data lookup.
        n_points: Number of photon energy points.
        n_slabs: Number of thin slabs for integration.

    Returns:
        Tuple of (photon_energies_mev, intensities).
    """
    k_min = 0.05 * electron_energy_mev
    k_max = 0.95 * electron_energy_mev
    photon_energies = list(np.linspace(k_min, k_max, n_points))

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

    Integrates I(k, phi_d) * 2*pi*sin(phi_d) over phi_d from 0 to pi/2.

    Args:
        electron_energy_mev: Incident electron kinetic energy (MeV).
        z: Atomic number.
        a: Atomic weight (g/mol).
        density_g_cm3: Material density (g/cm³).
        material_symbol: Material symbol for NASA data lookup.
        n_photon_points: Number of photon energy points.
        n_angle_points: Number of angle integration points.
        n_slabs: Number of thin slabs.

    Returns:
        Tuple of (photon_energies_mev, integrated_intensities).
        Integrated intensities in MeV/(MeV·electron).
    """
    angles_deg = list(np.linspace(0.0, 90.0, n_angle_points))
    d_angle_rad = math.radians(90.0 / (n_angle_points - 1))

    k_min = 0.05 * electron_energy_mev
    k_max = 0.95 * electron_energy_mev
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

    where intensity is in MeV/(MeV·sr·electron) and the result is
    in photons/(MeV·sr·s).

    Args:
        intensity: Bremsstrahlung intensity in MeV/(MeV·sr·electron).
        beam_current_ua: Beam current in microamperes.

    Returns:
        Photon rate in photons/(MeV·sr·s). Returns 0.0 if beam_current is 0.
    """
    if beam_current_ua == 0:
        return 0.0
    electrons_per_second = (beam_current_ua * 1.0e-6) / config.ELEMENTARY_CHARGE_C
    return intensity * electrons_per_second
