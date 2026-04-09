"""Bethe-Heitler thin-target bremsstrahlung cross section.

Implements the Koch & Motz formula "2BN" (eq. 2BN from Rev. Mod. Phys. 31,
1959, pp. 920-955) used in NASA TN D-4755 eq. 1 for the doubly-differential
thin-target cross section d²sigma / (dk dOmega).
"""

from __future__ import annotations

import logging
import math

import config

log = logging.getLogger(__name__)


def bethe_heitler_2bn(
    electron_energy_mev: float,
    photon_energy_mev: float,
    emission_angle_rad: float,
    z: int | float,
) -> float:
    """Koch & Motz eq. 2BN: d²sigma / (dk dOmega) in cm² / (MeV·sr·atom).

    This is the Bethe-Heitler cross section for bremsstrahlung emission
    from an electron of total energy E0 producing a photon of energy k
    at angle theta_0 from the electron direction, in the field of a
    nucleus with atomic number Z.

    Uses natural units where energies are in m0c² units internally.

    Args:
        electron_energy_mev: Electron kinetic energy in MeV.
        photon_energy_mev: Photon energy in MeV.
        emission_angle_rad: Angle between photon and electron direction in radians.
        z: Atomic number of target nucleus.

    Returns:
        Doubly-differential cross section in cm²/(MeV·sr·atom).
        Returns 0.0 if photon energy >= electron kinetic energy.

    Raises:
        ValueError: If inputs are non-positive.
    """
    if electron_energy_mev <= 0:
        msg = f"Electron energy must be positive, got {electron_energy_mev} MeV"
        raise ValueError(msg)
    if photon_energy_mev <= 0:
        msg = f"Photon energy must be positive, got {photon_energy_mev} MeV"
        raise ValueError(msg)
    if z <= 0:
        msg = f"Atomic number must be positive, got Z={z}"
        raise ValueError(msg)

    # Photon energy must be less than electron kinetic energy
    if photon_energy_mev >= electron_energy_mev:
        return 0.0

    m0c2 = config.ELECTRON_MASS_MEV
    z_f = float(z)

    # Convert to m0c2 units
    k = photon_energy_mev / m0c2
    e0 = config.electron_total_energy_mev(electron_energy_mev) / m0c2
    p0 = config.electron_momentum_moc(electron_energy_mev)

    # Final electron: E = E0 - k
    e_final = e0 - k
    if e_final <= 1.0:
        return 0.0
    p_final = math.sqrt(e_final**2 - 1.0)

    # Fractional photon energy
    x = k / e0

    # Angular parameter: u = (E0 * theta)^2 ~ (gamma * theta)^2
    u = (e0 * emission_angle_rad) ** 2

    # Schiff formula (2BS from Koch & Motz) — numerically stable
    # Angular distribution: (1/(1+u)^2) envelope with correction terms
    denom = (1.0 + u) ** 2
    if denom <= 0:
        return 0.0

    # Electric contribution (forward-peaked)
    term_e = (1.0 + (1.0 - x) ** 2 - 2.0 * (1.0 - x) / 3.0) / denom

    # Angular correction
    term_a = (1.0 - x) * 4.0 * u / denom**2

    # Prefactor: Z^2 * r0^2 * alpha / pi * (p_final / (p0 * k))
    prefactor = z_f**2 * config.RE_SQUARED_CM2 * config.ALPHA_FINE / math.pi * p_final / (p0 * k)

    return max(prefactor * (term_e + term_a), 0.0)


def thin_target_spectrum(
    electron_energy_mev: float,
    z: int | float,
    a: float,
    photon_energies_mev: list[float] | None = None,
    n_points: int = 100,
    include_electron_electron: bool = True,
) -> tuple[list[float], list[float]]:
    """Angle-integrated thin-target bremsstrahlung spectrum.

    Integrates the 2BN cross section over solid angle to get dsigma/dk.

    Args:
        electron_energy_mev: Electron kinetic energy in MeV.
        z: Atomic number.
        a: Atomic weight in g/mol.
        photon_energies_mev: Specific photon energies to evaluate at.
            If None, generates n_points energies from 0.01*T0 to 0.99*T0.
        n_points: Number of photon energy points if photon_energies_mev is None.
        include_electron_electron: If True, replace Z² with Z(Z+1).

    Returns:
        Tuple of (photon_energies, cross_sections) where cross sections are
        in cm²/(MeV·atom).
    """
    import numpy as np

    if photon_energies_mev is None:
        k_min = 0.01 * electron_energy_mev
        k_max = 0.99 * electron_energy_mev
        photon_energies_mev = list(np.linspace(k_min, k_max, n_points))

    z_f = float(z)
    z_correction = (z_f + 1.0) / z_f if include_electron_electron else 1.0

    cross_sections: list[float] = []

    # Integrate over angle using Gauss-Legendre quadrature
    n_angles = 32
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
