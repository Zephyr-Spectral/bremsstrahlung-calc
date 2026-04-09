"""Photon attenuation and buildup in target materials.

Implements the photon transport model from NASA TN D-4755 (eq. 12):
  transmitted = B(mu*t) * exp(-mu*t)

where B is the Taylor buildup factor and mu is the mass attenuation coefficient.
"""

from __future__ import annotations

import json
import logging
import math

import numpy as np
from scipy.interpolate import interp1d

import config

log = logging.getLogger(__name__)

# Module-level cache for loaded attenuation data
_attenuation_data: dict[str, object] | None = None
_buildup_data: dict[str, object] | None = None


def _load_nasa_data() -> None:
    """Load attenuation and buildup data from NASA JSON."""
    global _attenuation_data, _buildup_data
    if _attenuation_data is not None:
        return

    if not config.NASA_DATA_PATH.exists():
        log.warning("NASA data not found, using parametric model only")
        return

    with config.NASA_DATA_PATH.open() as f:
        data = json.load(f)

    _attenuation_data = data.get("mass_attenuation")
    _buildup_data = data.get("buildup_coefficients")


def mass_attenuation_coefficient(
    photon_energy_mev: float,
    z: int | float,
    material_symbol: str | None = None,
) -> float:
    """Mass attenuation coefficient mu/rho in cm²/g.

    Uses NASA tabulated data for known materials, or parametric model
    for arbitrary Z.

    Args:
        photon_energy_mev: Photon energy in MeV.
        z: Atomic number.
        material_symbol: Material symbol for NASA data lookup.

    Returns:
        Mass attenuation coefficient in cm²/g.
    """
    if photon_energy_mev <= 0:
        msg = f"Photon energy must be positive, got {photon_energy_mev} MeV"
        raise ValueError(msg)

    # Try NASA data first
    _load_nasa_data()
    if _attenuation_data is not None and material_symbol in (
        _attenuation_data.get("materials", {})
    ):
        energies = _attenuation_data["photon_energy_mev"]
        values = _attenuation_data["materials"][material_symbol]
        return float(_interpolate_log(energies, values, photon_energy_mev))

    # Parametric model: Compton + photoelectric + pair production
    return _parametric_attenuation(photon_energy_mev, z)


def buildup_factor(
    photon_energy_mev: float,
    mu_x: float,
    material_symbol: str | None = None,
) -> float:
    """Photon buildup factor B(mu*x).

    Taylor form: B = A1*exp(alpha1*mu*x) + A2*exp(alpha2*mu*x)
    Simplified to single exponential when A2 data unavailable.

    Args:
        photon_energy_mev: Photon energy in MeV.
        mu_x: Product of attenuation coefficient and thickness (dimensionless).
        material_symbol: Material symbol for NASA data lookup.

    Returns:
        Buildup factor (>= 1.0).
    """
    if mu_x <= 0:
        return 1.0

    _load_nasa_data()
    if _buildup_data is not None and material_symbol in (_buildup_data.get("materials", {})):
        mat_data = _buildup_data["materials"][material_symbol]
        energies = _buildup_data["photon_energy_mev"]

        # Interpolate A1 and alpha1
        a1_values = mat_data["A1"]
        alpha1_values = mat_data["alpha1"]

        a1 = float(_interpolate_log(energies, a1_values, photon_energy_mev))
        alpha1 = float(_interpolate_log(energies, alpha1_values, photon_energy_mev))

        # Simplified Taylor form with single exponential
        # B ≈ 1 + A1 * alpha1 * mu_x (linear approximation for small mu_x)
        # or full form for larger mu_x
        return 1.0 + (a1 - 1.0) * (1.0 - math.exp(-alpha1 * mu_x))

    # Default: simple exponential buildup
    return 1.0 + 0.5 * mu_x


def photon_transmission(
    photon_energy_mev: float,
    remaining_depth_g_cm2: float,
    detection_angle_deg: float,
    z: int | float,
    material_symbol: str | None = None,
) -> float:
    """Photon transmission factor through remaining target material.

    Combines exponential attenuation with buildup factor.
    Accounts for the slant path through the target at angle phi_d.

    Args:
        photon_energy_mev: Photon energy in MeV.
        remaining_depth_g_cm2: Remaining target depth in g/cm².
        detection_angle_deg: Detection angle from normal in degrees.
        z: Atomic number.
        material_symbol: Material symbol for NASA data lookup.

    Returns:
        Transmission factor (0 to ~1, can exceed 1 with buildup).
    """
    if remaining_depth_g_cm2 <= 0 or photon_energy_mev <= 0:
        return 1.0

    mu_rho = mass_attenuation_coefficient(photon_energy_mev, z, material_symbol)

    # Slant path correction: effective thickness = t / cos(phi_d)
    cos_phi = math.cos(math.radians(detection_angle_deg))
    if cos_phi <= 0.01:  # near 90 degrees, very long path
        cos_phi = 0.01

    effective_depth = remaining_depth_g_cm2 / cos_phi
    mu_x = mu_rho * effective_depth

    b = buildup_factor(photon_energy_mev, mu_x, material_symbol)
    return b * math.exp(-mu_x)


def _parametric_attenuation(photon_energy_mev: float, z: int | float) -> float:
    """Parametric mass attenuation coefficient for arbitrary Z.

    Combines Compton scattering (Klein-Nishina), photoelectric absorption,
    and pair production contributions.

    Args:
        photon_energy_mev: Photon energy in MeV.
        z: Atomic number.

    Returns:
        mu/rho in cm²/g.
    """
    z_f = float(z)
    k = photon_energy_mev

    # Compton scattering (Klein-Nishina, per electron, scaled by Z/A ~ 0.5)
    # sigma_KN ≈ sigma_T * f(k/m0c2) where sigma_T = 0.665 barn
    x = k / config.ELECTRON_MASS_MEV
    if x < 0.01:
        f_kn = 1.0
    else:
        f_kn = (3.0 / (4.0 * x)) * (
            (1.0 + x) / x**3 * (2.0 * x * (1.0 + x) / (1.0 + 2.0 * x) - math.log(1.0 + 2.0 * x))
            + math.log(1.0 + 2.0 * x) / (2.0 * x)
            - (1.0 + 3.0 * x) / (1.0 + 2.0 * x) ** 2
        )
    sigma_compton = 0.0597 * z_f * f_kn  # cm²/g approximately (Z/A * sigma_T * N_A / A)

    # Photoelectric (approximate: tau ~ Z^4.5 / k^3.5 for k > K-edge)
    # Normalized to reproduce known values
    sigma_pe = 0.0
    if k < 3.0:
        sigma_pe = 2.0e-4 * z_f**4.5 / (k**3.5 * (z_f * 2.0))  # very rough

    # Pair production (threshold at 1.022 MeV)
    sigma_pair = 0.0
    if k > 2.0 * config.ELECTRON_MASS_MEV:
        sigma_pair = 5.0e-5 * z_f**2 * math.log(k / config.ELECTRON_MASS_MEV) / (z_f * 2.0)

    return sigma_compton + sigma_pe + sigma_pair


def _interpolate_log(
    x_data: list[float],
    y_data: list[float],
    x_query: float,
) -> float:
    """Log-linear interpolation with extrapolation clamping."""
    x_arr = np.array(x_data)
    y_arr = np.array(y_data, dtype=float)

    # Filter out any non-positive values for log interpolation
    mask = y_arr > 0
    if not np.any(mask):
        return 0.0

    x_valid = x_arr[mask]
    y_valid = y_arr[mask]

    if len(x_valid) < 2:
        return float(y_valid[0])

    # Clamp query to data range
    x_clamped = max(float(x_valid[0]), min(float(x_valid[-1]), x_query))

    log_y = np.log(y_valid)
    interp_fn = interp1d(x_valid, log_y, kind="linear", fill_value="extrapolate")
    return float(np.exp(interp_fn(x_clamped)))
