"""Photon attenuation and buildup in target materials.

Primary data source: NIST XCOM photon cross sections (xcom_elements.json),
which includes the full absorption edge structure (K, L, M edges).

Transmission model (NASA TN D-4755 eq. 12):
    T = B(mu*x) * exp(-mu*x)

where B is the Taylor buildup factor and mu is the mass attenuation
coefficient.  For the thick-target integration, x is the slant-path
thickness from the emission point to the target exit.
"""

from __future__ import annotations

import json
import logging
import math
from typing import Any

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import-untyped]

import config
from server.data_access import get_nasa_data
from server.physics._validation import require_positive_energy

log = logging.getLogger(__name__)

# Module-level XCOM cache: loaded once at first use
_xcom_cache: dict[str, Any] | None = None


def _get_xcom_data() -> dict[str, Any]:
    """Load and cache NIST XCOM element data."""
    global _xcom_cache
    if _xcom_cache is None:
        with config.XCOM_DATA_PATH.open() as f:
            _xcom_cache = json.load(f)
    return _xcom_cache


def absorption_edges(
    material_symbol: str,
) -> list[float]:
    """Return absorption edge energies (MeV) for a material from XCOM data.

    Edges are identified by duplicate energy entries in the XCOM table.
    Returns a sorted list of edge energies (K, L, M edges as available).
    """
    if material_symbol not in config.XCOM_ELEMENT_NAMES:
        return []

    element_name = config.XCOM_ELEMENT_NAMES[material_symbol]
    xcom = _get_xcom_data()
    if element_name not in xcom:
        return []

    coeffs = xcom[element_name]["coefficients"]
    energies: list[float] = [c["energy_MeV"] for c in coeffs]
    edges: list[float] = []
    for i in range(len(energies) - 1):
        if energies[i] == energies[i + 1] and energies[i] not in edges:
            edges.append(float(energies[i]))
    return sorted(edges)


def mass_attenuation_coefficient(
    photon_energy_mev: float,
    z: int | float,
    material_symbol: str | None = None,
) -> float:
    """Mass attenuation coefficient mu/rho in cm^2/g.

    Uses NIST XCOM data (full edge structure) for all 10 NASA materials.
    Falls back to parametric model for unknown materials (SS304/316, etc.).

    XCOM data includes K/L/M absorption edges — the sharp jumps in mu/rho
    at element-specific energies are physically correct and cause step
    structure in the bremsstrahlung spectrum for high-Z materials (e.g.
    W K-edge at 0.069 MeV).

    Args:
        photon_energy_mev: Photon energy in MeV.
        z: Atomic number.
        material_symbol: Material symbol (e.g. 'W', 'Cu').

    Returns:
        Mass attenuation coefficient in cm^2/g.
    """
    require_positive_energy(photon_energy_mev, "Photon energy")

    # Try XCOM data first for known NASA materials
    if material_symbol in config.XCOM_ELEMENT_NAMES:
        element_name = config.XCOM_ELEMENT_NAMES[material_symbol]
        xcom = _get_xcom_data()
        if element_name in xcom:
            coeffs = xcom[element_name]["coefficients"]
            energies = [c["energy_MeV"] for c in coeffs]
            mu_rho = [c["mu_over_rho"] for c in coeffs]
            return float(_interpolate_xcom(energies, mu_rho, photon_energy_mev))

    # Fallback: parametric model for SS304/316 and unknown materials
    return _parametric_attenuation(photon_energy_mev, z)


def buildup_factor(
    photon_energy_mev: float,
    mu_x: float,
    material_symbol: str | None = None,
) -> float:
    """Photon buildup factor B(mu*x).

    Taylor form from NASA Table III for known materials.
    Simple polynomial fallback otherwise.

    Args:
        photon_energy_mev: Photon energy in MeV.
        mu_x: Product of attenuation coefficient and thickness (dimensionless).
        material_symbol: Material symbol for NASA data lookup.

    Returns:
        Buildup factor (>= 1.0).
    """
    if mu_x <= 0:
        return 1.0

    buildup_data = _get_buildup_table()
    if buildup_data is not None and material_symbol in buildup_data.get("materials", {}):
        mat_data = buildup_data["materials"][material_symbol]
        energies = buildup_data["photon_energy_mev"]

        a1 = float(_interpolate_log(energies, mat_data["A1"], photon_energy_mev))
        alpha1 = float(_interpolate_log(energies, mat_data["alpha1"], photon_energy_mev))
        return 1.0 + (a1 - 1.0) * (1.0 - math.exp(-alpha1 * mu_x))

    # Simple linear buildup estimate for unknown materials
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
    Accounts for slant path through target at angle phi_d from normal.

    Args:
        photon_energy_mev: Photon energy in MeV.
        remaining_depth_g_cm2: Remaining target depth in g/cm^2.
        detection_angle_deg: Detection angle from beam axis (degrees).
        z: Atomic number.
        material_symbol: Material symbol for data lookup.

    Returns:
        Transmission factor (0 to ~1; can slightly exceed 1 with buildup).
    """
    if remaining_depth_g_cm2 <= 0 or photon_energy_mev <= 0:
        return 1.0

    mu_rho = mass_attenuation_coefficient(photon_energy_mev, z, material_symbol)

    cos_phi = max(math.cos(math.radians(detection_angle_deg)), config.MIN_COSINE_SLANT)
    effective_depth = remaining_depth_g_cm2 / cos_phi
    mu_x = mu_rho * effective_depth

    b = buildup_factor(photon_energy_mev, mu_x, material_symbol)
    return b * math.exp(-mu_x)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_buildup_table() -> dict[str, Any] | None:
    """Return the buildup_coefficients sub-table from NASA data, or None."""
    data = get_nasa_data()
    return data.get("buildup_coefficients") if data else None


def _interpolate_xcom(
    energies: list[float],
    mu_rho: list[float],
    energy_query: float,
) -> float:
    """Log-log interpolation of XCOM data, preserving absorption edges.

    XCOM stores two entries at each absorption edge energy: one for the
    cross section just below the edge and one just above.  These represent
    real physics — the photoelectric cross section jumps discontinuously
    at each shell binding energy.

    Strategy:
      1. Build a list of segments between consecutive distinct energies.
         Each segment lies between two points with no edge in between.
      2. At an edge (duplicate energy), jump straight from the pre-edge
         value to the post-edge value — no interpolation across the gap.
      3. Within a segment, use standard log-log interpolation.

    Args:
        energies: Energy grid in MeV (may have duplicate entries at edges).
        mu_rho: mu/rho values in cm^2/g corresponding to energies.
        energy_query: Photon energy to interpolate at.

    Returns:
        Interpolated mu/rho in cm^2/g.
    """
    n = len(energies)
    if n == 0:
        return 0.0
    if n == 1:
        return mu_rho[0]

    # Clamp to table range
    if energy_query <= energies[0]:
        return mu_rho[0]
    if energy_query >= energies[-1]:
        return mu_rho[-1]

    # Find the correct interpolation interval.
    # Walk from the top of the table downward to find the highest index i
    # where energies[i] <= energy_query AND energies[i+1] >= energy_query.
    # When two entries share the same energy (an edge), they define a
    # discontinuity — never interpolate across them.
    idx = -1
    for i in range(n - 2, -1, -1):
        if energies[i] <= energy_query <= energies[i + 1]:
            # Check: is this an edge pair (duplicate energy)?
            if energies[i] == energies[i + 1]:
                # Query is exactly at an edge.  Return the post-edge
                # (higher-mu) value — the photon sees the full shell.
                return mu_rho[i + 1]
            idx = i
            break

    if idx < 0:
        # Should not reach here after clamping, but be safe
        return mu_rho[0]

    e0, e1 = energies[idx], energies[idx + 1]
    m0, m1 = mu_rho[idx], mu_rho[idx + 1]

    if e0 <= 0 or e1 <= 0 or m0 <= 0 or m1 <= 0:
        return m0

    # Log-log interpolation within this edge-free segment
    log_e = math.log(energy_query / e0) / math.log(e1 / e0)
    return float(m0 * (m1 / m0) ** log_e)


def _parametric_attenuation(photon_energy_mev: float, z: int | float) -> float:
    """Parametric mass attenuation coefficient for arbitrary Z.

    Combines Compton scattering (Klein-Nishina), photoelectric absorption,
    and pair production contributions.  Used for SS304/316 and any material
    not in the XCOM table.

    Args:
        photon_energy_mev: Photon energy in MeV.
        z: Atomic number.

    Returns:
        mu/rho in cm^2/g.
    """
    z_f = float(z)
    k = photon_energy_mev

    # Compton (Klein-Nishina)
    x = k / config.ELECTRON_MASS_MEV
    if x < 0.01:
        f_kn = 1.0
    else:
        f_kn = (3.0 / (4.0 * x)) * (
            (1.0 + x) / x**3 * (2.0 * x * (1.0 + x) / (1.0 + 2.0 * x) - math.log(1.0 + 2.0 * x))
            + math.log(1.0 + 2.0 * x) / (2.0 * x)
            - (1.0 + 3.0 * x) / (1.0 + 2.0 * x) ** 2
        )
    sigma_compton = 0.0597 * z_f * f_kn

    # Photoelectric (approximate)
    sigma_pe = 0.0
    if k < 3.0:
        sigma_pe = 2.0e-4 * z_f**4.5 / (k**3.5 * (z_f * 2.0))

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

    mask = y_arr > 0
    if not np.any(mask):
        return 0.0

    x_valid = x_arr[mask]
    y_valid = y_arr[mask]

    if len(x_valid) < 2:
        return float(y_valid[0])

    x_clamped = max(float(x_valid[0]), min(float(x_valid[-1]), x_query))
    log_y = np.log(y_valid)
    interp_fn = interp1d(x_valid, log_y, kind="linear", fill_value="extrapolate")
    return float(np.exp(interp_fn(x_clamped)))
