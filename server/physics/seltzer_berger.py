"""Seltzer-Berger cross section correction for bremsstrahlung.

The Seltzer-Berger (1986) tables give a scaled DCS chi(T, kappa) for each element,
capturing Coulomb corrections, exact screening, and partial-wave effects beyond
the Born approximation (2BN/2BH).

Physical cross section from table (Geant4 EMLOW convention):
    d_sigma/d_kappa  [cm^2]      = Z^2 * chi(T, kappa) * (16 alpha r_0^2 / 3)
    d_sigma/dk  [cm^2/MeV]       = Z^2 * chi(T, kappa) * (16 alpha r_0^2 / 3) / T

Normalization: The EMLOW tables may not reproduce the NIST ESTAR radiative
stopping power exactly due to normalization convention differences.  We
therefore compute a per-element, per-energy correction factor K(T, Z) that
ensures the S-B correction integrates to match ESTAR.  This gives a physically
consistent correction that:
  - Preserves the S-B spectral shape chi(kappa) vs kappa
  - Normalizes the total to match ESTAR stopping power
  - Is >1 for heavy elements (Coulomb correction increases cross section)
  - Is approx 1 for light elements (Born is already accurate)

Correction factor derivation:
    S_rad_ESTAR = (N_A/A) * integral k * d_sigma_corrected/dk dk
                = S_rad_ESTAR * integral_0^1 kappa * chi d_kappa / Omega
                = S_rad_ESTAR  (by construction)

    where Omega(T, Z) = integral_0^1 kappa * chi(T, kappa) d_kappa

    d_sigma_corrected/dk = S_rad_ESTAR * chi(T, kappa)
                           / [(N_A/A) * T^2 * Omega(T, Z)]

    f_SB(T, k, Z) = d_sigma_corrected/dk / d_sigma_BH/dk

Effect on thick-target spectrum:
    - At low  kappa = k/T: f_SB < 1  (Born overestimates low-energy photons)
    - At med  kappa approx 0.3: f_SB approx 0.7-1.0
    - At high kappa -> 1:   f_SB >> 1 (Born severely underestimates near endpoint)

References:
    Seltzer SM & Berger MJ, Atom. Data Nucl. Data Tables 35, 345-418 (1986).
    Data from G4EMLOW 8.6.1 (GEANT4), brem_SB/br{Z} files.
"""

from __future__ import annotations

import json
import logging
import math
from functools import lru_cache
from typing import Any

import numpy as np
from scipy.interpolate import RegularGridInterpolator, interp1d  # type: ignore[import-untyped]

import config
from server.physics._validation import require_positive_energy, require_positive_z

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# (16/3) alpha r_0^2  [cm^2] -- Geant4 bremFactor convention (raw, un-normalized)
_BREM_FACTOR_CM2: float = (16.0 / 3.0) * config.FINE_STRUCTURE_CONSTANT * config.RE_SQUARED_CM2

# Correction factor clamp range (physics-based bounds)
_F_SB_MIN: float = 0.1  # never less than 10% of Born
_F_SB_MAX: float = 20.0  # near endpoint at high Z can be large

# ---------------------------------------------------------------------------
# Data caches
# ---------------------------------------------------------------------------
_sb_data_cache: dict[str, Any] | None = None
_sb_interp_cache: dict[int, RegularGridInterpolator] = {}
_sb_omega_cache: dict[int, interp1d] = {}  # Omega(T) interpolators per Z


def _get_sb_data() -> dict[str, Any]:
    """Load and cache Seltzer-Berger JSON data."""
    global _sb_data_cache
    if _sb_data_cache is None:
        with config.SB_DATA_PATH.open() as f:
            _sb_data_cache = json.load(f)
    return _sb_data_cache


def _get_sb_chi_interpolator(z: int) -> RegularGridInterpolator | None:
    """2D interpolator for chi(ln_T, kappa). Returns None if Z not in table."""
    if z in _sb_interp_cache:
        return _sb_interp_cache[z]

    data = _get_sb_data()
    key = str(z)
    if key not in data:
        return None

    entry = data[key]
    y_arr = np.array(entry["y_logT"])  # ln(T/MeV), 57 points
    x_arr = np.array(entry["x_kappa"])  # kappa = k/T, 32 points
    chi_arr = np.array(entry["chi"])  # shape (57, 32)

    interp = RegularGridInterpolator(
        (y_arr, x_arr),
        chi_arr,
        method="linear",
        bounds_error=False,
        fill_value=None,  # use boundary values outside range
    )
    _sb_interp_cache[z] = interp
    return interp


def _get_sb_omega_interp(z: int) -> interp1d | None:
    """1D interpolator for Omega(T) = integral_0^1 kappa * chi(T, kappa) d_kappa vs ln(T).

    Pre-computed at the 57 table grid points to avoid repeated integration.
    Returns None if Z not in table.
    """
    if z in _sb_omega_cache:
        return _sb_omega_cache[z]

    data = _get_sb_data()
    key = str(z)
    if key not in data:
        return None

    entry = data[key]
    y_arr = np.array(entry["y_logT"])  # 57 ln(T) values
    x_arr = np.array(entry["x_kappa"])  # 32 kappa values
    chi_arr = np.array(entry["chi"])  # (57, 32)

    # Compute Omega at each logT grid point: vectorized trapezoid over kappa axis
    kappa_chi_2d = x_arr[np.newaxis, :] * chi_arr  # (57, 32)
    omega_arr = np.maximum(np.trapezoid(kappa_chi_2d, x_arr, axis=1), 1e-10)

    fn = interp1d(
        y_arr,
        omega_arr,
        kind="linear",
        bounds_error=False,
        fill_value=(omega_arr[0], omega_arr[-1]),
    )
    _sb_omega_cache[z] = fn
    return fn


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def sb_chi(z: int, t_mev: float, kappa: float) -> float:
    """Interpolated scaled DCS chi(T, kappa) from Seltzer-Berger tables.

    Args:
        z: Atomic number (integer; must be one of the 10 NASA elements).
        t_mev: Electron kinetic energy in MeV.
        kappa: Reduced photon energy kappa = k/T  (0 < kappa < 1).

    Returns:
        Dimensionless chi >= 0, or 0.0 if element not in table.
    """
    if t_mev <= 0 or kappa <= 0 or kappa >= 1:
        return 0.0

    interp = _get_sb_chi_interpolator(z)
    if interp is None:
        return 0.0

    log_t = math.log(t_mev)
    # Clamp to table bounds (boundary extrapolation is stable for log-space tables)
    data = _get_sb_data()
    entry = data[str(z)]
    log_t = float(np.clip(log_t, entry["y_logT"][0], entry["y_logT"][-1]))
    kappa = float(np.clip(kappa, entry["x_kappa"][1], entry["x_kappa"][-2]))

    result = float(interp([[log_t, kappa]])[0])
    return max(result, 0.0)


def sb_omega(z: int, t_mev: float) -> float:
    """Omega(T, Z) = integral_0^1 kappa * chi(T, kappa) d_kappa at this electron energy.

    Args:
        z: Atomic number.
        t_mev: Electron kinetic energy in MeV.

    Returns:
        Dimensionless Omega > 0, or 0.0 if element not in table.
    """
    if t_mev <= 0:
        return 0.0
    fn = _get_sb_omega_interp(z)
    if fn is None:
        return 0.0
    log_t = math.log(t_mev)
    return max(float(fn(log_t)), 1e-10)


@lru_cache(maxsize=1024)
def _bh_dk_cached(t_r: float, k_r: float, z: int, n_theta: int) -> float:
    """Angle-integrated 2BN d_sigma/dk [cm^2/MeV] (cached)."""
    from server.physics.bethe_heitler import bethe_heitler_2bn  # avoid circular import

    cos_nodes, weights = np.polynomial.legendre.leggauss(n_theta)
    integral = 0.0
    for ct, w in zip(cos_nodes, weights, strict=True):
        theta = math.acos(max(-1.0, min(1.0, float(ct))))
        ds = bethe_heitler_2bn(t_r, k_r, theta, float(z))
        integral += w * 2.0 * math.pi * ds
    return max(integral, 0.0)


def bh_dk_cm2_mev(t_mev: float, k_mev: float, z: int | float, n_theta: int = 20) -> float:
    """Angle-integrated 2BN d_sigma/dk [cm^2/MeV/atom].

    Rounds T and k to 5 significant figures for cache stability.
    """
    require_positive_energy(t_mev, "Electron energy")
    require_positive_energy(k_mev, "Photon energy")
    require_positive_z(z)
    z_int = round(z)
    return _bh_dk_cached(float(f"{t_mev:.5g}"), float(f"{k_mev:.5g}"), z_int, n_theta)


def sb_correction_factor(
    t_mev: float,
    k_mev: float,
    z: int | float,
    a: float,
    n_theta: int = 20,
) -> float:
    """Seltzer-Berger correction factor f_SB = d_sigma_SB/dk / d_sigma_BH/dk.

    The S-B tables give a more accurate energy-differential cross section than
    the Born approximation (2BN).  This function computes the ratio, normalized
    to reproduce the NIST ESTAR radiative stopping power when integrated over k.

    The correction is:
        f_SB < 1  at low  kappa = k/T  (Born overestimates slow photons)
        f_SB > 1  at high kappa -> 1   (Born misses the near-endpoint enhancement)
        f_SB approx 1  for light elements at intermediate kappa

    Returns 1.0 if Z not in S-B table (pure 2BN fallback).
    Clamped to [_F_SB_MIN, _F_SB_MAX].

    Args:
        t_mev: Electron kinetic energy in MeV.
        k_mev: Photon energy in MeV (must satisfy 0 < k < T).
        z: Atomic number.
        a: Atomic weight (g/mol).
        n_theta: Angle quadrature points for integrating the 2BN denominator.

    Returns:
        Dimensionless correction factor.
    """
    from server.physics.stopping_power import radiative_stopping_power  # avoid circular

    if k_mev <= 0 or k_mev >= t_mev or t_mev <= 0:
        return 1.0

    z_int = round(z)
    kappa = k_mev / t_mev

    # S-B shape factor chi(T, kappa)
    chi = sb_chi(z_int, t_mev, kappa)
    if chi <= 0:
        return 1.0  # Z not in table -- fall through to pure 2BN

    # Omega(T, Z) = integral_0^1 kappa chi d_kappa  (normalization denominator)
    omega = sb_omega(z_int, t_mev)
    if omega <= 0:
        return 1.0

    # ESTAR radiative stopping power (MeV cm^2/g) -- absolute normalization
    s_rad = radiative_stopping_power(t_mev, z_int, a)
    if s_rad <= 0:
        return 1.0

    # N_A/A [atoms/g]
    na_over_a = config.AVOGADRO / a

    # Normalized S-B DCS: d_sigma_SB/dk = S_rad * chi / (N_A/A * T^2 * Omega)
    # Derivation: S_rad = (N_A/A) integral k d_sigma_SB/dk dk
    #           = (N_A/A) integral kappa*T * d_sigma/d_kappa * T d_kappa
    #           -> d_sigma/d_kappa = S_rad * chi / (N_A/A * T * Omega)
    #           -> d_sigma/dk = d_sigma/d_kappa / T
    ds_sb_dk = s_rad * chi / (na_over_a * t_mev**2 * omega)

    # Denominator: angle-integrated 2BN
    ds_bh_dk = bh_dk_cm2_mev(t_mev, k_mev, z_int, n_theta)
    if ds_bh_dk <= 0:
        return 1.0

    factor = ds_sb_dk / ds_bh_dk
    if not math.isfinite(factor):
        log.warning("S-B correction factor non-finite: T=%.3f k=%.3f Z=%d", t_mev, k_mev, z_int)
        return 1.0

    return max(_F_SB_MIN, min(_F_SB_MAX, factor))
