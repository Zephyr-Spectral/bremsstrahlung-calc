"""BremsLib v2.0 doubly-differential bremsstrahlung cross sections.

Provides d^2 sigma/(dk dOmega) [cm^2/(sr MeV atom)] from relativistic
partial-wave calculations (Poskus, Comp. Phys. Comm. 2018) for all
elements Z=1-100, electron energies 10 eV - 30 MeV, all angles.

Replaces the Koch & Motz 2BN Born approximation with exact partial-wave
results that include Coulomb corrections, screening, and finite-nucleus
effects.

The data is stored as a pre-computed 4D array in ddcs_all_elements.npz
(8 MB), loaded once and cached.  Runtime interpolation uses
scipy.interpolate.RegularGridInterpolator on (log T1, kappa, theta).

Reference:
    Poskus A, Comp. Phys. Comm. 232, 237-255 (2018).
    BremsLib v2.0: https://web.vu.lt/ff/a.poskus/brems/
"""

from __future__ import annotations

import logging
import math

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator  # type: ignore[import-untyped]

import config

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_NPZ_PATH = config.DATA_DIR / "bremslib" / "ddcs_all_elements.npz"

# Cached data arrays (loaded once on first call)
_data: dict[str, npt.NDArray[np.float64]] | None = None
_interp_cache: dict[int, RegularGridInterpolator] = {}


def _load_data() -> dict[str, npt.NDArray[np.float64]]:
    """Load the pre-computed DDCS table from .npz (8 MB, one-time)."""
    global _data
    if _data is not None:
        return _data
    npz = np.load(_NPZ_PATH)
    _data = {
        "ddcs_scaled": npz["ddcs_scaled"],  # (100, n_T1, n_kappa, n_theta) float32
        "t1_grid": npz["t1_grid"],  # (n_T1,) float64
        "kappa_grid": npz["kappa_grid"],  # (n_kappa,) float64
        "theta_grid": npz["theta_grid"],  # (n_theta,) float64
    }
    log.info(
        "Loaded BremsLib DDCS: shape=%s, T1=[%.3g, %.3g] MeV, %d kappa, %d theta",
        _data["ddcs_scaled"].shape,
        _data["t1_grid"][0],
        _data["t1_grid"][-1],
        len(_data["kappa_grid"]),
        len(_data["theta_grid"]),
    )
    return _data


def _get_interpolator(z: int) -> RegularGridInterpolator:
    """Build or retrieve cached 3D interpolator for element Z.

    Interpolation axes: (log(T1/MeV), kappa, theta_deg).
    Values: DDCS_scaled in mb/sr  [= k/Z^2 * d^2sigma/(dk dOmega)].
    """
    if z in _interp_cache:
        return _interp_cache[z]

    data = _load_data()
    ddcs_z = data["ddcs_scaled"][z - 1]  # (n_T1, n_kappa, n_theta)
    log_t1 = np.log(data["t1_grid"])
    kappa = data["kappa_grid"]
    theta = data["theta_grid"]

    # Use log(DDCS) for interpolation where DDCS > 0 (smoother in log space)
    # Clamp tiny/zero values to a floor to avoid log(0)
    floor = 1e-20
    safe_ddcs = np.maximum(ddcs_z.astype(np.float64), floor)
    log_ddcs = np.log(safe_ddcs)

    interp = RegularGridInterpolator(
        (log_t1, kappa, theta),
        log_ddcs,
        method="linear",
        bounds_error=False,
        fill_value=np.log(floor),
    )
    _interp_cache[z] = interp
    return interp


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def bremslib_ddcs(
    t_mev: float,
    k_mev: float,
    theta_deg: float,
    z: int,
) -> float:
    """Scalar DDCS: d^2 sigma/(dk dOmega) [cm^2/(sr MeV atom)].

    Args:
        t_mev: Electron kinetic energy in MeV.
        k_mev: Photon energy in MeV (must be 0 < k < T).
        theta_deg: Photon emission angle in degrees.
        z: Atomic number (1-100).

    Returns:
        Doubly-differential cross section in cm^2/(sr MeV atom).
    """
    if k_mev <= 0 or k_mev >= t_mev or t_mev <= 0 or z < 1 or z > 100:
        return 0.0

    kappa = k_mev / t_mev
    interp = _get_interpolator(z)
    log_t = math.log(t_mev)
    log_ddcs_scaled = float(interp([[log_t, kappa, theta_deg]])[0])
    ddcs_scaled = math.exp(log_ddcs_scaled)  # mb/sr

    # Convert: d^2sigma/(dk dOmega) = ddcs_scaled * Z^2 / k * 1e-27 [cm^2/sr/MeV]
    return ddcs_scaled * z * z / k_mev * 1e-27


def bremslib_ddcs_vec(
    t_mev: float,
    k_mev: float,
    theta_rad: npt.NDArray[np.float64],
    z: int | float,
) -> npt.NDArray[np.float64]:
    """Vectorized DDCS for an array of emission angles.

    Drop-in replacement for bethe_heitler_2bn_vec().

    Args:
        t_mev: Electron kinetic energy in MeV.
        k_mev: Photon energy in MeV.
        theta_rad: Array of emission angles in RADIANS (any shape).
        z: Atomic number.

    Returns:
        d^2 sigma/(dk dOmega) in cm^2/(sr MeV atom), same shape as theta_rad.
    """
    z_int = round(z)
    if k_mev <= 0 or k_mev >= t_mev or t_mev <= 0 or z_int < 1 or z_int > 100:
        return np.zeros_like(theta_rad)

    kappa = k_mev / t_mev
    interp = _get_interpolator(z_int)
    log_t = math.log(t_mev)

    # Convert radians to degrees for the interpolation
    theta_deg = np.degrees(theta_rad)
    orig_shape = theta_deg.shape
    flat_theta = theta_deg.ravel()

    # Build query points: (log_t, kappa, theta_deg) for each angle
    n = len(flat_theta)
    pts = np.empty((n, 3), dtype=np.float64)
    pts[:, 0] = log_t
    pts[:, 1] = kappa
    pts[:, 2] = flat_theta

    # Interpolate in log space, then exp
    log_ddcs_scaled = interp(pts)  # (n,)
    ddcs_scaled = np.exp(log_ddcs_scaled)  # mb/sr

    # Convert to cm^2/sr/MeV
    ddcs = ddcs_scaled * (z_int * z_int / k_mev * 1e-27)

    return ddcs.reshape(orig_shape)  # type: ignore[no-any-return]


def clear_cache() -> None:
    """Clear all cached data and interpolators."""
    global _data
    _data = None
    _interp_cache.clear()
