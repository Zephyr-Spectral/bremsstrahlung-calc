"""Interpolation over NASA TN D-4755 tabulated data.

Provides interpolated bremsstrahlung spectra at arbitrary electron energies
and detection angles within (and slightly beyond) the NASA data grid:
  - Electron energies: 0.50, 0.75, 1.00, 2.00, 3.00 MeV
  - Detection angles: 0, 30, 60 degrees
"""

from __future__ import annotations

import json
import logging
import math

from scipy.interpolate import interp1d

import config

log = logging.getLogger(__name__)

# Module-level cache
_nasa_intensity_data: dict[str, object] | None = None


def _load_intensity_data() -> dict[str, object]:
    """Load and cache NASA intensity data."""
    global _nasa_intensity_data
    if _nasa_intensity_data is not None:
        return _nasa_intensity_data

    if not config.NASA_DATA_PATH.exists():
        msg = f"NASA data file not found at {config.NASA_DATA_PATH}"
        raise FileNotFoundError(msg)

    with config.NASA_DATA_PATH.open() as f:
        data = json.load(f)

    _nasa_intensity_data = data["intensity"]
    return _nasa_intensity_data


def interpolate_nasa_spectrum(
    electron_energy_mev: float,
    detection_angle_deg: float,
    material_symbol: str,
) -> tuple[list[float], list[float]]:
    """Interpolate NASA data for arbitrary energy and angle.

    Uses log-linear interpolation in electron energy and linear
    interpolation in angle.

    Args:
        electron_energy_mev: Electron kinetic energy in MeV.
        detection_angle_deg: Detection angle from normal in degrees.
        material_symbol: Material symbol (must be one of the 10 NASA materials).

    Returns:
        Tuple of (photon_energies_mev, intensities).

    Raises:
        ValueError: If material not in NASA data.
        FileNotFoundError: If NASA data file missing.
    """
    data = _load_intensity_data()
    materials = data["materials"]

    if material_symbol not in materials:
        msg = f"Material '{material_symbol}' not in NASA data. Available: {list(materials.keys())}"
        raise ValueError(msg)

    mat_data = materials[material_symbol]
    nasa_energies = config.NASA_ELECTRON_ENERGIES_MEV
    nasa_angles = config.NASA_DETECTION_ANGLES_DEG

    # Clamp to NASA range
    e_clamped = max(nasa_energies[0], min(nasa_energies[-1], electron_energy_mev))
    a_clamped = max(nasa_angles[0], min(nasa_angles[-1], detection_angle_deg))

    # Find bracketing electron energies
    e_below, e_above = _find_bracket(nasa_energies, e_clamped)

    # Find bracketing angles
    a_below, a_above = _find_bracket(nasa_angles, a_clamped)

    # Get spectra at the 4 corner points (or fewer if on grid)
    corners = _build_corners(mat_data, {e_below, e_above}, {a_below, a_above})

    if not corners:
        log.warning(
            "No valid NASA data for %s at E=%s MeV, angle=%s deg",
            material_symbol,
            electron_energy_mev,
            detection_angle_deg,
        )
        return [], []

    # Common photon energy grid: use the finest available
    all_k = set()
    for k_vals, _ in corners.values():
        all_k.update(k_vals)
    common_k = sorted(all_k)

    if len(common_k) < 2:
        return list(common_k), [next(iter(corners.values()))[1][0]]

    # Interpolate each corner spectrum onto common grid, then bilinear interpolate
    interpolated_corners: dict[tuple[float, float], list[float]] = {}
    for (e, a), (k_vals, i_vals) in corners.items():
        if len(k_vals) >= 2:
            log_i = [math.log(max(v, 1e-30)) for v in i_vals]
            fn = interp1d(k_vals, log_i, kind="linear", fill_value="extrapolate")
            interpolated_corners[(e, a)] = [float(math.exp(fn(k))) for k in common_k]
        else:
            interpolated_corners[(e, a)] = [float(i_vals[0])] * len(common_k)

    # Bilinear interpolation
    result = _bilinear_interpolate(
        interpolated_corners,
        e_clamped,
        a_clamped,
        e_below,
        e_above,
        a_below,
        a_above,
        len(common_k),
    )

    return common_k, result


def get_nasa_spectrum_at_grid_point(
    electron_energy_mev: float,
    detection_angle_deg: float,
    material_symbol: str,
) -> tuple[list[float], list[float]]:
    """Get NASA spectrum at exact grid point (no interpolation).

    Args:
        electron_energy_mev: Must be one of [0.50, 0.75, 1.00, 2.00, 3.00].
        detection_angle_deg: Must be one of [0, 30, 60].
        material_symbol: NASA material symbol.

    Returns:
        Tuple of (photon_energies, intensities) with null values filtered.
    """
    data = _load_intensity_data()
    mat_data = data["materials"].get(material_symbol, {})

    e_key = f"{electron_energy_mev:.2f}"
    a_key = str(int(detection_angle_deg))

    if e_key not in mat_data or a_key not in mat_data[e_key]:
        return [], []

    k_data = mat_data[e_key]["photon_energy_mev"]
    i_data = mat_data[e_key][a_key]

    valid = [(k, v) for k, v in zip(k_data, i_data, strict=False) if v is not None and v > 0]
    if not valid:
        return [], []

    k_vals, i_vals = zip(*valid, strict=False)
    return list(k_vals), list(i_vals)


def clear_cache() -> None:
    """Clear the interpolation data cache."""
    global _nasa_intensity_data
    _nasa_intensity_data = None


def _build_corners(
    mat_data: dict[str, object],
    energies: set[float],
    angles: set[float],
) -> dict[tuple[float, float], tuple[list[float], list[float]]]:
    """Extract valid (k, intensity) pairs at each (energy, angle) corner."""
    corners: dict[tuple[float, float], tuple[list[float], list[float]]] = {}
    for e in energies:
        for a in angles:
            e_key = f"{e:.2f}"
            a_key = str(int(a))
            if e_key in mat_data and a_key in mat_data[e_key]:
                k_data = mat_data[e_key]["photon_energy_mev"]
                i_data = mat_data[e_key][a_key]
                valid = [
                    (k, v) for k, v in zip(k_data, i_data, strict=False) if v is not None and v > 0
                ]
                if valid:
                    corners[(e, a)] = tuple(zip(*valid, strict=False))  # type: ignore[arg-type]
    return corners


def _find_bracket(grid: list[float], value: float) -> tuple[float, float]:
    """Find the two grid points bracketing value."""
    for i in range(len(grid) - 1):
        if grid[i] <= value <= grid[i + 1]:
            return grid[i], grid[i + 1]
    # At boundary
    if value <= grid[0]:
        return grid[0], grid[0]
    return grid[-1], grid[-1]


def _bilinear_interpolate(
    corners: dict[tuple[float, float], list[float]],
    e_query: float,
    a_query: float,
    e_lo: float,
    e_hi: float,
    a_lo: float,
    a_hi: float,
    n_points: int,
) -> list[float]:
    """Bilinear interpolation between up to 4 corner spectra."""
    # Weight factors
    t_e = math.log(e_query / e_lo) / math.log(e_hi / e_lo) if e_hi > e_lo else 0.0

    t_a = (a_query - a_lo) / (a_hi - a_lo) if a_hi > a_lo else 0.0

    result = [0.0] * n_points

    # Bilinear: f = (1-te)(1-ta)*f00 + te*(1-ta)*f10 + (1-te)*ta*f01 + te*ta*f11
    weights = {
        (e_lo, a_lo): (1.0 - t_e) * (1.0 - t_a),
        (e_hi, a_lo): t_e * (1.0 - t_a),
        (e_lo, a_hi): (1.0 - t_e) * t_a,
        (e_hi, a_hi): t_e * t_a,
    }

    total_weight = 0.0
    for key, w in weights.items():
        if key in corners and w > 0:
            for j in range(n_points):
                result[j] += w * corners[key][j]
            total_weight += w

    # Normalize if not all corners available
    if total_weight > 0 and total_weight < 0.99:
        result = [v / total_weight for v in result]

    return result
