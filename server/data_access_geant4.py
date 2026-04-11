"""Data access layer for Geant4 batch lookup table.

Lazy-loads geant4_lookup.npz on first access and provides functions
to query spectra, angular distributions, and metadata from the
pre-computed 10M-event simulation campaign.

Lookup table axes:
    materials:  11 elements (C, Mg, Al, Ti, Mn, Fe, Ni, Cu, W, Au, Pb)
    energies:   20 values (0.5-10.0 MeV in 0.5 MeV steps)
    theta:      36 bins (0-180 deg in 5 deg steps)
    k:          40 log-spaced photon energy bins per electron energy
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from numpy.typing import NDArray

import config

log = logging.getLogger(__name__)

# Module-level cache (loaded once on first access)
_g4_cache: dict[str, Any] | None = None


def get_g4_lookup() -> dict[str, Any]:
    """Load and return the Geant4 lookup table, caching after first read."""
    global _g4_cache
    if _g4_cache is not None:
        return _g4_cache

    path = config.GEANT4_LOOKUP_PATH
    if not path.exists():
        log.warning("Geant4 lookup file not found at %s", path)
        msg = f"Geant4 lookup file not found: {path}"
        raise FileNotFoundError(msg)

    data = np.load(str(path), allow_pickle=False)
    _g4_cache = {
        "intensity": data["intensity"],  # (11, 20, 36, 40)
        "counts": data["counts"],  # (11, 20, 36, 40)
        "materials": list(data["materials"]),  # ['C', 'Mg', ...]
        "energies_mev": data["energies_mev"].astype(float),  # [0.5, 1.0, ...]
        "theta_bin_edges": data["theta_bin_edges"].astype(float),  # [0, 5, ..., 180]
        "n_events": int(data["n_events"]),
    }
    log.info(
        "Loaded Geant4 lookup: %d materials, %d energies, %d theta bins, %d events/run",
        len(_g4_cache["materials"]),
        len(_g4_cache["energies_mev"]),
        _g4_cache["intensity"].shape[2],
        _g4_cache["n_events"],
    )
    return _g4_cache


def clear_g4_cache() -> None:
    """Clear the lookup cache (useful for testing)."""
    global _g4_cache
    _g4_cache = None


# ---------------------------------------------------------------------------
# Metadata accessors
# ---------------------------------------------------------------------------


def g4_materials() -> list[str]:
    """Return list of materials in the Geant4 lookup table."""
    return list(get_g4_lookup()["materials"])


def g4_energies() -> list[float]:
    """Return list of electron energies (MeV) in the lookup table."""
    return list(get_g4_lookup()["energies_mev"])


def g4_info() -> dict[str, Any]:
    """Return metadata about the lookup table."""
    lut = get_g4_lookup()
    theta_edges = lut["theta_bin_edges"]
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    return {
        "materials": lut["materials"],
        "energies_mev": list(lut["energies_mev"]),
        "theta_bin_edges": list(theta_edges),
        "theta_bin_centers": list(theta_centers),
        "n_events": lut["n_events"],
        "n_materials": len(lut["materials"]),
        "n_energies": len(lut["energies_mev"]),
        "n_theta_bins": lut["intensity"].shape[2],
        "n_k_bins": lut["intensity"].shape[3],
    }


# ---------------------------------------------------------------------------
# Index resolution helpers
# ---------------------------------------------------------------------------


def _material_index(material: str) -> int:
    """Resolve material symbol to lookup table index."""
    materials: list[str] = get_g4_lookup()["materials"]
    try:
        return int(materials.index(material))
    except ValueError:
        msg = f"Material '{material}' not in Geant4 lookup. Available: {materials}"
        raise ValueError(msg) from None


def _energy_index(t0_mev: float) -> int | None:
    """Return exact energy index if t0_mev is on-grid, else None."""
    energies = get_g4_lookup()["energies_mev"]
    diffs = np.abs(energies - t0_mev)
    idx = int(np.argmin(diffs))
    if diffs[idx] < 0.01:  # within 10 keV tolerance
        return idx
    return None


def _energy_bracket(t0_mev: float) -> tuple[int, int, float]:
    """Find bracketing energy indices and interpolation weight.

    Returns (idx_lo, idx_hi, weight) where weight is the fraction
    toward idx_hi: intensity = (1 - w) * I[lo] + w * I[hi].
    """
    energies = get_g4_lookup()["energies_mev"]
    if t0_mev <= energies[0]:
        return 0, 0, 0.0
    if t0_mev >= energies[-1]:
        n = len(energies) - 1
        return n, n, 0.0
    idx_hi = int(np.searchsorted(energies, t0_mev))
    idx_lo = idx_hi - 1
    e_lo, e_hi = energies[idx_lo], energies[idx_hi]
    weight = float((t0_mev - e_lo) / (e_hi - e_lo))
    return idx_lo, idx_hi, weight


def _theta_index(angle_deg: float) -> int:
    """Map detection angle to nearest theta bin index."""
    theta_edges = get_g4_lookup()["theta_bin_edges"]
    centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    idx = int(np.argmin(np.abs(centers - angle_deg)))
    return idx


# ---------------------------------------------------------------------------
# Photon energy bin reconstruction
# ---------------------------------------------------------------------------


def g4_photon_energy_bins(t0_mev: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Reconstruct log-spaced photon energy bin edges and centers for a given T0.

    Must match the formula in batch_run.py:
        k_min = max(G4_BATCH_K_MIN_ABS, G4_BATCH_K_MIN_FRAC * T0)
        k_max = G4_BATCH_K_MAX_FRAC * T0
        edges = logspace(log10(k_min), log10(k_max), G4_BATCH_K_BINS + 1)

    Returns (bin_centers, bin_widths) as 1D arrays of length G4_BATCH_K_BINS.
    """
    k_min = max(config.G4_BATCH_K_MIN_ABS, config.G4_BATCH_K_MIN_FRAC * t0_mev)
    k_max = config.G4_BATCH_K_MAX_FRAC * t0_mev
    edges = np.logspace(np.log10(k_min), np.log10(k_max), config.G4_BATCH_K_BINS + 1)
    centers = np.sqrt(edges[:-1] * edges[1:])  # geometric mean
    widths = np.diff(edges)
    return centers, widths


# ---------------------------------------------------------------------------
# Spectrum queries
# ---------------------------------------------------------------------------


def g4_spectrum(
    material: str,
    t0_mev: float,
    angle_deg: float,
) -> dict[str, Any]:
    """Return Geant4 batch spectrum at (material, energy, angle).

    For energies on the 0.5 MeV grid, returns exact data. For off-grid
    energies, interpolates between bracketing grid points in log-intensity
    space with proper uncertainty propagation.

    Returns dict with keys:
        photon_energy_mev, intensity, uncertainty, relative_error,
        counts, n_events, theta_range_deg, is_interpolated
    """
    lut = get_g4_lookup()
    mat_idx = _material_index(material)
    theta_idx = _theta_index(angle_deg)
    n_events = lut["n_events"]
    theta_edges = lut["theta_bin_edges"]
    theta_lo = float(theta_edges[theta_idx])
    theta_hi = float(theta_edges[theta_idx + 1])

    exact_idx = _energy_index(t0_mev)
    if exact_idx is not None:
        # Exact grid point
        intensity = lut["intensity"][mat_idx, exact_idx, theta_idx, :].copy()
        counts = lut["counts"][mat_idx, exact_idx, theta_idx, :].copy()
        k_centers, _ = g4_photon_energy_bins(float(lut["energies_mev"][exact_idx]))
        is_interpolated = False
    else:
        # Interpolate between bracketing energies
        idx_lo, idx_hi, weight = _energy_bracket(t0_mev)
        i_lo = lut["intensity"][mat_idx, idx_lo, theta_idx, :]
        i_hi = lut["intensity"][mat_idx, idx_hi, theta_idx, :]
        c_lo = lut["counts"][mat_idx, idx_lo, theta_idx, :]
        c_hi = lut["counts"][mat_idx, idx_hi, theta_idx, :]

        # Interpolate on common energy axis (use target T0 bins)
        k_lo, _ = g4_photon_energy_bins(float(lut["energies_mev"][idx_lo]))
        k_hi, _ = g4_photon_energy_bins(float(lut["energies_mev"][idx_hi]))
        k_centers, _ = g4_photon_energy_bins(t0_mev)

        # Remap both onto the target k axis via log-space interpolation
        i_lo_remap = _safe_interp(k_centers, k_lo, i_lo)
        i_hi_remap = _safe_interp(k_centers, k_hi, i_hi)
        c_lo_remap = _safe_interp(k_centers, k_lo, c_lo.astype(float))
        c_hi_remap = _safe_interp(k_centers, k_hi, c_hi.astype(float))

        # Linear blend
        w = weight
        intensity = (1.0 - w) * i_lo_remap + w * i_hi_remap
        counts = ((1.0 - w) * c_lo_remap + w * c_hi_remap).astype(float)
        is_interpolated = True

    # Compute uncertainties from Poisson statistics
    with np.errstate(divide="ignore", invalid="ignore"):
        relative_error = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    uncertainty = intensity * relative_error

    # Filter out empty bins
    mask = intensity > 0
    return {
        "photon_energy_mev": k_centers[mask].tolist(),
        "intensity": intensity[mask].tolist(),
        "uncertainty": uncertainty[mask].tolist(),
        "relative_error": relative_error[mask].tolist(),
        "counts": counts[mask].tolist(),
        "n_events": n_events,
        "theta_range_deg": [theta_lo, theta_hi],
        "is_interpolated": is_interpolated,
    }


def g4_angular_distribution(
    material: str,
    t0_mev: float,
    photon_energy_mev: float,
) -> dict[str, Any]:
    """Return intensity vs angle at fixed photon energy from Geant4 batch data.

    Returns dict with keys:
        angles_deg, intensity, uncertainty, n_events, k_bin_range_mev
    """
    lut = get_g4_lookup()
    mat_idx = _material_index(material)
    n_events = lut["n_events"]
    theta_edges = lut["theta_bin_edges"]
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    exact_idx = _energy_index(t0_mev)
    if exact_idx is not None:
        e_idx = exact_idx
        k_centers, _ = g4_photon_energy_bins(float(lut["energies_mev"][e_idx]))
        # Find the photon energy bin closest to the requested value
        k_idx = int(np.argmin(np.abs(k_centers - photon_energy_mev)))
        intensity = lut["intensity"][mat_idx, e_idx, :, k_idx].copy()
        counts = lut["counts"][mat_idx, e_idx, :, k_idx].copy()
    else:
        idx_lo, idx_hi, weight = _energy_bracket(t0_mev)
        k_lo, _ = g4_photon_energy_bins(float(lut["energies_mev"][idx_lo]))
        k_hi, _ = g4_photon_energy_bins(float(lut["energies_mev"][idx_hi]))
        k_idx_lo = int(np.argmin(np.abs(k_lo - photon_energy_mev)))
        k_idx_hi = int(np.argmin(np.abs(k_hi - photon_energy_mev)))

        i_lo = lut["intensity"][mat_idx, idx_lo, :, k_idx_lo]
        i_hi = lut["intensity"][mat_idx, idx_hi, :, k_idx_hi]
        c_lo = lut["counts"][mat_idx, idx_lo, :, k_idx_lo]
        c_hi = lut["counts"][mat_idx, idx_hi, :, k_idx_hi]

        intensity = (1.0 - weight) * i_lo + weight * i_hi
        counts = ((1.0 - weight) * c_lo + weight * c_hi).astype(float)
        k_centers, _ = g4_photon_energy_bins(t0_mev)
        k_idx = int(np.argmin(np.abs(k_centers - photon_energy_mev)))

    k_centers_final, _ = g4_photon_energy_bins(
        t0_mev if exact_idx is None else float(lut["energies_mev"][exact_idx])
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        rel_err = np.where(counts > 0, 1.0 / np.sqrt(np.maximum(counts, 1)), 0.0)
    uncertainty = intensity * rel_err

    mask = intensity > 0
    return {
        "angles_deg": theta_centers[mask].tolist(),
        "intensity": intensity[mask].tolist(),
        "uncertainty": uncertainty[mask].tolist(),
        "n_events": n_events,
        "k_bin_center_mev": float(k_centers_final[k_idx]),
    }


def g4_heatmap(
    material: str,
    t0_mev: float,
) -> dict[str, Any]:
    """Return full 2D intensity grid (theta x k) from Geant4 data.

    Returns dict with keys:
        photon_energy_mev, angles_deg, intensity (2D list), n_events
    """
    lut = get_g4_lookup()
    mat_idx = _material_index(material)
    theta_edges = lut["theta_bin_edges"]
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])

    exact_idx = _energy_index(t0_mev)
    if exact_idx is not None:
        grid = lut["intensity"][mat_idx, exact_idx, :, :].copy()
        k_centers, _ = g4_photon_energy_bins(float(lut["energies_mev"][exact_idx]))
    else:
        idx_lo, idx_hi, weight = _energy_bracket(t0_mev)
        g_lo = lut["intensity"][mat_idx, idx_lo, :, :]
        g_hi = lut["intensity"][mat_idx, idx_hi, :, :]
        grid = (1.0 - weight) * g_lo + weight * g_hi
        k_centers, _ = g4_photon_energy_bins(t0_mev)

    return {
        "photon_energy_mev": k_centers.tolist(),
        "angles_deg": theta_centers.tolist(),
        "intensity": grid.tolist(),
        "n_events": lut["n_events"],
    }


def g4_integrated_spectrum(
    material: str,
    t0_mev: float,
) -> dict[str, Any]:
    """Return angle-integrated spectrum from Geant4 batch data.

    Integrates I(k, theta) * 2*pi*sin(theta)*d_theta over the full 0-180 deg range.
    Result units: MeV / (MeV * electron) — the sr factor is integrated out.

    Returns dict with keys:
        photon_energy_mev, intensity, n_events
    """
    lut = get_g4_lookup()
    mat_idx = _material_index(material)
    theta_edges = lut["theta_bin_edges"]

    exact_idx = _energy_index(t0_mev)
    if exact_idx is not None:
        grid = lut["intensity"][mat_idx, exact_idx, :, :].copy()  # (36, 40)
        k_centers, _ = g4_photon_energy_bins(float(lut["energies_mev"][exact_idx]))
    else:
        idx_lo, idx_hi, weight = _energy_bracket(t0_mev)
        g_lo = lut["intensity"][mat_idx, idx_lo, :, :]
        g_hi = lut["intensity"][mat_idx, idx_hi, :, :]
        grid = (1.0 - weight) * g_lo + weight * g_hi
        k_centers, _ = g4_photon_energy_bins(t0_mev)

    # Solid angle of each theta bin: delta_omega = 2*pi*|cos(lo) - cos(hi)|
    cos_lo = np.cos(np.radians(theta_edges[:-1]))
    cos_hi = np.cos(np.radians(theta_edges[1:]))
    solid_angles = 2.0 * np.pi * np.abs(cos_lo - cos_hi)  # (36,)

    # Integrate: I_int(k) = sum_theta I(k,theta) * delta_omega(theta)
    integrated = np.einsum("tk,t->k", grid, solid_angles)  # (40,)

    mask = integrated > 0
    return {
        "photon_energy_mev": k_centers[mask].tolist(),
        "intensity": integrated[mask].tolist(),
        "n_events": lut["n_events"],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_interp(
    x_new: NDArray[np.float64],
    x_old: NDArray[np.float64],
    y_old: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Interpolate in log-space, handling zeros gracefully."""
    # Work in log-space for better accuracy on log-spaced grids
    safe_y = np.maximum(y_old, 1e-30)
    log_y = np.log(safe_y)
    log_interp = np.interp(np.log(x_new), np.log(x_old), log_y)
    result = np.exp(log_interp)
    # Zero out values that were originally zero
    result[result < 1e-25] = 0.0
    return np.asarray(result, dtype=np.float64)
