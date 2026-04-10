"""Geant4 thick-target bremsstrahlung interface.

Runs the compiled thick_target_brems executable via subprocess, parses
the per-photon CSV output, and bins into an energy spectrum at the
requested detection angle.

The executable uses G4EmStandardPhysics_option4 (Seltzer-Berger model
with best low-energy EM physics).  Environment variables for Geant4
data paths are set via the g4env.sh wrapper script.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import Any

import numpy as np

import config
from server.monte_carlo.cache import get_cached, save_cache
from server.physics.electron_range import csda_range

log = logging.getLogger(__name__)

_ANGLE_HALF_WIDTH: float = 5.0  # +/- degrees for angle binning


def _parse_g4_csv(csv_text: str) -> tuple[list[float], list[float]]:
    """Parse Geant4 CSV output into (k_mev, theta_deg) lists."""
    k_vals: list[float] = []
    theta_vals: list[float] = []
    for line in csv_text.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split(",")
        if len(parts) >= 2:  # k_MeV, theta_deg
            try:
                k_vals.append(float(parts[0]))
                theta_vals.append(float(parts[1]))
            except ValueError:
                continue
    return k_vals, theta_vals


def _bin_spectrum(
    k_all: list[float],
    theta_all: list[float],
    angle_deg: float,
    t0_mev: float,
    n_events: int,
    n_bins: int = 30,
) -> tuple[list[float], list[float]]:
    """Bin photons into energy spectrum at a specific detection angle.

    Selects photons within +/- ANGLE_HALF_WIDTH of the detection angle,
    histograms by energy, and converts counts to intensity units
    [MeV / (MeV sr electron)].
    """
    k_arr = np.array(k_all)
    theta_arr = np.array(theta_all)

    # Angle selection
    ang_lo = max(angle_deg - _ANGLE_HALF_WIDTH, 0.0)
    ang_hi = angle_deg + _ANGLE_HALF_WIDTH
    mask = (theta_arr >= ang_lo) & (theta_arr < ang_hi)

    k_sel = k_arr[mask]
    if len(k_sel) == 0:
        return [], []

    # Solid angle of the angular bin
    cos_lo = np.cos(np.radians(ang_lo))
    cos_hi = np.cos(np.radians(ang_hi))
    solid_angle = 2.0 * np.pi * abs(cos_lo - cos_hi)

    # Energy bins (log-spaced from 5% to 95% of T0)
    k_min = max(0.05 * t0_mev, 0.02)
    k_max = 0.95 * t0_mev
    bin_edges = np.logspace(np.log10(k_min), np.log10(k_max), n_bins + 1)
    counts, edges = np.histogram(k_sel, bins=bin_edges)

    k_centers = 0.5 * (edges[:-1] + edges[1:])
    dk = np.diff(edges)

    # Intensity: I(k) = k * counts / (N_electrons * dk * solid_angle)
    with np.errstate(divide="ignore", invalid="ignore"):
        intensity = k_centers * counts / (n_events * dk * solid_angle)

    # Filter out zero-count bins
    nonzero = counts > 0
    return k_centers[nonzero].tolist(), intensity[nonzero].tolist()


def _validate_g4_inputs(material: str) -> dict[str, Any] | None:
    """Validate Geant4 inputs, returning an error dict or None if valid."""
    if config.G4_MATERIAL_NAMES.get(material) is None:
        return {"status": "error", "message": f"No Geant4 material mapping for {material}"}
    if not config.GEANT4_EXE.exists():
        return {"status": "error", "message": "Geant4 executable not found"}
    if not config.GEANT4_ENV_SCRIPT.exists():
        return {"status": "error", "message": "Geant4 env script not found"}
    mat_props = config.NASA_MATERIALS.get(material, {})
    if int(mat_props.get("Z", 0)) == 0:
        return {"status": "error", "message": f"Unknown material {material}"}
    return None


def _execute_g4(material: str, t0_mev: float, n_events: int, timeout: int) -> dict[str, Any]:
    """Execute Geant4 subprocess and return raw result dict."""
    g4_name = config.G4_MATERIAL_NAMES[material]
    mat_props = config.NASA_MATERIALS[material]
    z, a = int(mat_props["Z"]), float(mat_props["A"])
    density = float(mat_props["density"])
    thickness_cm = csda_range(t0_mev, z, a) / density

    cmd = [
        str(config.GEANT4_ENV_SCRIPT),
        str(config.GEANT4_EXE),
        g4_name,
        str(t0_mev),
        str(n_events),
        f"{thickness_cm:.4f}",
    ]

    log.info("Running Geant4: %s %s %.1f MeV, %d events", material, g4_name, t0_mev, n_events)
    t_start = time.perf_counter()

    try:
        proc = subprocess.run(  # noqa: S603 — inputs are from config, not user
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(config.GEANT4_EXE.parent),
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": f"Geant4 timed out after {timeout}s"}

    runtime = time.perf_counter() - t_start
    if proc.returncode != 0:
        return {
            "status": "error",
            "message": f"Geant4 exit code {proc.returncode}",
            "stderr": proc.stderr[:500] if proc.stderr else "",
        }

    k_all, theta_all = _parse_g4_csv(proc.stdout)
    if not k_all:
        return {"status": "error", "message": "No photons in Geant4 output"}

    return {
        "k_all": k_all,
        "theta_all": theta_all,
        "n_events": n_events,
        "n_photons": len(k_all),
        "runtime_seconds": round(runtime, 2),
        "status": "ok",
    }


def run_geant4(
    material: str,
    t0_mev: float,
    angle_deg: float,
    n_events: int = config.MC_DEFAULT_EVENTS,
    timeout: int = config.MC_TIMEOUT_SECONDS,
    use_cache: bool = True,
) -> dict[str, Any]:
    """Run Geant4 thick-target bremsstrahlung simulation.

    Returns a dict with keys:
        photon_energy_mev, intensity, n_events, n_photons,
        runtime_seconds, status ("ok" | "error" | "cached")
    """
    if use_cache:
        cached = get_cached("geant4", material, t0_mev, angle_deg)
        if cached is not None:
            cached["status"] = "cached"
            return cached

    err = _validate_g4_inputs(material)
    if err is not None:
        return err

    raw = _execute_g4(material, t0_mev, n_events, timeout)
    if raw["status"] != "ok":
        return raw

    k_spec, i_spec = _bin_spectrum(raw["k_all"], raw["theta_all"], angle_deg, t0_mev, n_events)

    result: dict[str, Any] = {
        "photon_energy_mev": k_spec,
        "intensity": i_spec,
        "n_events": raw["n_events"],
        "n_photons": raw["n_photons"],
        "runtime_seconds": raw["runtime_seconds"],
        "status": "ok",
    }

    if use_cache:
        save_cache("geant4", material, t0_mev, angle_deg, result)

    return result
