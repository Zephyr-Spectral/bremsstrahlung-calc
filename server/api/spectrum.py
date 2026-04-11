"""Bremsstrahlung spectrum calculation API endpoints."""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query

import config
from server.api import resolve_material
from server.data_access_experimental import get_experimental_spectrum
from server.data_access_geant4 import (
    g4_angular_distribution,
    g4_heatmap,
    g4_integrated_spectrum,
    g4_spectrum,
)
from server.physics.attenuation import absorption_edges
from server.physics.interpolation import (
    get_nasa_spectrum_at_grid_point,
    interpolate_nasa_spectrum,
)
from server.physics.thick_target import (
    angle_integrated_spectrum,
    intensity_to_photon_rate,
    thick_target_intensity,
    thick_target_spectrum,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/spectrum", tags=["spectrum"])


@router.get("/calculate")
async def calculate_spectrum(
    material: str = Query(description="Material symbol (e.g., Cu, Al, W)"),
    electron_energy_mev: float = Query(ge=0.1, le=10.0, description="Electron energy in MeV"),
    angle_deg: float = Query(ge=0.0, le=180.0, default=0.0, description="Detection angle (deg)"),
    beam_current_ua: float = Query(
        ge=0.0,
        default=0.0,
        description="Beam current in uA (0=per electron)",
    ),
    mode: str = Query(
        default="both",
        description="calculated, interpolated, geant4, monte_carlo, all, or both",
    ),
    n_points: int = Query(ge=10, le=500, default=50, description="Number of photon energy points"),
) -> dict[str, object]:
    """Compute bremsstrahlung energy spectrum at a fixed detection angle."""
    mat = resolve_material(material)
    z = int(mat["Z"])
    a_val = float(mat["A"])
    density = float(mat["density"])

    result: dict[str, object] = {
        "parameters": {
            "material": material,
            "electron_energy_mev": electron_energy_mev,
            "angle_deg": angle_deg,
            "beam_current_ua": beam_current_ua,
            "mode": mode,
        },
    }

    if mode in ("calculated", "both", "all"):
        k_calc, i_calc = thick_target_spectrum(
            electron_energy_mev,
            angle_deg,
            z,
            a_val,
            density,
            material_symbol=material if material in config.NASA_MATERIALS else None,
            n_points=n_points,
            n_slabs=50,
        )
        result["calculated"] = {
            "photon_energy_mev": k_calc,
            "intensity": i_calc,
        }
        if beam_current_ua > 0:
            result["calculated"]["photon_rate"] = [  # type: ignore[index]
                intensity_to_photon_rate(i, beam_current_ua) for i in i_calc
            ]

    if mode in ("interpolated", "both", "all") and material in config.NASA_MATERIALS:
        k_interp, i_interp = interpolate_nasa_spectrum(
            electron_energy_mev,
            angle_deg,
            material,
        )
        result["interpolated"] = {
            "photon_energy_mev": k_interp,
            "intensity": i_interp,
        }
        if beam_current_ua > 0:
            result["interpolated"]["photon_rate"] = [  # type: ignore[index]
                intensity_to_photon_rate(i, beam_current_ua) for i in i_interp
            ]

    if mode in ("geant4", "monte_carlo", "all"):
        try:
            g4_data = g4_spectrum(material, electron_energy_mev, angle_deg)
            result["geant4"] = {
                "photon_energy_mev": g4_data["photon_energy_mev"],
                "intensity": g4_data["intensity"],
                "uncertainty": g4_data["uncertainty"],
                "n_events": g4_data["n_events"],
                "status": "batch_lookup",
                "is_interpolated": g4_data["is_interpolated"],
            }
        except (ValueError, FileNotFoundError) as exc:
            log.warning("Geant4 batch lookup failed: %s", exc)

    # Experimental data overlay (Dance et al. 1968)
    exp_data = get_experimental_spectrum(material, electron_energy_mev, angle_deg)
    if exp_data is not None:
        result["experimental"] = exp_data

    return result


@router.get("/angular")
async def angular_distribution(
    material: str = Query(description="Material symbol"),
    electron_energy_mev: float = Query(ge=0.1, le=10.0),
    photon_energy_mev: float = Query(gt=0.0, description="Photon energy in MeV"),
    n_angles: int = Query(ge=5, le=91, default=19),
    mode: str = Query(
        default="both",
        description="calculated, geant4, all, or both",
    ),
) -> dict[str, object]:
    """Compute intensity vs detection angle at fixed photon energy."""
    mat = resolve_material(material)
    z = int(mat["Z"])
    a_val = float(mat["A"])
    density = float(mat["density"])

    result: dict[str, object] = {
        "parameters": {
            "material": material,
            "electron_energy_mev": electron_energy_mev,
            "photon_energy_mev": photon_energy_mev,
        },
    }

    if mode in ("calculated", "both", "all"):
        angles = list(np.linspace(0.0, 180.0, n_angles))
        intensities = [
            thick_target_intensity(
                electron_energy_mev,
                photon_energy_mev,
                angle,
                z,
                a_val,
                density,
                material_symbol=material if material in config.NASA_MATERIALS else None,
                n_slabs=50,
            )
            for angle in angles
        ]
        result["calculated"] = {"angles_deg": angles, "intensity": intensities}

    if mode in ("geant4", "monte_carlo", "all"):
        try:
            g4_data = g4_angular_distribution(material, electron_energy_mev, photon_energy_mev)
            result["geant4"] = {
                "angles_deg": g4_data["angles_deg"],
                "intensity": g4_data["intensity"],
                "uncertainty": g4_data["uncertainty"],
                "n_events": g4_data["n_events"],
                "k_bin_center_mev": g4_data["k_bin_center_mev"],
            }
        except (ValueError, FileNotFoundError) as exc:
            log.warning("Geant4 angular lookup failed: %s", exc)

    return result


@router.get("/integrated")
async def integrated_spectrum(
    material: str = Query(description="Material symbol"),
    electron_energy_mev: float = Query(ge=0.1, le=10.0),
    n_points: int = Query(ge=10, le=200, default=30),
    mode: str = Query(
        default="both",
        description="calculated, interpolated, geant4, all, or both",
    ),
) -> dict[str, object]:
    """Compute angle-integrated bremsstrahlung spectrum."""
    mat = resolve_material(material)
    z = int(mat["Z"])
    a_val = float(mat["A"])
    density = float(mat["density"])

    result: dict[str, object] = {
        "parameters": {"material": material, "electron_energy_mev": electron_energy_mev},
    }

    if mode in ("calculated", "both", "all"):
        k, intensity = angle_integrated_spectrum(
            electron_energy_mev,
            z,
            a_val,
            density,
            material_symbol=material if material in config.NASA_MATERIALS else None,
            n_photon_points=n_points,
            n_angle_points=19,
            n_slabs=30,
            max_angle_deg=180.0,
        )
        result["calculated"] = {"photon_energy_mev": k, "intensity": intensity}

    if mode in ("geant4", "monte_carlo", "all"):
        try:
            g4_data = g4_integrated_spectrum(material, electron_energy_mev)
            result["geant4"] = {
                "photon_energy_mev": g4_data["photon_energy_mev"],
                "intensity": g4_data["intensity"],
                "n_events": g4_data["n_events"],
                "status": "batch_lookup",
            }
        except (ValueError, FileNotFoundError) as exc:
            log.warning("Geant4 integrated lookup failed: %s", exc)

    return result


@router.get("/compare")
async def compare_materials(
    materials: str = Query(description="Comma-separated material symbols"),
    electron_energy_mev: float = Query(ge=0.1, le=10.0),
    angle_deg: float = Query(ge=0.0, le=180.0, default=0.0),
    n_points: int = Query(ge=10, le=200, default=30),
    mode: str = Query(
        default="both",
        description="calculated, geant4, all, or both",
    ),
) -> dict[str, object]:
    """Compare bremsstrahlung spectra across multiple materials."""
    mat_list = [m.strip() for m in materials.split(",")]
    if len(mat_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 materials for comparison")

    calc_results: dict[str, dict[str, list[float]]] = {}
    g4_results: dict[str, dict[str, object]] = {}

    for symbol in mat_list:
        if mode in ("calculated", "both", "all"):
            mat = resolve_material(symbol)
            z = int(mat["Z"])
            a_val = float(mat["A"])
            density = float(mat["density"])

            k, intensity = thick_target_spectrum(
                electron_energy_mev,
                angle_deg,
                z,
                a_val,
                density,
                material_symbol=symbol if symbol in config.NASA_MATERIALS else None,
                n_points=n_points,
                n_slabs=30,
            )
            calc_results[symbol] = {"photon_energy_mev": k, "intensity": intensity}

        if mode in ("geant4", "monte_carlo", "all"):
            try:
                g4_data = g4_spectrum(symbol, electron_energy_mev, angle_deg)
                g4_results[symbol] = {
                    "photon_energy_mev": g4_data["photon_energy_mev"],
                    "intensity": g4_data["intensity"],
                    "uncertainty": g4_data["uncertainty"],
                    "n_events": g4_data["n_events"],
                }
            except (ValueError, FileNotFoundError):
                pass

    result: dict[str, object] = {
        "parameters": {
            "materials": mat_list,
            "electron_energy_mev": electron_energy_mev,
            "angle_deg": angle_deg,
        },
    }
    if calc_results:
        result["spectra"] = calc_results
    if g4_results:
        result["geant4_spectra"] = g4_results

    return result


@router.get("/heatmap")
async def heatmap_spectrum(
    material: str = Query(description="Material symbol"),
    electron_energy_mev: float = Query(ge=0.1, le=10.0),
    n_points: int = Query(ge=5, le=100, default=20, description="Photon energy points"),
    n_angles: int = Query(ge=3, le=37, default=9, description="Angle grid points"),
    mode: str = Query(
        default="calculated",
        description="calculated or geant4",
    ),
) -> dict[str, object]:
    """2-D intensity grid: photon energy vs detection angle."""
    if mode in ("geant4", "monte_carlo"):
        try:
            g4_data = g4_heatmap(material, electron_energy_mev)
            return {
                "parameters": {
                    "material": material,
                    "electron_energy_mev": electron_energy_mev,
                    "source": "geant4",
                },
                "photon_energy_mev": g4_data["photon_energy_mev"],
                "angles_deg": g4_data["angles_deg"],
                "intensity": g4_data["intensity"],
            }
        except (ValueError, FileNotFoundError) as exc:
            log.warning("Geant4 heatmap failed, falling back to calculated: %s", exc)

    mat = resolve_material(material)
    z = int(mat["Z"])
    a_val = float(mat["A"])
    density = float(mat["density"])

    angles = list(np.linspace(0.0, 180.0, n_angles))
    k_min = electron_energy_mev * config.THICK_SPECTRUM_K_FRACTION_MIN
    k_max = electron_energy_mev * config.THICK_SPECTRUM_K_FRACTION_MAX
    photon_energies = list(np.logspace(np.log10(k_min), np.log10(k_max), n_points))
    mat_sym: str | None = material if material in config.NASA_MATERIALS else None

    # Inject edge-adjacent points for K/L/M edge resolution
    if mat_sym is not None:
        edge_delta = 1e-5
        for e_edge in absorption_edges(mat_sym):
            if k_min < e_edge < k_max:
                photon_energies.append(e_edge - edge_delta)
                photon_energies.append(e_edge + edge_delta)
        photon_energies = sorted(set(photon_energies))

    intensity_grid: list[list[float]] = [
        [
            thick_target_intensity(
                electron_energy_mev,
                k,
                angle,
                z,
                a_val,
                density,
                material_symbol=mat_sym,
                n_slabs=30,
            )
            for k in photon_energies
        ]
        for angle in angles
    ]

    return {
        "parameters": {
            "material": material,
            "electron_energy_mev": electron_energy_mev,
            "source": "calculated",
        },
        "photon_energy_mev": photon_energies,
        "angles_deg": angles,
        "intensity": intensity_grid,
    }


@router.get("/nasa-data")
async def nasa_grid_data(
    material: str = Query(description="NASA material symbol"),
    electron_energy_mev: float = Query(description="Must be one of: 0.50, 0.75, 1.00, 2.00, 3.00"),
    angle_deg: float = Query(description="Must be one of: 0, 30, 60"),
) -> dict[str, object]:
    """Get raw NASA tabulated data at exact grid point."""
    if material not in config.NASA_MATERIALS:
        raise HTTPException(status_code=404, detail=f"Material '{material}' not in NASA data")

    k, intensity = get_nasa_spectrum_at_grid_point(
        electron_energy_mev,
        angle_deg,
        material,
    )

    if not k:
        raise HTTPException(status_code=404, detail="No data at this grid point")

    return {
        "parameters": {
            "material": material,
            "electron_energy_mev": electron_energy_mev,
            "angle_deg": angle_deg,
        },
        "photon_energy_mev": k,
        "intensity": intensity,
    }
