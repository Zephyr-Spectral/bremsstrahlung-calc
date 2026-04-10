"""Bremsstrahlung spectrum calculation API endpoints."""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query

import config
from server.api import resolve_material
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
    angle_deg: float = Query(ge=0.0, le=90.0, default=0.0, description="Detection angle (deg)"),
    beam_current_ua: float = Query(
        ge=0.0,
        default=0.0,
        description="Beam current in uA (0=per electron)",
    ),
    mode: str = Query(default="both", description="calculated, interpolated, or both"),
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

    if mode in ("calculated", "both"):
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

    if mode in ("interpolated", "both") and material in config.NASA_MATERIALS:
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

    return result


@router.get("/angular")
async def angular_distribution(
    material: str = Query(description="Material symbol"),
    electron_energy_mev: float = Query(ge=0.1, le=10.0),
    photon_energy_mev: float = Query(gt=0.0, description="Photon energy in MeV"),
    n_angles: int = Query(ge=5, le=91, default=19),
) -> dict[str, object]:
    """Compute intensity vs detection angle at fixed photon energy."""
    mat = resolve_material(material)
    z = int(mat["Z"])
    a_val = float(mat["A"])
    density = float(mat["density"])

    angles = list(np.linspace(0.0, 90.0, n_angles))
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

    return {
        "parameters": {
            "material": material,
            "electron_energy_mev": electron_energy_mev,
            "photon_energy_mev": photon_energy_mev,
        },
        "angles_deg": angles,
        "intensity": intensities,
    }


@router.get("/integrated")
async def integrated_spectrum(
    material: str = Query(description="Material symbol"),
    electron_energy_mev: float = Query(ge=0.1, le=10.0),
    n_points: int = Query(ge=10, le=200, default=30),
) -> dict[str, object]:
    """Compute angle-integrated bremsstrahlung spectrum."""
    mat = resolve_material(material)
    z = int(mat["Z"])
    a_val = float(mat["A"])
    density = float(mat["density"])

    k, intensity = angle_integrated_spectrum(
        electron_energy_mev,
        z,
        a_val,
        density,
        material_symbol=material if material in config.NASA_MATERIALS else None,
        n_photon_points=n_points,
        n_angle_points=10,
        n_slabs=30,
    )

    return {
        "parameters": {"material": material, "electron_energy_mev": electron_energy_mev},
        "photon_energy_mev": k,
        "intensity": intensity,
    }


@router.get("/compare")
async def compare_materials(
    materials: str = Query(description="Comma-separated material symbols"),
    electron_energy_mev: float = Query(ge=0.1, le=10.0),
    angle_deg: float = Query(ge=0.0, le=90.0, default=0.0),
    n_points: int = Query(ge=10, le=200, default=30),
) -> dict[str, object]:
    """Compare bremsstrahlung spectra across multiple materials."""
    mat_list = [m.strip() for m in materials.split(",")]
    if len(mat_list) > 5:
        raise HTTPException(status_code=400, detail="Maximum 5 materials for comparison")

    results: dict[str, dict[str, list[float]]] = {}
    for symbol in mat_list:
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
        results[symbol] = {"photon_energy_mev": k, "intensity": intensity}

    return {
        "parameters": {
            "materials": mat_list,
            "electron_energy_mev": electron_energy_mev,
            "angle_deg": angle_deg,
        },
        "spectra": results,
    }


@router.get("/heatmap")
async def heatmap_spectrum(
    material: str = Query(description="Material symbol"),
    electron_energy_mev: float = Query(ge=0.1, le=10.0),
    n_points: int = Query(ge=5, le=100, default=20, description="Photon energy points"),
    n_angles: int = Query(ge=3, le=19, default=9, description="Angle grid points"),
) -> dict[str, object]:
    """2-D intensity grid: photon energy vs detection angle."""
    mat = resolve_material(material)
    z = int(mat["Z"])
    a_val = float(mat["A"])
    density = float(mat["density"])

    angles = list(np.linspace(0.0, 90.0, n_angles))
    k_min = electron_energy_mev * config.THICK_SPECTRUM_K_FRACTION_MIN
    k_max = electron_energy_mev * config.THICK_SPECTRUM_K_FRACTION_MAX
    photon_energies = list(np.logspace(np.log10(k_min), np.log10(k_max), n_points))
    mat_sym: str | None = material if material in config.NASA_MATERIALS else None

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
        "parameters": {"material": material, "electron_energy_mev": electron_energy_mev},
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
