"""NASA data validation API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

import config
from server.data_access_experimental import get_experimental_spectrum, list_experimental_data
from server.data_access_geant4 import g4_spectrum
from server.physics.interpolation import get_nasa_spectrum_at_grid_point
from server.physics.thick_target import thick_target_spectrum

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/validation", tags=["validation"])


@router.get("/nasa-comparison")
async def nasa_comparison(
    material: str = Query(description="NASA material symbol"),
    electron_energy_mev: float = Query(description="0.50, 0.75, 1.00, 2.00, or 3.00"),
    angle_deg: float = Query(description="0, 30, or 60"),
) -> dict[str, object]:
    """Compare calculated spectrum against NASA tabulated data."""
    if material not in config.NASA_MATERIALS:
        raise HTTPException(status_code=404, detail=f"Material '{material}' not in NASA data")

    mat = config.NASA_MATERIALS[material]
    z = int(mat["Z"])
    a_val = float(mat["A"])
    density = float(mat["density"])

    # Get NASA data
    k_nasa, i_nasa = get_nasa_spectrum_at_grid_point(
        electron_energy_mev,
        angle_deg,
        material,
    )
    if not k_nasa:
        raise HTTPException(status_code=404, detail="No NASA data at this grid point")

    # Calculate at NASA photon energies
    k_calc, i_calc = thick_target_spectrum(
        electron_energy_mev,
        angle_deg,
        z,
        a_val,
        density,
        material_symbol=material,
        n_points=len(k_nasa),
        n_slabs=50,
    )

    return {
        "parameters": {
            "material": material,
            "electron_energy_mev": electron_energy_mev,
            "angle_deg": angle_deg,
        },
        "nasa": {"photon_energy_mev": k_nasa, "intensity": i_nasa},
        "calculated": {"photon_energy_mev": k_calc, "intensity": i_calc},
    }


@router.get("/geant4-comparison")
async def geant4_comparison(
    material: str = Query(description="Material symbol"),
    electron_energy_mev: float = Query(ge=0.5, le=10.0, description="Electron energy in MeV"),
    angle_deg: float = Query(ge=0.0, le=180.0, default=0.0, description="Detection angle (deg)"),
) -> dict[str, object]:
    """Three-way comparison: Geant4 batch vs Calculated vs NASA (if available)."""
    if material not in config.ALL_MATERIALS:
        raise HTTPException(status_code=404, detail=f"Unknown material '{material}'")

    mat = config.ALL_MATERIALS[material]
    z = int(mat["Z"])
    a_val = float(mat["A"])
    density = float(mat["density"])

    result: dict[str, object] = {
        "parameters": {
            "material": material,
            "electron_energy_mev": electron_energy_mev,
            "angle_deg": angle_deg,
        },
    }

    # Geant4 batch data
    try:
        g4_data = g4_spectrum(material, electron_energy_mev, angle_deg)
        result["geant4"] = {
            "photon_energy_mev": g4_data["photon_energy_mev"],
            "intensity": g4_data["intensity"],
            "uncertainty": g4_data["uncertainty"],
            "n_events": g4_data["n_events"],
            "is_interpolated": g4_data["is_interpolated"],
        }
    except (ValueError, FileNotFoundError) as exc:
        log.warning("Geant4 data unavailable for %s: %s", material, exc)

    # Semi-analytical calculation (at Geant4 photon energies if available)
    n_calc = 50
    k_calc, i_calc = thick_target_spectrum(
        electron_energy_mev,
        angle_deg,
        z,
        a_val,
        density,
        material_symbol=material if material in config.NASA_MATERIALS else None,
        n_points=n_calc,
        n_slabs=50,
    )
    result["calculated"] = {"photon_energy_mev": k_calc, "intensity": i_calc}

    # NASA tabulated data (only at grid points)
    if material in config.NASA_MATERIALS:
        try:
            k_nasa, i_nasa = get_nasa_spectrum_at_grid_point(
                electron_energy_mev, angle_deg, material
            )
            if k_nasa:
                result["nasa"] = {"photon_energy_mev": k_nasa, "intensity": i_nasa}
        except Exception:
            log.debug(
                "No NASA data at grid point %s %.2f MeV %.0f deg",
                material,
                electron_energy_mev,
                angle_deg,
            )

    # Experimental data (Dance et al. 1968)
    exp_data = get_experimental_spectrum(material, electron_energy_mev, angle_deg)
    if exp_data is not None:
        result["experimental"] = exp_data

    return result


@router.get("/experimental-data")
async def experimental_data_list() -> dict[str, object]:
    """List all available experimental validation data points."""
    available = list_experimental_data()
    return {
        "count": len(available),
        "datasets": available,
    }
