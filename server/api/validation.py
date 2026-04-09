"""NASA data validation API endpoints."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

import config
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
    z = mat["Z"]
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
