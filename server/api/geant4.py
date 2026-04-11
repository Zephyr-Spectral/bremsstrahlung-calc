"""Geant4 batch lookup API endpoints.

Serves pre-computed spectra from the 10M-event batch simulation campaign
via the geant4_lookup.npz lookup table.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query

from server.data_access_geant4 import (
    g4_angular_distribution,
    g4_heatmap,
    g4_info,
    g4_spectrum,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/geant4", tags=["geant4"])


@router.get("/spectrum")
async def geant4_spectrum(
    material: str = Query(description="Material symbol (e.g., Fe, Cu, W)"),
    electron_energy_mev: float = Query(ge=0.5, le=10.0, description="Electron energy in MeV"),
    angle_deg: float = Query(ge=0.0, le=180.0, default=0.0, description="Detection angle (deg)"),
) -> dict[str, object]:
    """Geant4 batch spectrum at given material, energy, and angle."""
    try:
        result = g4_spectrum(material, electron_energy_mev, angle_deg)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "parameters": {
            "material": material,
            "electron_energy_mev": electron_energy_mev,
            "angle_deg": angle_deg,
        },
        **result,
    }


@router.get("/angular")
async def geant4_angular(
    material: str = Query(description="Material symbol"),
    electron_energy_mev: float = Query(ge=0.5, le=10.0),
    photon_energy_mev: float = Query(gt=0.0, description="Photon energy in MeV"),
) -> dict[str, object]:
    """Angular distribution from Geant4 batch data at fixed photon energy."""
    try:
        result = g4_angular_distribution(material, electron_energy_mev, photon_energy_mev)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "parameters": {
            "material": material,
            "electron_energy_mev": electron_energy_mev,
            "photon_energy_mev": photon_energy_mev,
        },
        **result,
    }


@router.get("/heatmap")
async def geant4_heatmap_view(
    material: str = Query(description="Material symbol"),
    electron_energy_mev: float = Query(ge=0.5, le=10.0),
) -> dict[str, object]:
    """Full 2D intensity grid (theta x photon energy) from Geant4 batch data."""
    try:
        result = g4_heatmap(material, electron_energy_mev)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return {
        "parameters": {
            "material": material,
            "electron_energy_mev": electron_energy_mev,
        },
        **result,
    }


@router.get("/info")
async def geant4_lookup_info() -> dict[str, object]:
    """Return metadata about the Geant4 batch lookup table."""
    try:
        return g4_info()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
