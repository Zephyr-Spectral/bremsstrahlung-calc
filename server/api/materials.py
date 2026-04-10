"""Material property API endpoints."""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException, Query

import config
from server.physics.electron_range import csda_range
from server.physics.stopping_power import total_stopping_power

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/materials", tags=["materials"])


@router.get("")
async def list_materials() -> dict[str, object]:
    """List all available materials with properties."""
    materials = []
    for symbol, props in config.ALL_MATERIALS.items():
        materials.append(
            {
                "symbol": symbol,
                "name": props["name"],
                "Z": props["Z"],
                "A": props["A"],
                "density": props["density"],
                "is_nasa": symbol in config.NASA_MATERIALS,
            }
        )
    return {"materials": materials, "count": len(materials)}


@router.get("/{symbol}/stopping-power")
async def stopping_power_curve(
    symbol: str,
    n_points: int = Query(ge=5, le=200, default=50),
) -> dict[str, object]:
    """Compute stopping power vs electron energy for a material."""
    mat = _resolve_material(symbol)
    z = int(mat["Z"])
    a_val = float(mat["A"])

    energies = list(
        np.logspace(
            np.log10(config.MIN_ELECTRON_ENERGY_MEV),
            np.log10(config.MAX_ELECTRON_ENERGY_MEV),
            n_points,
        )
    )
    stopping = [total_stopping_power(e, z, a_val) for e in energies]

    return {
        "material": symbol,
        "electron_energy_mev": energies,
        "stopping_power_mev_cm2_g": stopping,
    }


@router.get("/{symbol}/range")
async def range_curve(
    symbol: str,
    n_points: int = Query(ge=5, le=50, default=20),
) -> dict[str, object]:
    """Compute electron CSDA range vs energy for a material."""
    mat = _resolve_material(symbol)
    z = int(mat["Z"])
    a_val = float(mat["A"])

    energies = list(
        np.logspace(
            np.log10(config.MIN_ELECTRON_ENERGY_MEV),
            np.log10(config.MAX_ELECTRON_ENERGY_MEV),
            n_points,
        )
    )
    ranges = [csda_range(e, z, a_val, n_steps=200) for e in energies]

    return {
        "material": symbol,
        "electron_energy_mev": energies,
        "range_g_cm2": ranges,
    }


def _resolve_material(symbol: str) -> config.MaterialProperties:
    """Look up material properties, raising 404 if not found."""
    if symbol in config.ALL_MATERIALS:
        return config.ALL_MATERIALS[symbol]
    raise HTTPException(
        status_code=404,
        detail=f"Unknown material '{symbol}'. Available: {list(config.ALL_MATERIALS.keys())}",
    )
