"""API routers for the bremsstrahlung calculator."""

from __future__ import annotations

from fastapi import HTTPException

import config


def resolve_material(symbol: str) -> config.MaterialProperties:
    """Look up material properties by symbol, raising HTTP 404 if not found.

    Args:
        symbol: Material symbol (e.g. 'Cu', 'Al', 'W', 'SS304').

    Returns:
        Material properties dict with Z, A, density, name.

    Raises:
        HTTPException: 404 if symbol not in ALL_MATERIALS.
    """
    if symbol in config.ALL_MATERIALS:
        return config.ALL_MATERIALS[symbol]
    raise HTTPException(
        status_code=404,
        detail=f"Unknown material '{symbol}'. Available: {list(config.ALL_MATERIALS.keys())}",
    )
